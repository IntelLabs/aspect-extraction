from typing import Dict
import logging
from torch.nn import CrossEntropyLoss
import torch

from torch.nn.modules.loss import BCELoss
from transformers import (
    RobertaForMaskedLM,
    AdamW,
    get_scheduler,
    default_data_collator
)
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput
from datasets import load_metric

from tqdm.auto import tqdm, trange
from torch.utils.data.dataloader import DataLoader
import math
import json
from absa_utils import AbsaPVP, DataCollatorForPatternLanguageModeling, compute_metrics, get_data_collator, get_labels

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
    
def predict(model, args, is_few_shot, test_dataset, split, data_collator, label_list, accelerator):
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    device = accelerator.device
    model.eval()
    
    metric = load_metric("seqeval")
    num_test_steps = int(len(test_dataset) / args.per_device_eval_batch_size)
    # progress_bar = tqdm(range(num_test_steps))
    all_preds = []

    with tqdm(range(num_test_steps)) as progress_bar:
        for step, batch in enumerate(test_dataloader):
            batch = {k: v.cpu().to(device=device) for k, v in batch.items()}
            with torch.no_grad():
                input_kwargs = dict(labeled_batch=batch) if is_few_shot else batch

                # Forward call
                outputs = model(**input_kwargs)

            preds = outputs.logits
            
            if not is_few_shot:
                preds = preds.argmax(dim=-1)

            preds = accelerator.gather(preds)

            if not is_few_shot:
                labels = batch["labels"]
                labels_gathered = accelerator.gather(labels)
                preds, refs = get_labels(accelerator.device, label_list, preds, labels_gathered)
                metric.add_batch(
                    predictions=preds,
                    references=refs,
                )  # predictions and references are expected to be a nested list of labels, not label_ids
                all_preds.extend(preds)
            else:
                all_preds.extend(preds)

            progress_bar.update(1)
            
        eval_metric = {}
        if not is_few_shot:
            eval_metric = compute_metrics(metric, split)
            accelerator.print(f"\nEvaluation Metrics:", eval_metric)
        else:
            all_preds = torch.stack(all_preds)

        return all_preds, eval_metric

class Trainer:
    def __init__(self, model, args, training_args, train_dataset, unlabeled_dataset,
            accelerator, tokenizer, train_conf: dict=None) -> None:
        self.model = model
        self.args = args
        self.tr_args = training_args
        self.train_dataset = train_dataset
        self.accelerator = accelerator
        self.train_conf = train_conf
        self.tokenizer = tokenizer
        self.unlabeled_dataset = unlabeled_dataset
        self.unlabeled_dataloader = None
        self.unlabeled_iter = None
        self.progress_bar = None
        self.lm_training = False
        self.logging_steps = training_args.logging_steps
        self.label_training = False
        self.per_device_train_batch_size = self.tr_args.per_device_train_batch_size

        if train_conf is not None:
            self.lm_training = train_conf.get('lm', False)
            self.tr_args.learning_rate = train_conf['lr']
            self.tr_args.max_train_steps = train_conf['steps']

            if "batch_size" in train_conf:
                self.per_device_train_batch_size = train_conf["batch_size"]

        self.prepare()

    def prepare(self):
        if self.args.few_shot:
            self.label_training = self.train_conf['label_loss']
            self.train_dataset = self.train_dataset.remove_columns(['P_x'])
        else:
            self.label_training = True

        self.data_collator = get_data_collator(self.args.few_shot, self.tokenizer)
        total_batch_size = self.per_device_train_batch_size * self.accelerator.num_processes * \
            self.tr_args.gradient_accumulation_steps
        train_dataloader = DataLoader(
            self.train_dataset, collate_fn=default_data_collator, batch_size=self.per_device_train_batch_size
        )

        if self.args.few_shot and self.lm_training:
            lm_data_collator = DataCollatorForPatternLanguageModeling(
                tokenizer=self.tokenizer, mlm_probability=self.args.mlm_prob) 
                # ignore_pattern_mask=self.tr_phase_conf['lm_ignore'])

            # we need unlabeled data both for auxiliary language modeling and for knowledge distillation
            assert self.unlabeled_dataset is not None
            self.unlabeled_dataloader = DataLoader(self.unlabeled_dataset, collate_fn=lm_data_collator,
                batch_size=self.args.per_device_unlabeled_batch_size)
            self.unlabeled_iter = self.unlabeled_dataloader.__iter__()

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.tr_args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.tr_args.learning_rate)

        # Prepare everything with our `accelerator`.
        self.model, self.optimizer, self.train_dataloader, self.unlabeled_dataloader = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, self.unlabeled_dataloader
        )

        # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
        # shorter in multiprocess)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.tr_args.gradient_accumulation_steps)
        if self.tr_args.max_train_steps is None:
            self.tr_args.max_train_steps = int(self.tr_args.num_train_epochs * num_update_steps_per_epoch)
        else:
            self.tr_args.num_train_epochs = math.ceil(self.tr_args.max_train_steps / num_update_steps_per_epoch)

        self.lr_scheduler = get_scheduler(
            name=self.tr_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=self.tr_args.max_train_steps,
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples (train_dataset) = {len(self.train_dataset)}")
        if self.lm_training and self.unlabeled_dataset is not None:
            logger.info(f"  Num examples (unlabeled dataset) = {len(self.unlabeled_dataset)}")
        logger.info(f"  max_train_steps = {self.tr_args.max_train_steps}")
        logger.info(f"  Num Epochs = {self.tr_args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.tr_args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.tr_args.max_train_steps}")
        logger.info(f"  Total optimization steps = {self.tr_args.max_train_steps}")
        logger.info("")

    def train(self):
        step, global_step = 0, 0
        tr_loss, logging_loss = 0.0, 0.0
        max_steps = self.tr_args.max_train_steps
        logged_losses = []
        use_tqdm = False
        epochs = int(self.tr_args.num_train_epochs)

        train_iterator = trange(epochs, desc="Epoch") if use_tqdm else range(epochs)

        # Only show the progress bar once on each machine.
        # self.progress_bar = tqdm(range(self.tr_args.max_train_steps), disable=not self.accelerator.is_local_main_process,
        #     miniters=10)

        for epoch in train_iterator:
            epoch_iterator = tqdm(self.train_dataloader, desc="Iteration") if use_tqdm else \
                self.train_dataloader

            for _, batch in enumerate(epoch_iterator):
                self.model.train()

                labeled_batch = None
                if self.label_training:
                    labeled_batch = {k: v.cpu().to(device=self.accelerator.device) for k, v in batch.items()}

                unlabeled_batch, alpha = None, None
                if self.lm_training:
                    while unlabeled_batch is None:
                        try:
                            unlabeled_batch = self.unlabeled_iter.__next__()
                        except StopIteration:
                            logger.info("Resetting unlabeled dataset")
                            self.unlabeled_iter = self.unlabeled_dataloader.__iter__()

                    unlabeled_batch = {k: t.to(self.accelerator.device) for k, t in unlabeled_batch.items()}
                    
                    if 'alpha' in self.train_conf:
                        alpha = self.train_conf['alpha']

                if self.args.few_shot:
                    train_kwargs=dict(labeled_batch=labeled_batch, unlabeled_batch=unlabeled_batch, alpha=alpha)
                else:
                    train_kwargs=labeled_batch

                outputs = self.model(**train_kwargs)

                loss = outputs.loss
                loss = loss / self.tr_args.gradient_accumulation_steps
                self.accelerator.backward(loss)
                tr_loss += loss.item()
                
                if step % self.tr_args.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    global_step += 1
                    
                if self.logging_steps > 0 and global_step % self.logging_steps == 0:
                    loss_scalar = (tr_loss - logging_loss) / self.logging_steps
                    logs = {'learning_rate': self.lr_scheduler.get_last_lr(), 'loss': loss_scalar}
                    logged_losses.append(loss_scalar)
                    logging_loss = tr_loss
                    print(json.dumps({**logs, **{'step': global_step}}))

                if 0 < max_steps < global_step:
                    if use_tqdm:
                        epoch_iterator.close()
                    break
                step += 1

            if 0 < max_steps < global_step:
                if use_tqdm:
                    train_iterator.close()
                break

        if self.tr_args.output_dir is not None:
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(self.tr_args.output_dir, save_function=self.accelerator.save)
        
        avg_tr_loss = (tr_loss / global_step if global_step > 0 else -1)
        return logged_losses, avg_tr_loss
        
class RobertaForPatternMaskedLM(RobertaForMaskedLM):
    def set_pvp(self, pvp: AbsaPVP) -> None:
        self.pvp = pvp

    def forward(
        self,
        labeled_batch=None,
        unlabeled_batch=None,
        alpha=None,
        **kwargs,
    ):
        assert labeled_batch is not None or unlabeled_batch is not None
        logits = None

        # LM Loss
        lm_loss = None
        if unlabeled_batch is not None:
            mlm_labels = unlabeled_batch['labels']

            # Standard MLM
            if self.pvp.lm_method == "pet":
                inputs = self.generate_inputs(unlabeled_batch)
                outputs = self.roberta(**inputs)
                logits = self.lm_head(outputs[0])
                logits = logits.view(-1, self.config.vocab_size)
                lm_loss = CrossEntropyLoss()(logits, mlm_labels.view(-1))

            # Label Conditioning
            if self.pvp.lm_method == "adapet":
                masked_input_ids = unlabeled_batch['input_ids']
                input_ids = unlabeled_batch["input_ids_orig"]

                # non_masked_idx = masked_input_ids != self.pvp.tokenizer.mask_token_id
                # np.copyto(input_ids, masked_input_ids, where=non_masked_idx)

                # Get softmaxed vocab probs
                outputs, probs = self.get_adapet_mlm_logits(input_ids, masked_input_ids) # [bs, max_seq_len]

                max_seq_len = probs.shape[1]
                is_correct = unlabeled_batch['is_correct'][:, None]
                is_correct = is_correct.repeat(1, max_seq_len).float() # [bs, max_seq_len]

                full_loss = BCELoss()(probs, is_correct)  # [bs, max_seq_len]

                mask_loss = masked_input_ids != input_ids  # [bs, max_seq_len]

                lm_loss = torch.sum(full_loss * mask_loss.float()) / torch.max(torch.sum(mask_loss),
                                                                            torch.tensor(1).to(self.pvp.device))
        # Label Loss
        label_loss = None
        if labeled_batch is not None:
            label_loss_func = 'bce_loss'
            labeled_inputs = self.generate_inputs(labeled_batch)
            outputs = self.roberta(**labeled_inputs)
            prediction_scores = self.lm_head(outputs[0])

            mlm_labels = labeled_batch['mlm_labels']
            logits = self.pvp.convert_mlm_logits_to_cls_logits(mlm_labels, prediction_scores)

            if 'cls_labels' in labeled_batch:
                cls_labels = labeled_batch['cls_labels']

                if label_loss_func == 'ce_loss':
                    label_loss = CrossEntropyLoss()(logits, cls_labels.view(-1))

                if label_loss_func == 'bce_loss':
                    probs = logits.softmax(dim=-1)
                    label_loss = BCELoss()(probs[:,1] , cls_labels.float())

        # TOTAL LOSS
        total_loss = None
        has_lm_loss, has_label_loss = lm_loss is not None, label_loss is not None
        if has_lm_loss or has_label_loss:
            if has_lm_loss and has_label_loss:
                # Both losses were computed -> Combine them
                total_loss = (1 - alpha) * label_loss + alpha * lm_loss
            else:
                # Use the single computed loss
                total_loss = lm_loss if has_lm_loss else label_loss

        ret_kwargs = dict(
            loss=total_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        return_obj = SequenceClassifierOutput if labeled_batch is not None else MaskedLMOutput
        return return_obj(**ret_kwargs)

    def get_adapet_mlm_logits(self, input_ids, masked_input_ids):
        '''
        Get logits for PET MLM objective

        :param input_ids: [bs, max_seq_len]
        :param masked_input_ids: [bs, max_seq_len]
        :return:
        '''
        outputs = self.roberta(masked_input_ids)
        pet_mask_rep = self.lm_head(outputs[0]) # [bs, max_seq_len, vocab_size]
        pet_mask_rep_vocab_prob = pet_mask_rep.softmax(dim=-1)  # [bs, max_num_lbl_tok, vocab_size]
        pet_mask_rep_correct_vocab_prob = torch.gather(pet_mask_rep_vocab_prob, 2, input_ids[:,:,None]).squeeze(2) # [bs, max_seq_len]
        return outputs, pet_mask_rep_correct_vocab_prob

    def generate_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Generate the default inputs required by almost every language model."""
        inputs = {'input_ids': batch['input_ids']}
        if self.config.model_type in ['bert', 'xlnet']:
            inputs['token_type_ids'] = batch['token_type_ids']
        return inputs
