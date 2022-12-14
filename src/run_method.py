import logging
from pathlib import Path
import json

from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    set_seed,
    default_data_collator
)

from modeling import (
    Trainer,
    predict,
    RobertaForPatternMaskedLM
)

from absa_utils import (
    AbsaPVP,
    Evaluation,
    write_P_x_to_file,
    set_up_logging,
    save_tr_loss_plots,
    log_and_plot_loss
)

from asp_cand_ext import(
    AddFineTunedPreds
)

from utils import (
    Arguments,
    Benchmark
)
from accelerate import Accelerator

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
CONF_DIR = ROOT_DIR / 'conf'
DATA_DIR = ROOT_DIR / 'data'


def main(hparams_path: str=None, seed=42, train=None, dev=None, test=None, unlabeled=None):

    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_json_file(json_file=hparams_path)

    training_args.do_train = train is not None
    training_args.do_eval = dev is not None
    training_args.do_predict = test is not None

    set_seed(seed)
    training_args.seed = seed

    accelerator = Accelerator()
    set_up_logging(accelerator)

    training_args.max_train_steps = args.max_train_steps
    training_args.num_train_epochs = int(training_args.num_train_epochs)
    
    # if args.super_smoke:
    #     training_args.max_train_steps = 10
    #     max_samples = {'train': 20, 'test': 25, 'dev': 15}     

    bench = Benchmark()
    track = bench.track

    with track('Total Run'):
        ############################ Load Data ####################################
        with track('Load Data'):
            data = dict(train=train, dev=dev, test=test, unlabeled=unlabeled)

            max_samples = {}
            for split in 'train', 'dev', 'test', 'unlabeled':
                if data[split] is None:
                    max_samples[split] = None
                    print(f"\n {split} split = None.")
                else:
                    split_samples = len(data[split])
                    max_samples[split] = split_samples
                    print(f"\n{split_samples} {split} samples loaded from disk.")
                    
            if args.pos_ex_only:
                # If enabled, select only exmaples containing tagged aspect(s)
                data['train'] = data['train'].filter(lambda ex: 'B-ASP' in ex['tags'])
            
            data['orig_train'] = data['train']

        ############################### Pre-process ###############################
        with track('Pre-process'):
            proxy = {} #dict(proxies={"https": "http://proxy-chain.intel.com:912"})

            with track('----Init tokenizer'):
                tokenizer = AutoTokenizer.from_pretrained(
                    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                    use_fast=True,
                    add_prefix_space=True,
                    **proxy
                )

            max_seq_len = min(args.max_seq_len, tokenizer.model_max_length)

            if args.few_shot:
                with track('Init Few-Shot'):
                    pvp = AbsaPVP(tokenizer=tokenizer, max_seq_len=max_seq_len, \
                        pattern_id=args.pattern_id, np_extractors=args.np_extractors,
                        device=accelerator.device, lm_method=args.lm_method, ace_using_model=args.ace_using_model)

                all_gold_bio, inference_tokens = {}, {}
                for split in 'test', 'dev':
                    if data[split] is not None:
                        all_gold_bio[split] = data[split]['tags']
                        inference_tokens[split] = data[split]['tokens']
                
            with track('----Tokenize + Extract Features'):
                def get_label_list():
                    label_list = ['B-ASP', 'I-ASP', 'O']
                    label_list.sort()
                    return label_list

                label_list = get_label_list()
                label_to_id = {l: i for i, l in enumerate(label_list)}
                num_labels = len(label_list)

                map_kwargs = dict(
                    batched=True,
                    batch_size = 10000,
                    num_proc=args.preprocessing_num_workers,
                    remove_columns=['tokens', 'tags', 'text'],
                    load_from_cache_file=not args.overwrite_cache)   

                map_kwargs_with_ace = dict(
                    batched=True,
                    batch_size = 10000,
                    num_proc=args.preprocessing_num_workers,
                    remove_columns=['tokens', 'tags', 'text', 'ace_preds'],
                    load_from_cache_file=not args.overwrite_cache)    

                map_kwargs_keep_cols = dict(
                    batched=True,
                    batch_size = 10000,
                    num_proc=args.preprocessing_num_workers,
                    load_from_cache_file=not args.overwrite_cache)

                #################### Preprocess Labeled Data for Model and Baseline #########################
                def preprocess_baseline(examples):                                                          #
                    tokenized_inputs = tokenizer(                                                           #
                            examples['tokens'],
                            padding="max_length",
                            truncation=True,
                            max_length=max_seq_len,
                            is_split_into_words=True)
                    labels = []
                    for i, label in enumerate(examples['tags']):
                        word_ids = tokenized_inputs.word_ids(batch_index=i)
                        previous_word_idx = None
                        label_ids = []
                        for word_idx in word_ids:
                            # Special tokens have a word id that is None. 
                            # We set the label to -100 so they are automatically
                            # ignored in the loss function.
                            if word_idx is None:
                                label_ids.append(-100)
                            # set the label for the first token of each word.
                            elif word_idx != previous_word_idx:
                                label_ids.append(label_to_id[label[word_idx]])
                            # For the other tokens in a word, we set the label to 
                            # either the current label or -100, depending on
                            # the label_all_tokens flag.
                            else:
                                label_ids.append(label_to_id[label[word_idx]] if \
                                    args.label_all_tokens else -100)
                            previous_word_idx = word_idx

                        labels.append(label_ids)

                    tokenized_inputs["labels"] = labels
                    return tokenized_inputs

                if training_args.do_train:
                    tr_prep_func = pvp.preprocess_train if args.few_shot else preprocess_baseline
                    data['labeled'] = data['train'].map(function=tr_prep_func, **map_kwargs)

                    data['lm'] = None
                    #############################################################################################

                    if args.few_shot:
                        pvp.ex_count = 0

                        lm_training = args.tr_phase_1_lm or (args.tr_phase_2 and args.tr_phase_2_lm)
                        if lm_training:
                            ####################### Preprocess Data for LM METHOD (PET/ADAPET) ################### 
                            def tokenize(examples):                              
                                tokenized = tokenizer(                                                              #
                                    examples['text'],                                                               #
                                    padding="max_length",                                                           #
                                    truncation=True,                                                                #
                                    max_length=max_seq_len,                                                         #
                                    return_special_tokens_mask=True,                                                #
                                )
                                tokenized['input_ids_orig'] = tokenized['input_ids']
                                return tokenized    

                            if args.lm_method == 'adapet':
                                data['lm'] = data['train'].map(function=pvp.preprocess_label_cond, **map_kwargs)       
                                data['lm'] = data['lm'].map(function=tokenize, remove_columns=['text'], **map_kwargs_keep_cols)    
                                      
                                                              
                            elif args.lm_method == 'pet':
                                data['lm'] = data['unlabeled'].map(function=tokenize, **map_kwargs)

                            else:
                                raise ValueError("Invalid value for '--lm_method'.")
                            #########################################################################################


                        # #####################################################################
                        # ############# ACE Model #############
                        # #####################################################################
                        if args.ace_using_model:
                            #data['labeled_for_ace_training'] = data['orig_train'].map(function=tokenize, remove_columns=['text'], **map_kwargs_keep_cols)  
                            data['labeled_for_ace_training'] = data['orig_train'].map(function=preprocess_baseline, **map_kwargs)                    
                            
                            print("\n******************************")
                            print("********** Ace step **********")
                            print("******************************")
                            config = AutoConfig.from_pretrained(
                                args.ace_model,
                                num_labels=num_labels,
                                label2id=label_to_id,
                                id2label={i: l for l, i in label_to_id.items()},
                                **proxy
                            )

                            model = AutoModelForTokenClassification.from_pretrained(
                                args.ace_model, 
                                config=config,
                                **proxy
                            ) 

                            # set baseline mode for asp cand ext model
                            args.few_shot=0
                            
                            training_kwargs = dict(model=model, 
                            args=args, 
                            training_args=training_args,
                            train_dataset=data['labeled_for_ace_training'],
                            unlabeled_dataset=data['lm'], 
                            accelerator=accelerator, 
                            tokenizer=tokenizer)

                            print(f"******* ACE step: sarted fine tuning *******")
                            with track('ACE-Model-Train - Finished'):
                                train_conf = {"steps": args.ace_steps, "lr": args.ace_lr, "batch_size": args.ace_bs}
                                trainer_ace_step = Trainer(**training_kwargs, train_conf=train_conf)
                                losses, avg_tr_loss = trainer_ace_step.train()
                            
                            # *********************** ACE step inference ****************************************
                            with track(f'ACE Step Inference'):
                                print(f"******* ACE step: started inference *******")
                                
                                training_args.eval_accumulation_steps = 20
                                training_args.per_device_eval_batch_size = 2
                                
                                if training_args.do_eval:
                                    print(f"******* ACE step: inferencing dev set *******")
                                    data['dev_for_ACE_step'] = data['dev'].map(function=preprocess_baseline, **map_kwargs)

                                    preds_ace, metrics = predict(
                                        model=model,
                                        args=training_args,
                                        is_few_shot=False,
                                        test_dataset=data['dev_for_ACE_step'],
                                        split='dev',
                                        data_collator=default_data_collator,
                                        label_list=label_list,
                                        accelerator=accelerator
                                    )                
                                
                                    addACEPreds = AddFineTunedPreds(preds_ace)
                                    prep_func = addACEPreds.add_ace_preds
                                    data['dev'] = (data['dev']).map(function=prep_func, **map_kwargs)
                                
                                if training_args.do_predict:
                                    print(f"******* ACE step: inferencing test set *******")
                                    data['test_for_ACE_step'] = data['test'].map(function=preprocess_baseline, **map_kwargs)
                                    preds_ace, metrics = predict(
                                        model=model,
                                        args=training_args,
                                        is_few_shot=False,
                                        test_dataset=data['test_for_ACE_step'],
                                        split='test',
                                        data_collator=default_data_collator,
                                        label_list=label_list,
                                        accelerator=accelerator
                                    ) 
                                    addACEPreds_test = AddFineTunedPreds(preds_ace)
                                    prep_func = addACEPreds_test.add_ace_preds
                                
                                    data['test'] = (data['test']).map(function=prep_func, **map_kwargs)
                                    #data['test'] = (data['test']).map(function=prep_func, **map_kwargs_not_batched)

                                args.few_shot=1

                        pvp.ex_count = 0
                        write_P_x_to_file(data=data, split='labeled', seed=seed, ex=max_samples['train'])

                inference_idx, all_pred_group = {}, {}

                def preprocess_for_inference(split):
                    func = pvp.preprocess_test if args.few_shot else preprocess_baseline
                    
                    # Start preprocess
                    if args.ace_using_model and args.few_shot:
                        kwargs = map_kwargs_with_ace
                    else:
                        kwargs = map_kwargs

                    data[split] = data[split].map(function=func, **kwargs)    
                    #data[split] = data[split].map(function=func, **map_kwargs)

                    # Pop metadata
                    if args.few_shot:
                        pvp.ex_count = 0
                        inference_idx[split] = data[split]['cand_idx']
                        all_pred_group[split] = data[split]['pred_group']
                        data[split] = data[split].remove_columns(['cand_idx', 'pred_group'])

                        write_P_x_to_file(data=data, split=split, seed=seed, ex=max_samples['train'])
                        data[split] = data[split].remove_columns(['P_x'])

                if training_args.do_predict:
                    preprocess_for_inference('test')
                if training_args.do_eval:
                    preprocess_for_inference('dev')

        ###################### Load Model and Trainer ############################
        with track('Load Model'):
            if args.few_shot:
                    model = RobertaForPatternMaskedLM.from_pretrained(
                        args.model_name_or_path,
                        **proxy
                        )
                    model.set_pvp(pvp)

            else: # Baseline #
                config = AutoConfig.from_pretrained(
                    args.model_name_or_path,
                    num_labels=num_labels,
                    label2id=label_to_id,
                    id2label={i: l for l, i in label_to_id.items()},
                    **proxy
                )

                model = AutoModelForTokenClassification.from_pretrained(
                    args.model_name_or_path,
                    config=config,
                    **proxy
                )

        ############################## Fine-Tune - Phase 1 #################################
        tr_loss_plot, tr_loss_plot_phase_2 = None, None
        avg_tr_loss, avg_tr_loss_phase_2, = None, None

        if training_args.do_train:
            training_kwargs = dict(model=model, args=args, training_args=training_args,
                train_dataset=data['labeled'], unlabeled_dataset=data['lm'], 
                accelerator=accelerator, tokenizer=tokenizer)

            with track('Fine-Tune'):
                with track('----Fine-Tune - Phase 1'):
                    if args.few_shot:
                        train_conf = {"lm": args.tr_phase_1_lm, "steps": args.tr_phase_1_steps,
                            "lr": args.tr_phase_1_lr, "label_loss": args.tr_phase_1_label_loss, "alpha": args.alpha}
                    else:
                        train_conf = None
                    trainer = Trainer(**training_kwargs, train_conf=train_conf)
                    losses, avg_tr_loss = trainer.train()
                    
                    tr_loss_plot = log_and_plot_loss(training_args.logging_steps, losses, avg_tr_loss)

        ############################## Fine-Tune - Phase 2 (Optional) #################################
                if args.few_shot and args.tr_phase_2:
                    if args.lm_method == 'adapet':
                        training_kwargs['unlabeled_dataset'] = data['unlabeled'].map(function=tokenize, **map_kwargs)

                    train_conf = {"lm": args.tr_phase_2_lm, "steps": args.tr_phase_2_steps,
                        "lr": args.tr_phase_2_lr, "label_loss": args.tr_phase_2_label_loss}

                    with track('----Fine-Tune - Phase 2'):
                        trainer = Trainer(**training_kwargs, train_conf=train_conf)
                        losses, avg_tr_loss_phase_2 = trainer.train()

                        tr_loss_plot_phase_2 = log_and_plot_loss(training_args.logging_steps, losses, avg_tr_loss_phase_2)

        ############################### Inference #################################
        def run_inference(split):
            with track(f'Inference on {split}'):
                logger.info(f"***** Running inference on {split} split *****")

                preds, metrics = None, {}
                if not args.npe_only:
                    if args.few_shot and args.tr_phase_1_lm or (args.tr_phase_2 and args.tr_phase_2_lm):
                        training_args.eval_accumulation_steps = 20
                        training_args.per_device_eval_batch_size = 2

                    preds, metrics = predict(
                        model=model,
                        args=training_args,
                        is_few_shot=args.few_shot,
                        test_dataset=data[split],
                        split=split,
                        data_collator=default_data_collator,
                        label_list=label_list,
                        accelerator=accelerator
                    )

                eval_args = (all_gold_bio, inference_idx, \
                    all_pred_group, inference_tokens) if args.few_shot else []

                metrics.update(dict(avg_tr_loss=avg_tr_loss, avg_tr_loss_phase_2=avg_tr_loss_phase_2))
                out_dir = Evaluation(args, seed, split, max_samples, ROOT_DIR).run(metrics, preds, *eval_args)
                
                # Log and Write HPARAMS to file
                hparams_str = f"***** Hyperparams *****\nArguments:\n{json.dumps(args.__dict__, indent=2)}\n\n{training_args}"
                with open(out_dir / 'hparams.txt', 'w') as f:
                    f.write(hparams_str)
                
                logger.info(hparams_str)
                
                return out_dir, metrics

        all_metrics = {}
        if training_args.do_predict:
            out_dir, test_metrics = run_inference('test')
            save_tr_loss_plots(out_dir, tr_loss_plot, tr_loss_plot_phase_2)
            all_metrics['test'] = test_metrics
            
        if training_args.do_eval:
            out_dir, dev_metrics = run_inference('dev')

            save_tr_loss_plots(out_dir, tr_loss_plot, tr_loss_plot_phase_2)
            all_metrics['dev'] = dev_metrics

        print('Run Completed.')
        return all_metrics

if __name__ == "__main__":
    main()