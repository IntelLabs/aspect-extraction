import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
from builtins import str
from collections import defaultdict
import shutil
from numpy import mean, std

from utils import Benchmark, verify_and_load_json_dataset


from transformers import (
    HfArgumentParser,
    AutoTokenizer,
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
    set_up_logging,
)

from utils import (
    Arguments,
    Benchmark
)
from accelerate import Accelerator


logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / 'config'
DATA_DIR = ROOT_DIR / "datasets"
MODELS_DIR = ROOT_DIR / 'models'

METRICS = ("precision", "recall", "f1")

    
class PATETrainer:
    def __init__(self, data, args, training_args, cli_args) -> None:
        self.args, self.training_args = args, training_args

        set_seed(training_args.seed)
        self.seed = training_args.seed

        self.accelerator = Accelerator()
        set_up_logging(self.accelerator)
        self.device = self.accelerator.device

        os.environ["CUDA_VISIBLE_DEVICES"] = str(cli_args.cuda_device)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.model_savename = cli_args.model_savename
        self.inference_model = cli_args.inference_model
        self.training_args.max_train_steps = self.args.max_train_steps
        self.training_args.num_train_epochs = int(self.training_args.num_train_epochs)
        self.data = data

        self.num_shot = len(data["train"])
        self.dataset_name = cli_args.dataset
        self.output_dir = self.init_output_dir(self.dataset_name)
        shutil.copyfile(CONFIG_DIR / cli_args.config, self.output_dir / 'config.json')
        assert data is not None
        self.track = Benchmark().track

        self.preprocess()

    def init_output_dir(self, data):
        output_root = ROOT_DIR / self.training_args.output_dir
        output_dir = output_root / (datetime.now().strftime("%d.%m-%H:%M:%S") + "_" + data)
        os.makedirs(output_dir)
        self.out_dir = output_dir
        latest_dir = output_root / f"{data}_latest"
        if os.path.exists(latest_dir):
            os.remove(latest_dir)
        os.symlink(output_dir, latest_dir)
        return output_dir
    
    def preprocess(self):
        with self.track('Pre-process'):
            with self.track('----Init tokenizer'):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.args.tokenizer_name if self.args.tokenizer_name else self.args.model_name_or_path,
                    use_fast=True,
                    add_prefix_space=True
                    )

            self.max_seq_len = min(self.args.max_seq_len, self.tokenizer.model_max_length)

            with self.track('Init Few-Shot'):
                self.pvp = AbsaPVP(tokenizer=self.tokenizer, max_seq_len=self.max_seq_len, \
                    pattern_id=self.args.pattern_id, np_extractors=self.args.np_extractors,
                    device=self.accelerator.device, lm_method=self.args.lm_method, ace_using_model=self.args.ace_using_model)

            self.all_gold_bio, self.inference_tokens = {}, {}

            if self.data['test'] is not None:
                self.all_gold_bio['test'] = self.data['test']['tags']
                self.inference_tokens['test'] = self.data['test']['tokens']
                
            with self.track('----Tokenize + Extract Features'):
                def get_label_list():
                    label_list = ['B-ASP', 'I-ASP', 'O']
                    label_list.sort()
                    return label_list

                self.label_list = get_label_list()

                self.map_kwargs = dict(
                    batched=True,
                    batch_size = 10000,
                    num_proc=self.args.preprocessing_num_workers,
                    remove_columns=['tokens', 'tags', 'text'],
                    load_from_cache_file=not self.args.overwrite_cache)   

    def preprocess_train(self):
        self.data['labeled'] = self.data['train'].map(function=self.pvp.preprocess_train, **self.map_kwargs)
        self.pvp.ex_count = 0

    def preprocess_test(self):
        self.inference_idx, self.all_pred_group = {}, {}

        def preprocess_for_inference(split):               
            # Start preprocess
            kwargs = self.map_kwargs
            self.data[split] = self.data[split].map(function=self.pvp.preprocess_test, **kwargs)    

            # Pop metadata
            self.pvp.ex_count = 0
            self.inference_idx[split] = self.data[split]['cand_idx']
            self.all_pred_group[split] = self.data[split]['pred_group']
            self.data[split] = self.data[split].remove_columns(['cand_idx', 'pred_group'])
            self.data[split] = self.data[split].remove_columns(['P_x'])

        preprocess_for_inference('test')

    def load_model(self):
        with self.track('Load Model'):
            self.model = RobertaForPatternMaskedLM.from_pretrained(
                self.args.model_name_or_path
                )
            self.model.to(device=self.device)
            self.model.set_pvp(self.pvp)

    def train(self):
        with self.track('Train'):
            self.preprocess_train()
            self.load_model()

            train_conf = {"steps": self.args.tr_phase_1_steps, "lr": self.args.tr_phase_1_lr, 
            "label_loss": self.args.tr_phase_1_label_loss, "alpha": self.args.alpha}

            trainer = Trainer(model=self.model, args=self.args, training_args=self.training_args,
                train_dataset=self.data['labeled'], accelerator=self.accelerator, tokenizer=self.tokenizer,
                train_conf=train_conf, save_path=self.model_savename, unlabeled_dataset=None)

            trainer.train()

    # def predict(self):
    #     with self.track('Predict'):
    #         self.preprocess_test()
    #         self.load_model()

    def eval(self):
        with self.track('Eval'):
            self.preprocess_test()
            self.args.model_name_or_path = self.inference_model
            self.load_model()

        def run_inference(split):
            with self.track(f'Inference on {split}'):
                logger.info(f"***** Running inference on {split} split *****")

                preds, metrics = None, {}

                # TODO: Try Removing this:
                self.training_args.eval_accumulation_steps = 20
                self.training_args.per_device_eval_batch_size = 2
                
                preds, metrics = predict(
                    model=self.model,
                    args=self.training_args,
                    is_few_shot=self.args.few_shot,
                    test_dataset=self.data[split],
                    split=split,
                    data_collator=default_data_collator,
                    label_list=self.label_list,
                    accelerator=self.accelerator
                )

                eval_args = (self.all_gold_bio, self.inference_idx, \
                    self.all_pred_group, self.inference_tokens) if self.args.few_shot else []

                Evaluation(self.out_dir, self.dataset_name, self.seed, split, self.num_shot)\
                    .run(metrics, preds, *eval_args)
                
                return metrics


        results = run_inference('test')

        scores = defaultdict(list)
        for metric in METRICS:
            scores[metric].append(results[f"test_{metric}"])
        agg_scores = {f"{m}_{f.__name__}": f(scores[m]) for f in (mean, std) for m in METRICS}
        return agg_scores

def load_json_dataset(ds_name: str):
    dataset = {}
    train_path, test_path = f"{DATA_DIR}/{ds_name}_train.json", f"{DATA_DIR}/{ds_name}_test.json"

    if os.path.exists(train_path):
        dataset["train"] = verify_and_load_json_dataset(train_path)
        dataset["orig_train"] = dataset["train"]
        print(f"Train dataset does not exist at: {train_path}")

    if os.path.exists(test_path):
        dataset["test"] = verify_and_load_json_dataset(test_path)
        print(f"Test dataset does not exist at: {test_path}")
       
    return dataset

def create_fewshot_dataset(ds_name, seed, sample_size):
    dataset = load_json_dataset(ds_name)
    dataset["train"] = dataset["train"].shuffle(seed=seed).select(range(sample_size))
    return dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="rest")
    parser.add_argument("--config", type=str, default="ex=32.json")
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_eval", default=False, action="store_true")
    parser.add_argument("--model_savename", type=str, default=MODELS_DIR / "finetuned")
    parser.add_argument("--inference_model", type=str, default=MODELS_DIR / "finetuned")

    parser.add_argument("--simulate_fewshot", default=False, action="store_true")
    parser.add_argument("--sample_size", type=int, default=16)

    args = parser.parse_args()
    return args

    
def main():
    cli_args = parse_args()
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_json_file(CONFIG_DIR / cli_args.config)

    if cli_args.simulate_fewshot:
        data = create_fewshot_dataset(cli_args.dataset, seed=training_args.seed, sample_size=cli_args.sample_size)
    else:
        data = load_json_dataset(cli_args.dataset)

    trainer = PATETrainer(data, args, training_args, cli_args)

    if cli_args.do_train:
        trainer.train()

    if cli_args.do_eval:
        trainer.eval()

if __name__ == "__main__":
    main()
