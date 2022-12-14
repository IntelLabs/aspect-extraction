from pathlib import Path
import string
from typing import Tuple, List, Union, Dict, Any, Set
import csv
from collections import defaultdict
import re
from os import PathLike, makedirs
import json
import logging

from transformers.data.data_collator import DataCollatorForTokenClassification
import seaborn as sns

import torch
import spacy
# import stanza
from spacy.tokens import Doc
from spacy import Language
from seqeval.metrics.sequence_labeling import get_entities
from seqeval.metrics.v1 import _precision_recall_fscore_support
from transformers import (
    AdamW,
    PreTrainedTokenizer,
    DataCollatorForLanguageModeling, 
    default_data_collator
)
import transformers
import datasets
from dataclasses import dataclass
from typing import Optional, Tuple
from datasets import load_metric
import numpy as np

logger = logging.getLogger(__name__)

SPACY_MODEL = None
ROW_FORMAT = lambda x: ' '.join((f'{k:{25}}' for k in x)) + '\n'
STANZA_MODEL = None

from typing import Tuple, List, Union, Dict, Any
FilledPattern = Tuple[List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]


def log_and_plot_loss(logging_steps, losses, avg_tr_loss):
    logger.info(f'\n\n****************************')
    logger.info(f'Training Loss: {avg_tr_loss:.4f}')
    logged_steps = [(i + 1) * logging_steps for i, _ in enumerate(losses)]
    tr_loss_plot = sns.lineplot(x=logged_steps, y=losses)
    tr_loss_plot.set(title=f'Training Loss (avg: {avg_tr_loss:.4f})', xlabel='Step', ylabel='Loss')
    return tr_loss_plot.figure


def save_tr_loss_plots(out_dir, tr_loss_plot, tr_loss_plot_phase_2):
    # Save Training Loss Plots
    if tr_loss_plot is not None:
        tr_loss_plot.savefig(out_dir / "tr_loss.pdf")
    if tr_loss_plot_phase_2 is not None:
        tr_loss_plot_phase_2.savefig(out_dir / "tr_loss_phase_2.pdf")


def get_verbalization_ids(word: str, tokenizer: PreTrainedTokenizer, force_single_token: bool) -> Union[int, List[int]]:
    """
    Get the token ids corresponding to a verbalization

    :param word: the verbalization
    :param tokenizer: the tokenizer to use
    :param force_single_token: whether it should be enforced that the verbalization corresponds to a single token.
           If set to true, this method returns a single int instead of a list and throws an error if the word
           corresponds to multiple tokens.
    :return: either the list of token ids or the single token id corresponding to this word
    """
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not force_single_token:
        return ids
    assert len(ids) == 1, \
        f'Verbalization "{word}" does not correspond to a single token, got {tokenizer.convert_ids_to_tokens(ids)}'
    verbalization_id = ids[0]
    assert verbalization_id not in tokenizer.all_special_ids, \
        f'Verbalization {word} is mapped to a special token {tokenizer.convert_ids_to_tokens(verbalization_id)}'
    return verbalization_id


def get_labels(device, label_list, predictions, references):
    # Transform predictions and references tensos to numpy arrays
    if device.type == "cpu":
        y_pred = predictions.detach().clone().numpy()
        y_true = references.detach().clone().numpy()
    else:
        y_pred = predictions.detach().cpu().clone().numpy()
        y_true = references.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    return true_predictions, true_labels

def compute_metrics(metric, split):
    results = metric.compute()
    return {
        f"{split}_precision": results["overall_precision"],
        f"{split}_recall": results["overall_recall"],
        f"{split}_f1": results["overall_f1"],
        f"{split}_accuracy": results["overall_accuracy"],
    }

def set_up_logging(accelerator):
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

def write_P_x_to_file(data, split, seed, ex):
    all_P_x = data[split]['P_x']
    write_dir = Path("data_P_x") / f"num_ex={ex}_seed={seed}.txt"
    makedirs(write_dir, exist_ok=True)
    with open(write_dir / split, 'w') as f:
        f.write('\n'.join(all_P_x))


@dataclass
class DataCollatorForPatternLanguageModeling(DataCollatorForLanguageModeling):

    ignore_pattern_mask: bool = True

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # if self.ex_countgnore_pattern_mask:
        #     labels[inputs == self.tokenizer.mask_token_id] = -100
        #TODO 
        # else:
        #     labels[inputs == self.tokenizer.mask_token_id] = label_token_ids

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

class PretokenizedTokenizer:
    #"""Custom tokenizer to be used in spaCy when the text is already pretokenized."""
    def __init__(self, vocab):
    #  """Initialize tokenizer with a given vocab
    #  :param vocab: an existing vocabulary (see https://spacy.io/api/vocab)
    #  """
        self.vocab = vocab
 
    def __call__(self, inp) -> Doc:
    # """Call the tokenizer on input `inp`.
    # :param inp: either a string to be split on whitespace, or a list of tokens
    # :return: the created Doc object
    # """
        if isinstance(inp, str):
            words = inp.split()
            spaces = [True] * (len(words) - 1) + ([True] if inp[-1].isspace() else [False])
            return Doc(self.vocab, words=words, spaces=spaces)
        elif isinstance(inp, list):
            return Doc(self.vocab, words=inp)
        else:
            raise ValueError("Unexpected input format. Expected string to be split on whitespace, or list of tokens.")

def init_spacy() -> Language:
    global SPACY_MODEL
    SPACY_MODEL = spacy.load('en_core_web_lg', disable=["ner", "vectors", "textcat", "parse", "lemmatizer", "textcat"])
    SPACY_MODEL.tokenizer = PretokenizedTokenizer(SPACY_MODEL.vocab)

    # global STANZA_MODEL
    # STANZA_MODEL = stanza.Pipeline('en', tokenize_pretokenized=True)

class AbsaPVP:
    VERBALIZER = {
        "0": ["No"],
        "1": ["Yes"]
    }

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int,
        pattern_id: int,
        np_extractors: str,
        device: torch.device,
        lm_method: str,
        ace_using_model: bool
    ) -> None:
        """
        Create a new AbsaPVP.

        :param tokenizer: the tokenizer for the underlying language model
        :param max_len: maximum sequence length for the underlying language model
        :param label_list: the labelsfor this language task
        :param pattern_id: the pattern id to use
        :param seed: a seed to be used for generating random numbers if necessary
        """
        self.tokenizer = tokenizer
        self.label_list = ["0", "1"]
        self.pattern_id = pattern_id
        self.max_seq_len = max_seq_len
        self.ex_count = 0
        self.device = device
        self.lm_method = lm_method
        self.ace_using_model = ace_using_model

        assert np_extractors.count('+') <= 1 # can only use 2 methods right now
        self.np_extractors = np_extractors.split('+')

        self.mlm_logits_to_cls_logits_tensor = self._build_mlm_logits_to_cls_logits_tensor()

        self.noun_pos = ('NOUN', 'PROPN')

        # INIT POS REGEX MATCHER
        self.pos_list = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN",
            "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "EOL", "SPACE"]
        self.pos_map = {pos: chr(i + 97) for i, pos in enumerate(self.pos_list)}

        # self.orig_patterns = ["ADJ" "NOUN", "ADJ" "PROPN", "(NOUN|PROPN)+"]
        self.orig_patterns = ["(ADJ" "(NOUN|PROPN))" "|((NOUN|PROPN)+)"]

        self.patterns = {}
        for orig_pattern in self.orig_patterns:
            pattern = orig_pattern
            for pos in self.pos_list:
                pattern = pattern.replace(pos, self.pos_map[pos])
            self.patterns[orig_pattern] = re.compile(pattern)

        self.func_map = {'pos': self.extract_phrases_by_pos, 'regex': self.extract_phrases_by_pos_regex,
        'chunker': self.extract_noun_chunks}

        init_spacy()

    def extract_noun_chunks(self, tokens):
        res = set()
        for np in SPACY_MODEL(' '.join(tokens)).noun_chunks:
            i = 0
            while i < len(np) and np[i].pos_ in ('DET', 'PRON', 'ADP'):
                i += 1
            
            if i < len(np):
                start = np.start + i
                end = np.end - 1
                
                if end >= start:
                    res.add((start, end))
        return res

        # return {(np.start + 1 if np[0].pos_ == 'DET' else np.start, np.end - 1) \
        #     for np in SPACY_MODEL(' '.join(tokens)).noun_chunks}

    # (ADJ)* [NOUN | PROPN] (ADP)* [NOUN | PROPN]

    # ADJ NOUN | PROPN   - COUNT: 298

    def matches(self, pos_tags):
        res = []
        for pattern in self.patterns:
            res.extend([pos_tags[slice(*m.span())] for m in re.finditer(pattern, ''.join((self.pos_map[p] for p in pos_tags)))])
        return set(res)

    def extract_phrases_by_pos_regex(self, tokens):
        # DEBUG
        all_idx = set()

        tokens_pos = [t.pos_ for t in SPACY_MODEL(tokens)]
        pos_str = ''.join((self.pos_map[p] for p in tokens_pos))
        indices = set()
        for orig_pattern in self.patterns.keys():
            pattern = self.patterns[orig_pattern]
            matches = [m.span() for m in re.finditer(pattern, pos_str)]
            for span in matches:
                
                indices.add((span[0], span[1] - 1))

        return indices

    def extract_phrases_by_pos(self, tokens, model='spacy'):
        indices, cur_idx = set(), None

        pos_tags = [t.pos_ for t in SPACY_MODEL(tokens)] if model == 'spacy' \
            else [w.upos for s in STANZA_MODEL(tokens).sentences for w in s.words]

        for i, pos in enumerate(pos_tags + ['END']):
            if cur_idx is not None and (pos == 'END' or pos not in self.noun_pos):
                indices.add((cur_idx, i - 1))
                cur_idx = None
            elif pos != 'END' and pos in self.noun_pos and cur_idx is None:
                cur_idx = i

        return indices

    def extract_gold_phrases(self, asp_labels):
        indices, cur_idx = set(), None

        for i, asp_label in enumerate(asp_labels + ['END']):
            if asp_label != "I-ASP" and cur_idx is not None:
                indices.add((cur_idx, i - 1))
                cur_idx = None
            if asp_label == "B-ASP":
                cur_idx = i

        return indices

    def merge_phrases_old(
            self,
            phrases_a: Set[Tuple[int]],
            phrases_b: Set[Tuple[int]],
            length: int = -1,
            disjoint=True) -> Set[Tuple[int]]:
        res = None

        if disjoint:
            phrase_sets = [set(range(s, e + 1)) for s, e in phrases_a]
            for phrase_b_start, phrase_b_end in phrases_b:
                add = True
                phrase_b_set = set(range(phrase_b_start, phrase_b_end + 1))
                for phrase_a in phrase_sets:
                    if phrase_b_set & phrase_a:
                        add = False
                        break
                if add:
                    phrase_sets.append(phrase_b_set)

            res = {(min(s), max(s)) for s in phrase_sets}

        else:
            bit_arr = [False for _ in range(length)]

            for start, end in phrases_a | phrases_b:
                bit_arr[start: end + 1] = [True] * (end - start + 1)

            res = set()
            in_phrase = False
            for i, bit in enumerate(bit_arr + [False]):
                if in_phrase and not bit:
                    res.add((start_i, i - 1))
                    in_phrase = False
                if bit and not in_phrase:
                    start_i, in_phrase = i, True
        return res

    def pad(self, inputs_ids_list: List[List[int]], token_type_ids_list: List[List[int]]):
        padded_inputs_ids, padded_token_type_ids = [], []

        for input_ids, token_type_ids in zip(inputs_ids_list, token_type_ids_list):
            padding_length = self.max_seq_len - len(input_ids)

            if padding_length < 0:
                raise ValueError(f"Maximum sequence length is too small, got {len(input_ids)} input ids")

            input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)

            assert len(input_ids) == len(token_type_ids) == self.max_seq_len
            padded_inputs_ids.append(input_ids)
            padded_token_type_ids.append(token_type_ids)

        return padded_inputs_ids, padded_token_type_ids

    def extract_indices_from_BIO_no_IASP(self, asp_labels):
        indices, cur_idx = set(), None
        start_flag = 1

        for i, asp_label in enumerate(asp_labels + ['END']):
            if asp_label != "I-ASP" and cur_idx is not None:
                if (asp_label != "B-ASP"):
                    indices.add((cur_idx, i - 1))
                    cur_idx = None
                    start_flag = 1
            if asp_label == "B-ASP":
                if start_flag == 1:
                    cur_idx = i
                    start_flag=0


        return indices   

    def remove_partly_overlap_cand(self, cand_idxes, gold_idxes):
        non_overlap_cands = []
        for cand_idx in cand_idxes:
            cand_idx_range = list(range(cand_idx[0], cand_idx[1]+1))
            overlap_flag = False
            for gold_idx in gold_idxes:                 
                gold_idx_range = list(range(gold_idx[0], gold_idx[1]+1))
                # if cand is same exact as 1 of the gold entities it's full overlap and not partly overplap
                if cand_idx_range == gold_idx_range:
                    break    
                for idx in cand_idx_range:
                    if idx in gold_idx_range: 
                        overlap_flag = True
                        break
            if overlap_flag==False:
                non_overlap_cands.append(cand_idx)

        return(set(non_overlap_cands)) 

    def assert_no_overlapping_candidates(self, cand_idx):
        all_idx = set()

        for idx_start, idx_end in cand_idx:
            cur_set = set(range(idx_start, idx_end + 1))

            if cur_set & all_idx:
                print('ERROR: Overlapping candidates!')
                assert False                
            all_idx |= cur_set

    def remove_overlapping_candidates(self, cand_idx):
        all_idx = set()
        filtered_cand = set()

        for idx_start, idx_end in cand_idx:
            cur_set = set(range(idx_start, idx_end + 1))

            if not cur_set & all_idx:
                filtered_cand.add((idx_start, idx_end))
                all_idx |= cur_set

        return filtered_cand

    def remove_overlapping_candidates_longer_wins(self, cand_idx, tokens):
        
        filtered_cand = []
        
        for begin,end in sorted(cand_idx):
            if filtered_cand and filtered_cand[-1][1] >= begin - 1:
                filtered_cand[-1][1] = max(filtered_cand[-1][1], end)
            else:
                filtered_cand.append([begin, end])
                
        # convert list to set
        filtered_cand_set = set()
        for begin, end in filtered_cand:
            filtered_cand_set.add((begin, end))

        return filtered_cand_set    

    def extract_aspect_candidates(self, tokens, split, tags, ace_pred):
        npe_func_1 = self.func_map[self.np_extractors[0]]
        npe_func_2 = self.func_map[self.np_extractors[1]] if len(self.np_extractors) > 1 else None

        cand_idx_a = npe_func_1(tokens)
        cand_idx_b = npe_func_2(tokens) if npe_func_2 else set()

        cand_idx = cand_idx_a | cand_idx_b

        if split == 'test' and ace_pred not in (None, 0): 
            # aspects cand according to fine_tuned model
            cand_idx_c = self.extract_indices_from_BIO_no_IASP(ace_pred)
            # unite asp cands NP extractor + fine_tuned model
            cand_idx = cand_idx | cand_idx_c

        all_phrase_texts, all_pred_groups, all_cls_labels = [], [], []

        cand_idx = self.remove_overlapping_candidates_longer_wins(cand_idx, tokens)
        # cand_idx = self.remove_overlapping_candidates(cand_idx)
        # self.assert_no_overlapping_candidates(cand_idx)

        if split == 'test':
            for cand in cand_idx:
                all_pred_groups.append(0 if cand in cand_idx_a else 1)
                all_phrase_texts.append(' '.join(tokens[cand[0]: cand[1] + 1]))
            all_phrases_idx = cand_idx

        # For training: add gold phrases as candidates, with their CLS (0/1) labels
        if split == 'train':
            gold_idx = self.extract_gold_phrases(tags)
            cand_idx = self.remove_partly_overlap_cand(cand_idx, gold_idx)
            all_phrases_idx = cand_idx | gold_idx
            for phrase in all_phrases_idx:
                pred_group = 0 if phrase in cand_idx_a else (1 if phrase in cand_idx_b else -1)
                all_pred_groups.append(pred_group)
                all_phrase_texts.append(' '.join(tokens[phrase[0]: phrase[1] + 1]))
                # all_cls_labels.append([0.0, 1.0] if phrase in gold_idx else [1.0, 0.0])
                all_cls_labels.append(1 if phrase in gold_idx else 0)

        return all_phrase_texts, all_phrases_idx, all_pred_groups, all_cls_labels

    def preprocess_test(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        return self.preprocess(examples, split='test')

    def preprocess_train(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        return self.preprocess(examples, split='train')

    def preprocess(self, examples: Dict[str, List[Any]], split: str) -> Dict[str, List[Any]]:
            all_input_ids, all_mlm_labels, all_token_type_ids, all_cls_labels,\
                all_cand_idx, all_pred_group, all_P_x = [], [], [], [], [], [], []
            
            #for text, tokens, tags in zip(examples['text'], examples['tokens'], examples['tags']):

            if split == 'test' and self.ace_using_model:
                   ace_preds = examples['ace_preds']
            else:
                   ace_preds = [0] * len(examples['text'])     

            for text, tokens, tags, ace_pred in zip(examples['text'], examples['tokens'], examples['tags'], ace_preds):
                                
                if split == 'test' and self.ace_using_model:
                    if len(ace_pred)>len(tokens):
                        ace_pred = ace_pred[:len(tokens)]
   
                phrase_texts, phrase_idx, pred_groups, cls_labels = \
                    self.extract_aspect_candidates(tokens, split, tags if split == 'train' else None, ace_pred)
                        
                phrase_input_ids, phrase_token_type_ids, P_x_text = self.encode(text, phrase_texts)
                padded_input_ids, padded_token_type_ids = self.pad(phrase_input_ids, phrase_token_type_ids)
                mlm_labels = [self.get_mask_positions(ids) for ids in padded_input_ids]

                # Append to aggregated result
                all_P_x.extend(P_x_text)
                all_pred_group.extend(pred_groups)
                all_input_ids.extend(padded_input_ids)
                all_token_type_ids.extend(padded_token_type_ids)
                all_mlm_labels.extend(mlm_labels)

                if split == 'test':
                    all_cand_idx.extend([(self.ex_count, phrase_i[0], phrase_i[1]) for phrase_i in phrase_idx])
                    self.ex_count += 1
                else:
                    all_cls_labels.extend(cls_labels)

            res = dict(input_ids=all_input_ids, mlm_labels=all_mlm_labels, token_type_ids=all_token_type_ids,
                        pred_group=all_pred_group, P_x=all_P_x)
            
            # For training only: add gold phrases with labels
            if split == 'train':
                res["cls_labels"] = all_cls_labels

            # For test only: add candidate indices list
            else:
                res['cand_idx'] = all_cand_idx

            self.check_same_lengths(res)
            return res

    def preprocess_label_cond(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        all_texts, all_is_correct = [], []
        
        for text, tokens, tags in zip(examples['text'], examples['tokens'], examples['tags']):
            phrase_texts, _, _, cls_labels = self.extract_aspect_candidates(tokens, 'train', tags, None)
                    
            for asp_candidate, label in zip(phrase_texts, cls_labels):
                # add correct example
                parts = self.get_parts(text, asp_candidate, label)
                correct_text = ' '.join(parts)
                all_texts.append(correct_text)
                all_is_correct.append(1)

                # add incorrect example
                parts = self.get_parts(text, asp_candidate, 1 - label)
                incorrect_text = ' '.join(parts)
                all_texts.append(incorrect_text)
                all_is_correct.append(0)
            
        res = dict(text=all_texts, is_correct=all_is_correct)
        
        self.check_same_lengths(res)
        return res

    def check_same_lengths(self, values_dict):
        lengths_dict = {}
        for k in values_dict.keys():
            lengths_dict[k] = str(len(values_dict[k]))

        if len(set(lengths_dict.values())) != 1:
            print(lengths_dict)
            assert False

    def encode(self, text: str, asp_candidates: List[str]) -> Tuple[List[int], List[int]]:
        """
        Encode an input example using this pattern-verbalizer pair.

        :param example: the input example to encode
        :param priming: whether to use this example for priming
        :param labeled: if ``priming=True``, whether the label should be appended to this example
        :return: A tuple, consisting of a list of input ids and a list of token type ids
        """
        tokenizer = self.tokenizer  # type: PreTrainedTokenizer

        all_input_ids, all_token_type_ids, all_P_x = [], [], []
        for asp_candidate in asp_candidates:
            parts = self.get_parts(text, asp_candidate)
            parts = [x if isinstance(x, tuple) else (x, False) for x in parts]
            parts = [(tokenizer.encode(x, add_special_tokens=False), s) for x, s in parts if x]

            self.truncate(parts, max_length=self.max_seq_len)
            tokens = [token_id for part, _ in parts for token_id in part]
            input_ids = tokenizer.build_inputs_with_special_tokens(tokens, None)
            token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens, None)
            P_x_text = self.tokenizer._decode(input_ids).replace('<s>', '').replace('</s>', '').strip()
            all_input_ids.append(input_ids)
            all_token_type_ids.append(token_type_ids)
            all_P_x.append(P_x_text)

        return all_input_ids, all_token_type_ids, all_P_x

    def get_parts(self, text: str, asp_candidate: str, mask_token_label=None) -> FilledPattern:
        # For PET Label Loss
        if mask_token_label is None:
            mask_token = self.mask
            text = self.shortenable(text)
        else:
            mask_token = "Yes" if mask_token_label is 1 else "No"

        if self.pattern_id == 0: ####
            return [text, ". So, does the review in the previous sentence focus on", asp_candidate, "?", mask_token]

        # if self.pattern_id == 1: 
        #     return [text, ". So, does the review focus on", asp_candidate, "?", mask_token]

        if self.pattern_id == 1: ####
            return [text, ". Is", asp_candidate, "an aspect?", mask_token]

        if self.pattern_id == 2: ####
            return [text, ". So, is the review about", asp_candidate, "?", mask_token]

        if self.pattern_id == 3: ####
            return [text, asp_candidate, mask_token]

        if self.pattern_id == 4:
            return [text, ". Is", asp_candidate, "the focus of the previous sentence?", mask_token]

        # if self.pattern_id == 4:
        #     return [text, ". Does this review talk about", asp_candidate, "?", mask_token]

        # elif self.pattern_id == 2:
        #     return [text, '. All in all, it was', self.mask, '.']
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    @staticmethod
    def _seq_length(parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    def truncate(self, parts: List[Tuple[str, bool]], max_length: int):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts)
        total_len += self.tokenizer.num_special_tokens_to_add()
        num_tokens_to_remove = total_len - max_length

        for _ in range(num_tokens_to_remove):
            self._remove_last(parts)

    def get_mask_positions(self, input_ids: List[int]) -> List[int]:
        label_idx = input_ids.index(self.tokenizer.mask_token_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels

    def convert_mlm_logits_to_cls_logits(self, mlm_labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        masked_logits = logits[mlm_labels >= 0]
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in masked_logits])
        return cls_logits.view(-1, len(self.label_list))

    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:
        m2c = self.mlm_logits_to_cls_logits_tensor.to(logits.device)
        # filler_len.shape() == max_fillers
        filler_len = torch.tensor([len(self.verbalize(label)) for label in self.label_list],
                                  dtype=torch.float)
        filler_len = filler_len.to(logits.device)

        # cls_logits.shape() == num_labels x max_fillers  (and 0 when there are not as many fillers).
        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)]
        cls_logits = cls_logits * (m2c > 0).float()

        # cls_logits.shape() == num_labels
        cls_logits = cls_logits.sum(axis=1) / filler_len
        return cls_logits

    def verbalize(self, label) -> List[str]:
        return AbsaPVP.VERBALIZER[label]

    @property
    def max_num_verbalizers(self) -> int:
        """Return the maximum number of verbalizers across all labels"""
        return max(len(self.verbalize(label)) for label in self.label_list)

    @staticmethod
    def shortenable(s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True

    @staticmethod
    def remove_final_punc(s: Union[str, Tuple[str, bool]]):
        """Remove the final punctuation mark"""
        if isinstance(s, tuple):
            return AbsaPVP.remove_final_punc(s[0]), s[1]
        return s.rstrip(string.punctuation)

    @staticmethod
    def lowercase_first(s: Union[str, Tuple[str, bool]]):
        """Lowercase the first character"""
        if isinstance(s, tuple):
            return AbsaPVP.lowercase_first(s[0]), s[1]
        return s[0].lower() + s[1:]

    @property
    def mask(self) -> str:
        """Return the underlying LM's mask token"""
        return self.tokenizer.mask_token

    def _build_mlm_logits_to_cls_logits_tensor(self):
        label_list = self.label_list
        m2c_tensor = torch.ones([len(label_list), self.max_num_verbalizers], dtype=torch.long) * -1

        for label_idx, label in enumerate(label_list):
            verbalizers = self.verbalize(label)
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = get_verbalization_ids(verbalizer, self.tokenizer, force_single_token=True)
                assert verbalizer_id != self.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

def fill_asp_bio(ex_bio, start, end):
    assert ex_bio[start] == 'O'
    ex_bio[start] = 'B-ASP'
    for i in range(start + 1, end + 1):
        assert ex_bio[i] == 'O'
        ex_bio[i] = 'I-ASP'

def calc_bio_metrics(pred, true):
    res = load_metric("seqeval").compute(predictions=pred, references=true)
    return {k.split('_')[1]: round(v, 4) for k, v in res.items() if k != 'ASP'}

class Evaluation:
    def __init__(
        self,
        args,
        seed: int,
        split: str,
        max_samples: dict,
        root: PathLike) -> None:

        self.out_dir = f"{args.results_dir}/{args.dataset}_ex={max_samples['train']}_seed={seed}_%s"
        self.is_few_shot = args.few_shot
        self.split = split
        self.max_samples = max_samples
        self.root = root
        
    def run(
        self,
        metrics: dict,
        preds=None,
        all_gold_bio=None,
        inference_idx=None,
        all_pred_group=None,
        inference_tokens=None
        ) -> None:

        if self.is_few_shot:
            split = self.split
            few_shot_metrics, step_1_err_counts, test_err_counts = \
                self.few_shot_eval(all_gold_bio[split], inference_idx[split], all_pred_group[split], inference_tokens[split], preds=preds)
            metrics.update(few_shot_metrics)
        else:
            step_1_err_counts, test_err_counts = None, None
            f1_key = f"{self.split}_f1"
            f1_str = f"{f1_key}={metrics[f1_key]}"
            self.out_dir = Path(self.out_dir % f1_str)

        metrics['test_samples'] = self.max_samples[self.split]
        self.write_metrics(metrics, step_1_err_counts, test_err_counts)
        return self.out_dir

    def few_shot_eval(self, all_gold_bio, candidate_idx, all_pred_group, test_tokens, preds=None):
        # Create bio lists in the same dimensions as gold_bio
        all_step_1_bio, all_step_1_bio_a, all_step_1_bio_b, all_pred_bio = [], [], [], []
        for gold_bio in all_gold_bio:
            for bio_list in all_step_1_bio, all_step_1_bio_a, all_step_1_bio_b, all_pred_bio:
                bio_list.append(['O'] * len(gold_bio))

        if preds is None:
            preds = [0] * len(candidate_idx)
        else:
            preds = [np.argmax(p.cpu()).item() for p in preds]

        for (ex_i, start, end), pred, pred_group in zip(candidate_idx, preds, all_pred_group):
            # Eval Step-1
            target_bio = all_step_1_bio_a if pred_group == 0 else all_step_1_bio_b
            fill_asp_bio(target_bio[ex_i], start, end)
            fill_asp_bio(all_step_1_bio[ex_i], start, end)

            # Final Eval
            if pred == 1:
                fill_asp_bio(all_pred_bio[ex_i], start, end)

        step_1_metrics = calc_bio_metrics(all_step_1_bio, all_gold_bio)
        metrics = {f'{self.split}_step_1_{k}': v for k, v in step_1_metrics.items()}

        step_1_overlapping_recall = self.overlapping_recall(y_true=all_gold_bio, \
            y_pred_a=all_step_1_bio_a, y_pred_b=all_step_1_bio_b)
        
        metrics['step_1_overlapping_recall'] = step_1_overlapping_recall

        if preds is not None:
            step_2_metrics = calc_bio_metrics(all_pred_bio, all_gold_bio)
            metrics.update({f'{self.split}_{k}': v for k, v in step_2_metrics.items()})
            f1_key = f"{self.split}_f1"
        else:
            f1_key = f"{self.split}_step_1_f1"

        f1_str = f"{f1_key}={metrics[f1_key]}"
        self.out_dir = Path(self.out_dir % f1_str)

        makedirs(self.out_dir, exist_ok=True)

        step_1_err_counts = self.write_error_analysis_csv(
            all_gold_bio, all_step_1_bio, test_tokens, 'step_1.csv', pos=True)
        test_err_counts = self.write_error_analysis_csv(
            all_gold_bio, all_pred_bio, test_tokens, 'test.csv')

        return metrics, step_1_err_counts, test_err_counts

    def write_metrics(self, metrics, step_1_err_counts=None, test_err_counts=None):
        makedirs(self.out_dir, exist_ok=True)

        for data, name in zip([metrics, step_1_err_counts, test_err_counts], \
            ["metrics", "step_1_err_counts", "test_err_counts"]):
            if data is not None:
                title = f"{self.split}_{name}"
                with open(self.out_dir / f'{title}.json', 'w') as f:
                    json.dump(data, f, indent=2)
                
                sep = f"\n{'*' * len(title)}\n"
                print(f"\n{sep}{title}{sep}{json.dumps(data, indent=2)}\n\n")


    def write_error_analysis_csv(self, gold, pred, test_tokens, out_path, pos=False):
        gold_ents = {(start, end) for (_, start, end) in get_entities(gold)}
        pred_ents = {(start, end) for (_, start, end) in get_entities(pred)}
        tokens_flat = [e for s in test_tokens for e in s + ['<END>']]
        token_to_sentence_idx = [i for i, s in enumerate(test_tokens) for _ in s + ['<END>']]

        if pos:
            pos_flat = [t if isinstance(t, str) else t.pos_ for s in test_tokens for t in list(SPACY_MODEL(s)) + ['END']]

        tp_set = gold_ents & pred_ents
        conf_matrix = defaultdict(list)

        for ent in gold_ents | pred_ents:
            err_type = 'TP' if ent in tp_set else ('FN' if ent in gold_ents else 'FP')
            ent_text = ' '.join(tokens_flat[ent[0]: ent[1] + 1])
            text = ' '.join(test_tokens[token_to_sentence_idx[ent[0]]])

            entry = [ent_text, text]
            if pos:
                entry.append(' '.join(pos_flat[ent[0]: ent[1] + 1]))

            conf_matrix[err_type].append(entry)

        with open(self.out_dir / out_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            header = ['Type', 'entitiy', 'text']
            if pos:
                header.append('POS')
            writer.writerow(header)

            for err_type, examples in conf_matrix.items():
                for ex in examples:
                    writer.writerow([err_type, *ex])

        err_counts = {err_type: len(errs) for err_type, errs in conf_matrix.items()}

        with open(str(self.out_dir / out_path) + '_DEBUG.csv' , 'w') as csvfile:
            writer = csv.writer(csvfile)
            header = ['y_true', 'y_pred', 'token']
            if pos:
                header += ['POS']
            writer.writerow(header)
            gold_flat = [e for s in gold for e in s + ['-----END-----']]
            pred_flat = [e for s in pred for e in s + ['-----END-----']]

            assert len(gold_flat) == len(pred_flat) == len(tokens_flat)

            if pos:
                for y_true, y_pred, token, pos in zip(gold_flat, pred_flat, tokens_flat, pos_flat):
                    writer.writerow([y_true, y_pred, token, pos])
            else:
                for y_true, y_pred, token in zip(gold_flat, pred_flat, tokens_flat):
                    writer.writerow([y_true, y_pred, token])

        return err_counts #ROW_FORMAT(err_counts) + ROW_FORMAT(map(str, err_counts.values()))

    @staticmethod
    def tp_overlapping(y_true, y_pred, suffix, *args):
        entities_true, entities_pred = defaultdict(set), defaultdict(set)
        for type_name, start, end in get_entities(y_true):
            entities_true[type_name].add((start, end))

        y_pred_a = [[t[0] for t in l] for l in y_pred]
        y_pred_b = [[t[1] for t in l] for l in y_pred]

        for type_name, start, end in get_entities(y_pred_a):
            entities_pred[type_name].add((start, end))

        for type_name, start, end in get_entities(y_pred_b):
            entities_pred[type_name].add((start, end))

        target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

        tp_sum = np.array([], dtype=np.int32)
        pred_sum = np.array([], dtype=np.int32)
        true_sum = np.array([], dtype=np.int32)
        for type_name in target_names:
            entities_true_type = entities_true.get(type_name, set())
            entities_pred_type = entities_pred.get(type_name, set())
            tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
            pred_sum = np.append(pred_sum, len(entities_pred_type))
            true_sum = np.append(true_sum, len(entities_true_type))
        return pred_sum, tp_sum, true_sum

    def overlapping_recall(self, y_true, y_pred_a, y_pred_b):
        y_preds = [list(zip(*lists)) for lists in zip(y_pred_a, y_pred_b)]

        return _precision_recall_fscore_support(y_true, \
            y_preds, extract_tp_actual_correct=self.tp_overlapping)[1][0]


def get_data_collator(is_few_shot, tokenizer):
    if is_few_shot:
        data_collator = DataCollatorForTokenClassification(tokenizer)
    else:
        data_collator = default_data_collator
    return data_collator
