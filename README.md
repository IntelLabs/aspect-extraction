# Pattern-based Aspect Term Extraction


This repository contains the code for the paper: ["Few-Shot Aspect Extraction using Prompt Training"](https://neurips2022-enlsp.github.io/papers/paper_17.pdf).

This method significantly outperforms the standard supervised training approach in few-shot setups on three datasets.

## Setup

First, create a virtual environment for the project and install all the requirments. We recommend conda for managing virtual enviroments.

```bash
conda create -n pate python==3.8; conda activate pate
pip install -r requirements.txt
python -m spacy download en_core_web_lg  # Spacy model used for noun-phrase extraction
```

### Data

The `datasets` direcory contains the following pre-processed datasets in `.jsonl` format:

 - Laptop (`lap`) - from SemEval-2014 and SemEval-2015 [[1,2]](#references)
 - Restaurant (`rest`) - from SemEval-2014 [[1]](#references)
 - Digital Device (`device`) - from (Hu and Liu, 2004) [[3]](#references)

New datasets may be added to this directory, with the same format used in existing datasets.

## Run Train / Predict

```bash
python src/run_fewshot.py --do_train --dataset=rest
```

Optional arguments:

 - `--config` - Name of config json file store in `config` directory. Defaults to `ex=32.json` use `smoke.json` for a sanity check.
 - `--model_savename` - Path to save the trained model. Defaults to `models/finetuned`.
 - `--inference_model` - Model file to load for inference. Defaults to `models/finetuned`.
 - `--do_train` - Perform few-shot training on using the train set. Save the trained model to `--model_savename`.
 - `--do_predict` - Perform inference on the test set. Save predictions to the `/out` directory.
 - `--do_eval` - Perform evaluation on the test set. Requires a test file with labels. Save metrics to the `/out` directory.
 - `--simulate_fewshot` - If enabled, take a small sample from a large train set to simulate a few-shot training setup. 
 - `--sample_size` - The sample size for `--simulate_fewshot`. Defaults to 64.

## Output

A timestamped directory with metrics and/or predictions is saved to the `/out` directory.

## Citation

```bibtex
@article{korat-etal-2022-fewshot,
    title = "Few-Shot Aspect Extraction using Prompt Training",
    author = "Korat, Daniel  and
      Pereg, Oren and
      Wasserblat, Moshe and
      Bar, Kfir",
    journal="Advances in Neural Information Processing Systems, 2022.",
    url="https://neurips2022-enlsp.github.io/papers/paper_17.pdf",
    year = "2022"
}
```

# References

[1] Maria Pontiki, Dimitris Galanis, John Pavlopoulos, Harris Papageorgiou, Ion Androutsopoulos, and Suresh Manandhar. 2014. SemEval-2014 task 4: Aspect-based sentiment analysis. In Proceedings of the 8th International Workshop on Semantic Evaluation (SemEval 2014), pages 27–35, Dublin,Ireland. Association for Computational Linguistics.

[2] Maria Pontiki, Dimitris Galanis, Haris Papageorgiou, Suresh Manandhar, and Ion Androutsopoulos. 2015. SemEval-2015 task 12: Aspect based sentiment analysis. In Proceedings of the 9th International Workshop on Semantic Evaluation (SemEval 2015), pages 486–495, Denver, Colorado. Association for Computational Linguistics.

[3] Minqing Hu and Bing Liu. 2004a. Mining and summarizing customer reviews. In Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining, pages 168–177.

[4] Jie Tang, Sebastian Ruder, and Zhilin Yang. 2022. FewNLU: Benchmarking state-of-the-art methods for few-shot natural language understanding. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 501–516, Dublin, Ireland. Association for Computational Linguistics. 
