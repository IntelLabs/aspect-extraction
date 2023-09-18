# DISCONTINUATION OF PROJECT #  
This project will no longer be maintained by Intel.  
This project has been identified as having known security escapes.  
Intel has ceased development and contributions including, but not limited to, maintenance, bug fixes, new releases, or updates, to this project.  
Intel no longer accepts patches to this project.  


# Pattern-based Aspect Term Extraction


This repository contains the code for the paper: ["Few-Shot Aspect Extraction using Prompt Training"](https://neurips2022-enlsp.github.io/papers/paper_19.pdf).

This method significantly outperforms the standard supervised training approach in few-shot setups on three datasets.

## Setup

First, create a virtual environment for the project and install all the requirments. We recommend conda for managing virtual enviroments.

```bash
conda create -n pate python==3.8
conda activate pate
pip install -r requirements.txt
python -m spacy download en_core_web_lg  # Spacy model used for noun-phrase extraction
```

### Data

The `data` folder contains the following pre-processed datasets:

 - Laptop - from SemEval-2014 and SemEval-2015 [[1,2]](#references)
 - Restaurant - from SemEval-2014 [[1]](#references)
 - Digital Device - from (Hu and Liu, 2004) [[3]](#references)

## Run Experiments

```bash
python src/run.py --dataset=[DATASET] --task=[TASK]
```

Where:
- `TASK` can be:
    - `tune` - tune model hyperparameters
    - `test` - train and evaluate model using tuned hyperparameters
    - `tune_base` - tune baseline model hyperparameters
    - `test_base` - train and evaluate baseline model using tuned hyperparameters
- `DATASET` can be `lap`/`rest`/`device` for Laptop, Restaurant and Digital Device

Tuning and testing tasks are performed according to the FewNLU [[4]](#references) paradigm.


## Output

A timestamped directory with full results is saved to `eval/test_results`. 
This directory contains `test_results.txt` with a table of avereage Precision/Recall/F1 for each training sample size.

## Citation

```bibtex
@article{korat-etal-2022-fewshot,
    title = "Few-Shot Aspect Extraction using Prompt Training",
    author = "Korat, Daniel  and
      Pereg, Oren and
      Wasserblat, Moshe and
      Bar, Kfir",
    journal="Advances in Neural Information Processing Systems, 2022.",
    url="https://neurips2022-enlsp.github.io/papers/paper_19.pdf",
    year = "2022"
}
```

# References

[1] Maria Pontiki, Dimitris Galanis, John Pavlopoulos, Harris Papageorgiou, Ion Androutsopoulos, and Suresh Manandhar. 2014. SemEval-2014 task 4: Aspect-based sentiment analysis. In Proceedings of the 8th International Workshop on Semantic Evaluation (SemEval 2014), pages 27–35, Dublin,Ireland. Association for Computational Linguistics.

[2] Maria Pontiki, Dimitris Galanis, Haris Papageorgiou, Suresh Manandhar, and Ion Androutsopoulos. 2015. SemEval-2015 task 12: Aspect based sentiment analysis. In Proceedings of the 9th International Workshop on Semantic Evaluation (SemEval 2015), pages 486–495, Denver, Colorado. Association for Computational Linguistics.

[3] Minqing Hu and Bing Liu. 2004a. Mining and summarizing customer reviews. In Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining, pages 168–177.

[4] Jie Tang, Sebastian Ruder, and Zhilin Yang. 2022. FewNLU: Benchmarking state-of-the-art methods for few-shot natural language understanding. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 501–516, Dublin, Ireland. Association for Computational Linguistics. 
