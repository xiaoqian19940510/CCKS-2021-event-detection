# CCKS-2021-event-detection

We only offer baseline model on [CCKS 2021：通用细粒度事件检测](https://www.biendata.xyz/competition/ccks_2021_maven/).  F1-score is at least 0.68.

本项目实现基于BERT的事件检测模型，后续会更新BERT+CNN、BERT+BiLSTM等模型。

This code is the implementation for [DMBERT](https://www.aclweb.org/anthology/N19-1105/) model. The implementations are based on [Huggingface's Transformers](https://github.com/huggingface/transformers), especially its example for the multiple-choice task.


## Requirements

- python==3.6.9

- torch==1.2.0

- transformers==2.8.0

- sklearn==0.20.2

  

## Usage

### On MAVEN:

1. Download MAVEN data files.
2. Run ```main.slurm``` for training and evaluation on the devlopment set, and geting predictions on the test set (dumped to ```results.jsonl```).

See the two scripts for more details.

### On ACE

1. Preprocess ACE 2005 dataset as in [this repo](https://github.com/thunlp/HMEAE).
2. Run ``main.slurm`` for training and evaluation.
