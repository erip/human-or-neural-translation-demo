# Human or Neural Translation Reproduction
A small repro of the ["Human or Neural Translation?"](https://aclanthology.org/2020.coling-main.576/) paper by Bhardwaj et al.

# Requirements

This repo assumes you have a GPU and have installed the relevant dependencies...

```
pip3 install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip3 install simpletransformers transformers sentencepiece scikit-learn pandas numpy sacremoses
```

## Preparing the dataset

Download and untar the da-en file found [here](https://www.statmt.org/europarl/). Once untarred, run `python split.py && python translate.py`.

## Running the experiments

Open the notebook in this directory with jupyter. If you have the dependencies installed and have a GPU, all should be well...
