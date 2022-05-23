# Label Correlation Based Graph Convolutional Network for Multi-label Text Classification

This repository is the official implementation of our [paper](). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper on RCV1, run this command:

```train
python train.py
```
## Evaluation

To test my model on RCV1, run:

```eval
python eval.py --checkpoint <path-to-checkpoint-file> --graph_feature <path-to-feature-of-graph>