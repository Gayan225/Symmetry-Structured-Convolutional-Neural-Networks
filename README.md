## Symmetry Structured Convolutional Neural Networks

The code is tested on a V100 GPU with Tensorflow 2.0.0 and Python 3.7.6.

## Requirements
```bash
pip install -r requirements.txt
```

## Implementation
RNA experiments on sCNN kernels use the half Normal Glorot initialization.

## Training sCNN models from scratch
To train sCNN model with hyperparameter match version on RNA problem with strand16 dataset
```
python3 machine_HM.py --dataset strand16s --weight 5 --lr 0.002 
```

## Acknowledgments
This project is built on top of [CNN for RNA problem](https://github.com/dwillmott/ss-inf).
