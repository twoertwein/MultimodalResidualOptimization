# MultimodalResidualOptimization

The code for ["Beyond Additive Fusion: Learning Non-Additive Multimodal Interactions"](paper.pdf) at Findings of EMNLP 2022.


## Setup
This code base heavily relies on [python-tools](https://bitbucket.org/twoertwein/python-tools/src/master/) to conduct the machine learning experiements.
```sh
git clone git@github.com:twoertwein/MultimodalResidualOptimization.git
cd MultimodalResidualOptimization
poetry update  # installs python-tools and all other dependencies
```

## Usage
For early fusion
```sh
# baseline (joint)
python train.py --dimension <dimension> --res_det --joint
# only tri-modal branch
python train.py --dimension <dimension> --res_det --joint --tri
# routing
python train.py --dimension <dimension> --routing 
# MRO
python train.py --dimension <dimension> --res_det
# sMRO
python train.py --dimension <dimension> --res_det --stepwise
```

For tensor fusion
```sh
# baseline (joint)
python train.py --dimension <dimension> --mult
# only tri-modal branch
python train.py --dimension <dimension> --mult --tri
# routing
python train.py --dimension <dimension> --routing --mult 
# MRO
python train.py --dimension <dimension> --mult --res_det
# sMRO
python train.py --dimension <dimension> --mult --res_det --stepwise
```
