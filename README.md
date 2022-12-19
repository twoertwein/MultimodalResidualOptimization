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

## Datasets

We will share the pre-processed features used for the machine-learning experiments when they are requested.

### TPOT

The creation of the Transitions in Parenting of Teens (TPOT) dataset was funded by NIH grant #5R01 HD081362 (awarded to Lisa B. Sheeber and Nicholas B. Allen). When referring to the TPOT dataset, please cite
```bibtex
@article{nelson2021psychobiological,
  title={Psychobiological markers of allostatic load in depressed and nondepressed mothers and their adolescent offspring},
  author={Nelson, Benjamin W and Sheeber, Lisa and Pfeifer, Jennifer and Allen, Nicholas B},
  journal={Journal of Child Psychology and Psychiatry},
  volume={62},
  number={2},
  pages={199--211},
  year={2021},
  publisher={Wiley Online Library}
}
```

We use the same [multimodal features](https://cmu.box.com/s/o2lvyd2jc0c72dreq0w3bvdg9wg6g309) as in the following paper

```bibtex
@inproceedings{wortwein2021human,
  title={Human-Guided Modality Informativeness for Affective States},
  author={W{\"o}rtwein, Torsten and Sheeber, Lisa B and Allen, Nicholas and Cohn, Jeffrey F and Morency, Louis-Philippe},
  booktitle={Proceedings of the 2021 International Conference on Multimodal Interaction},
  pages={728--734},
  year={2021}
}
```
