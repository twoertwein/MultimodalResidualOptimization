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
# MRO
python train.py --dimension <dimension> --res_det
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
# MRO
python train.py --dimension <dimension> --mult --res_det
# baseline (joint)
python train.py --dimension <dimension> --mult
# only tri-modal branch
python train.py --dimension <dimension> --mult --tri
# routing
python train.py --dimension <dimension> --routing --mult 
# sMRO
python train.py --dimension <dimension> --mult --res_det --stepwise
```

Where `<dimension>` can be: `uni` (unimodal sanity check), `bi` (bimodal sanity check), `mosi` (sentiemnt on MOSI), `sentiment` (MOSEI), `polarity` (MOSEI), `happiness` (MOSEI), `Arousal` (IEMOCAP), `Valence` (IEMOCAP), `arousal` (SEWA), `valence` (SEWA), `constructs` (TPOT), or `intent` (Instagram).

## Data

The pre-processed features used for the machine-learning experiments for the sanity checks and the Instagram dataset are part of this git repository. The features for MOSI, MOSEI, and TPOT are available [here](https://cmu.box.com/s/76pz4tbctt1az2ukvcpeqht97k6st5xc). If you want the features for [IEMOCAP](https://cmu.box.com/s/1sekj2jqyycvrygpzrajpgehr8kmit8e) and [SEWA](https://cmu.box.com/shared/static/5tbus1tb6dio2pw2v4yehwa53pek69vu.xz), please send us proof that you completed the data-sharing agreements required by those projects.

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

### Perception Study

The aggregated scores from the perception study on IEMOCAP are in `perception_study_arousal.csv` and `perception_study_arousal.csv`. To synchronize the scores with the model output, please use the fields `meta_id`, `meta_begin`, and `meta_end`.
