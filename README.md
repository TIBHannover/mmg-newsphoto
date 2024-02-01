## Multimodal Geolocation Estimation of News Photos
This is the official GitHub page for the paper ([link here](https://link.springer.com/chapter/10.1007/978-3-031-28238-6_14)):

Golsa Tahmasebzadeh, Sherzod Hakimov, Ralph Ewerth, Eric Müller-Budack: "Multimodal Geolocation Estimation of News Photos". In: European Conference on Information Retrieval (ECIR), Dublin, Ireland, 2023, 204–220.

## Contents
- [Installation](#Installation)
- [Download Data & Checkpoints](#Download_Data_&_Checkpoints)
- [Reproduce Results](#Reproduce_Results)
- [Training](#Training)

## Installation

``` bash
git clone https://github.com/TIBHannover/mmg-newsphoto.git
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Download Data & Checkpoints
Download the data and checkpoints for MMG-NewsPhoto from [here](https://tib.eu/cloud/s/zsdiw98easXY3Ax) and for BreakingNews from [here](https://tib.eu/cloud/s/2bANYcAw7eBtqGd).

## Reproduce Results
To evaluate the models based on MMG-NewsPhoto dataset: 
```bash
python mmg_news/eval.py --model_name <MODELNAME> --test_check_point <CHECKPOINT>
```
To evaluate the models based on Breakingnews dataset: 
```bash
python breakingnews/eval.py --model_name <MODELNAME> --test_check_point <CHECKPOINT>
```
## Training
To train the models based on MMG-NewsPhoto dataset:
```bash
python mmg_news/train.py \
--model_name <MODELNAME> \
--resume <CHECKPOINT>
```

To train the models based on BreakingNews dataset:
```bash
python breakingnews/bn_train.py \
--model_name <MODELNAME> \
--resume <CHECKPOINT> 
```

## Citation
```
@inproceedings{DBLP:conf/ecir/TahmasebzadehHEM23,
  author       = {Golsa Tahmasebzadeh and
                  Sherzod Hakimov and
                  Ralph Ewerth and
                  Eric M{\"{u}}ller{-}Budack},
  title        = {Multimodal Geolocation Estimation of News Photos},
  booktitle    = {Advances in Information Retrieval - 45th European Conference on Information
                  Retrieval, {ECIR} 2023, Dublin, Ireland, April 2-6, 2023, Proceedings,
                  Part {II}},
  series       = {Lecture Notes in Computer Science},
  volume       = {13981},
  pages        = {204--220},
  publisher    = {Springer},
  year         = {2023},
  url          = {https://doi.org/10.1007/978-3-031-28238-6\_14},
  doi          = {10.1007/978-3-031-28238-6\_14}
}
```
