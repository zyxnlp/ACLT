# ACLT
This repository is the PyTorch implementation of our ACLT model in EMNLP 2021 Paper "To be Closer: Learning to Link up Aspects with Opinions".
# Requirements and Installation
The code has been tested in the following enviroment:
* Ubuntu 18.04.4 LTS
* Python 3.8.3  
* torch 1.9.0
* cuda 10.2
* cudnn 7605

To install:  
* `conda create -n aclt python=3.8.3` 
* `source activate aclt`
* `git clone https://github.com/zyxnlp/ACLT.git`
* `cd ACLT`
* `pip install -r requirements.txt`  

Pre-trained model:  
* Downlaod bert-base model from [here](https://drive.google.com/file/d/1c3PFLniHY_DRLda5BVCBJQ1qoyBrIvFS/view?usp=sharing)
* Unzip the .zip file to the folder `ACLT/pretrain_model/`
# Running
* Change the access mode of the .sh file by `chmod 700 *.sh`
* `./run.sh`
# Citation
If you find our work or the code useful, please consider cite our paper using:
```
@inproceedings{zhou2021closer,
 title={To be Closer: Learning to Link up Aspects with Opinions},
 author={Zhou, Yuxiang and Liao, Lejian and Gao, Yang and Jie, Zhanming and Lu, Wei},
 booktitle={Proceedings of EMNLP},
 year={2021}
}
```

