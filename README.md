# This is the code for the paper "Genome-Phenome Association Prediction by Deep Factorizing Heterogeneous Molecular Network" on BIBM'2021 
(https://doi.org/10.1109/BIBM52615.2021.9669792).
If you use this code, please cite our paper:

```
@inproceedings{tan2021genome,
  title={Genome-Phenome Association Prediction by Deep Factorizing Heterogeneous Molecular Network},
  author={Tan, Haojiang and Qiu, Sichao and Wang, Jun and Yu, Guoxian and Guo, Wei and Guo, Maozu},
  booktitle={2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={211--216},
  year={2021},
  organization={IEEE}
}
```

##  Install dependencies


This code has been tested on the following environments with eight GeForce RTX 3090:

* Ubuntu 18.04
* Python 3.8.8
* Pytorch 1.7.0


Dataset:

* We constructed the biological heterogeneous molecular network using many data sources. Please refer to the link in the paper for specific data sources.
* For your convenience, we provide the first 2,000 genes out of 39,296 in directory 'data'.

## Evaluate the model with five cross validation
Run python cross_validation.py
## Obtain a training model through all the samples
Run python DeepGPA_caseStudy.py
