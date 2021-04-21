# Machine Learning Methods for Cervical Cancer Classifier

## Introduction 

This is a [challenge](https://www.kaggle.com/c/ima205challenge2021/overview) for our course IMA205. There are 2 tasks: binary classification and multi-class classification.

- For binary classification, I use ResNet101 based FPN (modified FPN)

- For multi-class classification, I use ResNet152

## Recommended environment

>python 3.6 \
>opencv-contrib-python 3.4.2. \
>opencv-python         4.1.2.30 \
>torch                 1.4.0 \
>torchfile             0.1.0 \
>torchvision           0.5.0

## Dataset [DownLoad](https://drive.google.com/file/d/1KAAGC6vucd3p0wOJ8RPF30jo8-zuDxbI/view?usp=sharing)

Change the data directory in the dala loader files (test_data_loader_seg.py, train_data_loader_seg.py, train_data_loader_multiclass_seg.py) to your own data directory.

Now I don't have the test labels

## Train

Before training, remember change the number of GPU in the train files according your number of GPU. 

- You can firstly have a look at the arguments of the train and evaluate python files. For example,

```shell
    python train_fpn_seg.py --help
```

- Training example

1. Binary classification:

```shell
    python train_fpn_seg.py --arch resnet101 --n_epoch 30 --schedule 10 20 30 --batch_size 96
```

2. Multiclass classification:

```shell
    python train_multiclass_seg.py --arch resnet152 --n_epoch 40 --schedule 10 20 30 --batch_size 96
```

- You can also write it in a shell file and then run it

```shell
    sh train_fpn_seg.sh
```

## Prediction

- You can firstly have a look at the arguments of the prediction python files. For example:

```shell
    python evaluate_fpn_seg.py --help
```

- Prediction example

1. Binary classification:

```shell
    python evaluate_fpn_seg.py --arch resnet101 --resume checkpoint.pth_97.304.tar
```

2. Multiclass classification:

```shell
    python evaluate_multiclass_seg.py --arch resnet152 --resume checkpoint.pth.tar
```

- You can also write it in a shell file and then run it. For example:

```shell
    sh evaluate_fpn_seg.sh
```

## Result

Here are some results

### Binary classification


| BackBone | Net | Public Score | Pretrain Model |
|  ----  | ----  |  ----  |  ----  |
| ResNet101 | FPN | 0.97304 | [DownLoad](https://drive.google.com/file/d/1ykwxyfU0vMtTAY2BhRKajoZwiIZWezLN/view?usp=sharing) |



### Multi-class classification


| BackBone | Public Score | Pretrain Model |
|  ----  | ----  |  ----  |
| ResNet152 | 0.79274 | [DownLoad](https://drive.google.com/file/d/1D4AdjC5_c76hcPLh5pGFyaxXS9X4hDKM/view?usp=sharing) |



## Reference

1.	https://github.com/Mael-zys/PSENet
2.	https://arxiv.org/pdf/1612.03144.pdf
3.	https://arxiv.org/abs/1908.02650

