# Machine Learning Methods for Cervical Cancer Classifier

## Introduction 

This is a [challenge](https://www.kaggle.com/c/ima205challenge2021/overview) for our course IMA205. There are 2 tasks: binary classification and multi-class classification.

- For binary classification, I use resnet101 based fpn

- For multi-class classification, I use resnet152

## Recommended environment

>python 3.6 \
>opencv-contrib-python 3.4.2. \
>opencv-python         4.1.2.30 \
>torch                 1.4.0 \
>torchfile             0.1.0 \
>torchvision           0.5.0

## Dataset [DownLoad](https://drive.google.com/file/d/1KAAGC6vucd3p0wOJ8RPF30jo8-zuDxbI/view?usp=sharing)

Change the data directory in the dala loader files to your data directory.

Now I don't have the test labels

## Train

- You can firstly have a look at the arguments of the train and evaluate python files

```
    python train_fpn_seg.py --help
```

- Training example

```
    python train_fpn_seg.py --arch resnet101 --n_epoch 30 --schedule 10 20 30 --batch_size 96
```

- You can also write it in a shell file and then run it

```
    sh train_fpn_seg.sh
```

## Prediction

- You can firstly have a look at the arguments of the prediction python files

```
    python evaluate_fpn_seg.py --help
```

- Prediction example

```
    python evaluate_fpn_seg.py --arch resnet101 --resume checkpoint.pth_97.304.tar
```

- You can also write it in a shell file and then run it

```
    sh evaluate_fpn_seg.sh
```

## Result

Here are some results

### Binary classification


| BackBone | Net | Public Score |
|  ----  | ----  |  ----  |
| ResNet101 | FPN | 0.97304 |



### Multi-class classification


| BackBone | Public Score |
|  ----  | ----  |
| ResNet101 | 0.74248 |
| ResNet152 | 0.79274 |



## Reference

To be continued