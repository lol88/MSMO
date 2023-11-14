# Multi-Scale Multi-Object Semi-Supervised Consistency Learning for Ultrasound Image Segmentation



## Introduction
Manual labeling of ultrasound images is a time-consuming and labor-intensive task. Semi-supervised learning (SSL) exploits large amounts of unlabeled data to improve model performance under limited labeled data. However, it faces two challenges: fusion of contextual information at multiple scales and confusion of spatial information between multiple objects. In this paper, we propose a multi-scale multi-object (MSMO) semi-supervised framework based on consistency learning for ultrasound image segmentation. MSMO tackles these challenges by employing a contextual-aware encoder and a decoder for multi-object semantic calibration and fusion. First, the encoder is used to extract multi-scale multi-objects context-aware features, and an attention module is also introduced to refine the feature map and enhance channel information interaction. Then, MSMO employs a hierarchical Convolutional Long Short-Term Memory (ConvLSTM) to calibrate the output features of the current object by using the hidden state of the previous object, and recursively fuse multi-object semantics at different scales. MSMO further reduces variations among multiple decoders in different perturbations through consistency constraints, thereby producing consistent predictions for highly uncertain areas. Extensive experiments demonstrate that MSMO achieves state-of-the-art performance for both single-object and multi-object ultrasound image segmentation. The proposed MSMO outperforms advanced SSL methods on 4 benchmark datasets. MSMO significantly reduces the burden of manual analysis of ultrasound images and holds great potential as a clinical tool.



## Datasets

Please put the [BUSI](https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset) dataset or your own dataset as the following architecture. 
```
├── data
        ├── BUSI
            ├── images
            |   ├── benign (10).png
            │   ├── malignant (17).png
            │   ├── normal (14).png
            │   ├── ...
            |
            └── masks
                ├── benign (10).png
                ├── malignant (17).png
                ├── normal (14).png
                ├── ...
        ├── your dataset
            ├── images
            |   ├── 0a7e06.png
            │   ├── 0aab0a.png
            │   ├── 0b1761.png
            │   ├── ...
            |
            └── masks
                ├── 0a7e06.png
                ├── 0aab0a.png
                ├── 0b1761.png
                ├── ...
```


## Training and Validation


You can first spilt your dataset:

```python
python spilt.py
```

Then, training your dataset:

```python
python train.py
```

[//]: # (## Citation)

[//]: # ()
[//]: # (If you use our code, please cite our paper:)

[//]: # (```tex)

[//]: # (@article{tang2023multilevel,)

[//]: # (  title={Multi-Level Global Context Cross Consistency Model for Semi-Supervised Ultrasound Image Segmentation with Diffusion Model},)

[//]: # (  author={Fenghe Tang and Jianrui Ding and Lingtao Wang and Min Xian and Chunping Ning},)

[//]: # (  journal={arXiv preprint arXiv:2305.09447},)

[//]: # (  year={2023})

[//]: # (})

[//]: # (```)

