# HVS-Inspired Signal Degradation Network for Just Noticeable Difference Estimation
![Figure 2](./assets/Fig.svg) \
This repository is the implementation of ["HVS-Inspired Signal Degradation Network for Just Noticeable Difference Estimation"](https://arxiv.org/abs/2208.07583) 

> **Abstract:**
> Significant improvement has been made on just noticeable difference (JND) modelling ([HVS-SD JND](https://arxiv.org/abs/2208.07583)) due to the development of deep neural networks, especially for the recently developed unsupervised-JND generation models. 
> However, they have a major drawback that the generated JND is assessed in the real-world signal domain instead of in the perceptual domain in the human brain. 
> There is an obvious difference when JND is assessed in such two domains since the visual signal in the real world is encoded before it is delivered into the brain with the human visual system (HVS). 
> Hence, we propose an HVS-inspired signal degradation network for JND estimation. 
> To achieve this, we carefully analyze the HVS perceptual process in JND subjective viewing to obtain relevant insights, and then design an HVS-inspired signal degradation (HVS-SD) network to represent the signal degradation in the HVS. 
> On the one hand, the well learnt HVS-SD enables us to assess the JND in the perceptual domain. On the other hand, it provides more accurate prior information for better guiding JND generation. 
> Additionally, considering the requirement that reasonable JND should not lead to visual attention shifting, a visual attention loss is proposed to control JND generation. Experimental results demonstrate that the proposed method achieves the SOTA performance for accurately estimating the redundancy of the HVS. 

### Running
`$ python test.py  --path /data/xy/data/kodak/ --lmbda=10` \
The three-channel-PSNR of the distorted images and original images will be generated.
`--lmbda` is the distortion parameter of the HVS-SD-JND.
`--path` is the testing data path.

### Running
The weight files canbe downloaded as the following link (https://pan.baidu.com/s/1jWw2G9Tlr1-SPghcSdAqiw?pwd=xv3a) with the password 'xv3a'. Then put all of the files into `./checkpoints`

## Citation
```bibtex
@article{jin2022hvs,
  title={Hvs-inspired signal degradation network for just noticeable difference estimation},
  author={Jin, Jian and Xue, Yuan and Zhang, Xingxing and Meng, Lili and Zhao, Yao and Lin, Weisi},
  journal={arXiv preprint arXiv:2208.07583},
  year={2022}
}
```
