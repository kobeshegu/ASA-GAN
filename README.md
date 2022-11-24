# Adversarial Semantic Augmentation for Training GANs under Limited Data

![image](./assets/teaser.png)

> **Adversarial Semantic Augmentation for Training GANs under Limited Data** <br>
> Mengping Yang, Zhe Wang, Ziqiu Chi, Dongdong Li, Wenli Du<br>
> paper will be available once accepted


## Abstract 
Generative adversarial networks have made remarkable achievements in synthesizing images in recent years. However, training GANs requires massive data, and the performance of GANs deteriorates significantly when training data is limited. To improve the generalization of GANs in low-data regimes, existing approaches utilize data augmentation techniques to enlarge the training samples, but these augmentation techniques may change the distribution of training data. To remedy this, we propose an adversarial semantic augmentation (ASA) technique to enlarge the training data at the semantic level instead of the image level. Concretely, we estimate the covariance matrices of semantic features for both real and generated images to find meaningful transformation directions. Such directions translate original features to another semantic representation, \emph{e.g.}, changing backgrounds or changing the expressions of the human face dataset. Moreover, we derive an upper bound of the expected adversarial loss. By optimizing the upper bound, our semantic augmentation is automatically performed. Such design avoids redundant sampling of the augmented features and introduces negligible computation burden, making our approach computation efficient. We perform extensive experiments on both large-scale and limited datasets. Our method outperforms the state-of-the-art approaches on $15/20$ few-shot datasets. Qualitative and quantitative results demonstrate that our method consistently alleviates overfitting and memorization of the discriminator, resulting in improved generalization performance.

## Usage 
Use ./run.sh to run the code.

The results and models will be automatically saved in /train_resutls folder.

## Contact
Fell free to contact me at kobeshegu@gmail.com if you have any questions or advices, thanks!

## Acknowledgment
Our code is built upon the excellent codebase of [FastGAN](https://github.com/odegeasslbc/FastGAN-pytorch), we thank a lot for their work.

## BibTeX
Coming soon

