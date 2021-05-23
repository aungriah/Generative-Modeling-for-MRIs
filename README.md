## Realistic Generation of Diffusion-Weighted Magnetic Resonance Brain Images with Deep Generative Models

This repository includes the official implemntation of *"Realistic Generation of Diffusion-Weighted Magnetic Resonance Brain Images with Deep Generative Models"*.

### Acknowledgements
The code presented in this network is heavily based on [bbeatrix's](https://github.com/bbeatrix/introvae.git) implementation of  "Introspective Variational Autoencoders for Photographic Image Synthesis" and on the [manicman1999](https://github.com/manicman1999/StyleGAN-Keras.git) adaptation of "A Style-Based Generator Architecture for Generative Adversarial Networks".

### Getting started

#### Data preparation

This code requires all training images to be of dimension (128x128x1) and to be saved into an .npy file, e.g. */data/normal_brains.npy*. Please refer to the official document for further information on the data acquisition and preprocessing process.

### Introspective Variational Autoencoder

#### Training
Set the hyperparameters for training in lines 19-62 of */introVAE/train.py* to train the network:
```
cd introVAE
python train.py
```
Intermediate weights and inference results will be saved at the locations specified.

#### Testing
The script introVAE/test.py will (i) select a number of images from a specified dataset and reconstruct them, (ii) interpolate MR brain images between pairs of selected images, and (iii) randomly sample from a gaussian distribution to generate artificial MR brain images with the network's encoder.

Speecify the path to the .npy containing images in line 221 and the path to your pretrained weights in line 53 for testing:

```
cd introVAE
python test.py
```
### Style-GAN

#### Training
Set the hyperparameters for training in lines 9-19 of */styleGAN/train.py* for training:
```
cd styleGAN
python train.py
```
Intermediate weights and inference results will be saved at the locations specified

#### Testing
The script */styleGAN/test.py* generates artifical MR brain images and saves them to an .npy file. Make sure to specify the path to your pretrained weights in line 18 to test:

```
cd styleGAN
python test.py
```
