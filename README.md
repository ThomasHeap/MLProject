# MLProject

Collection of scripts and models used by Ryan Egan (https://github.com/Regan831) and I in our final Machine Learning Practical project "Evaluating Relativisitic Generative Adversarial Networks in Artwork and Shoe Generation".

The directory structure has been changed since these files were used, making everything work again and writing a file that can call all the other interesting scripts is on my to-do list.

The RSGAN code is largely unchanged from https://github.com/AlexiaJM/RelativisticGAN  
The VAE code is based on https://github.com/sksq96/pytorch-vae  
The FID code is based on https://github.com/mseitzer/pytorch-fid  
The Inception Score code is based on https://github.com/sbarratt/inception-score-pytorch  

If anyone finds their code in this repo and I have forgotten to mention please let me know.

We made an original contribution (using modifications of the VAE and FID code) wherein the neural network used to get activations used to calculate FID scores were from a VAE trained on part of our training set (as opposed a model trained on unrelated imagenet images). 
