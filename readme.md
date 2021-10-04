# code for project about shared functional features

## conda environment
conda create --name sharedFuncFeat python=3.6.8  
conda install cudatoolkit=10.0.130  
conda install cudnn=7.6.0=cuda10.0_0  
pip install --upgrade tensorflow-gpu==1.14.0  
pip install keras==2.2.4  
pip install 'h5py==2.10.0' --force-reinstall 

## open access paper
https://www.sciencedirect.com/science/article/pii/S2666389921002038

## preprint
https://psyarxiv.com/jv3r2/

## data shared
stimuli (image files and GMF parameters) as well as responses  
https://osf.io/7yx28/

## data from
https://www.nature.com/articles/s41562-019-0625-3

## code for beta-VAE was adapted from:
https://github.com/alecGraves/BVAE-tf  
(unlicense license)

## code for triplet loss network was adapted from:
https://github.com/noelcodella/tripletloss-keras-tensorflow  
(MIT license)