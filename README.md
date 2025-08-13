# How do vision models navigate the binding trade-off?
## Installation
`conda create -n binding python=3.11`       
`conda activate binding`      
`python -m pip install -e .`    
## Parameters
All parameters are listed either at the top or bottom of each file with comments here and there.    
## Files here
### Dataset Generation
`bag_gen.py` for the original superposition dataset and all baselines except stable locations.        
`bag_stable_loc_gen.py` for the baseline where location is kept constant within each pair.      
### Capturing Activations
`superposition_cls_activations.py` to generate and save activations of the original dataset and all baselines except stable locations.      
`location_stable_cls_activations.py` to generate and save that of the stable location baseline.     
### Training the Probe
`supreposition_probe.py` will create a wandb project using your current account.    
`dotproduct_probe.py` trains a CLIP style probe on pairs of CLS tokens projected to joint embedding space.     
`dotproduct_roc.py` evaluates dot product of pairs directly without projection.    
### PCA experiments
`pca.ipynb`     
## Acknowledgements
Project built on [vit-prisma](https://github.com/Prisma-Multimodal/ViT-Prisma), a slight adaptation of which is included in this repository.        