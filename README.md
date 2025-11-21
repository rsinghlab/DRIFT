# DRIFT - Diffusion-based Representation Integration for Foundation models in spatial Transcriptomics

DRIFT is a scalable diffusion framework that denoises expression profiles and integrates the spatial topology of ST data into existing pretrained scRNA-seq and ST foundation models without additional retraining. Foundation models that do not explicitly model spatial information benefit from both denoising and spatial integration, while methods that do so leverage DRIFT's denoised output. DRIFT constructs a spatial adjacency graph among tissue spots and applies a heat-kernel diffusion process that propagates gene-expression signals across local neighborhoods while preserving tissue boundaries. This produces spatially coherent yet biologically meaningful representations that can be directly embedded into pretrained foundation models without retraining, making our approach much more computationally scalable and accessible. 

![STING Framework Overview](https://github.com/rsinghlab/DRIFT/blob/main/DRIFT_framework.png?raw=true)


## Requirements
To run the DRIFT step, you require the following libraries:

scanpy >= 1.9.1

numpy < 2.0.0

scipy

networkx

Python Optimal Transport >= 0.9.1

We suggest generating an environment (such as conda) to run the code. You can create the required conda environment directly by running the following lines sequentially in the shell.
```
conda create --name <env_name> python==3.11
conda activate <env_name>
pip install scanpy
pip install POT
pip install "numpy<2"
pip install scipy
pip install networkx
```

### Foundation Model Requirements

You will require additional libraries dependent on the foundation model you aim to use. Please refer to their code for any additional requirements necessary.
