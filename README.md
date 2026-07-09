# DRIFT - Diffusion-based Representation Integration for Foundation models in spatial Transcriptomics

DRIFT is a scalable diffusion framework that denoises expression profiles and integrates the spatial topology of ST data into existing pretrained scRNA-seq and ST foundation models without addition[...]

![STING Framework Overview](https://github.com/rsinghlab/DRIFT/blob/main/DRIFT_framework.jpg?raw=true)


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
pip install pycpd
```

### Foundation Model Requirements

You will need additional libraries depending on the foundation model you intend to use. Please refer to their code for any additional requirements necessary. The code repositories for the foundati[...]

**Geneformer** - <https://huggingface.co/ctheodoris/Geneformer>

**scFoundation** - <https://github.com/biomap-research/scFoundation>

**scGPT** - <https://github.com/bowang-lab/scgpt>

**Loki** - <https://github.com/GuangyuWangLab2021/Loki>

**Nicheformer** - <https://github.com/theislab/nicheformer>



### Run Code

To see how to run the code, please check our notebook tutorials.

run_drift.ipynb for running DRIFT to obtain diffused inputs.

run_annotation.ipynb for running annotation code.

run_alignment.ipynb for running alignment code.

For clustering, you need the embeddings from your foundation model. The embeddings can then be used in any clustering algorithm. For our work we used the mclust library in R.

## Data Availability (Dataset names reference the manuscript)

Dataset MERMB and MERMBA are available at the https://cellxgene.cziscience.com website. Dataset MERHH is available at the https://cells.ucsc.edu website. Datasets 10xHPC and MERMPH are available o[...]


## Publication

- Bioinformatics: https://academic.oup.com/bioinformatics/article/42/Supplement_1/btag259/8726325?login=false

