# Overcoming Topology Bias and Cold-Start Limitations in Drug Repurposing: A Clinical-Outcome-Aligned LLM Framework
# Not finished yet, still ongoing, I'm working hard with it :(

All the materials, including processed training and optimizing datasets, GNN model weights, evaluation datasets and results have been uploaded to [ScienceDB](https://www.scidb.cn).  
The model training and optimizing logs can be viewed at [Swanlab](https://swanlab.cn/@MasterCat/LLaMA-Factory/overview)  
The SFT and KTO model weights can be downloaded from [SFT model weights](https://www.modelscope.cn/collections/qwen3dr-sft-7203b24e95d14b) and [KTO model weights](https://www.modelscope.cn/collections/qwen3dr-8b-kto-8d1dc7eee6ed42). Please follow the instruction of [Modelscope](https://www.modelscope.cn/) for use.

> This repository contains the source code and data acquisition methods for the paper “Overcoming Topology Bias and Cold-Start Limitations in Drug Repurposing: A Clinical-Outcome-Aligned LLM Framework”.
![Graphical Abstract](graphical_abstract.png)

*Graphical abstract: Traditional Graph Neural Networks (GNNs) and vanilla LLMs suffer from Popularity Bias, disproportionately favoring high-degree nodes (e.g., Semagacestat) regardless of clinical efficacy, while overlooking sparse "cold-start" candidates (e.g., Thalidomide). Our framework incorporates a Clinical Arbiter mechanism via KTO Optimization. The model generates CoT for both candidates. Phase III clinical trial outcomes serve as ground-truth rewards: popular but failed drugs receive a penalty, while valid “cold” candidate receive a positive reinforcement. This feedback loop aligns the LLM's latent space with clinical reality, enabling it to reject scientific hype and identify hidden therapeutic gems.*
## 1. Environment Setup
#### (Optional, for GPU support) Check CUDA and NVCC version

To ensure compatibility with GPU acceleration, check your system's CUDA installation:

```bash
nvcc --version
# or
nvidia-smi
```

- The `nvcc --version` command shows the CUDA Toolkit version installed on your machine.
- The `nvidia-smi` command shows your GPU driver version and the currently installed CUDA version supported by your drivers.

#### (Optional) Install a Matching CUDA Toolkit in the Environment

If your project requires a specific CUDA version (e.g., CUDA 12.8), you can install it into your conda environment. Replace `11.7` with your matching CUDA version as needed:

```bash
conda install cudatoolkit=12.8
```

> Note: If you use PyTorch or other DL frameworks, it’s recommended to install the CUDA Toolkit **with PyTorch** according to the instructions on the [PyTorch official website](https://pytorch.org/get-started/locally/) to ensure full compatibility.

We strongly recommend using a virtual environment for reproducibility and dependency isolation. Here we provide instructions using [conda](https://docs.conda.io/en/latest/) with Python 3.13.

### Create a new conda environment

```bash
conda create -n drkto python=3.13
```

### Activate the environment

```bash
conda activate drkto
```

### Install Pytorch and PyG
For this project, we use PyTorch 2.7.0+cu128. To install, run:

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

PyG is also required:

```bash
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

You may adjust the framework versions according to your own environment. In such cases, please modify the relevant package versions in `requirements.txt` accordingly, and we cannot guarantee full reproducibility of the project under modified settings.
### Install relative requirements
You can now proceed to install the required dependencies as described in the following sections (e.g., using `pip install -r requirements.txt`).

### Configuring Large Language Model API

This project uses the Qwen3-max model for LLM-based evaluation.  
For detailed instructions on configuring the API, please refer to the official documentation:
[阿里云百炼](https://bailian.console.aliyun.com/) (mainland China) or [modelstudio (Aliyun)](https://modelstudio.console.aliyun.com/)

## 2. LLM supervised fine-tuning (SFT) and Kahneman-Tversky Optimization (KTO)

> Note: You need at least 4x A100 80G GPU on a Linux server to reproduce this procedure.

### Original data source
1. PrimeKG:
  [kg.csv](https://dataverse.harvard.edu/api/access/datafile/7144484) [node.csv](https://dataverse.harvard.edu/api/access/datafile/7144482) [edges.csv](https://dataverse.harvard.edu/api/access/datafile/7144483)
2. [RepoDB](https://unmtid-shinyapps.net/shiny/repodb/#:~:text=repoDB%20contains%20a%20standard%20set%20of%20drug%20repositioning,repoDB%20data%20was%20extracted%20from%20DrugCentral%20and%20ClinicalTrials.gov.)
3. [DrugRepoBank](https://awi.cuhk.edu.cn/DrugRepoBank/php/download.php)

### KG-RAG processes for training material generation
We utilized [MedReason](https://github.com/UCSC-VLAA/MedReason) framework for material generation. Please refer to their project for details. 
We provide the modified main code here.Please use the files in [medreason](medreason) dictionary to substitute the files in MedReason/src/data_generation/. for data generation.  

### SFT and KTO execution
We utilized [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory/) for training and optimizing. Please refer to their project for details.  
We provide our .yaml configurations in [llamafactory](llamafactory) dictionary.

