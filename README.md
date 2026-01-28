# Overcoming Topology Bias and Cold-Start Limitations in Drug Repurposing: A Clinical-Outcome-Aligned LLM Framework
# Mostly finished, if you encounter any problem, please let me know by issues. :)

All the materials, including processed training and optimizing datasets, GNN model weights, evaluation datasets and results have been uploaded to [ScienceDB](https://www.scidb.cn/s/3yIZ3i).  
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
```
or
```bash
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

## 3. Evaluation
### Drug Repositioning datasets
Original evaluation datasets were obtained from [Mirage](https://www.kaggle.com/datasets/ariasha/drug-repositioning?resource=download), as a test data after processing. Training for GNNs were performed on PrimeKG. The [drug_repositioning](drug_repositioning) dictionary contains all the scripts for data splitting, model training and evaluating.  
You need to place processed test datasets in drug_repositioning/data/benchmark/Kaggle_drug_repositioning/ and PrimeKG datasets in drug_repositioning/data/benchmark/PrimeKG/ for proper use.
For GNN model (RGCN, HGT, HAN), training and testing:
```bash
bash gnn.sh
```
For TxGNN training and testing:
```bash
python txgnn.py
```
and
```bash
python txgnn_evaluate.py
```
> Note: The relation id in cold-start and degree-matched datasets may not correct, we have provided the right relation id and designated them in evaluating scripts. There would be no mistake if you use the model weights provided by us.
> However, if you determine to train the model by yourself, please use[cold_start_check.py](drug_repositioning/src/cold_start_check.py) and [degree_matched_check.py](drug_repositioning/src/degree_matched_check.py) to confirm the right relation id and substitute them in [gnn.py](drug_repositioning/src/gnn.py) and [txgnn_evaluate.py](drug_repositioning/src/txgnn_evaluate.py).

For LLM testing, please use [llm_local_8bit.py](drug_repositioning/src/llm_local_8bit.py) for local model; [llm_logprob.py](drug_repositioning/src/llm_logprob.py) for remote api, and [llm_plot.py](drug_repositioning/src/llm_plot.py) for evaluation.

### General Biomedicine benchmark
Raw evaluation datasets: [Chemprot](https://huggingface.co/datasets/clinicalnlplab/chemprot_test); [BioASQ](https://huggingface.co/datasets/kroshan/BioASQ).
You need to place processed chemprot.json and BioASQ.json in biomedicine/data/evaluation/benchmark/ for proper use.
You need to download and place [BERT base model](https://huggingface.co/google-bert/bert-base-uncased) into biomedicine_benchmark/model/ for bert score calculation.
#### Chemprot benchmark
First, generate raw answers:
```bash
python chemprot_answer_api.py --model xx --output xx.json --threads 4 --log logs/xx-answer.log #enable_thinking (Qwen series model)
```
If you use local model, run:
```bash
python chemprot_answer.py --model xx --output xx.json --threads 4 --log logs/xx-answer.log #enable_thinking (Qwen series model)
```
Then, Summarize the raw answer:
```bash
python chemprot_test_api.py --input xx-answer.json --output xx-test.json --log logs/xx-test.log
```
Finally, output the results:
```bash
python chemprot_test_result.py --input xx-test.json -result xx-result.txt
```
#### BioASQ benchmark
The same, you need to generate the raw answer first. Then, you may directly calculate the bert relative scores:
```bash
python bioASQ_bert_result.py --answer xx-answer.json --result xx-bert.txt --log logs/xx-bert.log
```
For gpt score calculation:
```bash
python bioASQ_gpt_api.py --answer xx-answer.json --output xx-gpt.json --log logs/xx-gpt.log
```
and
```bash
python gpt_result.py --test xx-gpt.json --result xx-gpt.txt
```
### Sensitivity analysis
run:
```bash
python chemprot_answer_sp.py --model_dir your_local_model_dir --output xx.json --device cuda --log logs/xx.log --enable_thinking --temperature x --top_k x
python BioASQ_answer_sp.py --model_dir your_local_model_dir --output xx.json --device cuda --log logs/xx.log --enable_thinking --temperature x --top_k x
```
You may change the temperature and top-K freely. The subsequent result analysis are the same to previous section.

### Gradient attribution
Run:
```bash
python gradient_attribution.py --model your_local_model_dir
```
> Note: You need place the input file as the script denotes. You may find it in our uploaded files in ScienceDB.

### Molecular docking
We utilized [DrugReAlign](https://github.com/jinhang23/DrugReAlign) framework to perform molecular docking and assess our model. Please refer to their project for details.

## Citation

If you find this project helpful, please consider citing our work as follows:

```bibtex
@inproceedings{your2024paper,
  title={Overcoming Topology Bias and Cold-Start Limitations in Drug Repurposing: A Clinical-Outcome-Aligned LLM Framework},
  author={YourName, Collaborator1, Collaborator2, ...},
  booktitle={Proceedings of Some Conference or Journal},
  year={2024},
  address={Somewhere},
  publisher={PublisherName},
  url={https://arxiv.org/abs/xxxx.xxxxx}
}