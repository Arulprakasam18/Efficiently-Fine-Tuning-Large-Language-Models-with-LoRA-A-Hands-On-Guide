
# Efficiently Fine-Tuning Large Language Models with LoRA: A Hands-On Guide

## Overview

LoRA (Low-Rank Adaptation) is one of the most widely used **parameter-efficient fine-tuning (PEFT)** methods today. This repository provides a **step-by-step manual implementation** of LoRA, without using the `PEFT` package, to give you a clear understanding of how it works.

The implementation is designed to be **lightweight and efficient**, making it runnable on mainstream hardware with a small footprint. You can try it on **a single GPU**, such as Tesla T4 or consumer GPUs like NVIDIA RTX.

## Repository Contents

| Example                                                              | Description                                                             |
|----------------------------------------------------------------------|-------------------------------------------------------------------------|
| [01-finetune-opt-with-lora.ipynb](01-finetune-opt-with-lora.ipynb)   | Fine-tuning Meta's OPT-125M with LoRA, with an explanation of the method. |
| [02-finetune-gpt2-with-lora.ipynb](02-finetune-gpt2-with-lora.ipynb) | Fine-tuning OpenAI's GPT-2 small (124M) with LoRA.                     |

Unlike examples in the [official LoRA repository](https://github.com/microsoft/LoRA), this implementation **downloads pre-trained models** to focus solely on the LoRA fine-tuning process.

> **Note:** Hugging Face's API is used to download pre-trained models, and a standard PyTorch training loop is applied for fine-tuning instead of using Hugging Face's `Trainer` class.

---

## 1. Setup and Installation

To run this example, install the required software and set up your environment. The following setup assumes a **GPU-enabled virtual machine (VM)** running **Ubuntu Server 20.04 LTS** on Microsoft Azure.

### Install GPU Driver (CUDA)

Install CUDA (NVIDIA GPU driver) with the following commands:

```bash
# Update and install essential tools
sudo apt-get update
sudo apt install -y gcc make

# Install CUDA
wget https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_535.104.05_linux.run
sudo sh cuda_12.2.2_535.104.05_linux.run

# Set environment variables
echo -e "export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64" >> ~/.bashrc
source ~/.bashrc
```

### Install Required Packages

Install PyTorch, Hugging Face Transformers, and other required libraries:

```bash
# Install and upgrade pip
sudo apt-get install -y python3-pip
sudo -H pip3 install --upgrade pip

# Install required Python packages
pip3 install torch transformers pandas matplotlib

# Install Jupyter Notebook for running the examples
pip3 install jupyter
```

---

## 2. Fine-Tuning (Training)

### Clone the Repository

```bash
git clone https://github.com/tsmatz/finetune_llm_with_lora
```

### Run Jupyter Notebook

```bash
jupyter notebook
```

Open Jupyter Notebook in your browser and run the examples provided in this repository.

---

This repository provides a hands-on approach to fine-tuning LLMs efficiently using LoRA. Happy coding!

