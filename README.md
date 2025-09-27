<div align="center">

# CLEAR
<a href="https://arxiv.org/abs/2412.16112"><img src="https://img.shields.io/badge/arXiv-2412.16112-A42C25.svg" alt="arXiv"></a> 
</div>


> **CLEAR: Conv-Like Linearization Revs Pre-Trained Diffusion Transformers Up**
> <br>
> [Songhua Liu](http://huage001.github.io/), 
> [Zhenxiong Tan](https://scholar.google.com/citations?user=HP9Be6UAAAAJ&hl=en), 
> and 
> [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)
> <br>
> NeurIPS 2025
> <br>
> [xML Lab](https://sites.google.com/view/xml-nus), National University of Singapore
> <br>

![](./assets/teaser.png)

## 🔥News

**[2025/9/18]** CLEAR is accepted at NeurIPS 2025.

**[2024/12/20]** We release training and inference codes of CLEAR, a simple yet effective strategy to linearize the complexity of pre-trained diffusion transformers, such as FLUX and SD3.

## Introduction

Diffusion Transformers (DiT) have become a leading architecture in image generation. However, the quadratic complexity of attention mechanisms, which are responsible for modeling token-wise relationships, results in significant latency when generating high-resolution images. To address this issue, we aim at a linear attention mechanism in this paper that reduces the complexity of pre-trained DiTs to linear. We begin our exploration with a comprehensive summary of existing efficient attention mechanisms and identify four key factors crucial for successful linearization of pre-trained DiTs: locality, formulation consistency, high-rank attention maps, and feature integrity. Based on these insights, we introduce a convolution-like local attention strategy termed CLEAR, which limits feature interactions to a local window around each query token, and thus achieves linear complexity. 
Our experiments indicate that, by fine-tuning the attention layer on merely 10K self-generated samples for 10K iterations, we can effectively transfer knowledge from a pre-trained DiT to a student model with linear complexity, yielding results comparable to the teacher model. Simultaneously, it reduces attention computations by 99.5% and accelerates generation by 6.3 times for generating 8K-resolution images. Furthermore, we investigate favorable properties in the distilled attention layers, such as zero-shot generalization across various models and plugins, and improved support for multi-GPU parallel inference.

**TL;DR**: For pre-trained diffusion transformers, enforcing an image token to interact with only tokens within **a local window** can effectively reduce the complexity of the original models to a linear scale.

## Installation

* CLEAR requires ``torch>=2.5.0``, ``diffusers>=0.31.0``, and other packages listed in ``requirements.txt``. You can set up a new experiment with:

  ```bash
  conda create -n CLEAR python=3.12
  conda activate CLEAR
  pip install -r requirements.txt
  ```

* Clone this repo to your project directory:

  ``` bash
  git clone https://github.com/Huage001/CLEAR.git
  ```

## Supported Models

We release a series of variants for linearized [FLUX-1.dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) with various local window sizes. 

We experimentally find that when the local window size is small, e.g., 8, the model can produce repetitive patterns in many cases. To alleviate the problem, in some variants, we also include down-sampled key-value tokens besides local tokens for attention interaction.

The supported models and the download links are:

| window_size | down_factor |                             link                             |
| :---------: | :---------: | :----------------------------------------------------------: |
|     32      |     NA      | [here](https://huggingface.co/Huage001/CLEAR/resolve/main/clear_local_32.safetensors) |
|     16      |     NA      | [here](https://huggingface.co/Huage001/CLEAR/resolve/main/clear_local_16.safetensors) |
|      8      |     NA      | [here](https://huggingface.co/Huage001/CLEAR/resolve/main/clear_local_8.safetensors) |
|     16      |      4      | [here](https://huggingface.co/Huage001/CLEAR/resolve/main/clear_local_16_down_4.safetensors) |
|      8      |      4      | [here](https://huggingface.co/Huage001/CLEAR/resolve/main/clear_local_8_down_4.safetensors) |

You are encouraged to download the model weights you need to ``ckpt`` beforehand. For example:

```bash
mkdir ckpt
wget https://huggingface.co/Huage001/CLEAR/resolve/main/clear_local_8_down_4.safetensors
```

## Inference

* If you want to compare the linearized FLUX with the original model, please try ``inference_t2i.ipynb``.

* If you want to use CLEAR for high-resolution acceleration, please try ``inference_t2i_highres.ipynb``. We currently adopt the strategy of [SDEdit](https://huggingface.co/docs/diffusers/v0.30.2/en/api/pipelines/stable_diffusion/img2img#image-to-image). The basic idea is to generate a low-resolution result at first, based on which we gradually upscale the image.

* Please configure ``down_factor`` and ``window_size`` in the notebooks to use different variants of CLEAR. If you do not want to include down-sampled key-value tokens, specify ``down_factor=1``. The models will be downloaded automatically to ``ckpt`` if not downloaded.

* Currently, a GPU card with 48 GB VMem is recommended for high-resolution generation.


## Training

* Configure ``/path/to/t2i_1024`` in multiple ``.sh`` files.

* Download training images from [here](https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_1024_10K/data_000000.tar), which contains 10K 1024-resolution images generated by ``FLUX-1.dev`` itself, and unzip it to ``/path/to/t2i_1024``:

  ```
  tar -xvf data_000000.tar -C /path/to/t2i_1024
  ```

* [Optional but Recommended] Cache T5 and CLIP text embeddings and VAE features beforehand:

  ```bash
  bash cache_prompt_embeds.sh
  bash cache_latent_codes.sh
  ```

* Start Training:

  ```bash
  bash distill.sh
  ```

  By default, it uses 4 80G-VMem GPUs with ``train_batch_size=2`` and ``gradient_accumulation_steps=4``. Please feel free to configure them in ``distill.sh`` and ``deepspeed_config.yaml`` according to your situations.

## Acknowledgement

* [FLUX](https://blackforestlabs.ai/announcing-black-forest-labs/) for the source models.
* [flexattention](https://pytorch.org/blog/flexattention/) for kernel implementation.
* [diffusers](https://github.com/huggingface/diffusers) for the code base.
* [DeepSpeed](https://github.com/microsoft/DeepSpeed) for the training framework.
* [SDEdit](https://huggingface.co/docs/diffusers/v0.30.2/en/api/pipelines/stable_diffusion/img2img#image-to-image) for high-resolution image generation.
* [@Weihao Yu](https://github.com/yuweihao) and [@Xinyin Ma](https://github.com/horseee) for valuable discussions.
* NUS IT’s Research Computing group using grant numbers NUSREC-HPC-00001.

## Citation

If you find this repo helpful, please consider citing:

```bib
@article{liu2024clear,
    title     = {CLEAR: Conv-Like Linearization Revs Pre-Trained Diffusion Transformers Up},
    author    = {Liu, Songhua and Tan, Zhenxiong and Wang, Xinchao},
    journal   = {NeurIPS},
    year      = {2025},
}
```
