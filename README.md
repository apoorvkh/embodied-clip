# Embodied CLIP

Official repository for [Simple but Effective: CLIP Embeddings for Embodied AI](https://arxiv.org/abs/2111.09888). We present competitive performance on navigation-heavy tasks in Embodied AI using frozen visual representations from [CLIP](https://github.com/openai/CLIP).

This repository includes all code and pretrained models necessary to replicate the experiments in our paper:
- Baselines
  - [RoboTHOR ObjectNav](#robothor-objectnav) (Sec. 4.1)
  - [iTHOR Rearrangement](#ithor-rearrangement) (Sec. 4.2)
  - [Habitat ObjectNav](#navigation-in-habitat) (Sec. 4.3)
  - [Habitat PointNav](#navigation-in-habitat) (Sec. 4.4)
- [Probing for Navigational Primitives](#primitive-probing) (Sec. 5)
- [Zero-shot ObjectNav in RoboTHOR](#zeroshot-objectnav) (Sec. 7)

We have included forks of other repositories as branches of this repository, as we find this is a convenient way to centralize our experiments and track changes across codebases.

## RoboTHOR ObjectNav

### Installation

We've included instructions for installing the full AllenAct library (modifiable) with conda for [our branch](https://github.com/allenai/embodied-clip/tree/allenact), although you can also use the [official AllenAct repo (v0.5.0)](https://github.com/allenai/allenact/tree/v0.5.0) or perhaps newer.

```bash
git clone -b allenact --single-branch git@github.com:allenai/embodied-clip.git embclip-allenact
cd embclip-allenact

export EMBCLIP_ENV_NAME=embclip-allenact
export CONDA_BASE="$(dirname $(dirname "${CONDA_EXE}"))"
export PIP_SRC="${CONDA_BASE}/envs/${EMBCLIP_ENV_NAME}/pipsrc"
conda env create --file ./conda/environment-base.yml --name $EMBCLIP_ENV_NAME

# Install the appropriate cudatoolkit
conda env update --file ./conda/environment-<CUDA_VERSION>.yml --name $EMBCLIP_ENV_NAME
# OR for cpu mode
conda env update --file ./conda/environment-cpu.yml --name $EMBCLIP_ENV_NAME

# Install RoboTHOR and CLIP plugins
conda env update --file allenact_plugins/robothor_plugin/extra_environment.yml --name $EMBCLIP_ENV_NAME
conda env update --file allenact_plugins/clip_plugin/extra_environment.yml --name $EMBCLIP_ENV_NAME

# Download RoboTHOR dataset
bash datasets/download_navigation_datasets.sh robothor-objectnav

# Download CLIP model
python -c "import clip; clip.load('RN50')"
```

Please refer to the [official AllenAct installation instructions](https://allenact.org/installation/installation-allenact) for more details.

### Training

```bash
# ImageNet

# CLIP
PYTHONPATH=. python allenact/main.py -o storage/objectnav-robothor-rgb-clip-rn50 -b projects/objectnav_baselines/experiments/robothor/clip objectnav_robothor_rgb_clipresnet50gru_ddppo
```

### Using pretrained models

```bash
# ImageNet
curl -o pretrained_model_ckpts/objectnav-robothor-imagenet-rn50.195M.pt https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/navigation/exp_Objectnav-RoboTHOR-RGB-ResNet50GRU-DDPPO__stage_00__steps_000195242243.pt

# CLIP
curl -o pretrained_model_ckpts/objectnav-robothor-clip-rn50.130M.pt https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/navigation/exp_Objectnav-RoboTHOR-RGB-ClipResNet50GRU-DDPPO__stage_00__steps_000130091717.pt
```

You can use these models with the `python allenact/main.py` argument `-c pretrained_model_ckpts/objectnav-robothor-clip-rn50.130M.pt` (as an example).

### Evaluating 

Simply append the `--eval` argument to the above `python allenact/main.py` commands.

## iTHOR Rearrangement

## Navigation in Habitat

Please refer to the README in the [habitat branch](https://github.com/allenai/embodied-clip/tree/habitat), which has detailed instructions on installing Habitat and training/evaluating our models.

## Primitive Probing

## Zero-shot ObjectNav
