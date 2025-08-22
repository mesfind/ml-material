---
title: GNNs for Materials
teaching: 1
exercises: 0
questions:
- ""
objectives:
- ""
keypoints:
- ""
---

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>


The most recent, state of the art machine learned potentials in atomistic simulations are based on graph models that are trained on large (1M+) datasets. These models can be downloaded and used in a wide array of applications ranging from catalysis to materials properties. These pre-trained models can be used on their own, to accelerate DFT calculation, and they can also be used as a starting point to fine-tune new models for specific tasks. 

# Background on DFT and machine learning potentials

Density functional theory (DFT) has been a mainstay in molecular simulation, but its high computational cost limits the number and size of simulations that are practical. Over the past two decades machine learning has increasingly been used to build surrogate models to supplement DFT. We call these models machine learned potentials (MLP) In the early days, neural networks were trained using the cartesian coordinates of atomistic systems as features with some success. These features lack important physical properties, notably they lack invariance to rotations, translations and permutations, and they are extensive features, which limit them to the specific system being investigated. About 15 years ago, a new set of features called symmetry functions were developed that were intensive, and which had these invariances. These functions enabled substantial progress in MLP, but they had a few important limitations. First, the size of the feature vector scaled quadratically with the number of elements, practically limiting the MLP to 4-5 elements. Second, composition was usually implicit in the functions, which limited the transferrability of the MLP to new systems. Finally, these functions were "hand-crafted", with limited or no adaptability to the systems being explored, thus one needed to use judgement and experience to select them. While progess has been made in mitigating these limitations, a new approach has overtaken these methods.

Today, the state of the art in machine learned potentials uses graph convolutions to generate the feature vectors. In this approach, atomistic systems are represented as graphs where each node is an atom, and the edges connect the nodes (atoms) and roughly represent interactions or bonds between atoms. Then, there are machine learnable convolution functions that operate on the graph to generate feature vectors. These operators can work on pairs, triplets and quadruplets of nodes to compute "messages" that are passed to the central node (atom) and accumulated into the feature vector. This feature generate method can be constructed with all the desired invariances, the functions are machine learnable, and adapt to the systems being studied, and it scales well to high numbers of elements (the current models handle 50+ elements). These kind of MLPs began appearing regularly in the literature around 2016.

Today an MLP consists of three things:

1. A model that takes an atomistic system, generates features and relates those features to some output.
2. A dataset that provides the atomistic systems and the desired output labels. This label could be energy, forces, or other atomistic properties.
3. A checkpoint that stores the trained model for use in predictions.

# CHGNet Pretrained model

CHGNet (Crystal Hamiltonian Graph Neural Network) is an advanced graph neural network interatomic potential that integrates charge information directly into its modeling framework. Trained on static and relaxation trajectories obtained from GGA/GGA+U calculations in the Materials Project database, CHGNet excels at capturing electron interactions and charge distribution with near-DFT accuracy. In CHGNet, periodic crystal structures are represented as atom graphs by identifying neighboring atoms within a specified radius of each atom in the primitive cell, enabling detailed and accurate atomistic simulations. The short workflow descriptions for the CHGNet model is as follows

  - The input is a crystal structure with unknown atomic charges, which CHGNet uses to predict energy, forces, stress, and magnons, producing a charge-decorated structure.

  - Pairwise bond information between atoms is extracted to form a bond graph, while pairwise angle information between bonds is also captured.

  - An interaction block facilitates information sharing and updating among atoms, bonds, and angles.

  - The graphs are processed through basis expansions and embedding layers to generate detailed features for atoms, bonds, and angles.

  - In the atom convolution layer, neighboring atom and bond information is processed via weighted message passing and aggregated back to the atoms for enhanced representation.


# FAIR Chemistry models

FAIRChem provides a number of GNNs in this repository. Each model represents a different approach to featurization, and a different machine learning architecture. The models can be used for different tasks, and you will find different checkpoints associated with different datasets and tasks. Read the papers for details, but we try to hihglight here the core ideas and advancements from one model the next. 

**Since the Fairchem version 2.0.0, we are currently only supporting the UMA model code. For all other models please checkout fairchem version 1 of the repo while we bring them back to the new repo.**

## Universal Model for Atoms (UMA)

**Core Idea:** UMA is an equivariant GNN that leverages a novel technique called Mixture of Linear Experts (MoLE) to give it the capacity to learn the largest multi-modal dataset to date (500M examples and 50B atoms), while preserving energy conservation and inference speed. Even a 6M active parameter (145M total) UMA model is able to acheieve SOTA accuracy on a wide range of domains such as materials, molecules and catalysis. 

**Paper:** https://ai.meta.com/research/publications/uma-a-family-of-universal-models-for-atoms/ (arxiv link available soon)

## equivariant Smooth Energy Network (eSEN)

**Core Idea:** Scaling GNNs to train on hundreds of millions of structures required a number of engineering decisions that led to SOTA models for some tasks, but led to challenges in other tasks. eSEN started with the eSCN network, carefully analyzed which decisions were necessary to build smooth and energy conserving models, and used those learnings to train a new model that is SOTA (as of early 2025) across many domains. 

**Paper:** https://arxiv.org/abs/2502.12147

## Equivariant Transformer V2 (EquiformerV2)

**Core Idea:** We adapted and scaled the Equiformer model to larger datasets using a number of small tweaks/tricks to accelerate training and inference, and incorporating the eSCN convolution operation. This model was also the first shown to be SOTA on OC20 without requiring the underlying structures to be tagged as surface/subsurface atoms, a major improvement in usability. 

**Paper:** https://arxiv.org/abs/2306.12059

## Equivariant Spherical Channel Network (eSCN)

**Core Idea:** The SCN network was high performance, but the approach broke equivariance in the resulting models. eSCN enabled equivariance in these models, and introduced an SO(2) convolution operation that allowed the approach to scale to even higher order spherical harmonics. The model was shown to be equivariant in the limit of infinitely find grid for the convolution operation.

**Paper:** https://proceedings.mlr.press/v202/passaro23a.html

## Spherical Channel Network (SCN)

**Core Idea:** We developed a message convolution operation, inspired by the vision AI/ML community, that led to more scalable networks and allowed for higher-order spherical harmonics. This model was SOTA on OC20 on release, but introduced some limitations in equivariance addressed later by eSCN. 

**Paper:** https://proceedings.neurips.cc/paper_files/paper/2022/hash/3501bea1ac61fedbaaff2f88e5fa9447-Abstract-Conference.html

## GemNet-OC

**Core Idea:** GemNet-OC is a faster and more scalable version of GemNet, a model that incorporated some clever features like triplet/quadruplet information into GNNs, and provided SOTA performance when released on OC20. 

**Paper:** https://arxiv.org/abs/2204.02782

# Access to gated models on huggingface

To access gated models like UMA, you need to get a HuggingFace account and request access to the UMA models.

- Get and login to your [Huggingface account](https://huggingface.co/)
- Request access to https://huggingface.co/facebook/UMA
- Create a Huggingface token at https://huggingface.co/settings/tokens/ with the permission “Permissions: Read access to contents of all public gated repos you can access”
- Add the token as an environment variable (using huggingface-cli login or by setting the HF_TOKEN environment variable.

~~~
!pip install -U "huggingface_hub[cli]"
~~~
{: .python}

~~~
!huggingface-cli login
~~~
{: .python}

Let's start with the integration of FAIRChem's pretrained machine-learned interatomic potential (MLIP) with the Atomic Simulation Environment (ASE) for the purpose of geometrical optimization of an adsorbate on a metal surface. Initially, requisite modules are imported from the FAIRChem core and ASE packages, and the computational device is selected based on availability, prioritizing CUDA-enabled GPUs. Subsequently, a pretrained MLIP model titled "uma-s-1p1" is retrieved, serving as the predictive unit for atomic interactions. The system setup involves constructing a copper (Cu) slab exposing the (100) crystallographic face, with a 3x3x3 supercell and a vacuum spacing included to minimize interlayer interactions. A carbon monoxide (CO) molecule is instantiated as the adsorbate and placed upon the slab at the bridge adsorption site, set at an initial vertical distance of `2.0Å` from the surface.
~~~
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from ase.build import fcc100, add_adsorbate, molecule
from ase.optimize import LBFGS
import torch
device = str("cuda") if torch.cuda.is_available() else str("cpu")
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)
calc = FAIRChemCalculator(predictor, task_name="oc20")
# Set up your system as an ASE atoms object
slab = fcc100("Cu", (3, 3, 3), vacuum=8, periodic=True)
adsorbate = molecule("CO")
add_adsorbate(slab, adsorbate, 2.0, "bridge")
slab.calc = calc
# Set up LBFGS dynamics object
opt = LBFGS(slab)
opt.run(0.05, 100)
~~~
{: .python}


