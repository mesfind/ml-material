---
title: Materials Graph Library
teaching: 1
exercises: 1
questions:
- "What is a virtual environment?"
- "How do virtual environments help manage dependencies?"
- "How do you create and activate a virtual environment?"
objectives:
- "Learn about virtual environments and their purpose in Python projects."
- "Understand how to create and manage virtual environments."
- "Get hands-on experience with setting up and using a virtual environment."
keypoints:
- "Virtual environments isolate dependencies."
- "Using Conda or venv for creating environments."
- "Activating and deactivating virtual environments."
---

# Introduction


MatGL (Materials Graph Library) is a specialized graph deep learning library designed for materials science. Since mathematical graphs naturally represent collections of atoms, graph deep learning models are highly effective as surrogate models for predicting material properties, consistently delivering outstanding performance.

Built on top of the Deep Graph Library (DGL) and PyTorch, MatGL incorporates specific adaptations tailored to materials science applications. Its purpose is to provide an extensible platform for developing and sharing graph deep learning models for materials, such as the MatErials 3-body Graph Network (M3GNet) and its predecessor, MEGNet.

This project is a collaborative effort between the Materials Virtual Lab and Intel Labs, with contributions from Santiago Miret, Marcel Nassar, and Carmelo Gonzales.

MatGL is a key component of the MatML ecosystem, which also includes the maml (Materials Machine Learning) package, the MatPES (Materials Potential Energy Surface) dataset, and the MatCalc (Materials Calculator) tool.

## MatErials Graph Network (MEGNet)

MEGNet is a graph network implementation inspired by DeepMind’s approach, tailored for machine learning in materials science. It has proven highly effective, achieving low prediction errors across a wide range of properties for both molecules and crystals. Recent updates include advancements in multi-fidelity materials property modeling. Figure 1 illustrates the sequential update process of the graph network, where bonds, atoms, and global state attributes are iteratively updated by exchanging information, ultimately producing an updated output graph.

## M3GNet

The Materials 3-body Graph Network (M3GNet) is an enhanced graph neural network architecture that extends MEGNet by incorporating three-body interactions. A notable addition in M3GNet is the inclusion of atomic coordinates and the 3×3 lattice matrix for crystals, which are essential for calculating tensorial quantities like forces and stresses through auto-differentiation. 

M3GNet serves as a versatile framework with several key applications:

- **Interatomic potential development:** When trained on the same datasets, M3GNet matches the performance of state-of-the-art machine learning interatomic potentials (MLIPs). A major advantage of its graph-based representation is its scalability across diverse chemical spaces. Notably, M3GNet has enabled the creation of a universal interatomic potential that covers the entire periodic table, trained using relaxation data from the Materials Project.
  
- **Surrogate models for property prediction:** Building on the success of MEGNet, M3GNet can also be employed to build surrogate models for predicting material properties, often achieving accuracy equal to or better than other leading ML models.

For detailed benchmark results, please see the relevant publications in the References section.

MatGL offers a reimplementation of M3GNet using Deep Graph Library (DGL) and PyTorch. Compared to the original TensorFlow version, this implementation introduces several notable improvements:

- A more intuitive API and class structure powered by DGL.
- Support for multi-GPU training through PyTorch Lightning.

# Universal Potential with Property Prediction Models

There may be instances where you do not have access to a DFT relaxed structure. For instance, you may have a generated hypothetical structure or a structure obtained from an experimental source. In this notebook, we demonstrate how you can use the M3GNet universal potential to relax a crystal prior to property predictions.

This provides a pathway to "DFT-free" property predictions using ML models. It should be cautioned that this is not a substitute for DFT and errors can be expected. But it is sufficiently useful in some cases as a pre-screening tool for massive scale exploration of materials.

~~~
from __future__ import annotations

import warnings

import torch
from pymatgen.core import Lattice, Structure
from pymatgen.ext.matproj import MPRester

import matgl
from matgl.ext.ase import Relaxer

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
~~~
{: .python}

For the purposes of demonstration, we will use the perovskite SrTiO3 (STO). We will create a STO with an arbitrary lattice parameter of 4.5 A.

~~~
sto = Structure.from_spacegroup(
    "Pm-3m", Lattice.cubic(4.5), ["Sr", "Ti", "O"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
)
print(sto)
~~~
{: .python}

As a ground truth reference, we will also obtain the Materials Project DFT calculated SrTiO3 structure (mpid: mp-???) using pymatgen's interface to the Materials API.

~~~
mpr = MPRester()
doc = mpr.summary.search(material_ids=["mp-5229"])[0]
sto_dft = doc.structure
sto_dft_bandgap = doc.band_gap
sto_dft_forme = doc.formation_energy_per_atom
~~~
{: .python}

## Relaxing the crystal

~~~
pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")

relaxer = Relaxer(potential=pot)
relax_results = relaxer.relax(sto, fmax=0.01)
relaxed_sto = relax_results["final_structure"]
print(relaxed_sto)
~~~
{: .python}

You can compare the lattice parameter with the DFT one from MP. Quite clearly, the M3GNet universal potential does a reasonably good job on relaxing STO.

~~~
print(sto_dft)
~~~
{: .python}

## Formation energy prediction

To demonstrate the difference between making predictions with a unrelaxed vs a relaxed crystal, we will load the M3GNet formation energy model.

~~~
# Load the pre-trained MEGNet formation energy model.
model = matgl.load_model("M3GNet-MP-2018.6.1-Eform")
eform_sto = model.predict_structure(sto)
eform_relaxed_sto = model.predict_structure(relaxed_sto)

print(f"The predicted formation energy for the unrelaxed SrTiO3 is {float(eform_sto):.3f} eV/atom.")
print(f"The predicted formation energy for the relaxed SrTiO3 is {float(eform_relaxed_sto):.3f} eV/atom.")
print(f"The Materials Project formation energy for DFT-relaxed SrTiO3 is {sto_dft_forme:.3f} eV/atom.")
~~~
{: .python}

The predicted formation energy from the M3GNet relaxed STO is in fairly good agreement with the DFT value.

## Band gap prediction

We will repeat the above exericse but for the band gap.

~~~
model = matgl.load_model("MEGNet-MP-2019.4.1-BandGap-mfi")

# For multi-fidelity models, we need to define graph label ("0": PBE, "1": GLLB-SC, "2": HSE, "3": SCAN)
for i, method in ((0, "PBE"), (1, "GLLB-SC"), (2, "HSE"), (3, "SCAN")):
    graph_attrs = torch.tensor([i])
    bandgap_sto = model.predict_structure(structure=sto, state_attr=graph_attrs)
    bandgap_relaxed_sto = model.predict_structure(structure=relaxed_sto, state_attr=graph_attrs)

    print(f"{method} band gap")
    print(f"\tUnrelaxed STO = {float(bandgap_sto):.2f} eV.")
    print(f"\tRelaxed STO = {float(bandgap_relaxed_sto):.2f} eV.")
print(f"The PBE band gap for STO from Materials Project is {sto_dft_bandgap:.2f} eV.")

~~~
{: .python}

Again, you can see that using the unrelaxed SrTiO3 leads to large errors, predicting SrTiO3 to have very small band agps. Using the relaxed STO leads to predictions that are much closer to expectations. In particular, the predicted PBE band gap is quite close to the Materials Project PBE value. The experimental band gap is around 3.2 eV, which is reproduced very well by the GLLB-SC predict


# CHGnet

Crystal Hamiltonian Graph neural Network (CHGnet) is pretrained on the GGA/GGA+U static and relaxation trajectories from Materials Project, a comprehensive dataset consisting of more than 1.5 Million structures from 146k compounds spanning the whole periodic table.

CHGNet highlights its ability to study electron interactions and charge distribution in atomistic modeling with near DFT accuracy. The charge inference is realized by regularizing the atom features with DFT magnetic moments, which carry rich information about both local ionic environments and charge distribution.

Pretrained CHGNet achieves excellent performance on materials stability prediction from unrelaxed structures according to [Matbench Discovery repo](https://matbench-discovery.materialsproject.org/).

## predicting energy, force, stress, magmom

Examples for loading pre-trained CHGNet, predicting energy, force, stress, magmom as well as running structure optimization and MD.

~~~
import numpy as np
from pymatgen.core import Structure
# If the above line fails in Google Colab due to numpy version issue,
# please restart the runtime, and the problem will be solved
np.set_printoptions(precision=4, suppress=True)
from urllib.request import urlopen
url = "https://raw.githubusercontent.com/CederGroupHub/chgnet/main/examples/mp-18767-LiMnO2.cif"
cif = urlopen(url).read().decode("utf-8")
structure = Structure.from_str(cif, fmt="cif"
print(structure)
~~~
{: .python}

~~~
Full Formula (Li2 Mn2 O4)
Reduced Formula: LiMnO2
abc   :   2.868779   4.634475   5.832507
angles:  90.000000  90.000000  90.000000
pbc   :       True       True       True
Sites (8)
  #  SP      a    b         c
---  ----  ---  ---  --------
  0  Li+   0.5  0.5  0.37975
  1  Li+   0    0    0.62025
  2  Mn3+  0.5  0.5  0.863252
  3  Mn3+  0    0    0.136747
  4  O2-   0.5  0    0.360824
  5  O2-   0    0.5  0.098514
  6  O2-   0.5  0    0.901486
  7  O2-   0    0.5  0.639176
~~~
{: .output}

Define the model to predict energy, forces, stress and magmon

~~~
from chgnet.model import CHGNet
chgnet = CHGNet.load()
# Alternatively you can read your own model
# chgnet = CHGNet.from_file(model_path)
prediction = chgnet.predict_structure(structure)

for key, unit in [
    ("energy", "eV/atom"),
    ("forces", "eV/A"),
    ("stress", "GPa"),
    ("magmom", "mu_B"),
]:
    print(f"CHGNet-predicted {key} ({unit}):\n{prediction[key[0]]}\n")
~~~
{: .python}

~~~
CHGNet v0.3.0 initialized with 412,525 parameters
CHGNet will run on mps
CHGNet-predicted energy (eV/atom):
-7.3676910400390625

CHGNet-predicted forces (eV/A):
[[ 0.     -0.      0.0238]
 [ 0.      0.     -0.0238]
 [-0.      0.      0.0926]
 [-0.      0.     -0.0926]
 [-0.      0.     -0.0024]
 [-0.     -0.     -0.0131]
 [ 0.      0.      0.0131]
 [-0.      0.      0.0024]]

CHGNet-predicted stress (GPa):
[[-0.3041  0.     -0.    ]
 [ 0.      0.2232 -0.    ]
 [-0.     -0.     -0.1073]]

CHGNet-predicted magmom (mu_B):
[0.003  0.003  3.8694 3.8694 0.0441 0.0386 0.0386 0.0441]
~~~
{: .output}
