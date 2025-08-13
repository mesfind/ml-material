---
title: Adsorption Energies
teaching: 1
exercises: 0
questions:
- "How do I processing nc data?"
objectives:
- Understand xarray basics and capabilities.
- Load and manipulate data using xarray.
- Perform data processing and visualization tasks.
- Integrate xarray with other libraries.
- Optimize data processing pipelines.
- Explore best practices for data visualization.

---


# Catalysts

## Open Catalyst 2022 (OC22)

- All of these models are trained on various splits of the OC22 S2EF / IS2RE datasets. For details, see https://arxiv.org/abs/2206.08917 and 
- All OC22 models released here are trained on DFT total energies, in contrast to the OC20 models listed above, which are trained on adsorption energies.


## Pretrained models

**2025 recommendation:** We now suggest using the UMA model, trained on all of the FAIR chemistry datasets. The UMA model has a number of nice features over the previous checkpoints
1. It is state-of-the-art in out-of-domain prediction accuracy
2. It predicts total energies of a system, which are helpful for predicting properties beyond adsorption energies, and removes some ambiguity when your catalyst surface may reconstruct. 
3. The UMA small model is an energy conserving and smooth checkpoint, so should work much better for vibrational calculations, molecular dynamics, etc. 
4. The UMA model is most likely to be updated in the future.


### S2EF-Total models

| Model Name                        |Model	|Training	|Download	|val ID force MAE	|val ID energy MAE	|
|-----------------------------------|---	|---	|---	|---	|---	|
| GemNet-dT-S2EFS-OC22              |GemNet-dT | OC22	|[checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_09/oc22/s2ef/gndt_oc22_all_s2ef.pt) \| [config](https://github.com/facebookresearch/fairchem/blob/main/configs/oc22/s2ef/gemnet-dt/gemnet-dT.yml)	|0.032	|1.127	|
| GemNet-OC-S2EFS-OC22              | GemNet-OC                                                                                                                                                | OC22	|[checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_09/oc22/s2ef/gnoc_oc22_all_s2ef.pt) \| [config](https://github.com/facebookresearch/fairchem/blob/main/configs/oc22/s2ef/gemnet-oc/gemnet_oc.yml)	|0.030	|0.563	|
| GemNet-OC-S2EFS-OC20+OC22         | GemNet-OC                                                                                                                                                | OC20+OC22	|[checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_09/oc22/s2ef/gnoc_oc22_oc20_all_s2ef.pt) \| [config](https://github.com/facebookresearch/fairchem/blob/main/configs/oc22/s2ef/gemnet-oc/gemnet_oc_oc20_oc22.yml)	|0.027	|0.483	|
| GemNet-OC-S2EFS-nsn-OC20+OC22     | GemNet-OC <br><sub><sup>(trained with `enforce_max_neighbors_strictly=False`, [#467](https://github.com/facebookresearch/fairchem/pull/467))</sup></sub> | OC20+OC22	|[checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_05/oc22/s2ef/gnoc_oc22_oc20_all_s2ef.pt) \| [config](https://github.com/facebookresearch/fairchem/blob/main/configs/oc22/s2ef/gemnet-oc/gemnet_oc_oc20_oc22_degen_edges.yml)	|0.027	|0.458	|
| GemNet-OC-S2EFS-OC20->OC22        | GemNet-OC                                                                                                                                                | OC20->OC22	|[checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_09/oc22/s2ef/gnoc_finetune_all_s2ef.pt) \| [config](https://github.com/facebookresearch/fairchem/blob/main/configs/oc22/s2ef/gemnet-oc/gemnet_oc_finetune.yml)	|0.030	|0.417	|
| EquiformerV2-lE4-lF100-S2EFS-OC22 | EquiformerV2 ($\lambda_E$=4, $\lambda_F$=100)                                                                                                            | OC22  | [checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_10/oc22/s2ef/eq2_121M_e4_f100_oc22_s2ef.pt) \| [config](https://github.com/facebookresearch/fairchem/blob/main/configs/oc22/s2ef/equiformer_v2/equiformer_v2_N@18_L@6_M@2_e4_f100_121M.yml) | 0.023 | 0.447

## Intro to  adsorption energies

To introduce OCP we start with using it to calculate adsorption energies for a simple, atomic adsorbate where we specify the site we want to the adsorption energy for. Conceptually, you do this like you would do it with density functional theory. You create a slab model for the surface, place an adsorbate on it as an initial guess, run a relaxation to get the lowest energy geometry, and then compute the adsorption energy using reference states for the adsorbate.

You do have to be careful in the details though. Some OCP model/checkpoint combinations return a total energy like density functional theory would, but some return an "adsorption energy" directly. You have to know which one you are using. In this example, the model we use returns an "adsorption energy".

+++

## Intro to Adsorption energies

Adsorption energies are always a reaction energy (an adsorbed species relative to some implied combination of reactants). There are many common schemes in the catalysis literature. 

For example, you may want the adsorption energy of oxygen, and you might compute that from this reaction:

    1/2 O2 + slab -> slab-O

DFT has known errors with the energy of a gas-phase O2 molecule, so it's more common to compute this energy relative to a linear combination of H2O and H2. The suggested reference scheme for consistency with OC20 is a reaction

    x CO + (x + y/2 - z) H2 + (z-x) H2O + w/2 N2 + * -> CxHyOzNw*

Here, x=y=w=0, z=1, so the reaction ends up as

    -H2 + H2O + * -> O*

or alternatively,

    H2O + * -> O* + H2

It is possible through thermodynamic cycles to compute other reactions. If we can look up rH1 below and compute rH2

    H2 + 1/2 O2 -> H2O  re1 = -3.03 eV, from exp
    H2O + * -> O* + H2  re2  # Get from UMA 

Then, the adsorption energy for

    1/2O2 + * -> O*  

is just re1 + re2.

Based on https://atct.anl.gov/Thermochemical%20Data/version%201.118/species/?species_number=986, the formation energy of water is about -3.03 eV at standard state experimentally. You could also compute this using DFT, but you would probably get the wrong answer for this. 

The first step is getting a checkpoint for the model we want to use. UMA is currently the state-of-the-art model and will provide total energy estimates at the RPBE level of theory if you use the "OC20" task. 


````{admonition} Need to install fairchem-core or get UMA access or getting permissions/401 errors?
:class: dropdown


1. Install the necessary packages using pip, uv etc
```{code-cell} ipython3
:tags: [skip-execution]

! pip install fairchem-core fairchem-data-oc fairchem-applications-cattsunami
```

2. Get access to any necessary huggingface gated models 
    * Get and login to your Huggingface account
    * Request access to https://huggingface.co/facebook/UMA
    * Create a Huggingface token at https://huggingface.co/settings/tokens/ with the permission "Permissions: Read access to contents of all public gated repos you can access"
    * Add the token as an environment variable using `huggingface-cli login` or by setting the HF_TOKEN environment variable. 

```{code-cell} ipython3
:tags: [skip-execution]

# Login using the huggingface-cli utility
! huggingface-cli login

# alternatively,
import os
os.environ['HF_TOKEN'] = 'MY_TOKEN'
```

````

If you find your kernel is crashing, it probably means you have exceeded the allowed amount of memory. This checkpoint works fine in this example, but it may crash your kernel if you use it in the NRR example.

This next cell will automatically download the checkpoint from huggingface and load it. 

```{code-cell}
from __future__ import annotations

from fairchem.core import FAIRChemCalculator, pretrained_mlip

predictor = pretrained_mlip.get_predict_unit("uma-s-1")
calc = FAIRChemCalculator(predictor, task_name="oc20")
```

Next we can build a slab with an adsorbate on it. Here we use the ASE module to build a Pt slab. We use the experimental lattice constant that is the default. This can introduce some small errors with DFT since the lattice constant can differ by a few percent, and it is common to use DFT lattice constants. In this example, we do not constrain any layers.

```{code-cell}
from ase.build import add_adsorbate, fcc111
from ase.optimize import BFGS
```

```{code-cell}
# reference energies from a linear combination of H2O/N2/CO/H2!
atomic_reference_energies = {
    "H": -3.477,
    "N": -8.083,
    "O": -7.204,
    "C": -7.282,
}

re1 = -3.03

slab = fcc111("Pt", size=(2, 2, 5), vacuum=20.0)
slab.pbc = True

adslab = slab.copy()
add_adsorbate(adslab, "O", height=1.2, position="fcc")

slab.set_calculator(calc)
opt = BFGS(slab)
opt.run(fmax=0.05, steps=100)
slab_e = slab.get_potential_energy()

adslab.set_calculator(calc)
opt = BFGS(adslab)
opt.run(fmax=0.05, steps=100)
adslab_e = adslab.get_potential_energy()

# Energy for ((H2O-H2) + * -> *O) + (H2 + 1/2O2 -> H2) leads to 1/2O2 + * -> *O!
adslab_e - slab_e - atomic_reference_energies["O"] + re1
```

It is good practice to look at your geometries to make sure they are what you expect.

```{code-cell}
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

fig, axs = plt.subplots(1, 2)
plot_atoms(slab, axs[0])
plot_atoms(slab, axs[1], rotation=("-90x"))
axs[0].set_axis_off()
axs[1].set_axis_off()
```

```{code-cell}
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

fig, axs = plt.subplots(1, 2)
plot_atoms(adslab, axs[0])
plot_atoms(adslab, axs[1], rotation=("-90x"))
axs[0].set_axis_off()
axs[1].set_axis_off()
```

How did we do? We need a reference point. In the paper below, there is an atomic adsorption energy for O on Pt(111) of about -4.264 eV. This is for the reaction O + * -> O*. To convert this to the dissociative adsorption energy, we have to add the reaction:

    1/2 O2 -> O   D = 2.58 eV (expt)

to get a comparable energy of about -1.68 eV. There is about ~0.2 eV difference (we predicted -1.47 eV above, and the reference comparison is -1.68 eV) to account for. The biggest difference is likely due to the differences in exchange-correlation functional. The reference data used the PBE functional, and eSCN was trained on *RPBE* data. To additional places where there are differences include:

1. Difference in lattice constant
2. The reference energy used for the experiment references. These can differ by up to 0.5 eV from comparable DFT calculations.
3. How many layers are relaxed in the calculation

Some of these differences tend to be systematic, and you can calibrate and correct these, especially if you can augment these with your own DFT calculations. 

See [convergence study](#Convergence-study) for some additional studies of factors that influence this number.

+++

### Exercises

1. Explore the effect of the lattice constant on the adsorption energy.
2. Try different sites, including the bridge and top sites. Compare the energies, and inspect the resulting geometries.

+++

## Trends in adsorption energies across metals.

Xu, Z., & Kitchin, J. R. (2014). Probing the coverage dependence of site and adsorbate configurational correlations on (111) surfaces of late transition metals. J. Phys. Chem. C, 118(44), 25597–25602. http://dx.doi.org/10.1021/jp508805h

[Supporting information](https://pubs.acs.org/doi/suppl/10.1021/jp508805h/suppl_file/jp508805h_si_001.pdf).

These are atomic adsorption energies:

    O + * -> O*

We have to do some work to get comparable numbers from OCP

    H2 + 1/2 O2 -> H2O  re1 = -3.03 eV
    H2O + * -> O* + H2  re2   # Get from UMA
    O -> 1/2 O2         re3 = -2.58 eV

Then, the adsorption energy for

    O + * -> O*  

is just re1 + re2 + re3.

Here we just look at the fcc site on Pt. First, we get the data stored in the paper.

Next we get the structures and compute their energies. Some subtle points are that we have to account for stoichiometry, and normalize the adsorption energy by the number of oxygens.

+++

First we get a reference energy from the paper (PBE, 0.25 ML O on Pt(111)).

```{code-cell}
import json

with open("energies.json") as f:
    edata = json.load(f)

with open("structures.json") as f:
    sdata = json.load(f)

edata["Pt"]["O"]["fcc"]["0.25"]
```

Next, we load data from the SI to get the geometry to start from.

```{code-cell}
with open("structures.json") as f:
    s = json.load(f)

sfcc = s["Pt"]["O"]["fcc"]["0.25"]
```

Next, we construct the atomic geometry, run the geometry optimization, and compute the energy.

```{code-cell}
re3 = -2.58  # O -> 1/2 O2         re3 = -2.58 eV

from ase import Atoms

adslab = Atoms(sfcc["symbols"], positions=sfcc["pos"], cell=sfcc["cell"], pbc=True)

# Grab just the metal surface atoms
slab = adslab[adslab.arrays["numbers"] == adslab.arrays["numbers"][0]]
adsorbates = adslab[~(adslab.arrays["numbers"] == adslab.arrays["numbers"][0])]
slab.set_calculator(calc)
opt = BFGS(slab)
opt.run(fmax=0.05, steps=100)

adslab.set_calculator(calc)
opt = BFGS(adslab)

opt.run(fmax=0.05, steps=100)
re2 = (
    adslab.get_potential_energy()
    - slab.get_potential_energy()
    - sum([atomic_reference_energies[x] for x in adsorbates.get_chemical_symbols()])
)

nO = 0
for atom in adslab:
    if atom.symbol == "O":
        nO += 1
        re2 += re1 + re3

print(re2 / nO)
```

### Site correlations

This cell reproduces a portion of a figure in the paper. We compare oxygen adsorption energies in the fcc and hcp sites across metals and coverages. These adsorption energies are highly correlated with each other because the adsorption sites are so similar.

At higher coverages, the agreement is not as good. This is likely because the model is extrapolating and needs to be fine-tuned.

```{code-cell}
import time

from tqdm import tqdm

t0 = time.time()

data = {"fcc": [], "hcp": []}

refdata = {"fcc": [], "hcp": []}


for metal in ["Cu", "Ag", "Pd", "Pt", "Rh", "Ir"]:
    print(metal)
    for site in ["fcc", "hcp"]:
        for adsorbate in ["O"]:
            for coverage in tqdm(["0.25"]):

                entry = s[metal][adsorbate][site][coverage]

                adslab = Atoms(
                    entry["symbols"],
                    positions=entry["pos"],
                    cell=entry["cell"],
                    pbc=True,
                )

                # Grab just the metal surface atoms
                adsorbates = adslab[
                    ~(adslab.arrays["numbers"] == adslab.arrays["numbers"][0])
                ]

                slab = adslab[adslab.arrays["numbers"] == adslab.arrays["numbers"][0]]
                slab.set_calculator(calc)
                opt = BFGS(slab)
                opt.run(fmax=0.05, steps=100)

                adslab.set_calculator(calc)
                opt = BFGS(adslab)
                opt.run(fmax=0.05, steps=100)

                re2 = (
                    adslab.get_potential_energy()
                    - slab.get_potential_energy()
                    - sum(
                        [
                            atomic_reference_energies[x]
                            for x in adsorbates.get_chemical_symbols()
                        ]
                    )
                )

                nO = 0
                for atom in adslab:
                    if atom.symbol == "O":
                        nO += 1
                        re2 += re1 + re3

                data[site] += [re2 / nO]
                refdata[site] += [edata[metal][adsorbate][site][coverage]]

f"Elapsed time = {time.time() - t0} seconds"
```

First, we compare the computed data and reference data. There is a systematic difference of about 0.5 eV due to the difference between RPBE and PBE functionals, and other subtle differences like lattice constant differences and reference energy differences. This is pretty typical, and an expected deviation.

```{code-cell}
plt.plot(refdata["fcc"], data["fcc"], "r.", label="fcc")
plt.plot(refdata["hcp"], data["hcp"], "b.", label="hcp")
plt.plot([-5.5, -3.5], [-5.5, -3.5], "k-")
plt.xlabel("Ref. data (DFT)")
plt.ylabel("UMA-OC20 prediction");
```

Next we compare the correlation between the hcp and fcc sites. Here we see the same trends. The data falls below the parity line because the hcp sites tend to be a little weaker binding than the fcc sites.

```{code-cell}
plt.plot(refdata["hcp"], refdata["fcc"], "r.")
plt.plot(data["hcp"], data["fcc"], ".")
plt.plot([-6, -1], [-6, -1], "k-")
plt.xlabel("$H_{ads, hcp}$")
plt.ylabel("$H_{ads, fcc}$")
plt.legend(["DFT (PBE)", "UMA-OC20"]);
```

### Exercises

1. You can also explore a few other adsorbates: C, H, N. 
2. Explore the higher coverages. The deviations from the reference data are expected to be higher, but relative differences tend to be better. You probably need fine tuning to improve this performance. This data set doesn't have forces though, so it isn't practical to do it here.

+++

## Next steps

In the next step, we consider some more complex adsorbates in nitrogen reduction, and how we can leverage OCP to automate the search for the most stable adsorbate geometry. See [the next step](./NRR/NRR_example-gemnet).

+++

### Convergence study

In [Calculating adsorption energies](#Calculating-adsorption-energies) we discussed some possible reasons we might see a discrepancy. Here we investigate some factors that impact the computed energies.

In this section, the energies refer to the reaction 1/2 O2 -> O*.

+++

### Effects of number of layers

Slab thickness could be a factor. Here we relax the whole slab, and see by about 4 layers the energy is converged to ~0.02 eV.

```{code-cell}
for nlayers in [3, 4, 5, 6, 7, 8]:
    slab = fcc111("Pt", size=(2, 2, nlayers), vacuum=10.0)

    slab.pbc = True
    slab.set_calculator(calc)
    opt_slab = BFGS(slab, logfile=None)
    opt_slab.run(fmax=0.05, steps=100)
    slab_e = slab.get_potential_energy()

    adslab = slab.copy()
    add_adsorbate(adslab, "O", height=1.2, position="fcc")

    adslab.pbc = True
    adslab.set_calculator(calc)
    opt_adslab = BFGS(adslab, logfile=None)
    opt_adslab.run(fmax=0.05, steps=100)
    adslab_e = adslab.get_potential_energy()

    print(
        f"nlayers = {nlayers}: {adslab_e - slab_e - atomic_reference_energies['O'] + re1:1.2f} eV"
    )
```

### Effects of relaxation

It is common to only relax a few layers, and constrain lower layers to bulk coordinates. We do that here. We only relax the adsorbate and the top layer.

This has a small effect (0.1 eV).

```{code-cell}
from ase.constraints import FixAtoms

for nlayers in [3, 4, 5, 6, 7, 8]:
    slab = fcc111("Pt", size=(2, 2, nlayers), vacuum=10.0)

    slab.set_constraint(FixAtoms(mask=[atom.tag > 1 for atom in slab]))
    slab.pbc = True
    slab.set_calculator(calc)
    opt_slab = BFGS(slab, logfile=None)
    opt_slab.run(fmax=0.05, steps=100)
    slab_e = slab.get_potential_energy()

    adslab = slab.copy()
    add_adsorbate(adslab, "O", height=1.2, position="fcc")

    adslab.set_constraint(FixAtoms(mask=[atom.tag > 1 for atom in adslab]))
    adslab.pbc = True
    adslab.set_calculator(calc)
    opt_adslab = BFGS(adslab, logfile=None)
    opt_adslab.run(fmax=0.05, steps=100)
    adslab_e = adslab.get_potential_energy()

    print(
        f"nlayers = {nlayers}: {adslab_e - slab_e - atomic_reference_energies['O'] + re1:1.2f} eV"
    )
```

### Unit cell size

Coverage effects are quite noticeable with oxygen. Here we consider larger unit cells. This effect is large, and the results don't look right, usually adsorption energies get more favorable at lower coverage, not less. This suggests fine-tuning could be important even at low coverages.

```{code-cell}
for size in [1, 2, 3, 4, 5]:

    slab = fcc111("Pt", size=(size, size, 5), vacuum=10.0)

    slab.set_constraint(FixAtoms(mask=[atom.tag > 1 for atom in slab]))
    slab.pbc = True
    slab.set_calculator(calc)
    opt_slab = BFGS(slab, logfile=None)
    opt_slab.run(fmax=0.05, steps=100)
    slab_e = slab.get_potential_energy()

    adslab = slab.copy()
    add_adsorbate(adslab, "O", height=1.2, position="fcc")

    adslab.set_constraint(FixAtoms(mask=[atom.tag > 1 for atom in adslab]))
    adslab.pbc = True
    adslab.set_calculator(calc)
    opt_adslab = BFGS(adslab, logfile=None)
    opt_adslab.run(fmax=0.05, steps=100)
    adslab_e = adslab.get_potential_energy()

    print(
        f"({size}x{size}): {adslab_e - slab_e - atomic_reference_energies['O'] + re1:1.2f} eV"
    )
```

## Summary

As with DFT, you should take care to see how these kinds of decisions affect your results, and determine if they would change any interpretations or not.

# Expert adsorption energies


One of the most common tasks in computational catalysis is calculating the binding energies or adsorption energies of small molecules on catalyst surfaces.

````{admonition} Need to install fairchem-core or get UMA access or getting permissions/401 errors?
:class: dropdown


1. Install the necessary packages using pip, uv etc
```{code-cell} ipython3
:tags: [skip-execution]

! pip install fairchem-core fairchem-data-oc fairchem-applications-cattsunami
```

2. Get access to any necessary huggingface gated models 
    * Get and login to your Huggingface account
    * Request access to https://huggingface.co/facebook/UMA
    * Create a Huggingface token at https://huggingface.co/settings/tokens/ with the permission "Permissions: Read access to contents of all public gated repos you can access"
    * Add the token as an environment variable using `huggingface-cli login` or by setting the HF_TOKEN environment variable. 

```{code-cell} ipython3
:tags: [skip-execution]

# Login using the huggingface-cli utility
! huggingface-cli login

# alternatively,
import os
os.environ['HF_TOKEN'] = 'MY_TOKEN'
```

````

```{code-cell} ipython3
from __future__ import annotations

import os
import pickle
import time
from glob import glob

import ase.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.optimize import QuasiNewton
from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.data.oc.core import Adsorbate, AdsorbateSlabConfig, Bulk, Slab
from fairchem.data.oc.utils import DetectTrajAnomaly
from scipy.stats import linregress

# Set random seed to ensure adsorbate enumeration yields a valid candidate
# If using a larger number of random samples this wouldn't be necessary
np.random.seed(22)
```

## Introduction

We will reproduce Fig 6b from the following paper: Zhou, Jing, et al. "Enhanced Catalytic Activity of Bimetallic Ordered Catalysts for Nitrogen Reduction Reaction by Perturbation of Scaling Relations." ACS Catalysis 134 (2023): 2190-2201 (https://doi.org/10.1021/acscatal.2c05877).

The gist of this figure is a correlation between H* and NNH* adsorbates across many different alloy surfaces. Then, they identify a dividing line between these that separates surfaces known for HER and those known for NRR.

To do this, we will enumerate adsorbate-slab configurations and run ML relaxations on them to find the lowest energy configuration. We will assess parity between the model predicted values and those reported in the paper. Finally we will make the figure and assess separability of the NRR favored and HER favored domains.

+++

## Enumerate the adsorbate-slab configurations to run relaxations on

+++

Be sure to set the path in `fairchem/data/oc/configs/paths.py` to point to the correct place or pass the paths as an argument. The database pickles can be found in `fairchem/data/oc/databases/pkls` (some pkl files are only downloaded by running the command `python src/fairchem/core/scripts/download_large_files.py oc` from the root of the fairchem repo). We will show one explicitly here as an example and then run all of them in an automated fashion for brevity.

```{code-cell} ipython3
from pathlib import Path

import fairchem.data.oc

db = Path(fairchem.data.oc.__file__).parent / Path("databases/pkls/adsorbates.pkl")
db
```

### Work out a single example

We load one bulk id, create a bulk reference structure from it, then generate the surfaces we want to compute.

```{code-cell} ipython3
bulk_src_id = "oqmd-343039"
adsorbate_smiles_nnh = "*N*NH"
adsorbate_smiles_h = "*H"

bulk = Bulk(bulk_src_id_from_db=bulk_src_id, bulk_db_path="NRR_example_bulks.pkl")
adsorbate_H = Adsorbate(
    adsorbate_smiles_from_db=adsorbate_smiles_h, adsorbate_db_path=db
)
adsorbate_NNH = Adsorbate(
    adsorbate_smiles_from_db=adsorbate_smiles_nnh, adsorbate_db_path=db
)
slab = Slab.from_bulk_get_specific_millers(bulk=bulk, specific_millers=(1, 1, 1))
slab
```

We now need to generate potential placements. We use two kinds of guesses, a heuristic and a random approach. This cell generates 13 potential adsorption geometries.

```{code-cell} ipython3
# Perform heuristic placements
heuristic_adslabs = AdsorbateSlabConfig(slab[0], adsorbate_H, mode="heuristic")

# Perform random placements
# (for AdsorbML we use `num_sites = 100` but we will use 4 for brevity here)
random_adslabs = AdsorbateSlabConfig(
    slab[0], adsorbate_H, mode="random_site_heuristic_placement", num_sites=4
)

adslabs = [*heuristic_adslabs.atoms_list, *random_adslabs.atoms_list]
len(adslabs)
```

Let's see what we are looking at. It is a little tricky to see the tiny H atom in these figures, but with some inspection you can see there are ontop, bridge, and hollow sites in different places. This is not an exhaustive search; you can increase the number of random placements to check more possibilities. The main idea here is to *increase* the probability you find the most relevant sites.

```{code-cell} ipython3
from ase.visualize.plot import plot_atoms

fig, axs = plt.subplots(4, 4)

for i, slab in enumerate(adslabs):
    plot_atoms(slab, axs[i % 4, i // 4])
    axs[i % 4, i // 4].set_axis_off()

for i in range(16):
    axs[i % 4, i // 4].set_axis_off()

plt.tight_layout()
```

#### Run an ML relaxation

We will use an ASE compatible calculator to run these.

+++

Running the model with QuasiNewton prints at each relaxation step which is a lot to print. So we will just run one to demonstrate what happens on each iteration.

```{code-cell} ipython3
os.makedirs(f"data/{bulk_src_id}_{adsorbate_smiles_h}", exist_ok=True)

# Define the
predictor = pretrained_mlip.get_predict_unit("uma-s-1")
calc = FAIRChemCalculator(predictor, task_name="oc20")
```

Now we setup and run the relaxation.

```{code-cell} ipython3
t0 = time.time()
os.makedirs(f"data/{bulk_src_id}_H", exist_ok=True)
adslab = adslabs[0]
adslab.calc = calc
adslab.pbc = True
opt = QuasiNewton(adslab, trajectory=f"data/{bulk_src_id}_H/test.traj")
opt.run(fmax=0.05, steps=100)

print(f"Elapsed time {time.time() - t0:1.1f} seconds")
```

With a GPU this runs pretty quickly. It is much slower on a CPU.

+++

## Run all the systems

In principle you can run all the systems now. It takes about an hour though, and we leave that for a later exercise if you want. For now we will run the first two, and for later analysis we provide a results file of all the runs. Let's read in our reference file and take a look at what is in it.

```{code-cell} ipython3
with open("NRR_example_bulks.pkl", "rb") as f:
    bulks = pickle.load(f)

bulks
```

We have 19 bulk materials we will consider. Next we extract the `src-id` for each one.

```{code-cell} ipython3
bulk_ids = [row["src_id"] for row in bulks]
```

In theory you would run all of these, but it takes about an hour with a GPU. We provide the relaxation logs and trajectories in the repo for the next step.

These steps are embarrassingly parallel, and can be launched that way to speed things up. The only thing you need to watch is that you don't exceed the available RAM, which will cause the Jupyter kernel to crash.

+++

The goal here is to relax each candidate adsorption geometry and save the results in a trajectory file we will analyze later. Each trajectory file will have the geometry and final energy of the relaxed structure. 

It is somewhat time consuming to run this. We're going to use a small number of bulks for the testing of this documentation, but otherwise run all of the results for the actual documentation.

```{code-cell} ipython3
import os

fast_docs = os.environ.get("FAST_DOCS", "false").lower() == "true"
if fast_docs:
    num_bulks = 1
    num_sites = 5
    relaxation_steps = 20
else:
    num_bulks = -1
    num_sites = 20
    relaxation_steps = 300
```

```{code-cell} ipython3
import random
import time

from tqdm import tqdm

tinit = time.time()

random.seed(42)
random.shuffle(bulk_ids)

# Note we're just doing the first bulk_id!
for bulk_src_id in tqdm(bulk_ids[:num_bulks]):

    # Set up data directories
    os.makedirs("data/slabs/", exist_ok=True)
    os.makedirs(f"data/adslabs/{bulk_src_id}_H", exist_ok=True)
    os.makedirs(f"data/adslabs/{bulk_src_id}_NNH", exist_ok=True)

    # Enumerate slabs and establish adsorbates
    bulk = Bulk(bulk_src_id_from_db=bulk_src_id, bulk_db_path="NRR_example_bulks.pkl")
    slab = Slab.from_bulk_get_specific_millers(bulk=bulk, specific_millers=(1, 1, 1))

    slab_atoms = slab[0].atoms.copy()
    slab_atoms.calc = calc
    slab_atoms.pbc = True
    opt = QuasiNewton(
        slab_atoms,
        trajectory=f"data/slabs/{bulk_src_id}.traj",
        logfile=f"data/slabs/{bulk_src_id}.log",
    )
    opt.run(fmax=0.05, steps=relaxation_steps)
    print(
        f"  Elapsed time: {time.time() - t0:1.1f} seconds for data/slabs/{bulk_src_id} slab relaxation"
    )

    # Perform heuristic placements
    heuristic_adslabs_H = AdsorbateSlabConfig(
        slab[0],
        adsorbate_H,
        mode="random_site_heuristic_placement",
        num_sites=num_sites,
    )
    heuristic_adslabs_NNH = AdsorbateSlabConfig(
        slab[0],
        adsorbate_NNH,
        mode="random_site_heuristic_placement",
        num_sites=num_sites,
    )

    print(f"{len(heuristic_adslabs_H.atoms_list)} H slabs to compute for {bulk_src_id}")
    print(
        f"{len(heuristic_adslabs_NNH.atoms_list)} NNH slabs to compute for {bulk_src_id}"
    )

    for idx, adslab in enumerate(heuristic_adslabs_H.atoms_list):
        t0 = time.time()
        adslab.calc = calc
        adslab.pbc = True
        print(f"Running data/adslabs/{bulk_src_id}_H/{idx}")
        opt = QuasiNewton(
            adslab,
            trajectory=f"data/adslabs/{bulk_src_id}_H/{idx}.traj",
            logfile=f"data/adslabs/{bulk_src_id}_H/{idx}.log",
        )
        opt.run(fmax=0.05, steps=200)
        print(
            f"  Elapsed time: {time.time() - t0:1.1f} seconds for data/adslabs/{bulk_src_id}_H/{idx}"
        )

    for idx, adslab in enumerate(heuristic_adslabs_NNH.atoms_list):
        t0 = time.time()
        adslab.calc = calc
        adslab.pbc = True
        print(f"Running data/adslabs/{bulk_src_id}_NNH/{idx}")
        opt = QuasiNewton(
            adslab,
            trajectory=f"data/adslabs/{bulk_src_id}_NNH/{idx}.traj",
            logfile=f"data/adslabs/{bulk_src_id}_NNH/{idx}.log",
        )
        opt.run(fmax=0.05, steps=relaxation_steps)
        print(
            f"  Elapsed time: {time.time() - t0:1.1f} seconds for data/adslabs/{bulk_src_id}_NNH/{idx}"
        )

print(f"Elapsed time: {time.time() - tinit:1.1f} seconds")
```

## Parse the trajectories and post-process

As a post-processing step we check to see if:

1. the adsorbate desorbed
2. the adsorbate disassociated
3. the adsorbate intercalated
4. the surface has changed

We check these because they affect our referencing scheme and may result in energies that don't mean what we think, e.g. they aren't just adsorption, but include contributions from other things like desorption, dissociation or reconstruction. For (4), the relaxed surface should really be supplied as well. It will be necessary when correcting the SP / RX energies later. Since we don't have it here, we will ommit supplying it, and the detector will instead compare the initial and final slab from the adsorbate-slab relaxation trajectory. If a relaxed slab is provided, the detector will compare it and the slab after the adsorbate-slab relaxation. The latter is more correct!

To compute the adsorption energies using the total energy UMA-OC20 model, we'll need the gas-phase reference energies from OC20 (see the original paper!). You could also calculate these quickly in DFT using a linear combination of H2O, H2, N2, and CO.

```{code-cell} ipython3
# reference energies from a linear combination of H2O/N2/CO/H2!
atomic_reference_energies = {
    "H": -3.477,
    "N": -8.083,
    "O": -7.204,
    "C": -7.282,
}
```

In this loop we find the most stable (most negative) adsorption energy for each adsorbate on each surface and save them in a DataFrame.

```{code-cell} ipython3
# Iterate over trajs to extract results
min_E = []
for file_outer in glob("data/adslabs/*"):
    ads = file_outer.split("_")[1]
    bulk = file_outer.split("/")[-1].split("_")[0]

    slab = ase.io.read(f"data/slabs/{bulk}.traj")
    results = []
    for file in glob(f"{file_outer}/*.traj"):
        rx_id = file.split("/")[-1].split(".")[0]
        traj = ase.io.read(file, ":")

        # Check to see if the trajectory is anomolous
        detector = DetectTrajAnomaly(traj[0], traj[-1], traj[0].get_tags())
        anom = (
            detector.is_adsorbate_dissociated()
            or detector.is_adsorbate_desorbed()
            or detector.has_surface_changed()
            or detector.is_adsorbate_intercalated()
        )
        rx_energy = (
            traj[-1].get_potential_energy()
            - slab.get_potential_energy()
            - sum(
                [
                    atomic_reference_energies[x]
                    for x in traj[0][traj[0].get_tags() == 2].get_chemical_symbols()
                ]
            )
        )

        results.append(
            {
                "relaxation_idx": rx_id,
                "relaxed_atoms": traj[-1],
                "relaxed_energy_ml": rx_energy,
                "anomolous": anom,
            }
        )
    df = pd.DataFrame(results)

    df = df[~df.anomolous].copy().reset_index()
    min_e = min(df.relaxed_energy_ml.tolist())
    min_E.append({"adsorbate": ads, "bulk_id": bulk, "min_E_ml": min_e})

df = pd.DataFrame(min_E)
df_h = df[df.adsorbate == "H"]
df_nnh = df[df.adsorbate == "NNH"]
df_flat = df_h.merge(df_nnh, on="bulk_id")
```

## Make parity plots for values obtained by ML v. reported in the paper

```{code-cell} ipython3
# Add literature data to the dataframe
with open("literature_data.pkl", "rb") as f:
    literature_data = pickle.load(f)
df_all = df_flat.merge(pd.DataFrame(literature_data), on="bulk_id")
```

```{code-cell} ipython3
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
f.set_figheight(15)
x = df_all.min_E_ml_x.tolist()
y = df_all.E_lit_H.tolist()
ax1.set_title("*H parity")
ax1.plot([-3.5, 2], [-3.5, 2], "k-", linewidth=3)
slope, intercept, r, p, se = linregress(x, y)
ax1.plot(
    [-3.5, 2],
    [
        -3.5 * slope + intercept,
        2 * slope + intercept,
    ],
    "k--",
    linewidth=2,
)

ax1.legend(
    [
        "y = x",
        f"y = {slope:1.2f} x + {intercept:1.2f}, R-sq = {r**2:1.2f}",
    ],
    loc="upper left",
)
ax1.scatter(x, y)
ax1.axis("square")
ax1.set_xlim([-3.5, 2])
ax1.set_ylim([-3.5, 2])
ax1.set_xlabel("dE predicted UMA [eV]")
ax1.set_ylabel("dE NRR paper [eV]")


x = df_all.min_E_ml_y.tolist()
y = df_all.E_lit_NNH.tolist()
ax2.set_title("*N*NH parity")
ax2.plot([-3.5, 2], [-3.5, 2], "k-", linewidth=3)
slope, intercept, r, p, se = linregress(x, y)
ax2.plot(
    [-3.5, 2],
    [
        -3.5 * slope + intercept,
        2 * slope + intercept,
    ],
    "k--",
    linewidth=2,
)

ax2.legend(
    [
        "y = x",
        f"y = {slope:1.2f} x + {intercept:1.2f}, R-sq = {r**2:1.2f}",
    ],
    loc="upper left",
)
ax2.scatter(x, y)
ax2.axis("square")
ax2.set_xlim([-3.5, 2])
ax2.set_ylim([-3.5, 2])
ax2.set_xlabel("dE predicted UMA [eV]")
ax2.set_ylabel("dE NRR paper [eV]")
f.set_figwidth(15)
f.set_figheight(7)
```

## Make figure 6b and compare to literature results

```{code-cell} ipython3
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
x = df_all[df_all.reaction == "HER"].min_E_ml_y.tolist()
y = df_all[df_all.reaction == "HER"].min_E_ml_x.tolist()
comp = df_all[df_all.reaction == "HER"].composition.tolist()

ax1.scatter(x, y, c="r", label="HER")
for i, txt in enumerate(comp):
    ax1.annotate(txt, (x[i], y[i]))

x = df_all[df_all.reaction == "NRR"].min_E_ml_y.tolist()
y = df_all[df_all.reaction == "NRR"].min_E_ml_x.tolist()
comp = df_all[df_all.reaction == "NRR"].composition.tolist()
ax1.scatter(x, y, c="b", label="NRR")
for i, txt in enumerate(comp):
    ax1.annotate(txt, (x[i], y[i]))


ax1.legend()
ax1.set_xlabel("dE *N*NH predicted UMA [eV]")
ax1.set_ylabel("dE *H predicted UMA [eV]")


x = df_all[df_all.reaction == "HER"].E_lit_NNH.tolist()
y = df_all[df_all.reaction == "HER"].E_lit_H.tolist()
comp = df_all[df_all.reaction == "HER"].composition.tolist()

ax2.scatter(x, y, c="r", label="HER")
for i, txt in enumerate(comp):
    ax2.annotate(txt, (x[i], y[i]))

x = df_all[df_all.reaction == "NRR"].E_lit_NNH.tolist()
y = df_all[df_all.reaction == "NRR"].E_lit_H.tolist()
comp = df_all[df_all.reaction == "NRR"].composition.tolist()
ax2.scatter(x, y, c="b", label="NRR")
for i, txt in enumerate(comp):
    ax2.annotate(txt, (x[i], y[i]))

ax2.legend()
ax2.set_xlabel("dE *N*NH literature [eV]")
ax2.set_ylabel("dE *H literature [eV]")
f.set_figwidth(15)
f.set_figheight(7)
```
