---
title: Adsorption Energies 
teaching: 1
exercises: 0
questions:
  - How can machine learning models accelerate adsorption energy calculations
  - What are the best practices for computing reliable adsorption energies using ML
objectives:
  - Understand the structure and purpose of the OC22 dataset
  - Use the UMA model to predict total energies of adsorbate-slab systems
  - Compute adsorption energies using thermodynamic cycles and reference states
  - Analyze trends in oxygen adsorption across late transition metals
  - Perform convergence tests for slab thickness and relaxation depth
  - Evaluate model predictions against DFT and literature data
---

<!-- MathJax -->
<script type="text/javascript"
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>

## Introduction

The search for efficient electrocatalysts relies heavily on accurate predictions of how molecules bind to surfaces. The strength of this interaction is often quantified by the **adsorption energy**, a key descriptor in activity volcano plots and scaling relations. Traditionally, these energies are computed using density functional theory, a method that is accurate but computationally expensive.

The Open Catalyst Project addresses this bottleneck by releasing large datasets and pretrained machine learning interatomic potentials. The **Open Catalyst 2022 (OC22)** dataset was introduced to expand training data for **oxide electrocatalysts**, which are essential for the **oxygen evolution reaction (OER)**. It contains **62,331 DFT relaxations** and approximately **9.85 million single-point calculations** across diverse oxide materials, adsorbates, and surface coverages.

Unlike OC20, which focused on predicting adsorption energies directly, OC22 emphasizes **generalized total energy prediction**. This enables applications beyond static binding, including molecular dynamics, phonon calculations, and modeling of reconstructed or charged surfaces.

In this tutorial, we use the **Unified Multimodal Architecture (UMA)** model, trained on OC20, OC22, and other FAIR chemistry datasets, to compute adsorption energies of atomic oxygen on Pt(111) and across a range of late transition metals. We follow a workflow analogous to DFT: build a surface, place an adsorbate, relax the system using ML, and compute the energy change relative to reference states.

We also perform convergence studies and compare our results to published DFT data.



## Setting Up the Environment

Before we begin, we need to install the required packages and gain access to the UMA model, which is hosted on Hugging Face with gated access.

First, install the core packages from the FairChem suite

```{python}
# :tags: [skip-execution]
! pip install fairchem-core fairchem-data-oc fairchem-applications-cattsunami
```

Next, request access to the UMA model

- Visit [https://huggingface.co/facebook/UMA](https://huggingface.co/facebook/UMA)
- Log in and click “Request Access”
- After approval, generate a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- The token should have “Read access to public gated repos”

You can authenticate using the CLI or by setting an environment variable

```{python}
# :tags: [skip-execution]
# Option 1: CLI login
! huggingface-cli login

# Option 2: Set token in environment
import os
os.environ["HF_TOKEN"] = "your_hf_token_here"
```

Once authenticated, load the UMA-S-1 model for the OC20 total energy task

```{python}
from __future__ import annotations
from fairchem.core import FAIRChemCalculator, pretrained_mlip

predictor = pretrained_mlip.get_predict_unit("uma-s-1")
calc = FAIRChemCalculator(predictor, task_name="oc20")
```

This model predicts **RPBE-level total energies**, allowing us to compute adsorption energies using thermodynamic cycles.



## Computing Oxygen Adsorption on Pt(111)

We begin by constructing a Pt(111) surface and placing an oxygen atom in the fcc hollow site. We use the experimental lattice constant by default, though you can substitute a DFT-optimized value if preferred.

```{python}
from ase.build import add_adsorbate, fcc111
from ase.optimize import BFGS

slab = fcc111("Pt", size=(2, 2, 5), vacuum=20.0)
slab.pbc = True

adslab = slab.copy()
add_adsorbate(adslab, "O", height=1.2, position="fcc")
```

We now relax both the clean slab and the adsorbed system using the BFGS optimizer

```{python}
slab.set_calculator(calc)
opt = BFGS(slab)
opt.run(fmax=0.05, steps=100)
slab_e = slab.get_potential_energy()

adslab.set_calculator(calc)
opt = BFGS(adslab)
opt.run(fmax=0.05, steps=100)
adslab_e = adslab.get_potential_energy()
```

To compute the adsorption energy, we must define a reference state for atomic oxygen. DFT performs poorly on gas-phase O₂, so we use a thermochemical cycle based on water formation

- Formation of water from hydrogen and oxygen (experimental enthalpy change):
$$
\mathrm{H}_2 + \frac{1}{2}\mathrm{O}_2 \rightarrow \mathrm{H}_2\mathrm{O} \quad \Delta H = -3.03\, \text{eV}
$$

- Dissociation of atomic oxygen into half a molecule of oxygen gas (with given dissociation energy):
$$
\mathrm{O} \rightarrow \frac{1}{2} \mathrm{O}_2 \quad \Delta H = -2.58\, \text{eV}
$$

We also use atomic reference energies from the OC20 dataset

```{python}
atomic_reference_energies = {
    "H": -3.477,
    "N": -8.083,
    "O": -7.204,
    "C": -7.282
}
```

The adsorption energy is then

```{python}
re1 = -3.03
re3 = -2.58

adsorption_energy = (adslab_e - slab_e 
                     - atomic_reference_energies["O"] 
                     + re1 + re3)

print(f"Adsorption energy of O on Pt(111): {adsorption_energy:.3f} eV")
```

This gives a value of approximately -1.47 eV.

---

## Comparison with Literature

Xu and Kitchin reported a PBE-calculated adsorption energy of -4.264 eV for O + * → O* on Pt(111). Converting to the dissociative scale using the O₂ dissociation energy gives -1.684 eV.

Our ML prediction is about 0.21 eV higher. This difference is expected and primarily due to

- The use of RPBE in UMA versus PBE in the literature
- Slight differences in lattice constant
- Reference energy calibration

These systematic shifts are common and can be corrected with a small set of DFT calculations if needed.

---

## Visualizing the Structures

It is good practice to inspect the relaxed geometries

```{python}
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
plot_atoms(slab, axs[0], rotation="0x,0y,0z")
plot_atoms(adslab, axs[1], rotation="0x,0y,0z")
axs[0].set_title("Clean Pt(111)")
axs[1].set_title("O on fcc site")
for ax in axs:
    ax.set_axis_off()
plt.tight_layout()
plt.show()
```

---

## Trends Across Late Transition Metals

We now extend our analysis to a set of late transition metals: Cu, Ag, Pd, Pt, Rh, and Ir. We compare oxygen adsorption in fcc and hcp sites at 0.25 ML coverage, using structures and energies from Xu and Kitchin.

Load the reference data

```{python}
import json

with open("energies.json") as f:
    edata = json.load(f)
with open("structures.json") as f:
    sdata = json.load(f)
```

For each metal and site, we reconstruct the structure, relax it using UMA, and compute the adsorption energy

```{python}
data = {"fcc": [], "hcp": []}
refdata = {"fcc": [], "hcp": []}

for metal in ["Cu", "Ag", "Pd", "Pt", "Rh", "Ir"]:
    for site in ["fcc", "hcp"]:
        entry = sdata[metal]["O"][site]["0.25"]
        adslab = Atoms(entry["symbols"], positions=entry["pos"], cell=entry["cell"], pbc=True)
        slab = adslab[adslab.arrays["numbers"] == adslab.arrays["numbers"][0]]

        slab.set_calculator(calc)
        BFGS(slab).run(fmax=0.05)
        adslab.set_calculator(calc)
        BFGS(adslab).run(fmax=0.05)

        re2 = (adslab.get_potential_energy() 
               - slab.get_potential_energy() 
               - atomic_reference_energies["O"])
        energy = re2 + re1 + re3

        data[site].append(energy)
        refdata[site].append(edata[metal]["O"][site]["0.25"])
```

Plot the results against DFT values

```{python}
plt.figure(figsize=(6, 6))
plt.plot(refdata["fcc"], data["fcc"], "r.", label="fcc", ms=10)
plt.plot(refdata["hcp"], data["hcp"], "b.", label="hcp", ms=10)
plt.plot([-5.5, -3.5], [-5.5, -3.5], "k-", lw=2, label="y = x")
plt.xlabel("DFT (PBE) [eV]")
plt.ylabel("UMA-OC20 Prediction [eV]")
plt.legend()
plt.title("O Adsorption Energy: DFT vs ML")
plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.show()
```

The model captures the trend well, with a systematic offset due to the XC functional difference.

---

## Convergence Study

### Slab Thickness

We test convergence with respect to the number of layers in the Pt(111) slab

```{python}
for nlayers in [3, 4, 5, 6, 7, 8]:
    slab = fcc111("Pt", size=(2, 2, nlayers), vacuum=10.0)
    slab.pbc = True
    slab.set_calculator(calc)
    BFGS(slab).run(fmax=0.05)
    slab_e = slab.get_potential_energy()

    adslab = slab.copy()
    add_adsorbate(adslab, "O", height=1.2, position="fcc")
    adslab.set_calculator(calc)
    BFGS(adslab).run(fmax=0.05)
    adslab_e = adslab.get_potential_energy()

    energy = adslab_e - slab_e - atomic_reference_energies["O"] + re1
    print(f"nlayers = {nlayers}: {energy:.2f} eV")
```

The energy converges to within 0.02 eV by 5–6 layers.

### Unit Cell Size

We also test the effect of lateral coverage using larger unit cells

```{python}
for size in [1, 2, 3, 4, 5]:
    slab = fcc111("Pt", size=(size, size, 5), vacuum=10.0)
    slab.set_calculator(calc)
    BFGS(slab).run(fmax=0.05)
    slab_e = slab.get_potential_energy()

    adslab = slab.copy()
    add_adsorbate(adslab, "O", height=1.2, position="fcc")
    adslab.set_calculator(calc)
    BFGS(adslab).run(fmax=0.05)
    adslab_e = adslab.get_potential_energy()

    energy = adslab_e - slab_e - atomic_reference_energies["O"] + re1
    print(f"({size}x{size}): {energy:.2f} eV")
```

Adsorption energies become less favorable at lower coverage, which may indicate the need for fine-tuning at low coverages.

---

## Summary

This tutorial demonstrated how to use the UMA model and OC22 dataset to compute adsorption energies efficiently. Key points include

- UMA predicts total energies, enabling flexible post-processing
- Thermodynamic cycles are essential for consistent referencing
- Systematic differences from DFT arise from XC functional, lattice constant, and reference choices
- Convergence with respect to slab thickness and unit cell size must be tested
- ML models capture trends well but may require calibration for quantitative accuracy

The OC22 dataset and FairChem tools provide a powerful foundation for accelerating catalyst discovery.

---

## Exercises

1. Repeat the Pt(111) calculation with a DFT-optimized lattice constant (3.92 Å instead of 3.92 Å)
2. Compute adsorption energies for bridge and top sites on Pt(111)
3. Try different metals not in the original dataset, such as Ni or Co
4. Fine-tune UMA on a small DFT dataset to reduce the systematic error

---

## References

Shuaibi M, Liu Z, Goyal P, et al. The Open Catalyst 2022 (OC22) Dataset and Challenges for Oxide Electrocatalysts. *arXiv:2206.08917* [cond-mat.mtrl-sci]. 2023.  
Xu Z, Kitchin JR. Probing the Coverage Dependence of Site and Adsorbate Configurational Correlations on (111) Surfaces of Late Transition Metals. *J. Phys. Chem. C*. 2014;118(44):25597–25602.  
Hjorth Larsen A, et al. The atomic simulation environment—a Python library for working with atoms. *J. Phys.: Condens. Matter*. 2017;29(27):273002.

---

## Acknowledgments

- This work uses the OC22 dataset and UMA model from the Open Catalyst Project. 

- **Dataset**: [OC22 on arXiv](https://arxiv.org/abs/2206.08917) 


