---
title: Adsorption Energies
teaching: 1
exercises: 0
questions:
  - "How do I process NetCDF or trajectory data?"
objectives:
  - Understand the concept of adsorption energies and their role in catalysis.
  - Use pretrained machine learning models (e.g., UMA) from the Open Catalyst Project to predict total energies.
  - Compute adsorption energies using reference thermodynamic cycles.
  - Build and relax adsorbate-slab systems using ASE and ML-based force calculators.
  - Analyze trends in adsorption energies across different metals and adsorption sites.
  - Evaluate model predictions against DFT and literature results.
---

<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>

One of the most common tasks in computational catalysis is calculating **binding energies** or **adsorption energies** of small molecules on catalyst surfaces. These energies are crucial for understanding reaction mechanisms, activity trends, and designing new catalysts.

In this guide, we use machine learning (ML) models trained on the **Open Catalyst Project (OCP)** datasets—specifically **OC20** and **OC22**—to predict adsorption energies efficiently and accurately.


## Open Catalyst 2022 (OC22) Dataset

The [Open Catalyst 2022 (OC22)](https://arxiv.org/abs/2206.08917) dataset was introduced to address the lack of sufficient training data for **oxide electrocatalysts**, which are vital for the **oxygen evolution reaction (OER)**. Key features of the dataset:

- Contains **62,331 DFT relaxations** (~9.85 million single-point calculations)
- Covers a broad range of **oxide materials, adsorbates, and surface coverages**
- Includes **total energy**, **force**, and **stress** labels
- Designed to capture **long-range electrostatic and magnetic interactions**

OC22 enables generalized total energy prediction tasks beyond just adsorption energies, making it ideal for modeling complex systems like reconstructed surfaces or charged interfaces.

For more details, see:  
[Shuaibi et al., *The Open Catalyst 2022 (OC22) Dataset and Challenges for Oxide Electrocatalysts*, arXiv:2206.08917 (2022)](https://arxiv.org/abs/2206.08917)


## Pretrained Models

All models below are trained on various splits of the **OC22 S2EF (Structure to Energy and Forces)** and **IS2RE (Initial State to Relaxed Energy)** datasets.

> 🔍 **Note:** Unlike OC20 models, which are trained directly on **adsorption energies**, **OC22 models are trained on DFT total energies**. This allows predictions of properties beyond adsorption, such as stresses, phonons, and dynamics.

### ✅ 2025 Recommendation: Use the UMA Model

We recommend using the **Unified Multimodal Architecture (UMA)** model, trained across all FAIR chemistry datasets. UMA offers several advantages:

1. **State-of-the-art out-of-domain prediction accuracy**
2. Predicts **total energies** (RPBE-level DFT equivalence), enabling broader applications
3. Removes ambiguity in cases of surface reconstruction
4. UMA-small is **energy-conserving and smooth**, making it suitable for:
   - Vibrational frequency calculations
   - Molecular dynamics simulations
5. Likely to receive future updates and improvements


### S2EF-Total Energy Models (OC22)

| Model Name                              | Architecture       | Training Data       | Download                                                                                                 | val ID Force MAE (eV/Å) | val ID Energy MAE (meV/atom) |
|----------------------------------------|--------------------|---------------------|----------------------------------------------------------------------------------------------------------|--------------------------|-------------------------------|
| GemNet-dT-S2EFS-OC22                   | GemNet-dT          | OC22                | [checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_09/oc22/s2ef/gndt_oc22_all_s2ef.pt) \| [config](https://github.com/facebookresearch/fairchem/blob/main/configs/oc22/s2ef/gemnet-dt/gemnet-dT.yml) | 0.032                    | 1.127                         |
| GemNet-OC-S2EFS-OC22                   | GemNet-OC          | OC22                | [checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_09/oc22/s2ef/gnoc_oc22_all_s2ef.pt) \| [config](https://github.com/facebookresearch/fairchem/blob/main/configs/oc22/s2ef/gemnet-oc/gemnet_oc.yml) | 0.030                    | 0.563                         |
| GemNet-OC-S2EFS-OC20+OC22              | GemNet-OC          | OC20 + OC22         | [checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_09/oc22/s2ef/gnoc_oc22_oc20_all_s2ef.pt) \| [config](https://github.com/facebookresearch/fairchem/blob/main/configs/oc22/s2ef/gemnet-oc/gemnet_oc_oc20_oc22.yml) | 0.027                    | 0.483                         |
| GemNet-OC-S2EFS-nsn-OC20+OC22          | GemNet-OC<br><sub><sup>(`enforce_max_neighbors_strictly=False`, [PR #467](https://github.com/facebookresearch/fairchem/pull/467))</sup></sub> | OC20 + OC22         | [checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_05/oc22/s2ef/gnoc_oc22_oc20_all_s2ef.pt) \| [config](https://github.com/facebookresearch/fairchem/blob/main/configs/oc22/s2ef/gemnet-oc/gemnet_oc_oc20_oc22_degen_edges.yml) | 0.027                    | 0.458                         |
| GemNet-OC-S2EFS-OC20→OC22              | GemNet-OC (fine-tuned) | OC20 → OC22       | [checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_09/oc22/s2ef/gnoc_finetune_all_s2ef.pt) \| [config](https://github.com/facebookresearch/fairchem/blob/main/configs/oc22/s2ef/gemnet-oc/gemnet_oc_finetune.yml) | 0.030                    | 0.417                         |
| EquiformerV2-lE4-lF100-S2EFS-OC22      | EquiformerV2<br>($\lambda_E=4$, $\lambda_F=100$) | OC22             | [checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_10/oc22/s2ef/eq2_121M_e4_f100_oc22_s2ef.pt) \| [config](https://github.com/facebookresearch/fairchem/blob/main/configs/oc22/s2ef/equiformer_v2/equiformer_v2_N@18_L@6_M@2_e4_f100_121M.yml) | 0.023                    | 0.447                         |

> 📈 **Performance Insight**:  
> - Fine-tuning GemNet-OC on **both OC20 and OC22** improves energy predictions by **~36%**.
> - Joint training improves **total energy prediction on OC20 by ~19%** and **force prediction on OC22 by ~9%**.

## Introduction to Adsorption Energies

To compute adsorption energies using OCP models, follow a workflow analogous to DFT:

1. Build a **slab model** of the surface
2. Place an **adsorbate** at a candidate site
3. Run **ML relaxation** to find the lowest-energy configuration
4. Compute the **adsorption energy** relative to reference states

⚠️ **Important**: Some OCP checkpoints return **total energies** (like DFT), while others return **adsorption energies directly**. Always verify which type your model outputs.

In this guide, we use the **UMA-S-1** model with the **OC20 task**, which returns **total energies at RPBE level**, allowing flexible post-processing.


### Reference Schemes for Adsorption Energies

Adsorption energies are **reaction energies** defined relative to reference states. A common scheme (used in OC20) defines adsorption via:

$$
x\,\text{CO} + \left(x + \frac{y}{2} - z\right)\,\text{H}_2 + (z - x)\,\text{H}_2\text{O} + w/2\,\text{N}_2 + \ast \rightarrow \text{C}_x\text{H}_y\text{O}_z\text{N}_w\ast
$$

For oxygen adsorption ($x = y = w = 0, z = 1$):

$$
\text{H}_2\text{O} + \ast \rightarrow \text{O}\ast + \text{H}_2 \quad (\Delta E = \text{re2})
$$

Using known thermochemistry:

- Formation of water from hydrogen and oxygen (experimental enthalpy change):
$$
\mathrm{H}_2 + \frac{1}{2}\mathrm{O}_2 \rightarrow \mathrm{H}_2\mathrm{O} \quad \Delta H = -3.03\, \text{eV}
$$

- Dissociation of atomic oxygen into half a molecule of oxygen gas (with given dissociation energy):
$$
\mathrm{O} \rightarrow \frac{1}{2} \mathrm{O}_2 \quad \Delta H = -2.58\, \text{eV}
$$

Then the atomic oxygen adsorption energy is:

$$
\Delta E_{\text{ads}}(\text{O}) = \text{re1} + \text{re2} + \text{re3} = -3.03 + \text{re2} - 2.58
$$

Atomic reference energies (from OC20):

```python
atomic_reference_energies = {
    "H": -3.477,  # eV
    "N": -8.083,
    "O": -7.204,
    "C": -7.282,
}
```


### Setup: Install and Access UMA Model

 Need help installing packages or getting access to UMA?

1. Install required packages:
```python
!pip install fairchem-core fairchem-data-oc fairchem-applications-cattsunami
```

2. Request access to the gated Hugging Face model:
   - Go to [https://huggingface.co/facebook/UMA](https://huggingface.co/facebook/UMA)
   - Log in and request access
   - Generate a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with **"Read access to public gated repos"**
   - Set the token:
```python
# Option 1: CLI login
!huggingface-cli login
```
```python
# Option 2: Environment variable
import os
os.environ["HF_TOKEN"] = "your_hf_token_here"
```

Load the UMA model:
```python
from __future__ import annotations
from fairchem.core import FAIRChemCalculator, pretrained_mlip

predictor = pretrained_mlip.get_predict_unit("uma-s-1")
calc = FAIRChemCalculator(predictor, task_name="oc20")
```

---

### Example: Oxygen Adsorption on Pt(111)

```python
from ase.build import add_adsorbate, fcc111
from ase.optimize import BFGS

# Reference energies
re1 = -3.03  # H2O formation
re3 = -2.58  # O dissociation

# Build Pt(111) slab
slab = fcc111("Pt", size=(2, 2, 5), vacuum=20.0)
slab.pbc = True

# Add O in fcc site
adslab = slab.copy()
add_adsorbate(adslab, "O", height=1.2, position="fcc")

# Relax clean slab
slab.set_calculator(calc)
opt = BFGS(slab)
opt.run(fmax=0.05, steps=100)
slab_e = slab.get_potential_energy()

# Relax adsorbed system
adslab.set_calculator(calc)
opt = BFGS(adslab)
opt.run(fmax=0.05, steps=100)
adslab_e = adslab.get_potential_energy()

# Compute adsorption energy: O + * -> O*
ads_energy = (adslab_e - slab_e 
              - atomic_reference_energies["O"] 
              + re1 + re3)

print(f"Adsorption energy: {ads_energy:.3f} eV")
```

Visualize:
```python
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

fig, axs = plt.subplots(1, 2)
plot_atoms(slab, axs[0])
plot_atoms(adslab, axs[1], rotation="-90x")
for ax in axs:
    ax.set_axis_off()
plt.show()
```

#### Comparison to Literature

- Reference (PBE): ~-4.264 eV for O adsorption
- Converted to dissociative scale: ~-1.68 eV
- UMA (RPBE-trained): Predicts ~-1.47 eV → **~0.2 eV difference**

**Primary reasons for discrepancy**:
1. **XC functional**: PBE vs RPBE (known to differ by ~0.1–0.3 eV)
2. Lattice constant mismatch (experimental vs DFT)
3. Reference energy differences
4. Number of relaxed layers

---

### Exercises

1. **Lattice constant sensitivity**: Vary the Pt lattice constant and observe changes in adsorption energy.
2. **Site comparison**: Compare fcc, hcp, bridge, and top sites for O adsorption.
3. **Coverage effects**: Use larger unit cells (e.g., 3×3, 4×4) to study coverage dependence.

---

## Trends Across Metals

Reproducing results from:  
Xu, Z., & Kitchin, J. R. (2014). *J. Phys. Chem. C*, 118(44), 25597–25602. [DOI](http://dx.doi.org/10.1021/jp508805h)

We compare O adsorption in fcc and hcp sites across late transition metals (Cu, Ag, Pd, Pt, Rh, Ir) at 0.25 ML coverage.

```python
import json

with open("energies.json") as f:
    edata = json.load(f)
with open("structures.json") as f:
    sdata = json.load(f)

sfcc = sdata["Pt"]["O"]["fcc"]["0.25"]
```

```python
from ase import Atoms

adslab = Atoms(sfcc["symbols"], positions=sfcc["pos"], cell=sfcc["cell"], pbc=True)
slab = adslab[adslab.arrays["numbers"] == adslab.arrays["numbers"][0]]
adsorbates = adslab[~(adslab.arrays["numbers"] == adslab.arrays["numbers"][0])]

slab.set_calculator(calc)
BFGS(slab).run(fmax=0.05)

adslab.set_calculator(calc)
BFGS(adslab).run(fmax=0.05)

re2 = (adslab.get_potential_energy() 
       - slab.get_potential_energy() 
       - sum(atomic_reference_energies[x] for x in adsorbates.get_chemical_symbols()))
nO = sum(1 for atom in adslab if atom.symbol == "O")
ads_energy = (re2 + re1 + re3) / nO
print(f"Adsorption energy: {ads_energy:.3f} eV")
```

### Site Correlation Analysis

```{code-cell}
import time
from tqdm import tqdm

t0 = time.time()
data = {"fcc": [], "hcp": []}
refdata = {"fcc": [], "hcp": []}

for metal in ["Cu", "Ag", "Pd", "Pt", "Rh", "Ir"]:
    print(metal)
    for site in ["fcc", "hcp"]:
        entry = sdata[metal]["O"][site]["0.25"]
        adslab = Atoms(entry["symbols"], positions=entry["pos"], cell=entry["cell"], pbc=True)
        slab = adslab[adslab.arrays["numbers"] == adslab.arrays["numbers"][0]]
        adsorbates = adslab[~(adslab.arrays["numbers"] == adslab.arrays["numbers"][0])]

        slab.set_calculator(calc)
        BFGS(slab).run(fmax=0.05)
        adslab.set_calculator(calc)
        BFGS(adslab).run(fmax=0.05)

        re2 = (adslab.get_potential_energy() 
               - slab.get_potential_energy() 
               - sum(atomic_reference_energies[x] for x in adsorbates.get_chemical_symbols()))
        energy = (re2 + re1 + re3) / sum(1 for atom in adslab if atom.symbol == "O")

        data[site].append(energy)
        refdata[site].append(edata[metal]["O"][site]["0.25"])

print(f"Elapsed time: {time.time() - t0:.1f} seconds")
```

#### Parity Plots

```{code-cell}
import matplotlib.pyplot as plt

plt.plot(refdata["fcc"], data["fcc"], "r.", label="fcc")
plt.plot(refdata["hcp"], data["hcp"], "b.", label="hcp")
plt.plot([-5.5, -3.5], [-5.5, -3.5], "k-")
plt.xlabel("DFT (PBE) [eV]")
plt.ylabel("UMA-OC20 Prediction [eV]")
plt.legend()
plt.title("O Adsorption Energy: DFT vs ML")
plt.show()
```

---

## Convergence Study

### Effect of Slab Thickness

```{code-cell}
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

> ✅ **Converged by 5–6 layers** (~0.02 eV variation)

---

## Summary

- Use **UMA-S-1** for best performance and flexibility.
- Always account for **reference state definitions** and **XC functional differences**.
- Perform **convergence tests** on slab thickness, relaxation depth, and unit cell size.
- Be cautious of **surface reconstruction**, **dissociation**, or **desorption** during relaxation.

# Nitrogen Reduction Reaction (NRR)

