---
title:  UMA Models
teaching: 1
exercises: 0
questions:
- "What are the UMA models?"
- "How does UMA model is able to acheieve SOTA accuracy on a wide range of domains such as materials, molecules and catalysis?"
keypoints:
- "UMA is an equivariant GNN that leverages a novel technique called Mixture of Linear Experts (MoLE) to give it the capacity to learn the largest multi-modal dataset"
- "UMA is trained on 5 different DFT datasets with different levels of theory. An UMA task refers to a specific level of theory associated with that DFT dataset."
---

<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>


# [Universal Models for Atoms](https://scontent.fadd2-1.fna.fbcdn.net/v/t39.2365-6/520537587_1080318074038402_8373516189142708029_n.pdf?_nc_cat=107&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=l3xnwGRFsiIQ7kNvwGUMPLJ&_nc_oc=AdlRoyzTvLWzR5bSr4XHrR9I6fDTrGDfNvRrsGmYNM_VyykCJ41NtSpjJwNd3zZIp3Q&_nc_zt=14&_nc_ht=scontent.fadd2-1.fna&_nc_gid=HYvFYrfUd_ygtu9uY0AGlA&oh=00_AfU2wH8TNYOr6WHx1XnAGlymbw7rxmV0LGF2F-fxT_tLOg&oe=68A27139)

[UMA](https://ai.meta.com/research/publications/uma-a-family-of-universal-models-for-atoms/) is an equivariant GNN that leverages a novel technique called Mixture of Linear Experts (MoLE) to give it the capacity to learn the largest multi-modal dataset to date (500M DFT examples), while preserving energy conservation and inference speed. Even a 6M active parameter (145M total) UMA model is able to acheieve SOTA accuracy on a wide range of domains such as materials, molecules and catalysis. 

![UMA model architecture](uma.svg "UMA model architecture")


## The UMA Mixture-of-Linear-Experts routing function

The UMA model uses a Mixture-of-Linear-Expert (MoLE) architecture to achieve very high parameter count with fast inference speeds with a single output head. In order to route the model to the correct set parameters, the model must be given a set of inputs.  The following information are required for the input to the model.

* task, ie: omol, oc20, omat, odac, omc (this affects the level of theory and DFT calculations that the model is trying to emulate) see below
* charge - total known charge of the system (only used for omol task and defaults to 0)
* spin - total spin multiplicity of the system (only used for omol task and defaults to 1)
* elemental composition - The unordered total elemental composition of the system. Each element has an atom embedding and the composition embedding is the mean over all the atom embeddings. For example H2O2 would be assigned the same embedding regardless of its conformer configuration.


### The UMA task

UMA is trained on 5 different DFT datasets with different levels of theory. An UMA **task** refers to a specific level of theory associated with that DFT dataset. UMA learns an embedding for the given **task**. Thus at inference time, the user must specify which one of the 5 embeddings they want to use to produce an output with the DFT level of theory they want. See the following table for more details.

| Task    | Dataset | DFT Level of Theory | Relevant applications | Usage Notes |
| ------- | ------- | ----- | ------ | ----- |
| omol    | [Omol25](https://arxiv.org/abs/2505.08762) | wB97M-V/def2-TZVPD as implemented in ORCA6, including non-local dispersion. All solvation should be explicit.   |  Biology, organic chemistry, protein folding, small-molecule pharmaceuticals, organic liquid properties, homogeneous catalysis | total charge and spin multiplicity. If you don't know what these are, you should be very careful if modeling charged or open-shell systems. This can be used to study radical chemistry or understand the impact of magnetic states on the structure of a molecule. All training data is aperiodic, so any periodic systems should be treated with some caution. Probably won't work well for inorganic materials.  |
| omc     | Omc25 | PBE+D3 as implemented in VASP. | Pharmaceutical packaging, bio-inspired materials, organic electronics, organic LEDs | UMA has not seen varying charge or spin multiplicity for the OMC task, and expects total_charge=0 and spin multiplicity=0 as model inputs. |
| omat    | [Omat24](https://arxiv.org/abs/2410.12771) | PBE/PBE+U as implemented in VASP using Materials Project suggested settings, except with VASP 54 pseudopotentials. No dispersion.   | Inorganic materials discovery, solar photovoltaics, advanced alloys, superconductors, electronic materials, optical materials | UMA has not seen varying charge or spin multiplicity for the OMat task, and expects total_charge=0 and spin multiplicity=0 as model inputs. Spin polarization effects are included, but you can't select the magnetic state. Further, OMat24 did not fully sample possible spin states in the training data. |
| oc20    | [OC20*](https://arxiv.org/abs/2010.09990) | RPBE as implemented in VASP, with VASP5.4 pseudopotentials. No dispersion. | Renewable energy, catalysis, fuel cells, energy conversion, sustainable fertilizer production, chemical refining, plastics synthesis/upcycling | UMA has not seen varying charge or spin multiplicity for the OC20 task, and expects total_charge=0 and spin multiplicity=0 as model inputs. No oxides or explicit solvents are included in OC20. The model works surprisingly well for transition state searches given the nature of the training data, but you should be careful. RPBE works well for small molecules, but dispersion will be important for larger molecules on surfaces. |
| odac    | [ODac23](https://arxiv.org/abs/2311.00341) | PBE+D3 as implemented in VASP, with VASP5.4 pseudopotentials. | Direct air capture, carbon capture and storage, CO2 conversion, catalysis | UMA has not seen varying charge or spin multiplicity for the ODAC task, and expects total_charge=0 and spin multiplicity=0 as model inputs. The ODAC23 dataset only contains CO2/H2O water absorption, so anything more than might be inaccurate (e.g. hydrocarbons in MOFs). Further, there is a limited number of bare-MOF structures in the training data, so you should be careful if you are using a new MOF structure. |

*Note: OC20 is was updated from the original OC20 and recomputed to produce total energies instead of adsorption energies.


## Quick Start
The easiest way to use pretrained models is via the [ASE](https://wiki.fysik.dtu.dk/ase/) `FAIRChemCalculator`.
A single uma model can be used for a wide range of applications in chemistry and materials science by picking the
appropriate task name for domain specific prediction.

### Instantiate a calculator from a pretrained model
Make sure you have a Hugging Face account, have already applied for model access to the
[UMA model repository](https://huggingface.co/facebook/UMA), and have logged in to Hugging Face using an access token.
You can use the following to save an auth token,
```bash
huggingface-cli login
```

Models are referenced by their name, below are the currently supported models:

| Model Name | Description |
|---|---|
| uma-s-1p1 | Latest version of the UMA small model, fastest of the UMA models while still SOTA on most benchmarks (6.6M/150M active/total params) |
| uma-m-1p1 | Best in class UMA model across all metrics, but slower and more memory intensive than uma-s (50M/1.4B active/total params) |

### Set the task for your application and calculate

- **oc20:** use this for catalysis
- **omat:** use this for inorganic materials
- **omol:** use this for molecules
- **odac:** use this for MOFs
- **omc:** use this for molecular crystals

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

#### Relax an adsorbate on a catalytic surface,

~~~
from ase.build import fcc100, add_adsorbate, molecule
from ase.optimize import LBFGS
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
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

#### Relax an inorganic crystal,

~~~
from ase.build import bulk
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="omat")

atoms = bulk("Fe")
atoms.calc = calc

opt = LBFGS(FrechetCellFilter(atoms))
opt.run(0.05, 100)
~~~
{: .python}


#### Run molecular MD,

~~~
from ase import units
from ase.io import Trajectory
from ase.md.langevin import Langevin
from ase.build import molecule
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="omol")

atoms = molecule("H2O")
atoms.calc = calc

dyn = Langevin(
    atoms,
    timestep=0.1 * units.fs,
    temperature_K=400,
    friction=0.001 / units.fs,
)
trajectory = Trajectory("my_md.traj", "w", atoms)
dyn.attach(trajectory.write, interval=1)
dyn.run(steps=1000)
~~~
{: .python}

#### Calculate a spin gap,


~~~
from ase.build import molecule
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import matplotlib.pyplot as plt

# Load predictor
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cpu")

# Singlet CH2
singlet = molecule("CH2_s1A1d")
singlet.info.update({"spin": 1, "charge": 0})
singlet.calc = FAIRChemCalculator(predictor, task_name="omol")
energy_singlet = singlet.get_potential_energy()

# Triplet CH2
triplet = molecule("CH2_s3B1d")
triplet.info.update({"spin": 3, "charge": 0})
triplet.calc = FAIRChemCalculator(predictor, task_name="omol")
energy_triplet = triplet.get_potential_energy()

# Energy difference
delta_E = energy_triplet - energy_singlet
print(f"Singlet energy = {energy_singlet:.6f} eV")
print(f"Triplet energy = {energy_triplet:.6f} eV")
print(f"ΔE (triplet - singlet) = {delta_E:.6f} eV")

# Plot
bar_width = 0.6
x = [0, 1]
energies = [energy_singlet, energy_triplet]
colors = ["blue", "red"]
labels = ["Singlet CH₂", "Triplet CH₂"]

plt.figure(figsize=(6, 4))
bars = plt.bar(x, energies, width=bar_width, color=colors)

# Annotate each bar with its energy
for i, val in enumerate(energies):
    plt.text(x[i], val, f"{val:.3f}", ha="center", va="bottom")

# Annotate energy difference above the taller bar
top = max(energies)
plt.text((x[0]+x[1])/2, top + 0.05*top, f"ΔE = {delta_E:.3f} eV",
         ha="center", va="bottom", fontsize=10, fontweight='bold')

plt.xticks(x, labels)
plt.ylabel("Energy (eV)")
plt.title("CH₂ Singlet vs Triplet Energies")
plt.tight_layout()
plt.show()
~~~
{: .python}
