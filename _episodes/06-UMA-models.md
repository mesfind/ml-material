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




# Universal Models for Atoms (UMA)
[UMA](https://ai.meta.com/research/publications/uma-a-family-of-universal-models-for-atoms/) is an equivariant GNN that leverages a novel technique called Mixture of Linear Experts (MoLE) to give it the capacity to learn the largest multi-modal dataset to date (500M DFT examples), while preserving energy conservation and inference speed. Even a 6M active parameter (145M total) UMA model is able to acheieve SOTA accuracy on a wide range of domains such as materials, molecules and catalysis. 

![UMA model architecture](uma.svg "UMA model architecture")


## The UMA Mixture-of-Linear-Experts routing function

The UMA model uses a Mixture-of-Linear-Expert (MoLE) architecture to achieve very high parameter count with fast inference speeds with a single output head. In order to route the model to the correct set parameters, the model must be given a set of inputs.  The following information are required for the input to the model.

* task, ie: omol, oc20, omat, odac, omc (this affects the level of theory and DFT calculations that the model is trying to emulate) see below
* charge - total known charge of the system (only used for omol task and defaults to 0)
* spin - total spin multiplicity of the system (only used for omol task and defaults to 1)
* elemental composition - The unordered total elemental composition of the system. Each element has an atom embedding and the composition embedding is the mean over all the atom embeddings. For example H2O2 would be assigned the same embedding regardless of its conformer configuration.



### How to Access Gated Models on HuggingFace

To use gated models such as UMA, you must first create a HuggingFace account and request permission to access the UMA models.

- Get and login to your [Huggingface account](https://huggingface.co/)
- Request access to [UMA Models](https://huggingface.co/facebook/UMA)
- Create a [Huggingface token](https://huggingface.co/settings/tokens/) with the permission “Permissions: Read access to contents of all public gated repos you can access”
-  Install the  huggingface_hub Python package with `pip install -U "huggingface_hub[cli]"
- Add the token as an environment variable (using `hf auth login` or by setting the HF_TOKEN environment variable.

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

The MLIP calculator is then assigned to the slab to enable efficient evaluation of energies and atomic forces. For structural relaxation, the code employs the Limited-memory Broyden–Fletcher–Goldfarb–Shanno (LBFGS) algorithm, a quasi-Newton method well-suited for optimizing atomic coordinates in molecular simulations. The optimizer is configured to perform geometry relaxation until the maximum force on any atom falls below 0.05 eV/Å or a maximum of 100 steps is reached. This combination of ML-potentials with classical ASE tools demonstrates an advanced approach for accelerating surface chemistry simulations by leveraging accurate machine learning models within established computational chemistry workflows, enabling efficient exploration of adsorbate-surface interactions with reduced computational cost while maintaining predictive accuracy.



#### Relax an adsorbate on a catalytic surface,

Now time to use a pretrained FAIRChem machine-learned potential with ASE to relax a CO molecule adsorbed on a Cu(100) catalytic surface. It builds the slab and adsorbate, assigns the ML potential calculator, and performs geometry optimization using the LBFGS algorithm to minimize atomic forces efficiently.
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

~~~
     Step     Time          Energy          fmax
LBFGS:    0 10:40:21      -89.544140       11.491965
LBFGS:    1 10:40:22      -92.462341        6.582730
LBFGS:    2 10:40:22      -92.585173        7.551272
LBFGS:    3 10:40:23      -92.964017        3.752812
LBFGS:    4 10:40:23      -93.123250        3.560935
LBFGS:    5 10:40:24      -93.230204        2.254664
LBFGS:    6 10:40:24      -93.470363        1.136102
LBFGS:    7 10:40:25      -93.561651        0.999690
LBFGS:    8 10:40:25      -93.672328        0.700034
LBFGS:    9 10:40:26      -93.759912        0.503134
LBFGS:   10 10:40:26      -93.806733        0.364667
LBFGS:   11 10:40:27      -93.825594        0.344448
LBFGS:   12 10:40:27      -93.849672        0.485330
LBFGS:   13 10:40:28      -93.868526        0.431073
LBFGS:   14 10:40:28      -93.878749        0.161163
LBFGS:   15 10:40:29      -93.884701        0.168627
LBFGS:   16 10:40:29      -93.891148        0.204622
LBFGS:   17 10:40:30      -93.897887        0.250512
LBFGS:   18 10:40:30      -93.903948        0.175708
LBFGS:   19 10:40:31      -93.906856        0.055518
LBFGS:   20 10:40:31      -93.907350        0.042256
~~~
{: .output}

From the output result, the energy steadily decreases across the steps, showing that the system is lowering its energy and approaching stability. Correspondingly, the maximum force reduces from a high initial value (around `11.5 eV/Å`) down to below `0.05 eV/Å`, indicating that the atoms are reaching their equilibrium positions and the optimization is converging successfully. This process ensures that the atomic configuration is optimized to a local minimum in the potential energy surface, reflecting a physically realistic and stable structure.

## Pretrained Model for Omol

This workflow leverages the "uma-s-1p1" pretrained FAIRChem model specifically designed for organic molecules. It encompasses the creation or import of organic molecular structures into an ASE database, followed by batch data preparation and application of the model to predict molecular properties efficiently. This approach facilitates scalable and accurate predictions of organic molecular behavior using cutting-edge machine-learned interatomic potentials.

~~~
from ase import Atoms
from ase.db import connect
from torch.utils.data import DataLoader
from fairchem.core.datasets import AseDBDataset
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
from fairchem.core import pretrained_mlip
# create some atomic structure (e.g hydrogen molecule)
h2 = Atoms('H2', positions=[(0,0,0),(0,0, 0.74)])
db_path ='dataset.aselmdb'
db = connect(db_path, type='aselmdb',append=False) # to create new DB
db.write(h2, key_value_pairs={'description':'Hydrogen molecule'})
# Loading the ASE database with AseDBDataset and DataLoader
db_path ='dataset.aselmdb'
dataset = AseDBDataset(
    config=dict(src=db_path, a2g_args=dict(task_name="omol"))
)
loader = DataLoader(dataset, batch_size=20,
                    collate_fn=atomicdata_list_to_batch)
# load pretained model and do prediction
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1",device=device)
for batch in loader:
    preds = predictor.predict(batch)
    #print(preds)
print(preds["energy"][2])
    

~~~
{: .python}

~~~
~~~
{: .output}

## Polymers

This workflow demonstrates the construction and analysis of a polyethylene polymer using a pretrained FAIRChem machine-learned interatomic potential model. It begins by defining the polyethylene repeat unit, consisting of two carbon and four hydrogen atoms, and then generates a polymer chain by replicating and translating this unit to form a longer molecule. The polymer structure is saved into an ASE database for organized data handling. Subsequently, the polymer data is loaded into a dataset and prepared in batches for efficient processing. The pretrained "uma-s-1p1" model, designed for organic molecules, is used to predict molecular properties of the polymer chain. This approach enables scalable and accurate simulations of polymeric systems by integrating machine learning potentials with standard computational chemistry tools.

~~~
from ase import Atoms
from ase.db import connect
from torch.utils.data import DataLoader
from fairchem.core.datasets import AseDBDataset
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
from fairchem.core import pretrained_mlip
import torch
import numpy as np

# --- Build Polyethylene Chain ---
repeat_unit = Atoms(
    'C2H4',
    positions=[
        (0.00,  0.00, 0.00),  # C1
        (1.54,  0.00, 0.00),  # C2
        (-0.63,  0.90, 0.00),  # H1 (C1)
        (-0.63, -0.90, 0.00),  # H2 (C1)
        (2.17,  0.90, 0.00),  # H3 (C2)
        (2.17, -0.90, 0.00),  # H4 (C2)
    ]
)

# Build polymer with 5 repeat units
polymer_atoms = repeat_unit.copy()
for i in range(1, 5):
    unit = repeat_unit.copy()
    unit.translate((i * 2.54, 0.0, 0.0))  # ~C–C bond + buffer
    polymer_atoms.extend(unit)

# Save to ASE database
db_path = 'polymer_dataset.aselmdb'
with connect(db_path, type='aselmdb', append=False) as db:
    db.write(polymer_atoms, key_value_pairs={'description': 'Polyethylene, 5 repeat units'})

# --- Load Dataset & Predict ---
dataset = AseDBDataset(config=dict(
    src=db_path,
    a2g_args=dict(task_name='omol')
))
loader = DataLoader(dataset, batch_size=1, collate_fn=atomicdata_list_to_batch)  # Batch size 1 for clarity

# Device & Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)

# --- Process Predictions with Formatted Output ---
for i, batch in enumerate(loader):
    print(f"\n{'='*60}")
    print(f"Structure {i+1}")
    print(f"{'='*60}")

    # Move predictions to CPU and detach
    preds = predictor.predict(batch)
    energy = preds['energy'].cpu().detach().numpy().item()
    forces = preds['forces'].cpu().detach().numpy()  # Shape: (N, 3)
    stress = preds['stress'].cpu().detach().numpy() if 'stress' in preds else None

    # Get atomic info
    num_atoms = len(polymer_atoms)
    atom_symbols = polymer_atoms.get_chemical_symbols()

    # Print Energy
    print(f"\n Total Energy : {energy: .8f} eV")

    # Print Forces
    print(f"\nForces (eV/Å) on Atoms:")
    print(f"{'#':<3} {'Atom':<4} {'Fx':>10} {'Fy':>10} {'Fz':>10} {'|F|':>10}")
    print("-" * 55)
    for idx in range(num_atoms):
        fx, fy, fz = forces[idx]
        norm = np.linalg.norm(forces[idx])
        symbol = atom_symbols[idx]
        print(f"{idx:<3} {symbol:<4} {fx:10.6f} {fy:10.6f} {fz:10.6f} {norm:10.6f}")

    # Print Stress (if available)
    if stress is not None:
        stress = stress[0]  # Shape: (9,) or (6,)
        sxx, sxy, sxz, syx, syy, syz, szx, szy, szz = stress
        print(f"\nStress Tensor (eV/Å³):")
        print(f"    [{sxx:8.6f}  {sxy:8.6f}  {sxz:8.6f}]")
        print(f"    [{syx:8.6f}  {syy:8.6f}  {syz:8.6f}]")
        print(f"    [{szx:8.6f}  {szy:8.6f}  {szz:8.6f}]")

print(f"\nPrediction completed for {len(dataset)} structure(s).")

~~~
{: .python}

~~~
============================================================
Structure 1
============================================================

 Total Energy : -10357.98538294 eV

Forces (eV/Å) on Atoms:
#   Atom         Fx         Fy         Fz        |F|
-------------------------------------------------------
0   C      1.765896  -0.000000   0.000000   1.765896
1   C    -46.918243   0.000002  -0.000002  46.918243
2   H      1.401135  -0.021788   0.017874   1.401418
3   H      1.401135   0.021788  -0.017874   1.401418
4   H    222.772934  15.270905   0.144446 223.295761
5   H    222.772934 -15.270901  -0.144452 223.295761
6   C     43.055180   0.000009   0.000003  43.055180
7   C    -42.646687   0.000000  -0.000002  42.646687
8   H    -224.083084  15.226253  -0.376390 224.600113
9   H    -224.083115 -15.226264   0.376395 224.600143
10  H    213.545059  15.269310  -0.582799 214.091064
11  H    213.545105 -15.269321   0.582795 214.091110
12  C     43.606586   0.000001   0.000002  43.606586
13  C    -43.606598   0.000004  -0.000001  43.606598
14  H    -213.892731  15.599482   0.582262 214.461624
15  H    -213.892776 -15.599480  -0.582259 214.461670
16  H    213.892776  15.599501  -0.582262 214.461670
17  H    213.892746 -15.599491   0.582263 214.461624
18  C     42.646698  -0.000007  -0.000001  42.646698
19  C    -43.055168   0.000001  -0.000001  43.055168
20  H    -213.545151  15.269302   0.582800 214.091156
21  H    -213.545105 -15.269307  -0.582799 214.091110
22  H    224.083023  15.226274   0.376395 224.600052
23  H    224.083069 -15.226267  -0.376394 224.600098
24  C     46.918259  -0.000006   0.000000  46.918259
25  C     -1.765893   0.000001   0.000000   1.765893
26  H    -222.772888  15.270927  -0.144451 223.295731
27  H    -222.772934 -15.270930   0.144449 223.295776
28  H     -1.401136  -0.021787  -0.017875   1.401419
29  H     -1.401136   0.021787   0.017875   1.401419

Stress Tensor (eV/Å³):
    [-622.612671  -0.000010  -0.000001]
    [-0.000010  -220.839111  -0.000001]
    [-0.000001  -0.000001  0.000000]

Prediction completed for 1 structure(s).
~~~
{: .output}

The results present detailed predictions for a single polymer structure. The total energy of the system is about `-10357.99 eV`, indicating the overall stability of the configuration. The forces on each atom are listed with their components in the x, y, and z directions (Fx, Fy, Fz), along with the magnitude of the force |F|, which shows the strength and direction of the atomic interactions that may drive structural relaxation. Notably, several atoms experience large forces, suggesting sites of higher atomic stress or potential instability.

The stress tensor is given in units of `eV/Å³` and represents the internal stresses within the material, shown as a 3x3 matrix diagonal-dominated by negative values in two directions and near zero in the third, indicating anisotropic stress distribution within the polymer structure.

Together, these predictions offer insight into the energetic stability, atomic forces, and mechanical stress state of the polymer, which are critical for understanding its physical properties and behavior under various conditions.


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
