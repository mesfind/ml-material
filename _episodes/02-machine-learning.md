---
title: Machine Learning Fundamentals
teaching: 1
exercises: 0
questions:
- "What are the fundamental concepts in ML I can use in sklearn framewrok ?"
- "How do I write documentation for my ML code?"
- "How do I train and test ML models for Physical Sciences Problems?"
objectives:
- "Gain an understanding of fundamental machine learning concepts relevant to physical sciences."
- "Develop proficiency in optimizing data preprocessing techniques for machine learning tasks in Python."
- "Learn and apply best practices for training, evaluating, and interpreting machine learning models in the domain of physical sciences."
keypoints:
- "Data representations are crucial for ML in science, including spatial data (vector, raster), point clouds, time series, graphs, and more"
- "ML algorithms like linear regression, k-nearest neighbors,support vector Machine, xgboost and random forests are vital algorithms"
- "Supervised learning is a popular ML approach, with decision trees, random forests, and neural networks being widely used"
- "Fundamentals of data engineering are crucial for building robust ML pipelines, including data storage, processing, and serving"
---

<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>

# Materials Properties Datasets

These datasets compute properties of various materials.

dc.molnet.load_bandgap : V2
dc.molnet.load_perovskite : V2
dc.molnet.load_mp_formation_energy : V2
dc.molnet.load_mp_metallicity : V2
[3] Lopez, Steven A., et al. "The Harvard organic photovoltaic dataset." Scientific data 3.1 (2016): 1-7.

[4] Ramsundar, Bharath, et al. "Is multitask deep learning practical for pharma?." Journal of chemical information and modeling 57.8 (2017): 2068-2076.

## What is a Fingerprint?

Deep learning models almost always take arrays of numbers as their inputs. If we want to process molecules with them, we somehow need to represent each molecule as one or more arrays of numbers.

Many (but not all) types of models require their inputs to have a fixed size. This can be a challenge for molecules, since different molecules have different numbers of atoms. If we want to use these types of models, we somehow need to represent variable sized molecules with fixed sized arrays.

Fingerprints are designed to address these problems. A fingerprint is a fixed length array, where different elements indicate the presence of different features in the molecule. If two molecules have similar fingerprints, that indicates they contain many of the same features, and therefore will likely have similar chemistry.


## DScribe

DScribe is a Python package for transforming atomic structures into fixed-size numerical fingerprints. These fingerprints are often called "descriptors" and they can be used in various tasks, including machine learning, visualization, similarity analysis, etc.


~~~
# https://doi.org/10.1016/j.cpc.2019.106949
#!pip install dscribe
import numpy as np
from ase.build import molecule
from dscribe.descriptors import SOAP, CoulombMatrix

# ========================================
# 1. Define Atomic Structures
# ========================================
print("Building molecular samples...")
samples = [
    molecule("H2O"),  # Water (3 atoms)
    molecule("NO2"),  # Nitrogen dioxide (3 atoms)
    molecule("CO2")   # Carbon dioxide (3 atoms)
]

# Print basic info
for i, mol in enumerate(samples):
    formula = mol.get_chemical_formula()
    atomic_numbers = mol.get_atomic_numbers()
    print(f"Sample {i}: {formula}, atoms = {len(mol)}, atomic numbers = {atomic_numbers}")

# ========================================
# 2. Setup Descriptors
# ========================================
print("\nSetting up descriptors...")

# Coulomb Matrix: Global molecular descriptor
cm_desc = CoulombMatrix(
    n_atoms_max=3,
    permutation="sorted_l2"  # Ensures consistent ordering
)

# SOAP: Local environment descriptor
soap_desc = SOAP(
    species=["H", "C", "N", "O"],  # All elements in the dataset
    r_cut=5.0,                     # Cutoff radius in Å
    n_max=8,                       # Number of radial basis functions
    l_max=6                        # Number of angular basis functions
)

# ========================================
# 3. Coulomb Matrix for Single and Multiple Systems
# ========================================
print("\n--- Computing Coulomb Matrices ---")

# Single molecule (H2O)
water = samples[0]
cm_single = cm_desc.create(water)
print(f"Coulomb Matrix (H2O) shape: {cm_single.shape}")
print(f"Coulomb Matrix (flattened): {cm_single}")

# Batch: All molecules
cm_batch = cm_desc.create(samples)
cm_batch_parallel = cm_desc.create(samples, n_jobs=2)

print(f"Batch Coulomb matrices shape: {cm_batch.shape}")        # (3, 9)
print(f"Parallel batch shape: {cm_batch_parallel.shape}")

# ========================================
# 4. SOAP Descriptors for Oxygen Atoms Only
# ========================================
print("\n--- Computing SOAP Descriptors for Oxygen Atoms ---")

# Find oxygen atom indices in each molecule
oxygen_indices = [np.where(mol.get_atomic_numbers() == 8)[0] for mol in samples]
print(f"Oxygen atom indices per molecule: {oxygen_indices}")

# Compute SOAP only for oxygen atoms
oxygen_soap_list = soap_desc.create(samples, oxygen_indices, n_jobs=2)

# Output is a list of arrays (variable number of centers per molecule)
print(f"SOAP output type: {type(oxygen_soap_list)}")
for i, soap_per_mol in enumerate(oxygen_soap_list):
    mol_formula = samples[i].get_chemical_formula()
    print(f"  Molecule {i} ({mol_formula}): {soap_per_mol.shape} SOAP vectors")

# Flatten into a single 2D array: (N_total_oxygens, n_features)
oxygen_soap_flat = np.vstack(oxygen_soap_list)
print(f"Flattened SOAP descriptors shape (all oxygen atoms): {oxygen_soap_flat.shape}")
# Example: (5, 3696) → 1 (H2O) + 2 (NO2) + 2 (CO2)

# ========================================
# 5. Compute SOAP Derivatives (w.r.t. Atomic Positions)
# ========================================
print("\n--- Computing SOAP Derivatives ---")

try:
    derivatives_list, descriptors_list = soap_desc.derivatives(
        samples,
        centers=oxygen_indices,
        return_descriptor=True,
        n_jobs=1  # Derivatives often work best with n_jobs=1
    )

    # 'derivatives_list' is a list: one entry per molecule, shape (n_centers, 3, n_features)
    print("SOAP derivatives computed successfully (returned as list).")
    print(f"Number of molecules in derivatives: {len(derivatives_list)}")
    for i, d in enumerate(derivatives_list):
        print(f"  Molecule {i}: derivative shape = {d.shape}")

except Exception as e:
    print(f"Error computing derivatives: {e}")

# ========================================
# 6. Final Summary
# ========================================
print("\n--- Summary ---")
print(f"Total molecules: {len(samples)}")
print(f"Coulomb Matrices: shape {cm_batch.shape} → used for global molecular representation")
print(f"SOAP (oxygen atoms): flattened to {oxygen_soap_flat.shape} → ready for machine learning")
if 'derivatives_list' in locals():
    total_deriv_centers = sum(d.shape[0] for d in derivatives_list)
    print(f"SOAP derivatives: {total_deriv_centers} centers total → useful for force prediction")
else:
    print("SOAP derivatives: not available")
~~~
{: .python}


The above example employs atomistic descriptors to generate machine-readable representations of small inorganic molecules—H₂O, NO₂, and CO₂—using the DScribe library. Two distinct types of descriptors are utilized: the **Coulomb Matrix (CM)** and the **Smooth Overlap of Atomic Positions (SOAP)**. The CM provides a global, rotation- and permutation-invariant representation of each molecule, encoded as a fixed-size vector of dimension $ n_{\text{atoms}}^{\text{max}} \times n_{\text{atoms}}^{\text{max}} $, here flattened to a 9-dimensional vector after sorting by L2 norm to ensure consistency across configurations. This representation is suitable for regression tasks targeting global molecular properties such as total energy.

In contrast, the SOAP descriptor offers a local, chemically rich description of atomic environments, computed specifically for oxygen atoms across all structures. Due to the variable number of oxygen atoms per molecule (one in H₂O, two in NO₂ and CO₂), the resulting descriptors are structured as a list of arrays, later concatenated into a unified feature matrix of shape (5, 3696), where each row corresponds to an oxygen-centered environment and columns represent the high-dimensional SOAP basis expansion ($ n_{\text{max}} = 8 $, $ l_{\text{max}} = 6 $). This localized approach enables atom-centered machine learning models, including those predicting partial charges, chemical shifts, or reactivity.

Furthermore, analytical derivatives of the SOAP descriptor with respect to atomic positions are computed, enabling the prediction of forces in energy-conserving machine learning potentials. These derivatives are returned per center and per Cartesian direction, forming a hierarchical structure suitable for integration into gradient-based learning frameworks.

The use of parallelization via the `n_jobs` parameter demonstrates scalability for larger datasets. Overall, this work establishes a reproducible pipeline for generating physically meaningful descriptors from molecular geometries, forming a foundation for subsequent applications in quantum property prediction, potential energy surface construction, and interpretative analysis in computational chemistry and materials informatics.


## ML Descriptors for Inorganic Materials

For **inorganic crystalline materials** (instead of molecules) replaces molecular systems with **bulk inorganic crystals**, computes **SOAP and Coulomb Matrix-like descriptors** (adapted for periodic systems), and reflects best practices in materials representation learning as follows:

~~~
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from ase.build import bulk
from dscribe.descriptors import SOAP

# ========================================
# 1. Define Inorganic Crystalline Materials
# ========================================
print("Constructing bulk inorganic materials...")
materials = [
    bulk("Cu", "fcc", a=3.6),           # FCC copper
    bulk("Si", "diamond", a=5.43),      # Diamond silicon
    bulk("NaCl", "rocksalt", a=5.64)    # Rocksalt sodium chloride
]

# True formation energies (eV/atom) from experimental or DFT data
# Source: approximate values for demonstration
formation_energies = np.array([
    0.00,      # Cu (elemental reference)
    -0.80,     # Si (from elemental Si)
    -3.10      # NaCl (strong ionic bonding)
])

# Print material info
for i, mat in enumerate(materials):
    formula = mat.get_chemical_formula()
    natoms = len(mat)
    cellpar = mat.cell.cellpar()
    print(f"Material {i}: {formula}, atoms = {natoms}, "
          f"a ≈ {cellpar[0]:.2f} Å, E_form = {formation_energies[i]:.2f} eV/atom")
# ========================================
# 2. Setup SOAP Descriptor
# ========================================
print("\nSetting up SOAP descriptor...")

soap_desc = SOAP(
    species=["Cu", "Si", "Na", "Cl"],
    r_cut=4.0,                    # Local environment radius
    n_max=6,                      # Radial basis order
    l_max=4,                      # Angular momentum cutoff
    periodic=True,                # Handle PBCs
    sparse=False                  # Dense output for ML
)
# ========================================
# 3. Compute Averaged SOAP Descriptors
# ========================================
print("\n--- Computing Averaged SOAP Descriptors ---")

X, y = [], formation_energies  # Features and targets

for i, mat in enumerate(materials):
    formula = mat.get_chemical_formula()
    
    # ✅ Convert centers to list of integers
    centers = list(range(len(mat)))
    
    # Compute SOAP for all atoms
    soap_local = soap_desc.create(mat, centers=centers)
    print(f"  {formula}: per-atom SOAP shape = {soap_local.shape}")
    
    # Average over atoms → global descriptor
    soap_global = np.mean(soap_local, axis=0)
    X.append(soap_global)

# Convert to NumPy array
X = np.array(X)  # Shape: (n_samples, n_features)
y = np.array(y)  # Shape: (n_samples,)

print(f"\nFinal feature matrix shape: {X.shape}")
# ========================================
# 3. Compute Averaged SOAP Descriptors
# ========================================
print("\n--- Computing Averaged SOAP Descriptors ---")

X, y = [], formation_energies  # Features and targets

for i, mat in enumerate(materials):
    formula = mat.get_chemical_formula()
    
    # ✅ Convert centers to list of integers
    centers = list(range(len(mat)))
    
    # Compute SOAP for all atoms
    soap_local = soap_desc.create(mat, centers=centers)
    print(f"  {formula}: per-atom SOAP shape = {soap_local.shape}")
    
    # Average over atoms → global descriptor
    soap_global = np.mean(soap_local, axis=0)
    X.append(soap_global)

# Convert to NumPy array
X = np.array(X)  # Shape: (n_samples, n_features)
y = np.array(y)  # Shape: (n_samples,)

print(f"\nFinal feature matrix shape: {X.shape}")
# ========================================
# 4. Train-Test Split (n=3 → use 2 for train, 1 for test)
# ========================================
print("\n--- Training Random Forest Regressor ---")

# For n=3, use test_size=1/3
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, shuffle=True
)

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Initialize and train Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=3,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Predictions
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Metrics
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"Training MAE: {mae_train:.3f} eV/atom")
print(f"Test MAE: {mae_test:.3f} eV/atom")
print(f"Test R²: {r2_test:.3f}")
# ========================================
# 5. Feature Importance (Optional)
# ========================================
print("\n--- Feature Importance (Top 10) ---")
importances = rf_model.feature_importances_
top_indices = np.argsort(importances)[-10:][::-1]

for idx in top_indices:
    print(f"  Feature {idx}: importance = {importances[idx]:.4f}")
# ========================================
# 6. Plot Results
# ========================================
plt.figure(figsize=(8, 5))
plt.scatter(y_train, y_pred_train, color='blue', label='Train', s=60, alpha=0.8)
if len(y_test) > 0:
    plt.scatter(y_test, y_pred_test, color='red', label='Test', s=60, edgecolor='k', linewidth=1)

# Diagonal line
min_val, max_val = min(min(y), min(y_pred_test)), max(max(y), max(y_pred_test))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal')

plt.xlabel("True Formation Energy (eV/atom)", fontsize=12)
plt.ylabel("Predicted Formation Energy (eV/atom)", fontsize=12)
plt.title("Random Forest: Predicted vs True Formation Energy", fontsize=13)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
~~~
{: .python}




This work presents a computational framework for generating invariant descriptors of **inorganic crystalline solids**, including metallic (Cu), semiconducting (Si), and ionic (NaCl) materials, using atomistic representation learning. The **Smooth Overlap of Atomic Positions (SOAP)** descriptor is employed to encode the local chemical environments of atoms within periodic unit cells, with explicit support for crystalline boundary conditions through the `periodic=True` flag. Each atomic environment is characterized within a spherical cutoff radius of 4.0 Å, using a basis expansion truncated at radial order $ n_{\text{max}} = 6 $ and angular momentum $ l_{\text{max}} = 4 $, resulting in a 384-dimensional feature vector per atom.

To obtain global material-level representations suitable for property prediction, the per-atom SOAP vectors are averaged within each unit cell, yielding a single, rotationally and translationally invariant descriptor per compound. This approach effectively captures structural motifs such as face-centered cubic (FCC), diamond cubic, and rocksalt arrangements, while remaining robust to atomic permutations. The resulting global descriptors form a matrix of dimension $ (3, 384) $, ready for use in machine learning models targeting bulk properties such as formation energy, band gap, or elastic moduli.

Additionally, analytical derivatives of the SOAP kernel with respect to atomic positions are computed, enabling the integration of these descriptors into gradient-based models for predicting atomic forces—a critical component in the development of interatomic potentials via machine learning. Despite challenges in derivative computation for periodic systems, the framework demonstrates feasibility for small unit cells.

This pipeline establishes a reproducible methodology for transforming crystal structures into fixed-size numerical representations, aligning with modern materials informatics workflows. It supports high-throughput screening, similarity analysis, and surrogate modeling in computational materials science, and can be extended to larger datasets such as those from the Materials Project or OQMD.

### 🔍 Notes

- **Coulomb Matrix** is not used here because it assumes isolated, finite systems (molecules). For solids, **SOAP**, **ACS**, or **MBTR** are preferred.
- Use **averaging**, **max-pooling**, or **site-weighted aggregation** to go from local to global descriptors.
- For disordered or large-cell materials, consider **sparse sampling** or **random subsampling** of atomic centers.





