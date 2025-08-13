---
title:  Active Learning 
teaching: 1
exercises: 0
questions:
- ""
objectives:
- ""
---

<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>


# Active Learning for materials


Active learning is a data acquisition approach where the algorithm actively collaborates with a user to gather additional labeled data, improving its accuracy over time. Initially, the algorithm is trained on a small set of labeled examples and then identifies which new data points would be most beneficial to label. It queries the user to label these uncertain or informative samples, using the newly labeled data to enhance its performance. This cycle continues until the model reaches a desired level of accuracy.

**Example:** Consider training a machine learning model to recognize handwritten digits. The process begins with a limited set of labeled digits. The algorithm then selects digits it finds uncertain and asks the user to label them. By incorporating these newly labeled samples, the model refines its predictions, repeating this interactive labeling until it can reliably identify most handwritten digits.

## Derivative structure enumeration

~~~
from pymatgen import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.advanced_transformations import EnumerateStructureTransformation
from chgnet.model import CHGNet
from chgnet.utils import parse_vasp_dir
import numpy as np
import os

# Load and process structure
structure = Structure.from_file("EntryWithCollCode418490.cif")
print(structure)

# Modify disordered Li sites
for i, site in enumerate(structure):
    if not site.is_ordered:
        structure[i] = {"Li+": 0.5}
print("The composition after adjustments is %s." % structure.composition.reduced_formula)

# Symmetry reduction
analyzer = SpacegroupAnalyzer(structure)
prim_cell = analyzer.find_primitive()
print(prim_cell)

# Enumerate ordered structures
enum = EnumerateStructureTransformation()
enumerated = enum.apply_transformation(prim_cell, 100)
structures = [d["structure"] for d in enumerated]
print("%d structures returned." % len(structures))

# Load pre-trained CHGNet model
chgnet = CHGNet.load()

# Predict energies (and optionally forces/stresses)
predictions = []
for i, s in enumerate(structures):
    print(f"Predicting for structure {i+1}/{len(structures)}...")
    result = chgnet.predict_structure(s)
    predictions.append({
        "structure": s,
        "energy_per_atom": result["energy"],
        "forces": result["forces"],
        "magmoms": result["magmoms"],
        "total_energy": result["energy"] * len(s),  # optional
    })

# Convert to array for analysis
energies = np.array([p["energy_per_atom"] for p in predictions])

print(f"Energy range: {energies.min():.4f} — {energies.max():.4f} eV/atom")
print(f"Average energy: {energies.mean():.4f} eV/atom")
~~~
{: .python}
