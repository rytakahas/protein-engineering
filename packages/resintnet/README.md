# Residue Interaction Network (RIN) Analysis

This project builds **Residue Interaction Networks (RINs)** from protein structures, incorporating ligands and cofactors as weighted nodes based on the number of hydrogen bonds.

The pipeline can accept:
- **Experimental structures** (PDB IDs)
- **Predicted models** (e.g., AlphaFold2, ESMFold)

---

## Project Overview

- Parse protein 3D structures.
- Construct a graph where nodes = residues/ligands, edges = interactions.
- Ligand and cofactor connections are weighted by the number of hydrogen bonds.
- Calculate centrality measures (degree, closeness, betweenness) for each node.
- Analyze redidue and ligands/cofactors and protein-protein intreactions on the structural network.

---

## Directory Structure

```graphql
Res_Int_Net/
├── notebook/
│ ├── prot_net_ligs_h.ipynb # Graph-based analysis of residue-level structural networks
│ └── README.md
├── prot_api_flask/
│     ├── prot_api_flask.py # Flask API exposing residue graph analysis from structure
│     ├── residue_network.py # Builds residue interaction graph & computes centralities
│     ├── sample_input.json # Example request JSON with structure path
│     └── README.md # API usage instructions
└── README.md
```
---

## What This Module Does

This module:

- Parses protein structures (PDB or mmCIF)
- Builds a **residue interaction graph** (based on distance thresholds or contact rules)
- Computes **graph centrality metrics** such as:
  - Degree centrality
  - Betweenness centrality
  - Eigenvector centrality
- Can be extended to support ML-based classification or ranking of residues

---

## Flask API (prot_api_flask/)

### Endpoint

```http
POST /residue_graph_analysis
Example Request (sample_input.json)

{
  "structure_path": "/path/to/protein_structure.pdb"
}
```

Input
structure_path: Full path to a PDB or mmCIF structure file

Supported sources:

RCSB PDB

AlphaFold DB (AF3)

Your own predicted models (e.g., RoseTTAFold, ESMFold)

## Currently supports local file paths only. File upload support is planned.

Response

```json
{
  "residue_centralities": [
    {"residue": "ASP45", "betweenness": 0.83, "eigenvector": 0.51},
    {"residue": "TYR123", "betweenness": 0.91, "eigenvector": 0.63},
    ...
  ]
}
```

### File: residue_network.py
This module performs:

Parsing of .pdb or .mmcif structure files

Construction of a residue-level interaction graph (e.g., via distance-based edges)

Computation of standard graph centralities per residue

Optional export to JSON or integration into ML pipelines

Function Template

```python
# residue_network.py

def build_residue_graph(structure_path):
    """
    Parses a PDB or mmCIF file and builds a residue interaction graph.
    Returns a networkx.Graph object.
    """
    pass

def compute_centralities(G):
    """
    Computes centrality metrics (e.g., degree, betweenness, eigenvector).
    Returns a dictionary or list of residue-level values.
    """
    pass
```
  Related Notebook
  notebook/prot_net_ligs_h.ipynb shows:

How to create a graph from protein structure

How to compute centralities

How these metrics can support downstream residue-level predictions

## Future Integration with Seq_MLs
This graph-based module will be optionally merged with the sequence-based mutation predictor in:

  Seq_MLs/prot_api_flask/

Resulting in a unified protein API that supports:

Mutation-based prediction (e.g., activity)

Graph-based residue-level structure analysis

## Requirements
Python 3.8+

Biopython

NetworkX

gemmi or MDAnalysis

PyTorch or TensorFlow (optional for downstream ML)

Flask

## License
MIT License — free to use for research and commercial development.
---

## Planned API (Coming Soon)

### Example Request

```bash
curl -X POST http://127.0.0.1:5050/analyze_structure \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```
Input: JSON containing either:

A PDB ID (download automatically).

A file path to a local structure (PDB/mmCIF format).

Output: RIN graph metrics and centrality scores.

### TODO List
- Implement Flask API server.
- Accept either PDB ID or uploaded structure file.
- Build RIN graph dynamically with ligand interactions.
- Return JSON with centrality results and graph summaries.
- Dockerize for easy deployment.
- R&D: Protein Allosteric Memory Condition

  This repository explores the **memory condition** introduced in [Phys. Rev. Lett. 129, 028101 (2022)]  (https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.129.028101) and how it can be **applied across modeling paradigms** — from classical   normal mode analysis (ANM) to statistical mechanics (MD) and quantum mechanics (QM/MM).

---

#####  Memory Condition

  The core condition for **retention of signal or memory** along an edge (i.e., between two interacting elements such as atoms, residues, or sites) is:

  $\[
  \left\langle \frac{q_{ij}^2}{\kappa_{ij}} \right\rangle > \gamma
  \]$

  where:

  - $\( q_{ij} \)$: response to perturbation between nodes $\(i\)$ and $\(j\)$
  - $\( \kappa_{ij} \)$: stiffness or resistance of the interaction
  - $\( \gamma \)$: memory threshold

  If this condition is satisfied, the edge is considered to have **retained memory** of the perturbation.

  ---

##### Model-Agnostic Interpretation

  The condition is **model-agnostic**, and the terms can be redefined according to the physical layer:

| Modeling Domain         | $\( q_{ij} \)$: Response                   | $\( \kappa_{ij} \)$: Resistance/Stiffness       | Notes |
|-------------------------|------------------------------------------|-----------------------------------------------|-------|
| **Normal Mode Analysis (ANM)** | Relative displacement in a mode      | Spring constant (usually 1.0)                  | Classical elastic model |
| **Perturbation Response Scanning (PRS)** | Displacement due to force at node | Local mechanical resistance (can be Hessian)   | Linear force propagation |
| **Molecular Dynamics (MD)**   | Time-averaged distance fluctuation   | Effective stiffness (from fluctuations)        | Requires long simulations |
| **QM / QM-MM**                | Change in charge, bond length, or dipole | Force constant from PES, Hessian matrix        | Sensitive to electronic structure |

---

##### Application Scenarios

  - Identify **signal-retaining edges** or **residue-residue pathways**
  - Predict **allosteric hotspots** or **communication networks**
  - Combine with **centrality analysis** (e.g., edge betweenness) for robust signal path detection

---

##### 🧪 Examples (Coming Soon)

  - `prody_memory_condition.py`: Apply the condition using ANM from ProDy
  - `qm_memory_evaluator.py`: Parse Hessian output from Gaussian/ORCA to evaluate memory condition at binding sites
  - `networkx_memory_paths.ipynb`: Visualize predicted allosteric pathways with centrality overlays

---

##### References

- Ronellenfitsch, H., & Dunkel, J. (2022). Memory formation in flow networks. *Physical Review Letters*, 129(2), 028101.
- Bahar, I., Lezon, T. R., Yang, L. W., & Eyal, E. (2010). Global dynamics of proteins: bridging between structure and function. *Annual Review of Biophysics*, 39, 23–42.

---



### Requirements
- Python 3.8+
- Biopython
- NetworkX
- RDKit (optional, for more advanced ligand handling)
- Matplotlib / Seaborn (optional for plotting)

Install with:

```bash
pip install -r requirements.txt
```

Notes
Ligand weights enrich the standard RIN approach by capturing chemical environment effects.

This approach is suitable for small molecule docking evaluations, druggability assessments, and protein design tasks.
