# Computation of DH_2(G_{5,4}; F_2)

This repository contains the SageMath code used to reproduce the computer-assisted computation

\[
DH_2(G_{5,4};\mathbb F_2)=0
\]

The computation uses a support-rank filtration defined by based translates and standard parabolic subgroups of \(S_6\), together with cubical enumeration and linear algebra over \(\mathbb F_2\).

## Requirements

The computation has been tested with SageMath 10.8.

## Running the computation

From the repository root, run:

```bash
sage sage/compute.sage
```

## Repository structure

```text
parabolic/
├── README.md
├── LICENSE
├── sage/
│   ├── graph.sage
│   ├── cubical.sage
│   ├── gf2.py
│   ├── filtration.sage
│   └── compute.sage
└── results/
    └── compute_output.txt
```

### `sage/graph.sage`

Constructs the Coxeter group of type \(A_5\) and the graph \(G_{5,4}\), including lookup tables for group operations.

### `sage/cubical.sage`

Implements cubical faces, normalization, exact parabolic support, and based-cube enumeration, together with direct reference routines for verification.

### `sage/gf2.py`

Implements bitset linear algebra over \(\mathbb F_2\), including Gaussian elimination and kernel-relation tracking.

### `sage/filtration.sage`

Implements the support-rank filtration calculations, including local exact-support chain complexes, the first filtration stage, connecting homomorphisms, and relative boundary surjectivity computations.

### `sage/compute.sage`

Main reproducibility script. Runs the complete computation and verifies the conditions required for the final vanishing result.
