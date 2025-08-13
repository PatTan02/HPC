# Techniques of High-Performance Computing 

# Techniques of High-Performance Computing (HPC)

This is a 4th-year Theoretical Physics course at UCL. I received grades of **95.00%**, **85.00%**, **93.33%**, and **100.00%** for Assignments 1–4 respectively.

---

## Assignments

### 1) Matrix Multiplication & Performance
**Notebook:** [HPC_Assignment_1.ipynb](HPC_Assignment_1.ipynb)  
- Implemented a naïve triple-loop matrix–matrix product and verified correctness against NumPy.
- Accelerated the core loops using Numba JIT; compared against a vectorised baseline.
- Benchmarked runtimes across matrix sizes; discussed cache effects and memory layout (C vs Fortran order).
- Summarised complexity (\(O(n^3)\)) vs constant-factor gains from JIT and layout.

---

### 2) Sparse 1D Poisson & GPU Rod Heating
**Notebook:** [HPC_Assignment_2.ipynb](HPC_Assignment_2.ipynb)  
- Assembled the discrete 1D Poisson operator as a sparse matrix (COO → CSR) and solved \(Au=f\) with SciPy.
- Validated solutions against an analytic reference; measured errors and scaling with \(N\).
- Implemented a CUDA kernel for the 1D heat equation; matched CPU results and reported the first-crossing time at the rod centre.

---

### 3) Custom CSR Operator & Block Matrix
**Notebook:** [HPC_Assignment_3.ipynb](HPC_Assignment_3.ipynb)  
- Built a `CSRMatrix` (LinearOperator) with efficient `_matvec` and addition; tested vs SciPy/dense equivalents.
- Timed sparse matvec against dense matvec; showed regimes where sparsity wins.
- Implemented a custom block operator with an efficient composition; verified numerically and benchmarked.

---

### 4) 2D Heat Equation — Explicit, Implicit, and GPU
**Notebook:** [HPC_Assignment_4.ipynb](HPC_Assignment_4.ipynb)  
- Solved the 2D heat equation on a square with Dirichlet boundaries.
- Implemented explicit schemes (Numba loops, Kronecker formulation, and a GPU kernel); recorded time to reach a target centre temperature.
- Implemented implicit solves (direct sparse and CG with ILU preconditioning); compared performance and stability.
- Investigated CFL-type stability limits and trade-offs between CPU/GPU and explicit/implicit methods.
