# Techniques of High-Performance Computing (HPC)

This is a 4th-year Theoretical Physics course at UCL. I received grades of **95.00%**, **85.00%**, **93.33%**, and **100.00%** for Assignments 1–4 respectively.

---
# Assignment 1 — Matrix multiplication & performance
- Wrote a **naïve triple-loop matrix–matrix product** (`slow_matrix_product`) and verified correctness against NumPy.
- Implemented a **faster version** and then **Numba-JIT compiled** versions to accelerate the core loops.
- **Benchmarked** runtimes across a range of square sizes with `timeit`, plotting results (log scale).
- Investigated the effect of **memory layout** (**C vs Fortran order**) on speed and discussed cache-friendliness.
- Summarised takeaways about algorithmic complexity (still \(O(n^3)\)) vs constant-factor speedups from JIT and layout.
---
# Assignment 2 — Sparse 1D Poisson/heat setup & GPU rod heating
**Part 1**
- Built a routine to **assemble the 1D discrete Poisson operator** and RHS vector as a **sparse matrix** (COO → CSR).
- Performed **sanity checks** (e.g., `spy` visualisation, structure/indices).
- Solved \(A u = f\) with **`scipy.sparse.linalg.spsolve`** for several \(N\), compared against an **analytic sine solution**, and computed **errors**.
- From log–log error vs \(N\), **estimated the \(N\)** needed to achieve a target error (≈ \(10^{-8}\)) and **predicted runtime**, then **validated** by running the solve at the predicted \(N\).

**Part 2**
- Wrote a **CUDA GPU kernel** to time-step the **1D heat equation on a rod** and determined **when the midpoint temperature first exceeds 9.8**.
- Compared the GPU evolution with the CPU reference for correctness and reported the crossing time.
---
# Assignment 3 — Your own CSR and a custom block matrix
**Part 1: CSRMatrix**
- Implemented a **`CSRMatrix` class** (subclassing `LinearOperator`) that **converts COO → CSR**, stores `data/indices/indptr`, and implements **`_matvec`** and **`__add__`**.
- **Tested** your operators against SciPy/dense equivalents for correctness.
- **Timed** sparse matvec vs dense NumPy matvec across sizes and **plotted scaling** (log–log), commenting on when sparsity wins.

**Part 2: CustomMatrix**
- Implemented a **custom block structure**: top-left **diagonal** block and bottom-right **\(T(Wx)\)** composition, with an efficient **`_matvec`**.
- Verified results against a dense construction and **benchmarked** custom vs dense matvec, plotting performance and discussing complexity.
---
# Assignment 4 — 2D heat equation on a square plate (explicit, implicit, GPU)
- Set up the **2D heat equation** with Dirichlet boundaries and an initial condition on an \(N\times N\) grid.
- Implemented three **explicit** schemes:
  - **Non-matrix loop** version (Numba-accelerated),
  - **Matrix/Kronecker** formulation,
  - **GPU explicit kernel** using **shared memory**.
  - For each, evolved in time until the **centre reaches \(u=1\)**, recording **iteration count** and **time**.
- Implemented **implicit** schemes:
  - Direct sparse solve with **`spsolve`** (baseline),
  - **Conjugate Gradient with ILU preconditioning** (via `spilu`) wrapped as a linear operator (faster).
- **Compared computational performance** across all methods (CPU/GPU, explicit/implicit) and discussed trade-offs.
- **Investigated stability** of the explicit methods by varying the **Courant number** and grid size, documenting observed stability limits and behaviour.
