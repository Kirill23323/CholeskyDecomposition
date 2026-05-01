# Block Cholesky Decomposition (OpenMP)

Cholesky's method is one of the most effective methods for factoring symmetric, uniquely defined matrices and is widely used for solving systemic linear data, optimization problems, and computational methods in scientific computing. Its main advantage is its reduced computational complexity compared to more general methods such as LU decomposition.

## Requirements

Install required tools:

```bash
sudo apt update
sudo apt install -y build-essential g++ libomp-dev
```

Recommended:

* GCC 10+ (or newer)

---

## Compilation

### Basic (OpenMP enabled)

```bash
g++ -O3 -fopenmp -march=native cholesky.cpp -o cholesky
```

---


## Run

```bash
./cholesky
```

---

## Output

Program prints:

* execution time
* reconstruction error:
  ||A − L·Lᵀ||

---

## Notes

* Matrix is generated as symmetric positive definite (SPD)
* Block size = 64
* Parallelization is done using OpenMP
* Deterministic version (`_Consistent`) is included for validation

---

## Optional

Control number of threads:

```bash
export OMP_NUM_THREADS=4
```
