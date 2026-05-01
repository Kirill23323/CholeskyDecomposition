#include <iostream>
#include <cmath>
#include <chrono>
#include <random>
#include <cstring>
#include <omp.h>
#include <algorithm>


void cholDiagBlock(double* matrix_Ablock, int bs, int lda) {
    for(int index_i = 0; index_i < bs; ++index_i) {
        double* row_i = matrix_Ablock + index_i * lda;
        double sum = row_i[index_i];
        for(int index_k = 0; index_k < index_i; ++index_k) {
            double lik = row_i[index_k];
            sum -= lik * lik;
        }
        if (sum < 1e-12) {
            sum = 1e-12;
        }
        row_i[index_i] = std::sqrt(sum);
        double div = row_i[index_i];
        for(int index_j = index_i + 1; index_j < bs; ++index_j) {
            double* row_j = matrix_Ablock + index_j * lda;
            double val = row_j[index_i];
            for(int index_k = 0; index_k < index_i; ++index_k) {
                val -= row_j[index_k] * row_i[index_k];
            }
            row_j[index_i] = val / div;
        }
    }
}

void cholSolveBlockTriangularSystem(double* matrix_Lik, double* matrix_Aik, double* matrix_Lkk, int rows, int bs, int lda) {
    for(int index_c = 0; index_c < bs; ++index_c) {
        double lkk_cc = matrix_Lkk[index_c * lda + index_c];
        #pragma omp parallel for
        for(int index_r = 0; index_r < rows; ++index_r) {
            double val = matrix_Aik[index_r * lda + index_c];
            for(int index_k = 0; index_k < index_c; ++index_k) {
                val -= matrix_Lik[index_r * lda + index_k] * matrix_Lkk[index_c * lda + index_k];
            }
            matrix_Lik[index_r * lda + index_c] = val / lkk_cc;
        }
    }
}

void cholSolveBlockTriangularSystemСonsistent(double* matrix_Lik, double* matrix_Aik, double* matrix_Lkk, int rows, int bs, int lda) {
    for(int index_c = 0; index_c < bs; ++index_c) {
        double lkk_cc = matrix_Lkk[index_c * lda + index_c];
        for(int index_r = 0; index_r < rows; ++index_r) {
            double val = matrix_Aik[index_r * lda + index_c];
            for(int index_k = 0; index_k < index_c; ++index_k) {
                val -= matrix_Lik[index_r * lda + index_k] * matrix_Lkk[index_c * lda + index_k];
            }
            matrix_Lik[index_r * lda + index_c] = val / lkk_cc;
        }
    }
}

void Cholesky_Decomposition(double* matrix_A, double* matrix_L, int n) {
    const int block_size = 64;
    std::memset(matrix_L, 0, n * n * sizeof(double));
    for(int index_k = 0; index_k < n; index_k += block_size) {
        int bs = std::min(block_size, n - index_k);
        double* matrix_Akk = matrix_A + index_k * n + index_k;
        double* matrix_Lkk = matrix_L + index_k * n + index_k;
        cholDiagBlock(matrix_Akk, bs, n);
        for(int index_i = 0; index_i < bs; ++index_i) {
            for(int index_j = 0; index_j <= index_i; ++index_j) {
                matrix_Lkk[index_i * n + index_j] = matrix_Akk[index_i * n + index_j];
            }
        }
        if (index_k + bs < n) {
            #pragma omp parallel for
            for(int index_i = index_k + bs; index_i < n; ++index_i) {
                double* matrix_Aik = matrix_A + index_i * n + index_k;
                double* matrix_Lik = matrix_L + index_i * n + index_k;
                cholSolveBlockTriangularSystem(matrix_Lik, matrix_Aik, matrix_Lkk, 1, bs, n);
            }
        }
        if (index_k + bs < n) {
            #pragma omp parallel for collapse(2)
            for(int index_i = index_k + bs; index_i < n; ++index_i) {
                for(int index_j = index_k + bs; index_j <= index_i; ++index_j) {
                    double* matrix_Lik_row = matrix_L + index_i * n + index_k;
                    double* matrix_Ljk_row = matrix_L + index_j * n + index_k;
                    double* matrix_Aij_ptr = matrix_A + index_i * n + index_j;
                    double sum = 0.0;
                    for(int index_c = 0; index_c < bs; ++index_c) {
                        sum += matrix_Lik_row[index_c] * matrix_Ljk_row[index_c];
                    }
                    *matrix_Aij_ptr -= sum;
                }
            }
        }
    }
}

void Cholesky_Decomposition_Сonsistent(double* matrix_A, double* matrix_L, int n) {
    const int block_size = 64;
    std::memset(matrix_L, 0, n * n * sizeof(double));
    for(int index_k = 0; index_k < n; index_k += block_size) {
        int bs = std::min(block_size, n - index_k);
        double* matrix_Akk = matrix_A + index_k * n + index_k;
        double* matrix_Lkk = matrix_L + index_k * n + index_k;
        cholDiagBlock(matrix_Akk, bs, n);
        for(int index_i = 0; index_i < bs; ++index_i) {
            for(int index_j = 0; index_j <= index_i; ++index_j) {
                matrix_Lkk[index_i * n + index_j] = matrix_Akk[index_i * n + index_j];
            }
        }
        if (index_k + bs < n) {
            for(int index_i = index_k + bs; index_i < n; ++index_i) {
                double* matrix_Aik = matrix_A + index_i * n + index_k;
                double* matrix_Lik = matrix_L + index_i * n + index_k;
                cholSolveBlockTriangularSystemСonsistent(matrix_Lik, matrix_Aik, matrix_Lkk, 1, bs, n);
            }
        }
        if (index_k + bs < n) {
            for(int index_i = index_k + bs; index_i < n; ++index_i) {
                for(int index_j = index_k + bs; index_j <= index_i; ++index_j) {
                    double* matrix_Lik_row = matrix_L + index_i * n + index_k;
                    double* matrix_Ljk_row = matrix_L + index_j * n + index_k;
                    double* matrix_Aij_ptr = matrix_A + index_i * n + index_j;
                    double sum = 0.0;
                    for(int index_c = 0; index_c < bs; ++index_c) {
                        sum += matrix_Lik_row[index_c] * matrix_Ljk_row[index_c];
                    }
                    *matrix_Aij_ptr -= sum;
                }
            }
        }
    }
}

double СheckError(double* A, double* L, int n) {

    double error = 0.0;

    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {

            double sum = 0.0;

            for(int k = 0; k < n; ++k) {
                double Lik = (i >= k) ? L[i*n + k] : 0.0;
                double Ljk = (j >= k) ? L[j*n + k] : 0.0;
                sum += Lik * Ljk;
            }

            double diff = A[i*n + j] - sum;
            error += diff * diff;
        }
    }

    return std::sqrt(error);
}


void GenerateSPD(double* A, int n) {

    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    double* M = new double[n*n];

    for(int i = 0; i < n*n; ++i)
        M[i] = dis(gen);

    for(int i = 0; i < n; ++i) {
        for(int j = 0; j <= i; ++j) {

            double sum = 0.0;

            for(int k = 0; k < n; ++k)
                sum += M[i*n + k] * M[j*n + k];

            A[i*n + j] = sum;
            A[j*n + i] = sum;
        }
    }

    delete[] M;
}


int main() {

    int n = 5000;

    double* A = new double[n*n];
    double* L = new double[n*n];

    double* A_copy = new double[n * n];

    GenerateSPD(A, n);

    std::memcpy(A_copy, A, n * n * sizeof(double));

    auto start = std::chrono::high_resolution_clock::now();

    Cholesky_Decomposition(A, L, n);

    auto end = std::chrono::high_resolution_clock::now();

    double time = std::chrono::duration<double>(end - start).count();

    std::cout << "Time: " << time << " sec\n";

    double err = СheckError(A_copy, L, n);

    std::cout << "Error ||A - LL^T|| = " << err << "\n";

    delete[] A;
    delete[] L;

    return 0;
}