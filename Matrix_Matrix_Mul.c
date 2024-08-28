#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024 // Size of the matrix

void matrix_multiply(double **A, double **B, double **C, int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < size; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    int sizes[] = {512, 1024, 2048}; // Different matrix sizes
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int thread_counts[] = {1, 2, 4, 8}; // Different thread counts
    int num_thread_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);

    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];
        printf("Matrix size: %d x %d\n", size, size);

        // Allocate matrices
        double **A = (double **)malloc(size * sizeof(double *));
        double **B = (double **)malloc(size * sizeof(double *));
        double **C = (double **)malloc(size * sizeof(double *));
        for (int i = 0; i < size; i++) {
            A[i] = (double *)malloc(size * sizeof(double));
            B[i] = (double *)malloc(size * sizeof(double));
            C[i] = (double *)malloc(size * sizeof(double));
        }

        // Initialize matrices
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                A[i][j] = rand() % 100;
                B[i][j] = rand() % 100;
            }
        }

        for (int t = 0; t < num_thread_counts; t++) {
            int num_threads = thread_counts[t];

            double start_time = omp_get_wtime();
            matrix_multiply(A, B, C, size, num_threads);
            double end_time = omp_get_wtime();

            printf("Threads: %d, Time: %f seconds\n", num_threads, end_time - start_time);
        }

        // Free allocated memory
        for (int i = 0; i < size; i++) {
            free(A[i]);
            free(B[i]);
            free(C[i]);
        }
        free(A);
        free(B);
        free(C);

        printf("\n");
    }

    return 0;
}
