#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024 // Size of the matrix and vector

void matrix_vector_multiply(double **A, double *v, double *result, int size, int num_threads) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; i++) {
        result[i] = 0.0;
        for (int j = 0; j < size; j++) {
            result[i] += A[i][j] * v[j];
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

        // Allocate matrices and vectors
        double **A = (double **)malloc(size * sizeof(double *));
        double *v = (double *)malloc(size * sizeof(double));
        double *result = (double *)malloc(size * sizeof(double));
        for (int i = 0; i < size; i++) {
            A[i] = (double *)malloc(size * sizeof(double));
        }

        // Initialize matrix and vector
        for (int i = 0; i < size; i++) {
            v[i] = rand() % 100;
            for (int j = 0; j < size; j++) {
                A[i][j] = rand() % 100;
            }
        }

        for (int t = 0; t < num_thread_counts; t++) {
            int num_threads = thread_counts[t];

            double start_time = omp_get_wtime();
            matrix_vector_multiply(A, v, result, size, num_threads);
            double end_time = omp_get_wtime();

            printf("Threads: %d, Time: %f seconds\n", num_threads, end_time - start_time);
        }

        // Free allocated memory
        for (int i = 0; i < size; i++) {
            free(A[i]);
        }
        free(A);
        free(v);
        free(result);

        printf("\n");
    }

    return 0;
}
