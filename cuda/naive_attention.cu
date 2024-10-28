// Attention algorithm implement in CUDA

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_DIM 32

void init_matrix(float* M, int x, int y) {
    for (size_t i = 0; i < x*y; i++) {
        M[i] = (float)rand() / RAND_MAX;
    }
}

__global__ void matmul(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];
    
    float temp = 0.0f;
    for (int tile = 0; tile < K; tile += blockDim.x) {
        A_s[threadIdx.y][threadIdx.x] = A[row * K + tile + threadIdx.x];
        B_s[threadIdx.y][threadIdx.x] = B[tile * N + threadIdx.y * N + col];
        __syncthreads();
        
        for (int i = 0; i < TILE_DIM; i++) {
            temp += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
        }
        __syncthreads();
    } 
    C[row * N + col] = temp;
}

__global__ void softmax(float* in, float* out, int x, int y) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if (tid < x) {
        // Step 1: Find max value in the row for numerical stability
        float max_val = -INFINITY;
        for (int i = 0; i < y; i++) {
            max_val = max(max_val, in[tid * y + i]);
        }
        
        // Step 2: Compute exponentials and sum
        float sum = 0.0f;
        for (int i = 0; i < y; i++) {
            float exp_val = expf(in[tid * y + i] - max_val);
            out[tid * y + i] = exp_val;
            sum += exp_val;
        }
        
        // Step 3: Normalize by sum
        for (int i = 0; i < y; i++) {
            out[tid * y + i] /= sum;
        }
    }
}

__global__ void transpose(float* in, float* out, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        out[col * rows + row] = in[row * cols + col];
    }
}

__global__ void scale_kernel(float* matrix, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        matrix[idx] *= scale;
    }
}

int main() {
    const int N = 1 << 10;
    const int D = 1 << 6;
    float* Q = (float*)malloc(sizeof(float) * N * D);
    float* K = (float*)malloc(sizeof(float) * N * D);
    float* Kt = (float*)malloc(sizeof(float) * D * N);
    float* V = (float*)malloc(sizeof(float) * N * D);
    float* S = (float*)malloc(sizeof(float) * N * N);
    float* P = (float*)malloc(sizeof(float) * N * N);
    float* O = (float*)malloc(sizeof(float) * N * D);
    
    init_matrix(Q, N, D); init_matrix(K, N, D); init_matrix(V, N, D);
    
    float* Q_d, *K_d, *Kt_d, *V_d, *S_d, *P_d, *O_d;
    cudaMalloc(&Q_d, sizeof(float) * N * D);
    cudaMalloc(&K_d, sizeof(float) * N * D);
    cudaMalloc(&Kt_d, sizeof(float) * D * N);
    cudaMalloc(&V_d, sizeof(float) * N * D);
    cudaMalloc(&S_d, sizeof(float) * N * N);
    cudaMalloc(&P_d, sizeof(float) * N * N);
    cudaMalloc(&O_d, sizeof(float) * N * D);
    
    cudaMemcpy(Q_d, Q, sizeof(float) * N * D, cudaMemcpyHostToDevice);
    cudaMemcpy(K_d, K, sizeof(float) * N * D, cudaMemcpyHostToDevice);
    cudaMemcpy(V_d, V, sizeof(float) * N * D, cudaMemcpyHostToDevice);
    
    dim3 num_threads(TILE_DIM, TILE_DIM);
    dim3 num_blocks((N + num_threads.x - 1) / num_threads.x, (N + num_threads.y  - 1) / num_threads.y);

    transpose <<< num_blocks, num_threads >>> (K_d, Kt_d, N, D); // K -> Kt
    matmul <<< num_blocks, num_threads >>> (Q_d, Kt_d, S_d, N, N, D); // QKt -> S
    
    float scale = 1.0f / sqrt(D);
    int total_threads = 256;
    int total_blocks = (N * N + total_threads - 1) / total_threads;

    scale_kernel <<< total_blocks, total_threads >>> (S_d, scale, N * N); // S / sqrt(D) -> S
    
    softmax <<< (N + 255) / 256, 256 >>> (S_d, P_d, N, N); // softmax(S) -> P
    matmul <<< num_blocks, num_threads >>> (P_d, V_d, O_d, N, D, N); // PV -> O
    
    cudaMemcpy(O, O_d, sizeof(float) * N * D, cudaMemcpyDeviceToHost);
    
    printf("First few elements of output matrix:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%f ", O[i * D + j]);
        }
        printf("\n");
    }
    
    cudaFree(Q_d);
    cudaFree(K_d);
    cudaFree(Kt_d);
    cudaFree(V_d);
    cudaFree(S_d);
    cudaFree(P_d);
    cudaFree(O_d);
    
    free(Q);
    free(K);
    free(V);
    free(S);
    free(P);
    free(O);
    
    return 0;
}
