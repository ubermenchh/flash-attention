// Flash Attention implemented in CUDA

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,                     \
             cudaGetErrorString(err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  }

void init_matrix(float *matrix, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = (float)rand() / RAND_MAX;
  }
}

void init_array(float *arr, int length, float value) {
  for (size_t i = 0; i < length; i++) {
    arr[i] = value;
  }
}

void print_matrix(float *M, int rows, int cols) {
  printf("([\n");
  for (size_t i = 0; i < rows; i++) {
    printf("  [");
    for (size_t j = 0; j < cols; j++) {
      printf(" %f", M[i * cols + j]);
    }
    printf(" ]\n");
  }
  printf("], size=(%d, %d))\n", rows, cols);
}

__global__ void flash_attention(const float *Q, const float *K, const float *V,
                                const int N, const int D, const int Br,
                                const int Bc, const float scale, float *l,
                                float *m, float *O) {

  // Shared memory allocation
  extern __shared__ float shared_mem[];
  float *q_tile = shared_mem;              // (Br, D)
  float *k_tile = &shared_mem[Br * D];     // (Br, D)
  float *v_tile = &shared_mem[2 * Br * D]; // (Bc, D)
  float *s_tile = &shared_mem[3 * Br * D]; // (Br, Bc)

  // Number of blocks needed
  const int n_blocks = (N + Bc - 1) / Bc;

  // Process each block
  for (int block = 0; block < n_blocks; block++) {
    // Load K and V tiles to shared memory
    if (threadIdx.x < Bc) {
      const int col = block * Bc + threadIdx.x;
      if (col < N) {
        for (int d = 0; d < D; d++) {
          k_tile[threadIdx.x * D + d] = K[col * D + d];
          v_tile[threadIdx.x * D + d] = V[col * D + d];
        }
      }
    }
    __syncthreads();

    // Process each row in current block
    if (threadIdx.x < Br) {
      const int row = blockIdx.x * Br + threadIdx.x;
      if (row < N) {
        // Load Q tile
        for (int d = 0; d < D; d++) {
          q_tile[threadIdx.x * D + d] = Q[row * D + d];
        }

        // Load previous m and l values
        float prev_m = m[row];
        float prev_l = l[row];

        // Compute Sij = Qi * KjT
        float max_val = -INFINITY;
        for (int j = 0; j < Bc && (block * Bc + j) < N; j++) {
          float score = 0.0f;
          for (int d = 0; d < D; d++) {
            score += q_tile[threadIdx.x * D + d] * k_tile[j * D + d];
          }
          score *= scale;
          s_tile[threadIdx.x * Bc + j] = score;
          // mij = rowmax(Sij)
          max_val = max(max_val, score);
        }

        // Compute Pij = exp(Sij - mij), lij = rowsum(Pij)
        float sum = 0.0f;
        for (int j = 0; j < Bc && (block * Bc + j) < N; j++) {
          s_tile[threadIdx.x * Bc + j] =
              __expf(s_tile[threadIdx.x * Bc + j] - max_val);
          sum += s_tile[threadIdx.x * Bc + j];
        }

        // Compute mi, li_new
        const float new_m = max(prev_m, max_val);
        const float new_l =
            __expf(prev_m - new_m) * prev_l + __expf(max_val - new_m) * sum;

        // Compute and write to Oi
        for (int d = 0; d < D; d++) {
          float out = 0.0f;
          for (int j = 0; j < Bc && (block * Bc + j) < N; j++) {
            out += s_tile[threadIdx.x * Bc + j] * v_tile[j * D + d];
          }
          O[row * D + d] = (1.0f / new_l) *
                           (prev_l * __expf(prev_m - new_m) * O[row * D + d] +
                            __expf(max_val - new_m) * out);
        }

        // Write to m and l
        m[row] = new_m;
        l[row] = new_l;
      }
    }
    __syncthreads();
  }
}

int main() {
  srand(69);
  const int N = 1 << 10; // 1024
  const int D = 1 << 6;  // 64
  const int Br = 1 << 5; // 32
  const int Bc = 1 << 5; // 32
  const float scale = 1.0f / sqrtf(D);

  float *Q = (float *)malloc(sizeof(float) * N * D);
  float *K = (float *)malloc(sizeof(float) * N * D);
  float *V = (float *)malloc(sizeof(float) * N * D);
  float *O = (float *)malloc(sizeof(float) * N * D);
  float *l = (float *)malloc(sizeof(float) * N);
  float *m = (float *)malloc(sizeof(float) * N);

  init_matrix(Q, N, D);
  init_matrix(K, N, D);
  init_matrix(V, N, D);
  memset(O, 0, sizeof(float) * N * D);
  init_array(l, N, 0.0f);
  init_array(m, N, -INFINITY);

  float *Q_d, *K_d, *V_d, *O_d, *l_d, *m_d;
  CHECK_CUDA(cudaMalloc(&Q_d, sizeof(float) * N * D));
  CHECK_CUDA(cudaMalloc(&K_d, sizeof(float) * N * D));
  CHECK_CUDA(cudaMalloc(&V_d, sizeof(float) * N * D));
  CHECK_CUDA(cudaMalloc(&O_d, sizeof(float) * N * D));
  CHECK_CUDA(cudaMalloc(&l_d, sizeof(float) * N));
  CHECK_CUDA(cudaMalloc(&m_d, sizeof(float) * N));

  CHECK_CUDA(cudaMemcpy(Q_d, Q, sizeof(float) * N * D, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(K_d, K, sizeof(float) * N * D, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(V_d, V, sizeof(float) * N * D, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(O_d, O, sizeof(float) * N * D, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(l_d, l, sizeof(float) * N, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(m_d, m, sizeof(float) * N, cudaMemcpyHostToDevice));

  dim3 grid((N + Br - 1) / Br); // Number of thread blocks
  dim3 block(Br);               // Threads per block

  // size of shared memory
  size_t shared_mem_size = (3 * Br * D + Br * Bc) * sizeof(float);
  printf("Launching kernel with grid=%d, block=%d, shared_mem=%zu bytes\n",
         grid.x, block.x, shared_mem_size);

  flash_attention<<<grid, block, shared_mem_size>>>(Q_d, K_d, V_d, N, D, Br, Bc,
                                                    scale, l_d, m_d, O_d);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(O, O_d, sizeof(float) * N * D, cudaMemcpyDeviceToHost));

  printf("First few elements of output matrix:\n");
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      printf("%f ", O[i * D + j]);
    }
    printf("\n");
  }

  cudaFree(Q_d);
  cudaFree(K_d);
  cudaFree(V_d);
  cudaFree(O_d);
  cudaFree(l_d);
  cudaFree(m_d);

  free(Q);
  free(K);
  free(V);
  free(O);
  free(l);
  free(m);

  return 0;
}
