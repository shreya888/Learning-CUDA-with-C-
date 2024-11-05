#include <stdio.h>

// these are just for timing measurments
#include <time.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


const int DSIZE = 8192;  // Size of each dimension of the square matrices (8192 X 8192)
const int block_size = 32;  // CUDA maximum is 1024 *total* threads in block; Number of threads per block in each dimension (32x32)
const float A_val = 3.0f;  // Default values for initializing matrices A
const float B_val = 2.0f;  // Default values for initializing matrices B

// matrix multiply (naive) kernel with shared memory: C = A * B
__global__ void mmul(const float *A, const float *B, float *C, int ds) {

  // declare cache in shared memory
  // Shared memory arrays As and Bs hold sub-blocks of A and B in shared memory to allow fast access by threads within the same block.
  __shared__ float As[block_size][block_size];
  __shared__ float Bs[block_size][block_size];
  
  // Thread Indexing
  int idx = threadIdx.x+blockDim.x*blockIdx.x; // create thread x index  - col
  int idy = threadIdx.y+blockDim.y*blockIdx.y; // create thread y index  - row

  if ((idx < ds) && (idy < ds)){
    float temp_sum = 0;  // Sum calculation variable
    for (int i = 0; i < ds/block_size; i++) { //  Perform mul for every tile As, Bs

      // Load data into shared memory, aka tile of A and B called As and Bs
      // Row of A mul (vary in x) with column of B (vary in y), hence equate the indices accordingly
      // (i * block_size + threadIdx.<x/y>)- adjust for the position within the current tile in A and B
      // As - Global memory location in A that the thread reads and stores = Starting position for the row this thread will read + Offset within the row to select a specific column
      As[threadIdx.y][threadIdx.x] = A[idy*ds + (i*block_size+threadIdx.x)];
      // Bs - Global memory location in B that the thread reads and stores = Row offset in matrix B for the tile column + Global column position this thread reads (read the same column across multiple rows)
      Bs[threadIdx.y][threadIdx.x] = B[(i*block_size+threadIdx.y)*ds + idx];

      // Synchronize - Ensures all threads in the block have finished loading data into As and Bs before starting calculations
      __syncthreads();

      // Keep track of the running sum
      for (int k = 0; k < block_size; k++)
      	temp_sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];  // Partial dot product for the current tile (row, column = y, x)
      __syncthreads();

    }

    // Write to global memory
    C[idy*ds+idx] = temp_sum;  // Each thread writes its final result to the appropriate position in the global result matrix C
  }
}

int main(){

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;


  // these are just for timing
  clock_t t0, t1, t2;
  double t1sum=0.0;
  double t2sum=0.0;

  // start timing
  t0 = clock();

  h_A = new float[DSIZE*DSIZE];
  h_B = new float[DSIZE*DSIZE];
  h_C = new float[DSIZE*DSIZE];
  for (int i = 0; i < DSIZE*DSIZE; i++){
    h_A[i] = A_val;
    h_B[i] = B_val;
    h_C[i] = 0;}

  // Initialization timing
  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("Init took %f seconds.  Begin compute\n", t1sum);

  // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));
  cudaMalloc(&d_B, DSIZE*DSIZE*sizeof(float));
  cudaMalloc(&d_C, DSIZE*DSIZE*sizeof(float));
  cudaCheckErrors("cudaMalloc failure");  // cudaCheckErrors - Macro checks for any CUDA errors after API calls or kernel launches
  cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  // Cuda processing sequence step 1 is complete

  // Launch kernel
  dim3 block(block_size, block_size);  // dim3 variable holds 3 dimensions
  dim3 grid((DSIZE+block.x-1)/block.x, (DSIZE+block.y-1)/block.y);
  mmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);
  cudaCheckErrors("kernel launch failure");

  // Cuda processing sequence step 2 is complete

  // Copy results back to host
  cudaMemcpy(h_C, d_C, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);

  // GPU timing
  t2 = clock();
  t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
  printf ("Done. Compute took %f seconds\n", t2sum);

  // Cuda processing sequence step 3 is complete

  // Verify results
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");
  for (int i = 0; i < DSIZE*DSIZE; i++) if (h_C[i] != A_val*B_val*DSIZE) {printf("mismatch at index %d, was: %f, should be: %f\n", i, h_C[i], A_val*B_val*DSIZE); return -1;}
  printf("Success!\n"); 
  return 0;
}
  
