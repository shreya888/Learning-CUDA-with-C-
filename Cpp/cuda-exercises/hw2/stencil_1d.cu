#include <stdio.h>
#include <algorithm>

using namespace std;

// Define constants
#define N 4096  // Size of the input array
#define RADIUS 3  // Radius around each element for the stencil computation (output element is calculated using a window of 2 * RADIUS + 1)
#define BLOCK_SIZE 16

__global__ void stencil_1d(int *in, int *out) {
    // Shared memory to hold the elements for the stencil computation
    // The size includes the BLOCK_SIZE and additional elements for RADIUS on either side for stencil operation
    __shared__ int temp[BLOCK_SIZE + 2*RADIUS];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;  // Global index
    int lindex = threadIdx.x + RADIUS;  // Local index for accessing shared memory, indexing into the middle of the shared memory array (starting at RADIUS), allowing room for the preceding radius values

    // Read input elements into shared memory
    temp[lindex] = in[gindex];  // Load the main element into the correct position in shared memory
    if (threadIdx.x < RADIUS) {
      temp[lindex - RADIUS] = in[gindex - RADIUS];  // Load left neighbors
      temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE]; // Load right neighbors 
    }

    // Synchronize (ensure all the data is available)
    __syncthreads();

    // Apply the stencil operation by summing values from shared memory
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++)  // Offset defined by the stencil radius
      result += temp[lindex + offset];  // Sum the values from the stencil window

    // Store the result in global memory
    out[gindex] = result;
}

void fill_ints(int *x, int n) {
  fill_n(x, n, 1);  // Fill array x with '1' for n elements
}

int main(void) {
  // Declare pointers for host and device memory
  int *in, *out; // Host copies of in and out arrays
  int *d_in, *d_out; // Device copies of in and out arrays

  // Alloc space for host copies and setup values
  int size = (N + 2*RADIUS) * sizeof(int);  // Size includes the additional elements for the stencil
  in = (int *)malloc(size); 
  fill_ints(in, N + 2*RADIUS);
  out = (int *)malloc(size); 
  fill_ints(out, N + 2*RADIUS);

  // Alloc space for device copies
  cudaMalloc((void **)&d_in, size);
  cudaMalloc((void **)&d_out, size);

  // Copy to device
  cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);

  // Launch stencil_1d() kernel on GPU
  stencil_1d<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_in + RADIUS, d_out + RADIUS);  // in and out arrays on device while accounting for the additional radius added at the start of the arrays

  // Copy result back to host
  cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

  // Error Checking
  for (int i = 0; i < N + 2*RADIUS; i++) {
    if (i<RADIUS || i>=N+RADIUS){
      if (out[i] != 1)
    	printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i], 1);
    } else {
      if (out[i] != 1 + 2*RADIUS)
    	printf("Mismatch at index %d, was: %d, should be: %d\n", i, out[i], 1 + 2*RADIUS);
    }
  }

  // Cleanup
  free(in); free(out);
  cudaFree(d_in); cudaFree(d_out);
  printf("Success!\n");
  return 0;
}
