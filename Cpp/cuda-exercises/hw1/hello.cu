#include <stdio.h>

// CUDA kernel function that will run on the GPU
__global__ void hello(){
  // Print message showing which block and thread is currently executing this line
  printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);  // Index of block and thread
}

int main(){
  // Launching the CUDA kernel 'hello' by configuring it with a grid of 2 blocks and 2 threads per block
  hello<<<2,2>>>();  // <<<blocks, threads per block>>>

  // Ensure the CPU waits until the GPU completes the kernel execution
  cudaDeviceSynchronize();  // Synchronizes with CPU so kernel gets to print output before application termination
}
