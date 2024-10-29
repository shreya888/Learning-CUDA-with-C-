#include <stdio.h>

__global__ void hello(){

  printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);  // Index of block and thread
}

int main(){

  hello<<<2,2>>>();  // Configured the cuda kernel launch: <<<blocks, threads per blocks>>>
  cudaDeviceSynchronize();  // Synchronizes with CPU so kernel gets to print output before application termination
}

