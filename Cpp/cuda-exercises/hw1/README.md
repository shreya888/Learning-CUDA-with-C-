# Homework 1

In Homework 1, I practiced foundational CUDA programming skills, including GPU memory allocation, data transfer between the host CPU and GPU, and kernel launching. Each exercise provided a code skeleton, which I completed by filling in essential CUDA functions and logic.

## Exercise Overview
* **Hello World** - A simple parallel "Hello World" program using CUDA.
* **Vector Add** - A basic element-wise vector addition program.
* **Matrix Multiply (Naive)** - A naive implementation of matrix multiplication, using 2D threadblock/grid indexing.


## **1. Hello World**
The goal of this exercise was to create a simple CUDA "Hello World" program that prints a unique message from each thread. The initial code skeleton was provided in `hello.cu`.

### Insights

1. **CUDA Kernel (`__global__` keyword)**:
   - The `__global__` keyword before the `hello` function defines it as a CUDA kernel, which means it will run on the GPU instead of the CPU.
   - When we call `hello<<<2, 2>>>();`, we're launching this kernel on the GPU with specific grid and block dimensions.

2. **Grid and Block Configuration**:
   - The `<<<2, 2>>>` syntax is used to specify the **number of blocks and threads per block**. Here, we use 2 blocks and 2 threads per block, giving a total of 4 threads.
   - Each thread will execute the code inside the `hello` kernel function. The exact message printed will depend on the block and thread indices.

3. **Indices (`blockIdx` and `threadIdx`)**:
   - `blockIdx.x` and `threadIdx.x` are built-in variables in CUDA, providing the **current block index** and **thread index** in the x-dimension.
   - These indices help each thread know its position in the grid and can be essential for tasks where each thread needs to work on specific data.

4. **`cudaDeviceSynchronize()`**:
   - CUDA operations are **asynchronous**, meaning that by default, the CPU will not wait for the GPU kernel to finish execution.
   - `cudaDeviceSynchronize()` forces the CPU to wait until all GPU operations have completed, ensuring that the program doesn’t terminate before the kernel finishes and prints its output.

5. **Output on GPU**:
   - Unlike standard `printf` on the CPU, printing from the GPU may produce output in a non-deterministic order because threads may execute in parallel.

6. **Compilation and Execution**:
   - To compile and run the program use commands like below:
    ```
    nvcc -o hello hello.cu
    ./hello.exe
    ```
   - `nvcc` is the NVIDIA CUDA compiler, similar to `gcc/g++` for `C++` code. This will generate 3 files with extensions - `.exe`, `.exp`, `.lib`. Running hello.exe outputs unique messages from each thread.


### Expected Output
The expected output, with thread and block identifiers, looks like this (ordering may vary):
```
Hello from block: 0, thread: 0
Hello from block: 0, thread: 1
Hello from block: 1, thread: 0
Hello from block: 1, thread: 1
```


## **2. Vector Add**

This code performs element-wise vector addition on the GPU using CUDA. Here's an explanation of the key sections:

### Insights

1. **Error Checking Macro**:
   - The **`cudaCheckErrors`** macro is defined to check for any CUDA errors after each CUDA runtime API call. If an error is encountered, it prints an error message and aborts the program.
   - Good practice to rigorously check these error codes. The provided macro makes this job easier.

2. **Constant Definitions**:
   - **`DSIZE`**: Defines the size of the vectors being added, i.e., the number of elements in each vector.
   - **`block_size`**: Sets the number of threads per block (256 in this case).

3. **CUDA Kernel `vadd`**:
   - The `vadd` kernel performs element-wise addition of two vectors, A and B, storing the result in C. Each thread is responsible for one element.
   - **`ds`**: Kernel parameter in the vadd function used to pass the size of the vectors to the kernel.
   - **Global thread index `idx = threadIdx.x + blockDim.x * blockIdx.x`**:
     - `threadIdx.x`: The offset (local position) of a thread within its block.
     - `blockDim.x * blockIdx.x`: Computes the total number of threads in all preceding blocks. This product gives the offset needed to find the correct position of the first thread in the current block within the global array.

4. **Memory Management and Data Transfer**:
   - **`cudaMalloc`**: Allocates size bytes of linear memory on the GPU device and returns in *devPtr a pointer to the allocated memory.
     - Analogous to `malloc` in C for memory allocation on the host, but `cudaMalloc` allocates memory in the global memory space of the CUDA device.
     - Syntax: `cudaMalloc(void** devPtr, size_t size);`
       - `devPtr`: Pointer to the allocated device memory.
       - `size`: Requested memory allocation size in bytes.
     - e.g. `cudaMalloc(&d_A, DSIZE*sizeof(float));`: This allocates memory for `DSIZE`(=4096) floats on the device for A matrix.
     - Return Value: Returns a `cudaError_t` value that can be checked to ensure the allocation was successful. Common errors include `cudaErrorMemoryAllocation` if the GPU does not have enough memory to fulfill the request.
       
   - **`cudaMemcpy`**: Copies data (`count` bytes) between host and device memory and vice versa, depending on the direction specified by the `kind` parameter.
     - Syntax: `cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);`
       - `dst`: Destination memory address (can be either device or host memory).
       - `src`: Source memory address (can be either device or host memory).
       - `count`: Number of bytes to copy.
       - `kind`: Type of transfer, specified by one of the `cudaMemcpyKind` enum values - `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, `cudaMemcpyDeviceToDevice`, or `cudaMemcpyHostToHost`. This specifies the direction of the copy.
     - e.g. `cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice);`: This copies `DSIZE`(=4096) floats from host memory `h_A` to device memory `d_A`.
     - e.g. `cudaMemcpy(h_C, d_C, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);`: This copies `DSIZE`(=4096) floats from device memory `d_A` to host memory `h_A`.

### Output
The output when complete would looked like this:
```
A[0] = 0.001251
B[0] = 0.563585
C[0] = 0.564837
```


## **3. Matrix Multiply (naive)**

A skeleton naive matrix multiply was given in `matrix_mul.cu`. I am trying to complete it to get a correct result.

This exercise introduced 2D threadblock/grid indexing. This code included built-in error checking, so a correct result is indicated by the program.
