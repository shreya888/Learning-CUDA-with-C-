# Homework 1

In these exercises, I practiced some basic CUDA applications. I learnt how to allocate GPU memory, move data between the host and the GPU, and launch kernels.

## **1. Hello World**

The first task was to create a simple hello world application in CUDA. The code skeleton was provided in `hello.cu`. Editted that file (the FIXME locations), so that the output when run is like this:

```
Hello from block: 0, thread: 0
Hello from block: 0, thread: 1
Hello from block: 1, thread: 0
Hello from block: 1, thread: 1
```

(the ordering of the above lines may vary; ordering differences do not indicate an incorrect result)

Note the use of `cudaDeviceSynchronize()` after the kernel launch. In CUDA, kernel launches are *asynchronous* to the host thread. The host thread will launch a kernel but not wait for it to finish, before proceeding with the next line of host code. Therefore, to prevent application termination before the kernel gets to print out its message, we must use this synchronization function.

After editing the code, compile it using the following:

```
nvcc -o hello hello.cu
```

`nvcc` is the CUDA compiler invocation command. The syntax is generally similar to gcc/g++. This will generate 3 files with extensions - `.exe`, `.exp`, `.lib`.

To run the code use command:

```
./hello.exe
```


## **2. Vector Add**

Wrote a vector add program from the skeleton code given in `vector_add.cu`. Editted the code to build a complete vector_add program. Compiled and ran it similar to the method given in exercise 1.

This skeleton code included some CUDA error checking. Every CUDA runtime API call returns an error code. It's good practice (especially if you're having trouble) to rigorously check these error codes. A macro is given that will make this job easier. Note the special error checking method after a kernel call.

The output when complete would looked like this:
```
A[0] = 0.001251
B[0] = 0.563585
C[0] = 0.564837
```

## **3. Matrix Multiply (naive)**

A skeleton naive matrix multiply was given in `matrix_mul.cu`. I am trying to complete it to get a correct result.

This exercise introduced 2D threadblock/grid indexing. This code included built-in error checking, so a correct result is indicated by the program.
