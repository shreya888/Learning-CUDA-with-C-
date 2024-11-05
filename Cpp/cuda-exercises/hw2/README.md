# Homework 2

These exercises were to practice the concept of Shared Memory on the GPU.

## **1. 1D Stencil Using Shared Memory**

Created a 1D stencil application that uses shared memory. The code skeleton was provided and modified in *stencil_1d.cu*. A stencil operation is commonly used in signal processing, image processing, and numerical methods where each element of the output array depends on a small region of the input array. Editted that file by adding code and many comments. The code was verified against the output and would report any errors.

In a simple 1D stencil operation with a radius of 1, the output at each index `i` might be computed as:
```output[i]=input[i−1]+input[i]+input[i+1]```

## **2. 2D Matrix Multiply Using Shared Memory**

Applied shared memory to the 2D matrix multiply from hw1. The code skeleton was provided and modified provided in *matrix_mul_shared.cu*. Loaded the required data (A, B matrix) into shared memory as tiles As and Bs and then appropriately updated the dot product calculation in Cs then back to global C. Added many comments to understand the complete code and intuition behind the code. Shared memory increased the time to compute and initialize compared to the hw1 implementation of this 2D matrix multiply. From "1.104000" originally to "1.678000" to compute. Init increased from "0.050000" to "0.153000" seconds. One potential reason for this could be the overhead introduced by using shared memory (loading tiles, managing synchronization), which can outweigh the benefits if the problem isn't large or complex enough to offset these costs.
