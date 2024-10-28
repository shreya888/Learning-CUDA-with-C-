# Learning CUDA with C++ & PyTorch
My notes, exercises etc will be recorded here while learning CUDA with C++

1. Cuda is Asynchronous
   * Asynchronous programming is a technique that allows a program to start a long-running task while still being able to respond to other events.
   * Hence python's time module would not measure the actual time the kernal runs. Thus, `torch.cuda.event` is used to measure start and end time. CUDA events are synchronization markers that can be used to monitor the device’s progress, to accurately measure timing, and to synchronize CUDA streams.

2. Warmup run
   * The first kernel call takes longer, so it is a good practice to measure your benchmark run after at least one warmup run. It may involve a Just-in-time compilation which can take many seconds for complex kernels. This is why you generally call each kernel once before benchmarking. Sometimes it’s also referred to as “cooking” a kernel.
   * Similar to "clearing one's throat before giving a speech". Or "in car racing the drivers would put their car into gear first and rev the engine before the light turns green, giving a much faster start."

3. Synchronize the code
   * `torch.cuda.synchronize` is used to synchronize the current device and waits until all GPU work in all streams is finished thus blocking the host from advancing. As CUDA operations are executed asynchronously w.r.t. the CPU, this prevents CPU thread from proceeding until the previous works are done.

4. ATen (atten::<some_function_name>)
    * ATen is the Torch C++ tensor math library. This first row of operations represents the direct translation of our code calling the high-level operations in ATen.
    * Each successive row underneath represents a function called by the function in the row above. The operations become more specific, from aten::mul to aten::sqrt (example from torch.sqrt(vector * vector) implementation in pytorch_abs.py), until we are finally ready to send an instruction to the GPU.
  

References
1. https://github.com/gpu-mode/lectures/tree/main
2. https://www.youtube.com/@GPUMODE
3. https://github.com/CisMine/Parallel-Computing-Cuda-C
4. https://forums.developer.nvidia.com/
5. https://discuss.pytorch.org
6. https://pytorch.org/docs
7. https://wandb.ai/wandb/trace/reports/Using-the-PyTorch-Profiler-with-W-B--Vmlldzo5MDE3NjU
8. https://gist.github.com/mingfeima/e08310d7e7bb9ae2a693adecf2d8a916
