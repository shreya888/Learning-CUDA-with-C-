import torch

# Create a 1D tensor (vector) with 3 elements
a = torch.tensor([1., 2., 3.])

# Print the absolute values of the elements in tensor using different methods
print(torch.abs(a))
print(torch.sqrt(torch.square(a)))
print(torch.sqrt(a ** 2))
print(torch.sqrt(a * a))


def time_pytorch_function(func, input):
    """
    Measures the execution time of function 'func' on the input 'input'.
    The timing is performed using CUDA events, which are necessary because CUDA operations are asynchronous.
    """
    # CUDA events for timing, used to measure GPU execution time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warm-up loop to make sure the GPU is ready for better benchmarking (preventing timing the cold-start overhead)
    for _ in range(5):
        func(input)

    # Record the start event, then execute the function on input, followed by recoding end even after execution
    start.record()
    func(input)
    end.record()

    # Synchronize the CUDA stream to ensure all operations are finished
    torch.cuda.synchronize()

    # Return the elapsed time between the start and end events in ms
    return start.elapsed_time(end)


# Create a large random tensor on the GPU for benchmarking
b = torch.randn(10000, 10000).cuda()


# Alternative custom implementations to calculate the absolute value using different methods:
def abs_1(a):
    return torch.sqrt(torch.square(a))


def abs_2(a):
    return torch.sqrt(a * a)


def abs_3(a):
    return torch.sqrt(a ** 2)


# Time the built-in torch.abs function and the custom implementations
time_pytorch_function(torch.abs, b)
time_pytorch_function(abs_1, b)
time_pytorch_function(abs_2, b)
time_pytorch_function(abs_3, b)

# Use PyTorch's profiler to measure the performance of different implementation with same input
# The profiler captures CUDA kernel execution times and other relevant metrics
# Followed by printing the profiling results, sorted by total CUDA time
print("--------------------------------------------------")
print("Profiling torch.abs(vector))")
print("--------------------------------------------------")
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    torch.abs(b)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("--------------------------------------------------")
print("Profiling torch.sqrt(torch.square(vector))")
print("--------------------------------------------------")
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    abs_1(b)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("--------------------------------------------------")
print("Profiling torch.sqrt(vector * vector)")
print("--------------------------------------------------")
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    abs_2(b)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("--------------------------------------------------")
print("Profiling torch.sqrt(vector ** 2)")
print("--------------------------------------------------")
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    abs_3(b)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
