# CUDA Programming Learning Notes

## Overview

These notes document key concepts and learnings from CUDA GPU programming exercises.

---

## 1. Basic GPU Kernel Execution

**Key Concept**: Running functions on GPU instead of CPU

### Essential Components:

- **`__global__` keyword**: Marks a function that runs on GPU but is called from CPU
- **Kernel launch syntax**: `kernel_name<<<blocks, threads>>>(arguments);`
  - `blocks`: Number of thread blocks to execute
  - `threads`: Number of threads per block

### Example Structure:

```cuda
__global__ void hello_from_gpu() {
    // GPU code runs here
}

int main() {
    hello_from_gpu<<<4, 8>>>();  // 4 blocks, 8 threads per block
    cudaDeviceSynchronize();      // Wait for GPU to finish
}
```

### Important Built-in Variables:

- `threadIdx.x`: Thread index within the current block (0 to blockDim.x-1)
- `blockIdx.x`: Block index in grid (0 to gridDim.x-1)
- `blockDim.x`: Number of threads per block
- **Global ID calculation**: `blockIdx.x * blockDim.x + threadIdx.x`

---

## 2. Host-Device Memory Management

### Three Memory Types:

1. **Host Memory (CPU)**: Regular RAM accessed by CPU
2. **Device Memory (GPU)**: GPU's memory, much faster but limited
3. **Shared Memory**: Fast memory shared by threads in same block

### Memory Transfer Operations:

```cuda
// Allocate GPU memory
cudaMalloc(&d_data, size);

// Copy CPU → GPU (Host to Device)
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// Copy GPU → CPU (Device to Host)
cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

// Free GPU memory
cudaFree(d_data);
```

### Key Points:

- GPU cannot directly access CPU memory
- CPU cannot directly access GPU memory
- Data must be explicitly copied between them
- This is a bottleneck - minimize data transfers

---

## 3. Global Memory

**Definition**: Large GPU memory accessible by all threads (slower than shared memory)

### Characteristics:

- **Size**: Large (GBs typically) but slow ~50-100 cycles latency
- **Scope**: All threads and blocks can access
- **Persistence**: Data stays throughout kernel execution
- **Use case**: Storing large data structures

### Example:

```cuda
__global__ void addOne(int *data) {
    int i = threadIdx.x;
    data[i] = data[i] + 1;  // Direct global memory access
}
```

### Performance Issues:

- Memory coalescing problems if threads access non-consecutive addresses
- Bank conflicts reduce performance
- Better for sequential/grouped access patterns

---

## 4. Shared Memory

**Definition**: Fast memory shared among threads in the same block

### Characteristics:

- **Size**: Small (~48KB-96KB per block) but FAST (~1-4 cycles latency)
- **Scope**: Only threads within same block
- **Persistence**: Destroyed after kernel finishes
- **Allocation**: Declared with `__shared__` keyword

### Declaration:

```cuda
__shared__ int temp[8];  // 8-element array shared by all threads in block
```

### Synchronization Requirement:

```cuda
__syncthreads();  // Barrier - all threads must reach here before continuing
```

### Advantages:

- Much faster than global memory (10-100x speedup possible)
- Avoids redundant global memory accesses
- Good for collaborative data processing

### Typical Pattern:

1. Threads read from global memory into shared memory
2. Sync all threads
3. Process using fast shared memory
4. Sync again
5. Write results back to global memory

---

## 5. Synchronization

### `__syncthreads()`

- **Purpose**: Barrier synchronization within a block
- **Effect**: All threads wait until every thread reaches this point
- **Scope**: Only within same block (different blocks cannot sync directly)

### Usage:

```cuda
temp[i] = data[i];           // Each thread copies data
__syncthreads();             // Wait for all copies complete
temp[i] = temp[i] + 1;       // Now safe to use data from other threads
__syncthreads();             // Wait again before next step
```

### Critical Rule:

- Must use `__syncthreads()` before reading data written by other threads
- All threads must converge at sync point (no divergent control flow)

---

## 6. Common Pitfalls & Debugging

### Typo Errors:

- ❌ `blcokidx` → ✅ `blockIdx` (wrong identifier)
- ❌ `_global_` → ✅ `__global__` (need double underscores)
- ❌ `_syncthreads` → ✅ `__syncthreads__`

### Syntax Errors:

- Missing semicolons in CUDA code
- Improper function nesting (can't define functions inside functions)
- Comment syntax: Use `//` for C++ comments in CUDA, not `#`

### Missing Setup:

- Must `!pip install nvcc4jupyter` before using `%%cuda` magic
- Must `%load_ext nvcc4jupyter` to load the magic in notebook
- Must call `cudaDeviceSynchronize()` to wait for GPU completion

---

## 7. Performance Optimization Hierarchy

**From Slowest to Fastest:**

1. **Global Memory** (~50-100 cycles, large)
2. **Shared Memory** (~1-4 cycles, small)
3. **Registers** (~1 cycle, per-thread only)

**Strategy**: Move frequently accessed data to faster memory types

---

## 8. Practical Workflow

```cuda
// 1. Initialize data on CPU
int h_data[SIZE] = {...};
int *d_data;

// 2. Allocate GPU memory
cudaMalloc(&d_data, SIZE * sizeof(int));

// 3. Transfer data to GPU
cudaMemcpy(d_data, h_data, SIZE * sizeof(int), cudaMemcpyHostToDevice);

// 4. Launch kernel(s)
myKernel<<<numBlocks, threadsPerBlock>>>(d_data);

// 5. Wait for GPU
cudaDeviceSynchronize();

// 6. Get results back
cudaMemcpy(h_data, d_data, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

// 7. Process CPU results
for(int i = 0; i < SIZE; i++) printf("%d ", h_data[i]);

// 8. Cleanup
cudaFree(d_data);
```

---

## 9. Key Takeaways

✅ **Do's:**

- Use shared memory for fast collaborative computation
- Coalesce memory accesses for efficiency
- Synchronize threads when necessary
- Minimize host-device data transfers
- Use proper double underscores in CUDA keywords

❌ **Don'ts:**

- Don't access GPU memory from host (and vice versa) without transfer
- Don't forget `cudaDeviceSynchronize()` before reading results
- Don't trust data from other threads without `__syncthreads()`
- Don't use Python comments (`#`) in CUDA code blocks

---

## 10. Concepts for Future Learning

- **Global synchronization**: Inter-block communication techniques
- **Memory coalescing**: Optimizing access patterns
- **Bank conflicts**: Shared memory access optimization
- **Thread divergence**: Branch prediction and warp execution
- **Occupancy**: Maximizing active warps per SM
- **Atomic operations**: Thread-safe global memory updates
- **Streams**: Asynchronous execution and pipelining

---

_Last Updated: March 11, 2026_
_Based on: HPC-CUDA notebook exercises_
