//
// Example from:
// https://devblogs.nvidia.com/even-easier-introduction-cuda/
//
// Compiles as follows:
// nvcc add_v02.cu -o add_v2_cuda
//
// nvcc is set as follows:
// export PATH=/usr/local/cuda-10.0/bin:$PATH 

#include <iostream>
#include <math.h>

// CUDA *kernel* function to add the elements of two arrays
//
// The __global__ specifier tells the CUDA C++ compiler that 
// this is a function that runs on the GPU and can be called 
// from CPU code
//
// These __global__ functions are known as *kernels*, and code 
// that runs on the GPU is often called *device code*, while code 
// that runs on the CPU is *host code*
//
// Here we'll *spread* calculations over *multiple* threads
// which will reduce the execution time by quite a lot
//

__global__
void add(int n, float* x, float* y)
{
  
  // Properly spread calculations among threads (see
  // comments in main)
  //
  // CUDA C++ provides keywords that let kernels get 
  // the indices of the running threads. 
  // Specifically, threadIdx.x contains the index of 
  // the current thread within its block, and blockDim.x 
  // contains the number of threads in the block
  //
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride) 
  {
      y[i] = x[i] + y[i];
  }
}

int main(void)
{
  // left shift by 20 bits
  //
  int N = 1<<20; // (more than) 1M elements

  // Allocate memory in *CUDA*
  // it's Unified Memory that's accessible
  // from CPU and/or GPU 
  //
  float *x, *y;
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));  

  // initialize x and y arrays on the host N elements each
  //
  for (int i = 0; i < N; i++) 
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Launch the add() kernel on GPU
  // the <<< >>> syntax is CUDA kernel lanch(es)
  // Here we pick up 256 threads (from one block)
  // CUDA GPUs run kernels using blocks of threads 
  // that are a multiple of 32 in size, so 256 threads 
  // is a reasonable number to choose
  // However, some modifications to the add() kernel
  // are needed to spread the calculations properly
  // (see the add's code)
  //
  add <<< 1, 256 >>>(N, x, y); 
  
  // Last but not least !!!
  // Tell the *CPU* to wait until the kernel is done
  // before accessing results (because CUDA kernel lauches
  // don't block calling CPU thread)
  //
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  //
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
  {
    maxError = fmax(maxError, fabs(y[i]-3.0));
  }
  std::cout << "Max error: " << maxError << std::endl;

  // free memory in CUDA
  //
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
