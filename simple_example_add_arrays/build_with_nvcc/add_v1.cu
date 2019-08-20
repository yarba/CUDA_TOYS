//
// Example from:
// https://devblogs.nvidia.com/even-easier-introduction-cuda/
//
// Compiles as follows:
// nvcc add_v01.cu -o add_v1_cuda
//
// nvcc is set as follows:
// export PATH=/usr/local/cuda-10.0/bin:$PATH 

#include <iostream>
#include <math.h>

// NOTE (JVY): This kernel is only correct for a single thread, 
//             since every thread that runs it will perform the add 
//             on the whole array 
//             Moreover, there is a race condition since multiple 
//             parallel threads would both read and write the same 
//             locations.

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
__global__
void add(int n, float* x, float* y)
{
  for (int i = 0; i < n; i++) // i++)
  {
      y[i] = x[i] + y[i];
  }
}

int main(void)
{
  // left shift by 20 bits
  //
  int N = 1<<20; // (more than) 1M elements

  // memory allocation in C++/CPU) but we'll replace it
  // with allocation in CUDA - see below !
  //
  // float* x = new float[N];
  // float* y = new float[N];
  //
  // alocate memory in CUDA
  // it's ** Unified Memory ** accessible from CPU and/or GPU
  //
  // NOTE: (as seen in many other examples) one can use 
  //       cudaMalloc but that will allocate memory on the ** GPU ** 
  //       e.g.
  //       int* dev_x, dev_y;
  //       cudaMalloc( &dev_x, N*sizeof(int) );
  //       cudaMalloc( &dev_y, N*sizeof(int) );
  //       but then one should also copy arrays x and y to the GPU:
  //       cudaMemcpy( dev_x, x, N*sizeof(int), cudaMemoryHostToDevice );
  //       cudaMemcpy( dev_x, x, N*sizeof(int), cudaMemoryHostToDevice );
  //       and at the end (after running add(...)) one should copy 
  //       the new content of y back to CPU:
  //       cudaMemcpy( x, dev_x, N*sizeof(int), cudaMemoryDeviceToHost );
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

  // Run kernel on 1M elements on the CPU
  //
  // Q (JVY): why do they call it "kernel" ???
  // A : OK, they call it "kernel" because that's
  //     what it is in CUDA (__global__)
  //
  // This is how to "launch" the add(...) function (kernel) on CPU
  //
  // add(N, x, y);
  //
  // Launch the add(...) kernel on GPU
  // the <<< >>> syntax is CUDA kernel launch(es)
  // (apparently) this will launch on *one* thread
  // with 1 "block per grid" (don't know yet what it means)
  //
  add <<< 1, 1 >>>(N, x, y); 
  
  // Last but not least !!!
  // Tell the *CPU* to wait until the kernel is done
  // before accessing results (because CUDA kernel lauch
  // doesn't block calling CPU thread)
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

  // free memory (C++/CPU)
  //
  // delete [] x;
  // delete [] y;
  //
  // free memory in CUDA
  //
  cudaFree(x);
  cudaFree(y);
  
  return 0;
}
