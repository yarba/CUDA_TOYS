//
// Example inspired by:
// https://devblogs.nvidia.com/even-easier-introduction-cuda/
//

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
// Here we'll *spread* calculations over *multiple threads*
// AND *multiple block* which will reduce the execution time 
// even further
//

__global__ void add(int n, float* x, float* y)
{
  
  // Properly spread calculations among threads AND blocks
  //
  // CUDA GPUs have many parallel processors grouped into 
  // Streaming Multiprocessors (SM). Each SM can run multiple 
  // concurrent thread blocks
  // E.g. a Tesla P100 GPU based on the Pascal GPU Architecture 
  // has 56 SMs, each can support up to 2048 active threads
  // (so 256 threads is a relatively small number of threads
  // for such kind of architecture) 
  // Together, the blocks of parallel threads make up what is 
  // known as the *grid*. 
  // To take full advantage of all these threads, we should 
  // launch the kernel with multiple thread blocks
  //
  // CUDA C++ provides keywords that let kernels get 
  // the indices of the running threads. 
  // Specifically, threadIdx.x contains the index of 
  // the current thread within its block, and blockDim.x 
  // contains the number of threads in the block
  // CUDA also provides gridDim.x, which contains the number 
  // of blocks in the grid, and blockIdx.x, which contains 
  // the index of the current thread block in the grid
  //
  // In CUDA kernelthe type of loop shown below is often 
  // *grid-stride loop*
  //
  int index = blockIdx.x * blockDim.x + threadIdx.x;
//e.g.index =   (2)      *   (256)    +    (3)       = 515 
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) 
  {
      // As part of the exercise, try prinf()  
/*
      if ( i < 1000 )
      {
         printf( "Thread id=%d, block id=%d \n", threadIdx.x, blockIdx.x );
      }
*/
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
  // the <<< >>> syntax is CUDA kernel launch(es)
  // Here we pick up multiple blocks of threads,
  // with 256 threads per block
  // CUDA GPUs run kernels using blocks of threads 
  // that are a multiple of 32 in size
  // Modern architrecture can support more than 2000 threads
  // per block, so 256 threads is a reasonable number to choose
  // We have to calculate (estimate) number of blocks needed
  // to process N elements in parallel (i.e. how many blocks
  // we need to get at least N threads)
  // NOTE: since N is not necessarily a multiple of 256, 
  // we may need to round up the number of blocks
  // See more comments in the add's kernel code
  //
  int block_size = 256;
  int num_blocks = ( N + block_size -1 ) / block_size;
  add <<< num_blocks, block_size >>>( N, x, y ); 
  
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
