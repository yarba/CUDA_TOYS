//
// Example from:
// https://devblogs.nvidia.com/even-easier-introduction-cuda/
//
// Compiles as follows:
// g++ add_v0.cc -o add_v0

#include <iostream>
#include <math.h>

// function to add the elements of two arrays
//
void add(int n, float* x, float* y)
{
  for (int i = 0; i < n; i++)
  {
      y[i] = x[i] + y[i];
  }
}

int main(void)
{
  // left shift by 20 bits
  //
  int N = 1<<20; // (more than) 1M elements

  float* x = new float[N];
  float* y = new float[N];

  // initialize x and y arrays on the host N elements each
  //
  for (int i = 0; i < N; i++) 
  {
// --> original    x[i] = 1.0f;
// --> original    y[i] = 2.0f;
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  //
  // Q (JVY): why do they call it "kernel" ???
  //
  add(N, x, y);

  // Check for errors (all values should be 3.0f)
  //
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
  {
// --> original    maxError = fmax(maxError, fabs(y[i]-3.0f));
    maxError = fmax(maxError, fabs(y[i]-3.0));
  }
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  //
  delete [] x;
  delete [] y;

  return 0;
}
