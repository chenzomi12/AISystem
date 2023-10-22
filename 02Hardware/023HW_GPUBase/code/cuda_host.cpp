#include <iostream>
#include <math.h>
#include <sys/time.h>

// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}

int main(void)
{
  int N = 1<<25; // 30M elements

  float *x = new float[N];
  float *y = new float[N];

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  struct timeval t1,t2;
  double timeuse;
  gettimeofday(&t1,NULL);
  
  // Run kernel on 30M elements on the CPU
  add(N, x, y);

  // Free memory
  delete [] x;
  delete [] y;

  return 0;
}