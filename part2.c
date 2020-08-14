// Shared Variables (alphabetical order): a, b, c, chunk_size, N
// Private Variables (alphabetical order): i, nthreads, omp_id, tmp

#include <stdio.h>
#include <omp.h>
#define CHUNKSIZE 8
#define N 88888

void print_string(char str[]) 
{ 
    printf("(in alphabetical order) %s \n", str);
} 


int main(void)
{
  int i, chunk_size, nthreads, omp_id, tmp;
  float a[N], b[N], c[N];
  chunk_size = CHUNKSIZE;
  omp_set_num_threads(11);
  for (i = 0; i < N; i++) /* Initialize arrays a and b */
  {
    a[i] = i * 6.8;
    b[i] = i * 7.9;
  }
  char shared_variables[] = "Shared Variables: a, b, c, chunk_size, N ";
  char private_variables[] = "Private Variables: i, nthreads, omp_id, tmp ";
  print_string(shared_variables);
  print_string(private_variables);
  #pragma omp parallel shared(a, b, c, chunk_size) private(i, nthreads, omp_id, tmp)  /* Compute values of array c = a + b in parallel. */
  {
    omp_id = omp_get_thread_num();
    if (omp_id == 0)
    {
        nthreads = omp_get_num_threads();
    }

  // parallelize the for loop using dynamic schedule with no wait for threads to finish
  #pragma omp for schedule(dynamic, chunk_size), nowait
  for (i = 0; i < N; i++)
    {
        tmp = 2.0 * a[i];
        a[i] = tmp;
        c[i] = a[i] + b[i];
    }
  }
  //printf ("%f\n", c[N]);
  //printf ("%f\n", c[N-1]);
  //printf ("%f\n", c[N-2]);
  // printing elements of an array
  for(int i = 0; i < N; ++i)
  {
     printf("%f\n", c[i]);
  }
}
