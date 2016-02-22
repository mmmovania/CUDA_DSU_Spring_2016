#include <stdio.h>

__global__ void Add(int* a, int* b, int* c) {
   *c = *a + *b;
}

int main()
{
   //host memory
   int h_a = 10;
   int h_b = 20;
   int h_c =  0;

   //device memory pointers
   int* d_a = 0;
   int* d_b = 0;
   int* d_c = 0;

   //allocate device memory
   cudaMalloc(&d_a, sizeof(int));
   cudaMalloc(&d_b, sizeof(int));
   cudaMalloc(&d_c, sizeof(int));

   //copy data from host to device memory
   cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice);

   //launch kernel
   Add<<<1,1>>>(d_a, d_b, d_c);
   cudaDeviceSynchronize();

   //copy data from device to host memory
   cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

   //output result
   printf("Result: %3d \n", h_c);

   //release device memory
   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_c);

   cudaDeviceReset();

   return 0;
}