#include<iostream>

__global__ void myKernel()
{
     //should work on a new GPU
     //std::cout << "Hello world from GPU " << threadIdx.x << "," << blockIdx.x << std::endl;
    
     //we use printf so it should work
     printf("Hello world from GPU %d %d \n", threadIdx.x,blockIdx.x );
}

int main()
{
	std::cout << "Hello world!\n";
	myKernel<<< 1,32 >>>();
	cudaDeviceSynchronize();
	std::cout << "Hello world after kernel launch!\n";
	return 0;

}