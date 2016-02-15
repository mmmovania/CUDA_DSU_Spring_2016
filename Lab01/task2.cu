#include<iostream>

__global__ void myKernel()
{
     //should work on a new GPU
	//std::cout << "Hello world from GPU" << std::endl;

     printf("Hello world from GPU");
}

int main()
{
	std::cout << "Hello world!\n";
	myKernel<<< 1,32 >>>();
	cudaDeviceSynchronize();
	std::cout << "Hello world after kernel launch!\n");
	return 0;

}