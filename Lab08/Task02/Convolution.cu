/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include "common/book.h"
#include "common/cpu_bitmap.h"

const int WIDTH = 512;
const int HEIGHT = 512;

__global__ void ConvolutionKernel(unsigned char* d_Pin,
                             unsigned char* d_Pout,
                             int width, int height) { 
   //Calculate the row # 
   int y = blockIdx.y*blockDim.y + threadIdx.y; 

   //Calculate the column # 
   int x = blockIdx.x*blockDim.x + threadIdx.x; 
   
   int index = x + (height-1-y)*width;

   float kernel[3][3]={{-1,-1,-1},
		       {-1,9,-1},
		       {-1,-1,-1}};

   float sum = 0;
   
   for(int k=-1;k<=1;++k) {
      for(int m=-1;m<=1;++m) {
         
         int f = x + m;
         int g = y + k;
         if(f>0 && f < width && g>0 && g < height) { 
            int index1 = f+g*width;
            sum = sum + kernel[k+1][m+1]*d_Pin[index1];
         }
      }
   }

   d_Pout[index] = sum;
} 

// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};

void loadBinImage(const char* imageName, unsigned char* pData) {

   FILE* fp = fopen(imageName, "rb");
   fread(pData, 1, WIDTH*HEIGHT, fp);
   fclose(fp);
}

int main( void ) {
    DataBlock   data;
    CPUBitmap bitmap(WIDTH, HEIGHT, &data );
    unsigned char    *host_bitmap;
    unsigned char    *dev_bitmap;
    unsigned char    *dev_bitmap2;

    host_bitmap = (unsigned char*)malloc(bitmap.image_size() );
    memset(host_bitmap,0, bitmap.image_size());
   
    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap2, bitmap.image_size() ) );

    HANDLE_ERROR( cudaMemcpy( dev_bitmap2, host_bitmap, 
                              bitmap.image_size(),
                              cudaMemcpyHostToDevice ) ); 

    loadBinImage("Baboon.raw", host_bitmap);
    HANDLE_ERROR( cudaMemcpy( dev_bitmap, host_bitmap, 
                              bitmap.image_size(),
                              cudaMemcpyHostToDevice ) ); 
   
    data.dev_bitmap = dev_bitmap;

    dim3    blocksGrid;
    dim3    threadsBlock(16,16,1);
    blocksGrid.x = ceil(WIDTH/16.0);
    blocksGrid.y = ceil(HEIGHT/16.0);
 
    ConvolutionKernel<<<blocksGrid, threadsBlock>>>( dev_bitmap, dev_bitmap2, WIDTH, HEIGHT);

    HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap2,
                              bitmap.image_size(),
                              cudaMemcpyDeviceToHost ) );
                              
    HANDLE_ERROR( cudaFree( dev_bitmap ) );
    HANDLE_ERROR( cudaFree( dev_bitmap2 ) );
    free(host_bitmap);                              
    bitmap.display_and_exit();
}

