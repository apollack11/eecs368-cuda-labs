#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 1024

__global__ void UpSweepKernel(float *outArray, float *inArray, float *auxArray, int numElements, int numBlocks);
__global__ void AuxArrayScan(float *auxArrayScanned, float *auxArray, int numBlocks);
__global__ void DownSweepKernel(float *outArray, float *inArray, float *auxArray, int numElements);

// Lab4: Host Helper Functions (allocate your own data structure...)


// Lab4: Device Functions


// Lab4: Kernel Functions
__global__ void UpSweepKernel(float *outArray, float *inArray, float *auxArray, int numElements, int numBlocks)
{
    // computes total thread ID so we can access every element in the array
    int globalId = threadIdx.x + blockIdx.x * blockDim.x;
    
    // allocates a shared array which is the size of the block (defined above)
    __shared__ float scanArray[BLOCK_SIZE];

    // sets the value at index threadIdx.x to the value of the input array at the global ID
    // this should "chunk" the global array
    scanArray[threadIdx.x] = inArray[globalId];
    
    // syncs all threads to make sure the shared array has all values set
    __syncthreads();

    int stride = 1;
    while (stride < BLOCK_SIZE)
    {
        int index = (threadIdx.x + 1) * stride * 2 - 1;
        if (index < BLOCK_SIZE)
            scanArray[index] += scanArray[index-stride];
        stride = stride*2;
        __syncthreads();
    }
    
    outArray[globalId] = scanArray[threadIdx.x];

    if (threadIdx.x == BLOCK_SIZE - 1 && blockIdx.x < numBlocks - 1)
        auxArray[blockIdx.x + 1] = scanArray[threadIdx.x]; 
}

__global__ void AuxArrayScan( float* auxArrayScanned, float* auxArray, int numBlocks)
{
    for (int i=1; i < numBlocks; ++i)
    {
        auxArrayScanned[i] = auxArray[i] + auxArrayScanned[i-1];
    }
} 

__global__ void DownSweepKernel(float *outArray, float *inArray, float *auxArrayScanned, int numElements)
{
    int globalId = threadIdx.x + blockIdx.x * blockDim.x;

    // allocates a shared array which is the size of the block (defined above)
    __shared__ float scanArray[BLOCK_SIZE];

    // sets the value at index threadIdx.x to the value of the output array at the global ID
    // this should "chunk" the global array
    scanArray[threadIdx.x] = outArray[globalId];
    
    // syncs all threads to make sure the shared array has all values set
    __syncthreads();
   
    int stride = BLOCK_SIZE >> 1;
    while(stride > 0)
    {
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index < BLOCK_SIZE) {
            scanArray[index+stride] += scanArray[index];
        }
        stride = stride >> 1;
        __syncthreads();
    }
    
    if (threadIdx.x < BLOCK_SIZE - 1)
        outArray[globalId+1] = scanArray[threadIdx.x];
    __syncthreads();
    
    if (threadIdx.x == 0) {
        outArray[globalId] = 0;
    }
   
    outArray[globalId] += auxArrayScanned[blockIdx.x];
    

}

// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, float *auxArray, float *auxArrayScanned, int numElements, int numBlocks)
{ 
    dim3 gridSize(numBlocks,1);
    UpSweepKernel<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(outArray, inArray, auxArray, numElements, numBlocks);
    AuxArrayScan<<<1, 1>>>(auxArrayScanned, auxArray, numBlocks);
    DownSweepKernel<<<gridSize, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(outArray, inArray, auxArrayScanned, numElements);
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
