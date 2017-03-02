#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

// Adding a bunch of constants for testing
#define BIN_COUNT (HISTO_WIDTH * HISTO_HEIGHT)

#define INPUT_SIZE (INPUT_WIDTH * INPUT_HEIGHT)

#define WARP_COUNT 6

#define THREADBLOCK_MEMORY (WARP_COUNT * BIN_COUNT)

__global__ void HistogramSharedCoalesced(uint32_t* d_input, uint8_t* d_bins);
__global__ void HistogramShared(uint32_t* d_input, uint8_t* d_bins);
__global__ void HistogramFastest(uint32_t* d_input, uint8_t* d_bins);

void opt_2dhisto(uint32_t* d_input, uint8_t* d_bins)
{
    /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
    cudaMemset(d_bins, 0, BIN_COUNT * sizeof(uint8_t));

    // Working shared histogram with coalesced access: GPU TIME = 4.224s
    //dim3 sharedCoalescedDimGrid(13,1);
    //dim3 sharedCoalescedDimBlock(257,1); 
    //HistogramSharedCoalesced<<<sharedCoalescedDimGrid, sharedCoalescedDimBlock>>>(d_input, d_bins);
        
    // Working shared histogram: GPU TIME = 8.614s
    //dim3 sharedDimGrid(15,1);
    //dim3 sharedDimBlock(297,1);
    //HistogramShared<<<sharedDimGrid, sharedDimBlock>>>(d_input, d_bins);
    
    // Working histogram values, don't edit: GPU TIME = 3.569 
    dim3 fastestDimGrid(15,1);
    dim3 fastestDimBlock(257,1);
    HistogramFastest<<<fastestDimGrid, fastestDimBlock>>>(d_input, d_bins);
}

// Custom function to atomicIncrement with uint8_t input
__device__ void atomicIncCustom(uint8_t* addr)
{
    size_t addr_offset = (size_t)addr & 0x3;
    size_t bits_shift = addr_offset * 8;
    size_t mask = 0xff << bits_shift;
    unsigned int* aligned_addr = (unsigned int *)(addr - addr_offset);

    unsigned int old = *aligned_addr;
    unsigned int stored;
    unsigned int new_value;
    do{
        stored = old;  
        new_value = (stored >> bits_shift) & 0xff;
        if(new_value < 255) {
            new_value++;
        } else {
            return;
        }
        new_value = (new_value << bits_shift) | (stored & ~mask);
        old = atomicCAS(aligned_addr, stored, new_value);
     } while (stored != old);
}

// Custom function to perform atomicAdd with uint8_t input
__device__ void atomicAddCustom(uint8_t* addr, unsigned int val)
{
    if (val == 0) return;
    //Need the 4 byte aligned adress containing this uint8_t
    size_t addr_offset = (size_t)addr & 0x3;
    size_t bits_shift = addr_offset * 8;
    size_t mask = 0xff << bits_shift;
    unsigned int* aligned_addr = (unsigned int *)(addr - addr_offset);

    unsigned int old = *aligned_addr;
    unsigned int stored;
    unsigned int new_value;

    do{
        stored = old;
        new_value = (stored >> bits_shift) & 0xff;
        if(new_value == 255) {
            return;
        } else {
            new_value += val;
            if (new_value > 255) new_value = 255;
        }
        new_value = (new_value << bits_shift) | (stored & ~mask);
        old = atomicCAS(aligned_addr, stored, new_value);
     } while (stored != old);
}

// Trying new shared solution
__global__ void HistogramSharedCoalesced(uint32_t* d_input, uint8_t* d_bins)
{
    // Shared partial histogram computation (using coalesced access)
    const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;
    __shared__ uint8_t s_Hist[THREADBLOCK_MEMORY];

    const int elsPerThread = INPUT_SIZE / numThreads;

    uint32_t idx;

#pragma unroll
    for (int i = threadIdx.x; i < BIN_COUNT; i += blockDim.x) {
        s_Hist[i] = 0;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < elsPerThread; i++) {
        idx = globalTid * elsPerThread + i;
        atomicIncCustom(&s_Hist[d_input[idx]]);
    }
    __syncthreads();

#pragma unroll
    for (int i = threadIdx.x; i < BIN_COUNT; i += blockDim.x) {
        atomicAddCustom(&d_bins[i],s_Hist[i]);
    }
}

// Best Shared Solution So Far
__global__ void HistogramShared(uint32_t* d_input, uint8_t* d_bins)
{
    // Shared partial histogram computation (based on slides)
    const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;
    __shared__ uint8_t s_Hist[THREADBLOCK_MEMORY];

    for (int i = threadIdx.x; i < BIN_COUNT; i += blockDim.x) {
        s_Hist[i] = 0;
    }
    __syncthreads();

    for (int i = globalTid; i < INPUT_SIZE; i += numThreads) {
        atomicIncCustom(&s_Hist[d_input[i]]);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < BIN_COUNT; i += blockDim.x) {
        atomicAddCustom(&d_bins[i],s_Hist[i]);
    }
}

// Best Solution So Far
__global__ void HistogramFastest(uint32_t* d_input, uint8_t* d_bins)
{
    // Computing the histogram using strided access to the global array
    int globalTid = threadIdx.x + blockIdx.x * blockDim.x;
    int numThreads = blockDim.x * gridDim.x;
    
    // start at the global id of the thread and increment by the total number of threads
    // until all of the input has been computed 
    for (int i = globalTid; i < INPUT_SIZE; i += numThreads) {
        atomicIncCustom(&d_bins[d_input[i]]);
    }
}

/* Include below the implementation of any other functions you need */
// Allocate memory on GPU for arrays and copy arrays to GPU
void opt_2dhisto_setup(uint32_t*& d_input, uint32_t** input, uint8_t*& d_bins, uint8_t* kernel_bins) {
    // allocate memory on GPU
    cudaMalloc((void**)&d_input, INPUT_WIDTH * INPUT_HEIGHT * sizeof(uint32_t));
    cudaMalloc((void**)&d_bins, HISTO_WIDTH * HISTO_HEIGHT * sizeof(uint8_t));
    
    // copy arrays from CPU to GPU
    uint32_t* d_pointer = d_input;
    for (int i=0; i<INPUT_HEIGHT; i++) {
        cudaMemcpy(d_pointer, input[i], INPUT_WIDTH * sizeof(uint32_t), cudaMemcpyHostToDevice);
        d_pointer += INPUT_WIDTH;
    }
    cudaMemcpy(d_bins, kernel_bins, HISTO_WIDTH * HISTO_HEIGHT * sizeof(uint8_t), cudaMemcpyHostToDevice);  
}

// Copy bins from GPU to CPU and free memory on GPU
void opt_2dhisto_teardown(uint32_t*& d_input, uint8_t*& d_bins, uint8_t* kernel_bins) { 
    cudaMemcpy(kernel_bins, d_bins, HISTO_WIDTH * HISTO_HEIGHT * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_bins);   
}
