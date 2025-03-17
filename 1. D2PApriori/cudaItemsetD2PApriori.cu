#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <unordered_map>
#include <iostream>
#include <string.h>
#include <time.h>
#include "../include/commonmethods.h"

#define MAX_NODES 6000 // Maximum nodes in the FP-Tree
#define EMPTY -1

typedef struct
{
    int id;
    int processed; // 1 signifies it was processed
    int itemSet;
    int count;
    int parent;
    int nextSibling;
    int firstChild;
} Node;



// Calculates the distance between two instances
__device__ float generateItemSet(float *instance_A, float *instance_B, int num_attributes)
{
    float sum = 0;

    for (int i = 0; i < num_attributes - 1; i++)
    {
        float diff = instance_A[i] - instance_B[i];
        // printf("instance a and b were %.3f %.3f\n", instance_A[i] ,instance_B[i]);
        sum += diff * diff;
    }
    // printf("sum was %.3f\n,", sum);
    return sqrt(sum);
}

__global__ void processItemSets(char *inData, int minimumSetNum, int *d_Offsets, int totalRecords, int blocksPerGrid)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory is treated as a single contiguous block
    extern __shared__ int sharedMemory[];

    char *line = inData + d_Offsets[tid];
    bool inNumber = false;
    int itemCount = 0;
    int number = 0;
    int items[32];

    // Initialize the shared memory (done by thread 0 in each block)
    if (tid <= 10000)
    {
        printf("we are in tid %d\n", tid);
        // Extract items from the input line
        for (char *current = line; *current != '\n' && *current != '\0'; current++)
        {
            if (*current >= '0' && *current <= '9')
            {
                number = number * 10 + (*current - '0');
                inNumber = true;
            }
            else if (inNumber)
            {

                items[itemCount] = number;
                itemCount++;
                number = 0;
                inNumber = false;
            }
        }

        if (inNumber)
        {
            items[itemCount++] = number;
        }
        for (int i = 0; i < itemCount; i++)
        {
            printf("%d", items[i]);
        }
    }
    __syncthreads();

    // Parse the input and build the FP-Tree
    if (tid < totalRecords)
    {
    }
}



// Compute support by intersecting the bitsets of all items in 'set'
int computeSupport(ItemBitmap* set, int rowSize) {
    // // Start with the bitset of the first item
    // static uint32_t temp[rowSize];
    // memcpy(temp, bitsets[set->items[0]], rowSize * sizeof(uint32_t));

    // // Intersect with the rest
    // for (int i = 1; i < set->length; i++) {
    //     int item = set->items[i];
    //     for(int block = 0; block < rowSize; block++){
    //         temp[block] &= bitsets[item][block];
    //     }
    // }

    // // Count total bits set
    // int support = 0;
    // for(int block = 0; block < rowSize; block++){
    //     support += popcount32(temp[block]);
    // }

    // return support;
}

/*
removeLowFrequencyItems is used to remove the items in the bitmap that have low frequency
it will return a new bitmap database containing only items which were determined to have support
greater than the minimum threshold
*/
int *removeLowFrequencyItemsCPU(int *inBitMap, int rowLength, int minItemCount)
{
}

int *generateCandidates(int *inBitMap, int rowLength, int minItemCount){
    int k = 0;

    if(sizeof(inBitMap) == 0){
        
    }
}

// Implements a threaded kNN where for each candidate query an in-place priority queue is maintained to identify the nearest neighbors
int KNN()
{
    printf("we started\n");
    clock_t cpu_start_withSetup = clock();
    int *itemsBitmap = (int *)calloc(3125000, sizeof(int));
    clock_t setupTimeStart = clock();
    // int lineCountInDataset = 1692081;
    // int lineCountInDataset = 55012;

    int lineCountInDataset = 100000;
    const char *inDataFilePath = "../T10I4D100K.txt";

    FILE *file = fopen(inDataFilePath, "r");

    // Get the file size
    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    rewind(file);

    char *h_buffer = (char *)malloc(file_size);
    fread(h_buffer, 1, file_size, file);

    // Count the number of lines and create offsets
    int *h_offsets = (int *)malloc((file_size + 1) * sizeof(int));
    int lineCount = 0;
    h_offsets[lineCount++] = 0; // First line starts at the beginning

    for (size_t i = 0; i < file_size; i++)
    {
        // printf("are we in size?");
        if (h_buffer[i] == '\n')
        {
            // printf("we are in the newline stuff");
            h_offsets[lineCount++] = i + 1; // Next line starts after '\n'
        }
    }

    // Allocate memory to hold the file contents
    bool inNumber = false;

    char *h_text = (char *)malloc(file_size);
    int itemCount = 0;
    int number = 0;

    // Read the file into the host buffer
    fread(h_text, 1, file_size, file);
    // fclose(file);
    // size_t sharedMemSize = (6 * MAX_NODES) * sizeof(int) +  1 * sizeof(int) ;  // 5 arrays + nodeCounter

    /* So we are reading in every line
    We must find a way to generate the vertical database
    It is known the total number of possible items is 181682 (i counted this)
    */

    /* for the t100 database let's do that calculation
    100,000 transactions / 32 bits and we get 3125 numbers requiredto store that

    we want to keep our arrays 1-D (best memory eff) so let's say

    1000 items (known) * 3125 = 325000 total required array length

    */

    // keeping track of our location of where we are in the bitmap
    // we know we need 1718 ints because we have 55012 transactions, so we do 55012 / 32 bits and get 1717.12 but we need 1718 to cover the remainder

    int rowSize = 3125; // already known
    int countInBitmap = 0;
    // ItemBitmap firstBitmap[181682];

    // for(int i = 0; i < 181682; i++){
    //     firstBitmap[i].bitmap = (int*)calloc(1718, sizeof(int));
    // }

    //int* itemBitmap2 = generateBitmapCPU(h_buffer, h_offsets);
    int countOfItems = 0;
    // int items[55012];
    // printf("before for loop\n");
    for (int i = 0; i < 100000; i++)
    {
        // printf("line %d\n", i);
        int locationOfTransaction = i / 32;
        // int locationOfTransaction = (rowSize) - (i/32);
        // code for where to flip the bit
        //   = (i / 32) + ( i % 32)
        number = 0;
        // printf("we just entered the for loop\n");
        for (char *current = h_buffer + h_offsets[i]; *current != '\n' && *current != '\0'; current++)
        {
            // code for the bit flip
            //  |= (1 << (i - 1))

            if (*current >= '0' && *current <= '9')
            {
                // printf("test");
                number = number * 10 + (*current - '0');
                inNumber = true;
            }
            else if (inNumber)
            {
                int locationOfInsertion = locationOfTransaction + (number * rowSize);
                if (number == 999)
                {
                    int locationOfLine = i + 1;
                    printf("We found item 999 at line %d\n", locationOfLine);
                    printf("the location of insertion will be %d\b\n", locationOfInsertion);
                }

                // printf("Are we gonna segfault? + locaiton of insertion %d and number is %d\n", locationOfInsertion, number);
                itemsBitmap[locationOfInsertion] |= (1 << (i % 32));
                // firstBitmap[countInBitmap].id =  number;
                // firstBitmap[countInBitmap].bitmap[location] |= (1 << (i % 32));
                countInBitmap++;
                // printf("%d\n",number);
                countOfItems++;
                // items[itemCount] = number;
                itemCount++;
                number = 0;
                inNumber = false;
            }

            // printf("\n");
        }

        // printf("not segfaulted\n");
        if (number == 999)
        {
            printf("We found item 999 at %d\n", i);
        }
        // firstBitmap[countInBitmap].id =  number;
        // firstBitmap[countInBitmap].bitmap[location] |= (1 << (i % 32));
        int locationOfInsertion = locationOfTransaction + (number * rowSize);
        // printf("Are we gonna segfault OUT OF INNER? + locaiton of insertion %d\n", locationOfInsertion);
        itemsBitmap[locationOfInsertion] |= (1 << (i % 32));
        countInBitmap++;
        // printf("%d\n", number);
        countOfItems++;
    }

    printf("we escaped the for loop");
    int *itemAndCounts = (int *)calloc(1000, sizeof(int));
    for (int i = 0; i < 1000; i++)
    {
        int sumOfInts = 0;
        for (int j = 0; j < rowSize; j++)
        {

            sumOfInts += __builtin_popcount(itemsBitmap[i * 3125 + j]);
        }
        itemAndCounts[i] = sumOfInts;

        if (i == 999)
        {
            printf("\n Item %d: ", i);
            for (int j = 0; j < 3125; j++)
            {

                int temp = itemsBitmap[i * 3125 + j];

                int position = 100000 - (j * 32);
                int locationtracker = i * 3125 + j;
                if (itemsBitmap[i * 3125 + j] != 0)
                {
                    int temp = itemsBitmap[i * 3125 + j] & -itemsBitmap[i * 3125 + j];
                    int index = __builtin_ctz(temp); // Get index of LSB (0-based)
                    printf("Bit at index: %d\n", index);
                    printf("%d, tid %d , the location (which should match insertion where this is should be %d |||| ", itemsBitmap[locationtracker], position + index + 1, locationtracker);
                }
            }
        }
    }

    
    int countOfFreqItem = 0;
    for (int i = 0; i < 1000; i++)
    {
        if (itemAndCounts[i] >= 3)
        {
            countOfFreqItem++;
            printf("Items %d had a frequency >3 of %d\n", i, itemAndCounts[i]);
        }
    }

    printf("total number of frequent items is %d\n", countOfFreqItem);
    int* items = (int *)calloc(countOfFreqItem * 3125, sizeof(int));
    
    
    for (int i = 0; i < 1000; i++)
    {
        if (itemAndCounts[i] >= 3)
        {
            
            countOfFreqItem++;
            //printf("Items %d had a frequency >3 of %d\n", i, itemAndCounts[i]);
        }
    }

    int indexInArray = 0;
    ItemBitmap* cpuFreqBitmap = (ItemBitmap*)calloc(countOfFreqItem, sizeof(ItemBitmap));
    for(int i = 0; i < countOfFreqItem; i++){
        cpuFreqBitmap[i].bitmap = (int*)malloc(rowSize * sizeof(int));
    }

    for (int i = 0; i < 1000; i++)
    {
        if (itemAndCounts[i] >= 3)
        {   
            cpuFreqBitmap[indexInArray].item  = i;
            //cpuFreqBitmap[indexInArray] -> item = i;

            memcpy(cpuFreqBitmap[indexInArray].bitmap, &itemsBitmap[i * rowSize], rowSize * sizeof(int));
            indexInArray++;
            
            //printf("Items %d had a frequency >3 of %d\n", i, itemAndCounts[i]);
        }
    }

    for(int i = 0; i < indexInArray; i++){
        printf("Our frequent item is %d ", cpuFreqBitmap[i].item);
    }

    int k = 1;
    while(countOfFreqItem > 0){
        countOfFreqItem = 0;


    }

    //printf("total number of items is %d\n", countOfItems);
    // Allocate memory on the GPU
    char *d_text;
    int *d_offsets;
    cudaMalloc(&d_text, file_size);
    cudaMalloc(&d_offsets, lineCountInDataset * sizeof(int));

    // Copy the file contents to the GPU
    cudaMemcpy(d_text, h_buffer, file_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, lineCountInDataset * sizeof(int), cudaMemcpyHostToDevice);
    int threadsPerBlock = 32;
    int blocksPerGrid = ((lineCountInDataset + threadsPerBlock) - 1) / threadsPerBlock; // how do we know how many blocks we need to use?
    // printf("BlocksPerGrid = %d\n", blocksPerGrid);
    printf("number of threads is roughly %d\n", threadsPerBlock * blocksPerGrid);
    int countBitTest = 6;
    printf("result of buildin_popcount = %d\n", __builtin_popcount(countBitTest));

    int minItemCount = 3; // setting the minimum # of items to be considered an itemset

    // here I would want to generate all itemsets

    clock_t setupTimeEnd = clock();

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float cudaElapsedTime;

    cudaEventRecord(startEvent);
    // processItemSets<<<blocksPerGrid, threadsPerBlock>>>(d_text, minItemCount, d_offsets, lineCountInDataset, blocksPerGrid);
    cudaDeviceSynchronize();
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    // Print the elapsed time (milliseconds)
    cudaEventElapsedTime(&cudaElapsedTime, startEvent, stopEvent);
    printf("CUDA Kernel Execution Time: %.3f ms\n", cudaElapsedTime);

    // ensure there are no kernel errors
    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
    {
        fprintf(stderr, "processItemSets cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
        exit(EXIT_FAILURE);
    }

    clock_t retrieveGPUResultsStart = clock();
    clock_t retrieveGPUResultsEnd = clock();

    // global reduction will be written to file
    FILE *resultsFile = fopen("cudaItemSetMiningResults.txt", "w");
    if (resultsFile == NULL)
    {
        perror("Error opening results file");
        return 1;
    }

    // Record end time
    clock_t cpu_end_withSetup = clock();
    // Calculate elapsed time in milliseconds
    // float cpuElapsedTime = ((float)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0;
    // float cpuElapsedTimeSetup = ((float)(cpu_end_withSetup - cpu_start_withSetup)) / CLOCKS_PER_SEC * 1000.0;
    // float setupTime = ((float)(setupTimeEnd - setupTimeStart)) / CLOCKS_PER_SEC * 1000.0;
    // float gpuRetrievalTime = ((float)(retrieveGPUResultsEnd - retrieveGPUResultsStart)) / CLOCKS_PER_SEC * 1000.0;

    // printf("CPU Execution Time: %.3f ms\n", cpuElapsedTime);
    // printf("Total Runtime: %.3f ms\n", cudaElapsedTime + cpuElapsedTime);
    // printf("Total Runtime (with setup/file write): %.3f ms\n", cpuElapsedTimeSetup);
    // printf("Total Setup Time: %.3f ms\n", setupTime);
    // printf("Total GPU Results Retrieval Time: %.3f ms\n", gpuRetrievalTime);
    // printf("Proccessed %d nodes\n", totalNodes);
    // // Print the aggregated counts (if has no child then follow up to the parent)
    // printf("{ ");
    // for (const auto& [itemSet, count] : map) {
    //     std::cout << itemSet << ": " << count << '\n';
    // } printf("}");
    return 1;
}

int main(int argc, char *argv[])
{

    printf("test\n");
    int x = KNN();
    return -1;
}