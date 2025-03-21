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


__global__ void processItemsetOnGPU(ItemBitmap *items){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
        printf("tid %d 2 items in first item is %d \n",tid,  items[tid].item[0]);
    
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

int *generateCandidates(ItemBitmap *inBitmap, int rowLength, int minItemCount){
    int k = 0;

    if(sizeof(inBitmap) == 0){
        
    }
}

// Implements a threaded kNN where for each candidate query an in-place priority queue is maintained to identify the nearest neighbors
int KNN()
{
     int minItemCount = 3; // setting the minimum # of items to be considered an itemset
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
                
                    int locationOfLine = i + 1;
                    //printf("We found item 999 at line %d\n", locationOfLine);
                    //printf("the location of insertion will be %d\b\n", locationOfInsertion);
                

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
            //printf("We found item 999 at %d\n", i);
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
                    //printf("Bit at index: %d\n", index);
                    //printf("%d, tid %d , the location (which should match insertion where this is should be %d |||| ", itemsBitmap[locationtracker], position + index + 1, locationtracker);
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
            //printf("Items %d had a frequency >3 of %d\n", i, itemAndCounts[i]);
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
        cpuFreqBitmap[i].item = (int*)malloc(sizeof(int));
    }

    for (int i = 0; i < 1000; i++)
    {
        if (itemAndCounts[i] >= 3)
        {   
            memcpy(cpuFreqBitmap[indexInArray].item, &i, sizeof(int));
            //cpuFreqBitmap[indexInArray] -> item = i;
            
            memcpy(cpuFreqBitmap[indexInArray].bitmap, &itemsBitmap[i * rowSize], rowSize * sizeof(int));
            indexInArray++;
            
            //printf("Items %d had a frequency >3 of %d\n", i, itemAndCounts[i]);
        }
    }

    // for(int i = 0; i < indexInArray; i++){
    //     printf("Our frequent item is %d \n", cpuFreqBitmap[i].item[0]);
    //     for(int j = 0; j < 1000; j++){
    //         int value =  cpuFreqBitmap[i].bitmap[j];
    //         if(value != 0){
    //             printf(" index start:%d : %d ", j * 32 + 1, value);
    //         }
    //     }
    //     printf("\n");
    // }

    
    
    
    //generating 2 itemsets
    //calculating worst case max size needed for n=2 itemsets
    int numPairs = indexInArray * (indexInArray -1)/2;
    //printf("numpairs is %d\n", numPairs);
    ItemBitmap* cpu2Itemsets = (ItemBitmap*)calloc(numPairs, sizeof(ItemBitmap));

    for(int i = 0; i < numPairs; i++){
        cpu2Itemsets[i].item = (int *)malloc(2 * sizeof(int));
        cpu2Itemsets[i].bitmap = (int *)malloc(1 * rowSize * sizeof(int));
    }
    int countIndexInPairs = 0;

    
    /*generating the 2 length itemsets (I'm generating all of them)*/
    for(int i = 0; i < indexInArray; i++){
         
        for(int j = i + 1; j < indexInArray; j++){
            cpu2Itemsets[countIndexInPairs].item[0] = cpuFreqBitmap[i].item[0];
            cpu2Itemsets[countIndexInPairs].item[1] = cpuFreqBitmap[j].item[0];
            for(int k = 0; k < rowSize; k++){
                cpu2Itemsets[countIndexInPairs].bitmap[k] = cpuFreqBitmap[i].bitmap[k] & cpuFreqBitmap[j].bitmap[k];
            }
            countIndexInPairs++;
            
        }

    }

    int countOf2Itemsets = 0;
    for(int i = 0; i < numPairs; i++){
        int countOfBits = 0;
        //printf("pair is %d and %d\n", cpu2Itemsets[i].item[0], cpu2Itemsets[i].item[1]);
        for(int j = 0; j < rowSize; j++){
            countOfBits += __builtin_popcount(cpu2Itemsets[i].bitmap[j]);
        }
        if(countOfBits >= minItemCount){
            countOf2Itemsets++;
            //printf("2 frequent itemset is %d and %d \n", cpu2Itemsets[i].item[0], cpu2Itemsets[i].item[1]);
        }
    }

    printf("Count of frequent 2 itemsets is %d\n", countOf2Itemsets);
    int sizeOfQueueToGpu = 2 * countOf2Itemsets;
    ItemBitmap* queueToGpu = (ItemBitmap*)malloc(sizeOfQueueToGpu * sizeof(ItemBitmap));
    for(int i = 0; i < sizeOfQueueToGpu; i++){
        queueToGpu[i].item = (int *)malloc(sizeof(int));
        queueToGpu[i].bitmap = (int *)malloc(rowSize * sizeof(int));
    }

    
    printf("past malloc gpu queue\n");
    /*This is very inefficient (the array isn't flat), will need to be improved*/
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < countOf2Itemsets; j++){
                //printf("before item");
                int item = cpu2Itemsets[j].item[i];
                //printf("item is %d", item);
                //here we are getting the bitmap from the orginal 1 frequent items array
                queueToGpu[i*countOf2Itemsets + j].item[0] = item;
                queueToGpu[i*countOf2Itemsets + j].bitmap = &itemsBitmap[item];
        }
    }
    printf("we just filled the gpu queue");
    /* We are now ready to send this to the gpu*/
    ItemBitmap *d_2Itemsets;
    cudaMalloc(&(d_2Itemsets), sizeOfQueueToGpu * sizeof(ItemBitmap));
    for(int i = 0; i < sizeOfQueueToGpu; i++){
    //     cudaMemcpy(&(d_2Itemsets[i] ->item), ,1 * sizeof(int *), cudaMemcpyHostToDevice);
    //     cudaMemcpy(&(d_2Itemsets[i] ->bitmap), rowSize * sizeof(int *), cudaMemcpyHostToDevice);
    //     // cudaMalloc(&d_2Itemsets[i].item, sizeof(int));
        //cudaMalloc((void**)&((d_2Itemsets)[i].bitmap), rowSize * sizeof(int));
    }
    printf("we just past cudaMalloc\n");
    //cudaMemcpy(d_2Itemsets, queueToGpu, sizeOfQueueToGpu * sizeof(ItemBitmap), cudaMemcpyHostToDevice);
    printf("we got past firstmemcpy \n");
    ItemBitmap *h_2Itemsets = (ItemBitmap *)malloc(sizeOfQueueToGpu * sizeof(ItemBitmap));

    for (int i = 0; i < sizeOfQueueToGpu; i++) {
        // Allocate memory for struct members on the GPU
        cudaMalloc(&(h_2Itemsets[i].item), sizeof(int)); 
        cudaMalloc(&(h_2Itemsets[i].bitmap), rowSize * sizeof(int));

        // Copy actual bitmap data from host to device
        cudaMemcpy(h_2Itemsets[i].item, queueToGpu[i].item, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(h_2Itemsets[i].bitmap, queueToGpu[i].bitmap, rowSize * sizeof(int), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(d_2Itemsets, h_2Itemsets, sizeOfQueueToGpu * sizeof(ItemBitmap), cudaMemcpyHostToDevice);

    //printf("total number of items is %d\n", countOfItems);
    // Allocate memory on the GPU
    int threadsPerBlock = 32;
    int blocksPerGrid = 1; // how do we know how many blocks we need to use?
    // printf("BlocksPerGrid = %d\n", blocksPerGrid);
    printf("number of threads is roughly %d\n", threadsPerBlock * blocksPerGrid);
    int countBitTest = 6;
    printf("result of buildin_popcount = %d\n", __builtin_popcount(countBitTest));

   

    // here I would want to generate all itemsets

    clock_t setupTimeEnd = clock();

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    float cudaElapsedTime;

    cudaEventRecord(startEvent);
    // processItemSets<<<blocksPerGrid, threadsPerBlock>>>(d_text, minItemCount, d_offsets, lineCountInDataset, blocksPerGrid);
    processItemsetOnGPU<<<blocksPerGrid, threadsPerBlock>>>(d_2Itemsets);
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