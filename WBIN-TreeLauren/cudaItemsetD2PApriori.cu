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

typedef struct{
    TreeNode* table[1000];
} HashTable;

unsigned int hash(int vals, int tableSize){
    int hash = 0;
    hash = 31 * hash + vals;
    return hash % tableSize;
}



/* so essentially, each index is paired with it's assocaited index + countOf2Itemsets */
__global__ void processItemsetOnGPU(ItemBitmap *items, int countOf2Itemsets, int rowSize)
{
    extern __shared__ int side1[2];
    // extern __shared__ int side2[1];

    // hardcoded p value (cus we have 1000 items)
    int pValue = 4;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int sectionOfBitmap = tid % rowSize;
    // since our Vert database is so long, it will span multiple blocks
    bool isInFirstBlock = true;
    // printf("rowsize is %d", rowSize);

    // this is aka itemset number
    int verticalListIndex = tid / rowSize;

    int verticalListIndex2 = tid / rowSize + countOf2Itemsets;
    int item1 = items[verticalListIndex].item[0];
    int item2 = items[verticalListIndex2].item[0];
    if (tid / rowSize == 209375)
    {
        // printf("209372 FOUND FIRST VERTICAL LIST INDEX IS %d\n", verticalListIndex);
    }
    if (tid / rowSize == 209377 && tid % rowSize == 0)
    {
        printf("209377 first item should not be zero it's instead: %d\n", item1);
    }
    if (sectionOfBitmap > blockDim.x)
    {
        isInFirstBlock = false;
    }

    /* Here I am assigning a tid to handle a section of the vertical bitmap */
    // if(tid < rowSize){
    // launched 690575000 threads was that right?
    long encodedPair = 0;
    if (tid % rowSize == 0 || tid % 1024 == 0 && tid / rowSize < countOf2Itemsets)
    {

        // printf("I am tid %d and my items are %d and %d my first vertic list index is %d and vl 2 is %d\n", tid, item1, items[countOf2Itemsets + tid / rowSize].item[0], verticalListIndex, verticalListIndex2);
        encodedPair += item1;
        encodedPair += item2 * 10000LL;
        // printf("I am tid %d and my items are %d \n", tid, items[countOf2Itemsets + tid / rowSize].item[0]);
        // printf("I am tid %d and my encoded item is %d\n", tid, encodedPair);
        side1[0] = item1;
        // side2[0] = encodedPair;
    }
    __syncthreads();

}

/*
removeLowFrequencyItems is used to remove the items in the bitmap that have low frequency
it will return a new bitmap database containing only items which were determined to have support
greater than the minimum threshold
*/
int *removeLowFrequencyItemsCPU(int *inBitMap, int rowLength, int minItemCount)
{
}

int *generateCandidates(ItemBitmap *inBitmap, int rowLength, int minItemCount)
{
    int k = 0;

    if (sizeof(inBitmap) == 0)
    {
    }
}

int hashItemsetPointer(int* itemset, int rowSize){
    int seed = 0;
    for(int i = 0; i < rowSize; i++){
        seed ^= itemset[i] + (seed << 6) + (seed >> 2);
    }
    printf("seed for hash is %d\n", seed);
    return seed;
}


TreeNode *createNode(int *bitSet, int transactionNumber, int rowSize)
{
    int indexStart = transactionNumber * rowSize;
    // printf("We are right abotu to make a new node\n");
    TreeNode *newNode = (TreeNode *)malloc(sizeof(TreeNode));
    newNode->weight = (int *)malloc(rowSize * sizeof(int));
    if (newNode == NULL)
    {
        fprintf(stderr, "Memory allocation failed!\n");
        exit(1);
    }

    // copying the bitmap section for this node specifically
    for (int i = 0; i < rowSize; i++)
    {
        newNode->weight[i] = bitSet[indexStart + i];
    }

    newNode->count = 1;
    newNode->Child = NULL;
    newNode->Next = NULL;
    // printf("we tried\n");
    return newNode;
}

TreeNode *Weighted_Binary_Count_Tree(int *weightBitSet, int countOfTransactions, int rowSize)
{
    printf("[BUILD TREE STARTED]\n");
    TreeNode *root = createNode(0, 0, 0);
    int complete = 0;
    TreeNode *traverseNode = NULL;
    printf("%d\n", weightBitSet[64]);
    for (int i = 0; i < countOfTransactions; i++)
    {
        int transaction = i;
        // printf("have we crashed yet?\n ");
        int indexStart = rowSize * i;
        if (root->Child == NULL)
        {
            // printf("nope\n ");
            TreeNode *newNode = createNode(weightBitSet, i, rowSize);
            // root = newNode;
            root->Child = newNode;
            // traverseNode = root;
        }
        else
        {
            
            complete = 0;

            traverseNode = root->Child;
            while (complete != 1)
            {
                bool doesMatch = true;
                for (int j = 0; j < rowSize; j++)
                {
                    // printf("checking at lcation %d on item %d\n", transaction * rowSize + j, transaction);
                    if (traverseNode != NULL && weightBitSet[transaction * rowSize + j] != traverseNode->weight[j])
                    {
                        // printf("crashing here?\n");
                        doesMatch = false;
                        //break;
                    }
                }
                if (doesMatch)
                {
                    complete = 1;
                    traverseNode->count = traverseNode->count + 1;
                }
                else
                {
                    // printf("crashing here?\n");
                    bool isIncluded = true; // represents the ^ on line 12 in Algorithm 1
                    for (int j = 0; j < rowSize; j++)
                    {
                        // printf("crashing here?\n");
                        if (weightBitSet[transaction * rowSize + j] & traverseNode->weight[j] != traverseNode->weight[j])
                        {
                            isIncluded = false;
                            // break;
                        }
                    }
                    if (isIncluded)
                    {
                        // printf("crashing 172?\n");
                        if (traverseNode != NULL && traverseNode->Child == NULL)
                        {
                            
                            TreeNode *newNode = createNode(weightBitSet, transaction, rowSize);
                            traverseNode->Child = newNode;
                            complete = 1;
                        }
                        else
                        {
                            // printf("are we gonna crash?\n");
                            if (traverseNode != NULL)
                            {
                                // printf("are we gonna crash?\n");
                                //traverseNode->Child = traverseNode; 
                                traverseNode = traverseNode->Child;// Move to the child
                            }
                        }
                    }
                    else
                    {
                        bool isEqualWeightk = true;
                        // printf("crashing here?\n");
                        for (int j = 0; j < rowSize; j++)
                        {
                            // printf("crashing here?\n");
                            if (weightBitSet[transaction * rowSize + j] & traverseNode->weight[j] != weightBitSet[transaction * rowSize + j])
                            {
                                isEqualWeightk = false;
                                // break;
                            }
                        }
                        if (isEqualWeightk)
                        {
                            if(traverseNode->Next == NULL){
                                // printf("crashing here?\n");
                                TreeNode *newNode = createNode(weightBitSet, transaction, rowSize);
                                newNode->Child = traverseNode;
                                newNode->Next = traverseNode->Next;
                                //traverseNode = newNode;
                                
                                // traverseNode = newNode;
                                // traverseNode->Child = newNode;
                                complete = 1;
                            }
                        }
                        else
                        {
                            // printf("or here?\n");
                            if(traverseNode->Next != NULL){
                                traverseNode->Next = traverseNode;
                            }
                        }
                    }
                }
            }
        }
    }
    printf("root node count = %d count of transactions was %d\n", root->count, countOfTransactions);
    return root;
}

int countSetBits(int* bitSet, int rowSize){
    int k = 0;
    for(int i = 0; i < rowSize; i++){
        if(bitSet[i] > 0){
            k += __popc(bitSet[i]);
        }
    }   
    return k;
}

__global__ int* generate_Subset(int* nodeElements, int rowSize, int countElements)
{
    int tid = blockDim.x * blockIdx + threadIdx.x;
    for(int i = 0; i < countElements; i++){
        if()
    }
}

void depthFirstTraversal(TreeNode *wBinTree, int rowSize)
{
    // return 1;
    if (wBinTree != NULL)
    {   
        
        if(wBinTree->count > 2){
            printf("Count: %d\n", wBinTree->count);
            hashItemsetPointer(wBinTree->weight, rowSize);
            int countOfItems = countSetBits(wBinTree->weight, rowSize);
        }
        if (wBinTree->Child != NULL)
        {
            depthFirstTraversal(wBinTree->Child, rowSize);
        }

        TreeNode *sibling = wBinTree->Next;
        while (sibling)
        {
            depthFirstTraversal(sibling, rowSize);
            sibling = sibling->Next;
        }
    }
}

void *Weighted_Binary_Count_Tree_Mining(TreeNode *wBinTree, int rowSize)
{
    depthFirstTraversal(wBinTree, rowSize);
}

// Implements a threaded kNN where for each candidate query an in-place priority queue is maintained to identify the nearest neighbors
int KNN()
{
    int minItemCount = 3; // setting the minimum # of items to be considered an itemset
    printf("we started\n");
    clock_t cpu_start_withSetup = clock();
    // a 1-d array (flat) that will store all of the 1sized itemsets
    //  int *itemsBitmap = (int *)calloc(3125000, sizeof(int));
    clock_t setupTimeStart = clock();
    // int lineCountInDataset = 1692081;
    // int lineCountInDataset = 55012;
    int countOfItems = 1000;
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

    /*
    calculating the size of the array to store the bitset for this study
    ((total number of items in dataset + 32) / 32)  * (total number of transactions))
    (1000 + 32) / 32 * 100,000 = 3,200,000 ints needed
    rowSize = 32
    */

    int *itemsBitmap = (int *)calloc(3200000, sizeof(int));
    int rowSize = 32; // already known
    int countInBitmap = 0;
    // ItemBitmap firstBitmap[181682];

    // for(int i = 0; i < 181682; i++){
    //     firstBitmap[i].bitmap = (int*)calloc(1718, sizeof(int));
    // }

    // int* itemBitmap2 = generateBitmapCPU(h_buffer, h_offsets);
    // int countOfItems = 0;
    //  int items[55012];
    //  printf("before for loop\n");
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

                // let's say our number = 0, we will want it to be at i = 999 - number (aka 0)
                // from section IV A2 in the paper
                int locationOfInsertion = (countOfItems - 1 - number) / 32;
                int bitLocationOfFlip = number % 32;
                int intLocationToAdjust = (rowSize * i) + locationOfInsertion;
                if (number == 0)
                {
                    int locationOfLine = i + 1;
                    // printf("We found item 0 at line %d", locationOfLine);
                    // printf(" the location of insertion will be %d\b\n", intLocationToAdjust);
                }

                // printf("Are we gonna segfault? + locaiton of insertion %d and number is %d\n", locationOfInsertion, number);
                itemsBitmap[intLocationToAdjust] |= (1 << bitLocationOfFlip);
                // firstBitmap[countInBitmap].id =  number;
                // firstBitmap[countInBitmap].bitmap[location] |= (1 << (i % 32));
                countInBitmap++;
                // printf("%d\n",number);
                // countOfItems++;
                // items[itemCount] = number;
                itemCount++;
                number = 0;
                inNumber = false;
            }
            // if (number == 999)
            // {
            //     printf("We found item 999 at %d\n", i);
            //     int locationOfInsertion = locationOfTransaction + (number * rowSize);
            //     itemsBitmap[locationOfInsertion] |= (1 << (i % 32));
            //     countOfItems++;
            //     countInBitmap++;
            //     inNumber = false;
            // }
            // printf("\n");
        }
    }

    TreeNode* testNode = Weighted_Binary_Count_Tree(itemsBitmap, 100000, rowSize);
    depthFirstTraversal(testNode, rowSize);
    // int firstItemCount = 0;
    // for(int i = 0; i < rowSize; i++){
    //     if(itemsBitmap[0 * rowSize + i]!= 0){
    //         firstItemCount += __builtin_popcount(itemsBitmap[0 * rowSize + i]);
    //     }
    // }

    // printf("total number of 0 is %d\n", firstItemCount);
    // printf("we escaped the for loop");
    // int *itemAndCounts = (int *)calloc(1000, sizeof(int));
    // for (int i = 0; i < 1000; i++)
    // {
    //     int sumOfInts = 0;
    //     for (int j = 0; j < rowSize; j++)
    //     {
    //         sumOfInts += __builtin_popcount(itemsBitmap[i * 3125 + j]);
    //     }
    //     itemAndCounts[i] = sumOfInts;

    //     if (i == 0)
    //     {
    //         printf("\n Item %d: ", i);
    //         for (int j = 0; j < rowSize; j++)
    //         {

    //             int temp = itemsBitmap[i * 3125 + j];

    //             int position = 100000 - (j * 32);
    //             int locationtracker = i * 3125 + j;
    //             if (itemsBitmap[i * rowSize + j] != 0)
    //             {
    //                 int temp = itemsBitmap[i * rowSize + j] & -itemsBitmap[i * rowSize + j];
    //                 int index = __builtin_ctz(temp); // Get index of LSB (0-based)
    //                 //printf("Bit at index: %d\n", index);
    //                 if(locationtracker < 20){
    //                     printf("%d, tid %d , the location (which should match insertion where this is should be %d |||| ", itemsBitmap[locationtracker], position + index + 1, locationtracker);
    //                 }
    //             }
    //         }
    //     }
    // }

    // int countOfFreqItem = 0;
    // for (int i = 0; i < 1000; i++)
    // {
    //     if (itemAndCounts[i] >= minItemCount)
    //     {
    //         countOfFreqItem++;
    //         //printf("Items %d had a frequency >3 of %d\n", i, itemAndCounts[i]);
    //     }
    // }

    // printf("total number of frequent items is %d\n", countOfFreqItem);
    // int* items = (int *)calloc(countOfFreqItem * 3125, sizeof(int));

    // int indexInArray = 0;
    // ItemBitmap* cpuFreqBitmap = (ItemBitmap*)calloc(countOfFreqItem, sizeof(ItemBitmap));
    // for(int i = 0; i < countOfFreqItem; i++){
    //     cpuFreqBitmap[i].bitmap = (int*)malloc(rowSize * sizeof(int));
    //     cpuFreqBitmap[i].item = (int*)malloc(sizeof(int));
    // }

    // for (int i = 0; i < 1000; i++)
    // {
    //     if (itemAndCounts[i] >= minItemCount)
    //     {
    //         memcpy(cpuFreqBitmap[indexInArray].item, &i, sizeof(int));
    //         //cpuFreqBitmap[indexInArray] -> item = i;

    //         memcpy(cpuFreqBitmap[indexInArray].bitmap, &itemsBitmap[i * rowSize], rowSize * sizeof(int));
    //         indexInArray++;

    //         //printf("Items %d had a frequency >3 of %d\n", i, itemAndCounts[i]);
    //     }
    // }

    // //generating 2 itemsets
    // //calculating worst case max size needed for n=2 itemsets
    // int numPairs = indexInArray * (indexInArray -1)/2;
    // //printf("numpairs is %d\n", numPairs);
    // ItemBitmap* cpu2Itemsets = (ItemBitmap*)calloc(numPairs, sizeof(ItemBitmap));

    // for(int i = 0; i < numPairs; i++){
    //     cpu2Itemsets[i].item = (int *)malloc(2 * sizeof(int));
    //     cpu2Itemsets[i].bitmap = (int *)malloc(1 * rowSize * sizeof(int));
    // }
    // int countIndexInPairs = 0;

    // /*generating the 2 length itemsets (I'm generating all of them)*/
    // for(int i = 0; i < indexInArray; i++){

    //     for(int j = i + 1; j < indexInArray; j++){
    //         cpu2Itemsets[countIndexInPairs].item[0] = cpuFreqBitmap[i].item[0];
    //         cpu2Itemsets[countIndexInPairs].item[1] = cpuFreqBitmap[j].item[0];
    //         for(int k = 0; k < rowSize; k++){
    //             cpu2Itemsets[countIndexInPairs].bitmap[k] = cpuFreqBitmap[i].bitmap[k] & cpuFreqBitmap[j].bitmap[k];
    //         }
    //         countIndexInPairs++;

    //     }

    // }

    // int countOf2Itemsets = 0;
    // for(int i = 0; i < numPairs; i++){
    //     int countOfBits = 0;
    //     //printf("pair is %d and %d\n", cpu2Itemsets[i].item[0], cpu2Itemsets[i].item[1]);
    //     for(int j = 0; j < rowSize; j++){
    //         countOfBits += __builtin_popcount(cpu2Itemsets[i].bitmap[j]);
    //         if(cpu2Itemsets[i].item[0] == 0){
    //             //printf("count of bits was %d\n", __builtin_popcount(cpu2Itemsets[i].bitmap[j]));
    //             //printf("2 frequent itemset is %d and %d \n", cpu2Itemsets[i].item[0], cpu2Itemsets[i].item[1]);
    //         }
    //     }
    //     if(countOfBits >= minItemCount){
    //         countOf2Itemsets++;

    //     }
    // }

    // printf("Count of frequent 2 itemsets is %d\n", countOf2Itemsets);
    // int sizeOfQueueToGpu = 2 * countOf2Itemsets;
    // ItemBitmap* queueToGpu = (ItemBitmap*)malloc(sizeOfQueueToGpu * sizeof(ItemBitmap));
    // for(int i = 0; i < sizeOfQueueToGpu; i++){
    //     queueToGpu[i].item = (int *)malloc(sizeof(int));
    //     queueToGpu[i].bitmap = (int *)malloc(rowSize * sizeof(int));
    // }

    // printf("past malloc gpu queue\n");
    // /*This is very inefficient (the array isn't flat), will need to be improved*/
    // for(int i = 0; i < 2; i++){
    //     for(int j = 0; j < countOf2Itemsets; j++){
    //         //printf("before item");
    //         int item = cpu2Itemsets[j].item[i];
    //         //printf("item is %d", item);
    //         //here we are getting the bitmap from the orginal 1 frequent items array
    //         queueToGpu[i*countOf2Itemsets + j].item[0] = item;
    //         if(item == 0){
    //             // printf("Copying from itemsBitmap[%d * %d] -> queueToGpu[%d], size = %zu\n",
    //             // item, rowSize, (i*countOf2Itemsets + j), rowSize * sizeof(int));
    //             // printf(" itemsBitmap capacity = ???, item = %d, rowSize = %d\n", item, rowSize);
    //             for(int k = 0; k < rowSize; k++){
    //                 if(itemsBitmap[item * rowSize + k] != 0 && k < 25){
    //                     //printf("found item: %d near index %d\n", item, k * 32);
    //                 }
    //             }
    //         }

    //         memcpy(queueToGpu[i*countOf2Itemsets + j].bitmap,
    //         &itemsBitmap[item * rowSize],
    //         rowSize * sizeof(int));

    //     }
    // }
    // //printf("we just filled the gpu queue\n");
    // /* We are now ready to send this to the gpu*/
    // ItemBitmap *d_2Itemsets;
    // cudaMalloc(&(d_2Itemsets), sizeOfQueueToGpu * sizeof(ItemBitmap));
    // printf("item in bitmap to gpu queuetoGPU bitmap %d\n", queueToGpu[0].item[0]);
    // for(int i = 0; i < rowSize; i++){
    //     if(queueToGpu[0].bitmap[i]!= 0 && i < 20){
    //         //printf("found item %d at around index %d int value was %d\n", queueToGpu[0].item[0], i * 32, queueToGpu[0].bitmap[i]);
    //     }
    // }

    // printf("we just past cudaMalloc\n");
    // //cudaMemcpy(d_2Itemsets, queueToGpu, sizeOfQueueToGpu * sizeof(ItemBitmap), cudaMemcpyHostToDevice);
    // printf("we got past firstmemcpy \n");
    // ItemBitmap *h_2Itemsets = (ItemBitmap *)malloc(sizeOfQueueToGpu * sizeof(ItemBitmap));

    // for (int i = 0; i < 2 * countOf2Itemsets; i++) {
    //     // Allocate memory for struct members on the GPU
    //     cudaMalloc(&(h_2Itemsets[i].item), sizeof(int));
    //     cudaMalloc(&(h_2Itemsets[i].bitmap), rowSize * sizeof(int));

    //     // Copy actual bitmap data from host to device
    //     cudaMemcpy(h_2Itemsets[i].item, queueToGpu[i].item, sizeof(int), cudaMemcpyHostToDevice);
    //     if(i < countOf2Itemsets && queueToGpu[i].item[0] == 0){
    //         //printf("we copied an item %d and i was %d\n", queueToGpu[i].item[0], i);
    //     }
    //     cudaMemcpy(h_2Itemsets[i].bitmap, queueToGpu[i].bitmap, rowSize * sizeof(int), cudaMemcpyHostToDevice);
    // }

    // cudaMemcpy(d_2Itemsets, h_2Itemsets, sizeOfQueueToGpu * sizeof(ItemBitmap), cudaMemcpyHostToDevice);

    // /* setting up our grid to determine how many threads we will need */
    // int threadsPerBlock = 1024;
    // int blocksPerGrid = ((sizeOfQueueToGpu * rowSize + threadsPerBlock) / threadsPerBlock ) / 2; // how do we know how many blocks we need to use?
    // printf("BlocksPerGrid = %d\n", blocksPerGrid);
    // printf("number of threads is roughly %d\n", threadsPerBlock * blocksPerGrid);
    // int countBitTest = 6;
    // printf("result of buildin_popcount = %d\n", __builtin_popcount(countBitTest));

    // // here I would want to generate all itemsets

    // clock_t setupTimeEnd = clock();

    // cudaEvent_t startEvent, stopEvent;
    // cudaEventCreate(&startEvent);
    // cudaEventCreate(&stopEvent);
    // float cudaElapsedTime;

    // cudaEventRecord(startEvent);
    // // processItemSets<<<blocksPerGrid, threadsPerBlock>>>(d_text, minItemCount, d_offsets, lineCountInDataset, blocksPerGrid);
    // processItemsetOnGPU<<<blocksPerGrid, threadsPerBlock>>>(d_2Itemsets, countOf2Itemsets, rowSize);
    // cudaDeviceSynchronize();
    // cudaEventRecord(stopEvent);
    // cudaEventSynchronize(stopEvent);

    // // Print the elapsed time (milliseconds)
    // cudaEventElapsedTime(&cudaElapsedTime, startEvent, stopEvent);
    // printf("CUDA Kernel Execution Time: %.3f ms\n", cudaElapsedTime);

    // // ensure there are no kernel errors
    // cudaError_t cudaError = cudaGetLastError();
    // if (cudaError != cudaSuccess)
    // {
    //     fprintf(stderr, "processItemSets cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
    //     exit(EXIT_FAILURE);
    // }

    // clock_t retrieveGPUResultsStart = clock();
    // clock_t retrieveGPUResultsEnd = clock();

    // // global reduction will be written to file
    // FILE *resultsFile = fopen("cudaItemSetMiningResults.txt", "w");
    // if (resultsFile == NULL)
    // {
    //     perror("Error opening results file");
    //     return 1;
    // }

    // Record end time
    // clock_t cpu_end_withSetup = clock();
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
    printf("finished\n");
    return -1;
}