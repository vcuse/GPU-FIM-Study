typedef struct
{
    int *item;
    int *bitmap;
} ItemBitmap;

struct TreeNode {
    int *weight; //bitset representation 
    int count;
    struct TreeNode* Child; // Pointer to the first child node
    struct TreeNode* Next;  // Pointer to the next sibling node
};

// Define a structure to hold your list of integers
typedef struct {
    int *data;
    size_t len;
} int_list_t;

int* generateBitmapCPU(char* h_buffer, int* h_offsets);

uint64_t hash_int_list(int* list, int listLen);