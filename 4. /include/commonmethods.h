typedef struct
{
    int item;
    int *bitmap;
} ItemBitmap;

int* generateBitmapCPU(char* h_buffer, int* h_offsets);