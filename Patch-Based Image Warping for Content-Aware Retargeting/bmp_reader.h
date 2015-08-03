#ifndef BMP_READER_H_
#define BMP_READER_H_

#include <windows.h>

typedef unsigned char * BMPImagePointer;

BMPImagePointer LoadBitmapFile(char *file_name, BITMAPINFO *image_bitmap_info);

#endif