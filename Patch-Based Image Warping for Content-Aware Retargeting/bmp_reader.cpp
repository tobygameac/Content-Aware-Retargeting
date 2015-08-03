#include "bmp_reader.h"

#include <cstdio>

BMPImagePointer LoadBitmapFile(char *file_name, BITMAPINFO *image_bitmap_info) {
  FILE *file_pointer = fopen(file_name, "rb");

  if (!file_pointer) {
    printf("File %s not found.\n", file_name);
    exit(-1);
  }

  BITMAPFILEHEADER bitmap_file_header;

  fread(&bitmap_file_header, sizeof(BITMAPFILEHEADER), 1, file_pointer);

  int information_size = bitmap_file_header.bfOffBits - sizeof(BITMAPFILEHEADER);
  fread(image_bitmap_info, information_size, 1, file_pointer);

  int bitmap_size = image_bitmap_info->bmiHeader.biSizeImage;
  BMPImagePointer bitmap_image = (BYTE *)malloc(sizeof(BYTE) * bitmap_size);
  fread(bitmap_image, 1, bitmap_size, file_pointer);

  fclose(file_pointer);

  int pixel_count = (image_bitmap_info->bmiHeader.biWidth) * (image_bitmap_info->bmiHeader.biHeight);

  // BGR -> RGB
  for (int index = 0; index < pixel_count; ++index) {
    unsigned char temp = bitmap_image[index * 3];
    bitmap_image[index * 3] = bitmap_image[index * 3 + 2];
    bitmap_image[index * 3 + 2]  = temp;
  }

  return bitmap_image;
}