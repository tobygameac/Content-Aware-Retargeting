#ifndef IMAGE_H_
#define IMAGE_H_

#include <iostream>
#include <fstream>
#include <vector>

enum ColorType {
  RED,
  GREEN,
  BLUE
};

template <class T>
class Pixel {
public:
  T r, g, b, a;
};

template <class T>
class Image {

public:
  Image() {}

  Image(int width, int height) : width(width), height(height) {
    pixel.resize(height);

    for (int r = 0; r < height; ++r) {
      pixel[r].resize(width);
      for (int c = 0; c < width; ++c) {
        pixel[r][c].r = 0;
        pixel[r][c].g = 0;
        pixel[r][c].b = 0;
      }
    }
  }

  Image(int width, int height, T *image_pointer) : width(width), height(height) {
    pixel.resize(height);

    for (int r = 0; r < height; ++r) {
      pixel[r].resize(width);
      for (int c = 0; c < width; ++c) {
        int index = r * width + c;
        pixel[r][c].r = image_pointer[index * 3];
        pixel[r][c].g = image_pointer[index * 3 + 1];
        pixel[r][c].b = image_pointer[index * 3 + 2];
      }
    }
  }

  bool write_BMP_image(const char *file_name) {
    const int HEADER_SIZE = 54;

    char header[HEADER_SIZE] = {
      0x42, 0x4d, 0, 0, 0, 0, 0, 0, 0, 0,
      54, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0, 
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
      0, 0, 0, 0
    };

    int file_size = width * height * 3 + 54;

    header[2] = file_size &0x000000ff;
    header[3] = (file_size >> 8) & 0x000000ff;
    header[4] = (file_size >> 16) & 0x000000ff;
    header[5] = (file_size >> 24) & 0x000000ff;

    header[18] = width & 0x000000ff;
    header[19] = (width >> 8) & 0x000000ff;
    header[20] = (width >> 16) & 0x000000ff;
    header[21] = (width >> 24) & 0x000000ff;

    header[22] = height & 0x000000ff;
    header[23] = (height >> 8) & 0x000000ff;
    header[24] = (height >> 16) & 0x000000ff;
    header[25] = (height >> 24) & 0x000000ff;

    std::ofstream file(file_name, std::ios::out | std::ios::binary);
    if (!file) {
      return false;
    }
    file.write(header, HEADER_SIZE);

    for (int r = 0; r < height; ++r) {
      for (int c = 0; c < width; ++c) {
        file.put(pixel[r][c].b).put(pixel[r][c].g).put(pixel[r][c].r);
      }
    }

    file.close();
    return true;
  }

  int width, height;
  std::vector<std::vector<Pixel<T> > > pixel;
};
#endif