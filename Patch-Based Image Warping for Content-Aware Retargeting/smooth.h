#ifndef SMOOTH_H_
#define SMOOTH_H_

#include "image.h"

typedef Image<unsigned char> ImageType;

ImageType GaussianSmoothing(const ImageType &image, double sigma) {
  const int FILTER_SIZE = 3;

  double filter[FILTER_SIZE][FILTER_SIZE];
  double filter_sum = 0;

  for (int filter_r = 0; filter_r < FILTER_SIZE; ++filter_r) {
    for (int filter_c = 0; filter_c < FILTER_SIZE; ++filter_c) {
      int dr = filter_r - FILTER_SIZE / 2;
      int dc = filter_c - FILTER_SIZE / 2;
      filter[filter_r][filter_c] = (1.0 / (2 * acos(-1) * sigma * sigma)) * exp(-1 * ((dr * dr + dc * dc) / (2 * sigma * sigma)));
      filter_sum += filter[filter_r][filter_c];
    }
  }

  ImageType new_image(image.width, image.height);

  for (int color = 0; color < 3; ++color) {
    for (int r = 0; r < image.height; ++r) {
      for (int c = 0; c < image.width; ++c) {
        double new_pixel_r_value = 0;
        double new_pixel_g_value = 0;
        double new_pixel_b_value = 0;
        for (int filter_r = 0; filter_r < FILTER_SIZE; ++filter_r) {
          for (int filter_c = 0; filter_c < FILTER_SIZE; ++filter_c) {
            int dr = filter_r - FILTER_SIZE / 2;
            int dc = filter_c - FILTER_SIZE / 2;
            int target_r = r + dr;
            int target_c = c + dc;
            if (target_r >= 0 && target_r < image.height && target_c >= 0 && target_c < image.width) {
              if (color == RED) {
                new_pixel_r_value += image.pixel[target_r][target_c].r * filter[filter_r][filter_c];
              }
              if (color == GREEN) {
                new_pixel_g_value += image.pixel[target_r][target_c].g * filter[filter_r][filter_c];
              }
              if (color == BLUE) {
                new_pixel_b_value += image.pixel[target_r][target_c].b * filter[filter_r][filter_c];
              }
            }
          }
        }
        new_pixel_r_value /= (double)filter_sum;
        new_pixel_g_value /= (double)filter_sum;
        new_pixel_b_value /= (double)filter_sum;
        if (color == RED) {
          new_image.pixel[r][c].r = (unsigned char)new_pixel_r_value;
        }
        if (color == GREEN) {
          new_image.pixel[r][c].g = (unsigned char)new_pixel_g_value;
        }
        if (color == BLUE) {
          new_image.pixel[r][c].b = (unsigned char)new_pixel_b_value;
        }
      }
    }
  }

  return new_image;
}

#endif