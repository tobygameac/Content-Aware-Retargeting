#ifndef SALIENCY_H_
#define SALIENCY_H_

#include "image.h"

#include <utility>
#include <vector>
#include <algorithm>

class CIELabPixel {
public:
  CIELabPixel () : L(0), a(0), b(0) {}
  CIELabPixel (double L, double a, double b) : L(L), a(a), b(b) {}
  double L, a, b;
};

class SaliencyPatch {
public:
  SaliencyPatch(Image<unsigned char> &image, int center_r, int center_c, int patch_width, int patch_height) : image(image), center_r(center_r), center_c(center_c), patch_width(patch_width), patch_height(patch_height) {
    pixel_coordinates.clear();
    for (int delta_r = -patch_height / 2; delta_r <= patch_height / 2; ++delta_r) {
      for (int delta_c = -patch_width / 2; delta_c <= patch_width / 2; ++delta_c) {
        int pixel_r = center_r + delta_r;
        int pixel_c = center_c + delta_c;
        if (pixel_r >= 0 && pixel_r < image.height && pixel_c >= 0 && pixel_c < image.width) {
          pixel_coordinates.push_back(std::make_pair(pixel_r, pixel_c));
        }
      }
    }
  }

  CIELabPixel CalculatePixelAverageValue(const std::vector<std::vector<CIELabPixel> > &image_pixel_in_CIELab) const {
    CIELabPixel patch_pixel_average_value(0, 0, 0);
    for (int i = 0; i < pixel_coordinates.size(); ++i) {
      double height_scale = image.height / (double)image_pixel_in_CIELab.size();
      double width_scale = image.width / (double)image_pixel_in_CIELab[0].size();
      int original_r = pixel_coordinates[i].first / height_scale;
      int original_c = pixel_coordinates[i].second / width_scale;
      patch_pixel_average_value.L += image_pixel_in_CIELab[original_r][original_c].L;
      patch_pixel_average_value.a += image_pixel_in_CIELab[original_r][original_c].a;
      patch_pixel_average_value.b += image_pixel_in_CIELab[original_r][original_c].b;
    }
    if (pixel_coordinates.size()) {
      patch_pixel_average_value.L /= (double)pixel_coordinates.size();
      patch_pixel_average_value.a /= (double)pixel_coordinates.size();
      patch_pixel_average_value.b /= (double)pixel_coordinates.size();
    }
    return patch_pixel_average_value;
  }

  void CalculateSaliency(const std::vector<SaliencyPatch> &saliency_patches, const std::vector<std::vector<CIELabPixel> > &image_pixel_in_CIELab, const double C, const double K) {
    const int SAMPLE_RATE = 5;
    int index_gap = saliency_patches.size() / (double)(K * SAMPLE_RATE);

    std::vector<double> distance_list;
    for (int i = 0; i < saliency_patches.size(); i += index_gap) {
      distance_list.push_back(DistanceOfPatches(*this, saliency_patches[i], image_pixel_in_CIELab, C));
    }

    std::sort(distance_list.begin(), distance_list.end());

    saliency = 0;
    for (int i = 0; i< K; ++i) {
      saliency += distance_list[i];
    }

    saliency = 1 - exp((-1 / K) * saliency);
  }

  std::vector<std::pair<int ,int> > pixel_coordinates;
  int center_r, center_c;
  int patch_width, patch_height;
  double saliency;
  Image<unsigned char> &image;

private:
  inline double DistanceOfPatchesColor(const SaliencyPatch &patch1, const SaliencyPatch &patch2, const std::vector<std::vector<CIELabPixel> > &image_pixel_in_CIELab) {
    CIELabPixel patch1_pixel_average_value = patch1.CalculatePixelAverageValue(image_pixel_in_CIELab);
    CIELabPixel patch2_pixel_average_value = patch2.CalculatePixelAverageValue(image_pixel_in_CIELab);
    double distance_L = patch1_pixel_average_value.L - patch2_pixel_average_value.L;
    double distance_a = patch1_pixel_average_value.a - patch2_pixel_average_value.a;
    double distance_b = patch1_pixel_average_value.b - patch2_pixel_average_value.b;
    return sqrt(pow(distance_L, 2) + pow(distance_a, 2) + pow(distance_b, 2)) / sqrt(3.0);
  }

  inline double DistanceOfPatchesPosition(const SaliencyPatch &patch1, const SaliencyPatch &patch2) {
    double distance_r = (patch1.center_r / (double)patch1.image.height) - (patch2.center_r / (double)patch2.image.height);
    double distance_c = (patch1.center_c / (double)patch1.image.width) - (patch2.center_c / (double)patch2.image.width);
    return sqrt(pow(distance_r, 2) + pow(distance_c, 2)) / sqrt(2.0);
  }

  inline double DistanceOfPatches(const SaliencyPatch &patch1, const SaliencyPatch &patch2, const std::vector<std::vector<CIELabPixel> > &image_pixel_in_CIELab, const double C) {
    return DistanceOfPatchesColor(patch1, patch2, image_pixel_in_CIELab) / (1 + C * DistanceOfPatchesPosition(patch1, patch2));
  }
};

inline double CIE_F(double t) {
  return (t > 0.008856) ? pow(t, 1 / 3.0) : (7.787 * t + (16.0 / 116.0));
}

Image<unsigned char> CalculateContextAwareSaliencyMap(const Image<unsigned char> &image, std::vector<std::vector<double> > &saliency_map, const double SALIENCY_C, const double SALIENCY_K) {
  const double RGB_TO_CIELAB_MATRIX[3][3] = {
    {0.412453, 0.357580, 0.180423},
    {0.212671, 0.715160, 0.072169},
    {0.019334, 0.119193, 0.950227}
  };

  const double CIE_X_N = 0.9515;
  const double CIE_Y_N = 1.0000;
  const double CIE_Z_N = 1.0886;

  const double PIXEL_VALUE_NORMALIZED_CONSTANT = 1 / 255.0;

  std::vector<std::vector<double> > CIE_X(image.height, std::vector<double>(image.width));
  std::vector<std::vector<double> > CIE_Y(image.height, std::vector<double>(image.width));
  std::vector<std::vector<double> > CIE_Z(image.height, std::vector<double>(image.width));

  for (int r = 0; r < image.height; ++r) {
    for (int c = 0; c < image.width; ++c) {
      CIE_X[r][c] = image.pixel[r][c].r * RGB_TO_CIELAB_MATRIX[0][0] + image.pixel[r][c].g * RGB_TO_CIELAB_MATRIX[0][1] + image.pixel[r][c].b * RGB_TO_CIELAB_MATRIX[0][2];
      CIE_Y[r][c] = image.pixel[r][c].r * RGB_TO_CIELAB_MATRIX[1][0] + image.pixel[r][c].g * RGB_TO_CIELAB_MATRIX[1][1] + image.pixel[r][c].b * RGB_TO_CIELAB_MATRIX[1][2];
      CIE_Z[r][c] = image.pixel[r][c].r * RGB_TO_CIELAB_MATRIX[2][0] + image.pixel[r][c].g * RGB_TO_CIELAB_MATRIX[2][1] + image.pixel[r][c].b * RGB_TO_CIELAB_MATRIX[2][2];

      CIE_X[r][c] *= PIXEL_VALUE_NORMALIZED_CONSTANT;
      CIE_Y[r][c] *= PIXEL_VALUE_NORMALIZED_CONSTANT;
      CIE_Z[r][c] *= PIXEL_VALUE_NORMALIZED_CONSTANT;
    }
  }

  std::vector<std::vector<CIELabPixel> > image_pixel_in_CIELab(image.height, std::vector<CIELabPixel>(image.width));

  CIELabPixel max_pixel_value_in_CIELab(-2e9, -2e9, -2e9);
  CIELabPixel min_pixel_value_in_CIELab(2e9, 2e9, 2e9);

  for (int r = 0; r < image.height; ++r) {
    for (int c = 0; c < image.width; ++c) {
      const double X_DIVIDE_BY_X_N = CIE_X[r][c] / CIE_X_N;
      const double Y_DIVIDE_BY_Y_N = CIE_Y[r][c] / CIE_Y_N;
      const double Z_DIVIDE_BY_Z_N = CIE_Z[r][c] / CIE_Z_N;
      if (Y_DIVIDE_BY_Y_N > 0.008856) {
        image_pixel_in_CIELab[r][c].L = 116 * pow(Y_DIVIDE_BY_Y_N, 1 / 3.0) - 16;
      } else {
        image_pixel_in_CIELab[r][c].L = 903.3 * Y_DIVIDE_BY_Y_N;
      }

      image_pixel_in_CIELab[r][c].a = 500.0 * (CIE_F(X_DIVIDE_BY_X_N) - CIE_F(Y_DIVIDE_BY_Y_N));
      image_pixel_in_CIELab[r][c].b = 200.0 * (CIE_F(Y_DIVIDE_BY_Y_N) - CIE_F(Z_DIVIDE_BY_Z_N));

      max_pixel_value_in_CIELab.L = max_pixel_value_in_CIELab.L > image_pixel_in_CIELab[r][c].L ? max_pixel_value_in_CIELab.L : image_pixel_in_CIELab[r][c].L;
      max_pixel_value_in_CIELab.a = max_pixel_value_in_CIELab.a > image_pixel_in_CIELab[r][c].a ? max_pixel_value_in_CIELab.a : image_pixel_in_CIELab[r][c].a;
      max_pixel_value_in_CIELab.b = max_pixel_value_in_CIELab.b > image_pixel_in_CIELab[r][c].b ? max_pixel_value_in_CIELab.b : image_pixel_in_CIELab[r][c].b;

      min_pixel_value_in_CIELab.L = min_pixel_value_in_CIELab.L < image_pixel_in_CIELab[r][c].L ? min_pixel_value_in_CIELab.L : image_pixel_in_CIELab[r][c].L;
      min_pixel_value_in_CIELab.a = min_pixel_value_in_CIELab.a < image_pixel_in_CIELab[r][c].a ? min_pixel_value_in_CIELab.a : image_pixel_in_CIELab[r][c].a;
      min_pixel_value_in_CIELab.b = min_pixel_value_in_CIELab.b < image_pixel_in_CIELab[r][c].b ? min_pixel_value_in_CIELab.b : image_pixel_in_CIELab[r][c].b;
    }
  }

  double pixel_value_gap_L = max_pixel_value_in_CIELab.L - min_pixel_value_in_CIELab.L;
  double pixel_value_gap_a = max_pixel_value_in_CIELab.a - min_pixel_value_in_CIELab.a;
  double pixel_value_gap_b = max_pixel_value_in_CIELab.b - min_pixel_value_in_CIELab.b;

  // Normalize CIE L*a*b to [0, 1]
  for (int r = 0; r < image.height; ++r) {
    for (int c = 0; c < image.width; ++c) {
      image_pixel_in_CIELab[r][c].L = (image_pixel_in_CIELab[r][c].L - min_pixel_value_in_CIELab.L) / pixel_value_gap_L;
      image_pixel_in_CIELab[r][c].a = (image_pixel_in_CIELab[r][c].a - min_pixel_value_in_CIELab.a) / pixel_value_gap_a;
      image_pixel_in_CIELab[r][c].b = (image_pixel_in_CIELab[r][c].b - min_pixel_value_in_CIELab.b) / pixel_value_gap_b;
    }
  }

  const int MAX_IMAGE_SIZE = 250;

  const double BASIC_IMAGE_SCALE = MAX_IMAGE_SIZE / (double)((image.width > image.height) ? image.width : image.height);

  const int IMAGE_SCALES_COUNT = 3;
  const double IMAGE_SCALES[IMAGE_SCALES_COUNT] = {BASIC_IMAGE_SCALE * 1.0, BASIC_IMAGE_SCALE * 0.5, BASIC_IMAGE_SCALE * 0.25};

  const int PATCH_SIZE = 7;
  const int PATCH_SCALES_COUNT = 4;
  const double PATCH_SCALES[PATCH_SCALES_COUNT] = {1.0, 0.8, 0.5, 0.3};

  const double OVERLAP_RATE = 0.5;

  std::vector<Image<unsigned char> > image_with_scales(IMAGE_SCALES_COUNT);
  for (int i = 0; i < IMAGE_SCALES_COUNT; ++i) {
    double scale = IMAGE_SCALES[i];
    image_with_scales[i] = image;
    image_with_scales[i].Resize(scale);
  }

  std::vector<std::vector<std::vector<double> > > saliency_map_with_scales(IMAGE_SCALES_COUNT);
  for (int i = 0; i < IMAGE_SCALES_COUNT; ++i) {
    double scale = IMAGE_SCALES[i];
    saliency_map_with_scales[i] = std::vector<std::vector<double> >(image_with_scales[i].height, std::vector<double>(image_with_scales[i].width));
  }

  std::vector<int> patch_size_with_scales(PATCH_SCALES_COUNT);
  for (int i = 0; i < PATCH_SCALES_COUNT; ++i) {
    int patch_size = PATCH_SIZE * PATCH_SCALES[i];
    // Adjust the patch size to odd value
    patch_size = patch_size + !(patch_size & 1);
    patch_size_with_scales[i] = patch_size;
  }

  std::vector<SaliencyPatch> saliency_patches;
  for (int i = 0; i < IMAGE_SCALES_COUNT; ++i) {
    for (int patch_scales_index = 0; patch_scales_index < PATCH_SCALES_COUNT; ++patch_scales_index) {
      int patch_size = patch_size_with_scales[patch_scales_index];
      double gap = (OVERLAP_RATE * patch_size) * 2;
      if (gap < 1) {
        continue;
      }
      for (double r = gap * (patch_scales_index & 1); r < saliency_map_with_scales[i].size(); r += gap) {
        for (double c = gap * (patch_scales_index & 1); c < saliency_map_with_scales[i][r].size(); c += gap) {
          saliency_patches.push_back(SaliencyPatch(image_with_scales[i], (int)r, (int)c, patch_size, patch_size));
        }
      }
    }
  }


  saliency_map.clear();
  saliency_map = std::vector<std::vector<double> >(image.height, std::vector<double>(image.width));

  std::vector<std::vector<int> > saliency_map_summation_times(image.height, std::vector<int>(image.width));

  for (int i = 0; i < saliency_patches.size(); ++i) {
    saliency_patches[i].CalculateSaliency(saliency_patches, image_pixel_in_CIELab, SALIENCY_C, SALIENCY_K);
    for (int j = 0; j < saliency_patches[i].pixel_coordinates.size(); ++j) {
      double height_scale = image.height / (double)image_pixel_in_CIELab.size();
      double width_scale = image.width / (double)image_pixel_in_CIELab[0].size();
      int original_r = saliency_patches[i].pixel_coordinates[j].first / height_scale;
      int original_c = saliency_patches[i].pixel_coordinates[j].second / width_scale;

      saliency_map[original_r][original_c] += saliency_patches[i].saliency;
      saliency_map_summation_times[original_r][original_c] += 1;
    }
  }

  for (int r = 0; r < image.height; ++r) {
    for (int c = 0; c < image.width; ++c) {
      if (saliency_map_summation_times[r][c]) {
        saliency_map[r][c] /= saliency_map_summation_times[r][c];
      }
    }
  }

  /*const int NEAREST_PIXEL_MASK_SIZE = sqrt(SALIENCY_K);

  for (int scale_index = 0; scale_index < IMAGE_SCALES_COUNT; ++scale_index) {
  double scale = IMAGE_SCALES[scale_index];
  for (int r = 0; r < image_with_scales[scale_index].height; ++r) {
  for (int c = 0; c < image_with_scales[scale_index].width; ++c) {
  int neighbor_count = 0;
  for (int delta_r = -NEAREST_PIXEL_MASK_SIZE / 2; delta_r <= NEAREST_PIXEL_MASK_SIZE / 2; ++delta_r) {
  for (int delta_c = -NEAREST_PIXEL_MASK_SIZE / 2; delta_c <= NEAREST_PIXEL_MASK_SIZE / 2; ++delta_c) {
  if (delta_r || delta_c) {
  int neighbor_r = r + delta_r;
  int neighbor_c = c + delta_c;

  int original_r = r / scale;
  int original_c = c / scale;
  int original_neighbor_r = neighbor_r / scale;
  int original_neighbor_c = neighbor_c / scale;
  if (original_neighbor_r >= 0 && original_neighbor_r < image.height && original_neighbor_c >= 0 && original_neighbor_c < image.width) {
  neighbor_count = neighbor_count + 1;
  double difference_color = sqrt(pow((image_pixel_in_CIELab[original_r][original_c].L - image_pixel_in_CIELab[original_neighbor_r][original_neighbor_c].L), 2) + pow((image_pixel_in_CIELab[original_r][original_c].a - image_pixel_in_CIELab[original_neighbor_r][original_neighbor_c].a), 2) + pow((image_pixel_in_CIELab[original_r][original_c].b - image_pixel_in_CIELab[original_neighbor_r][original_neighbor_c].b), 2)) / sqrt(3.0);
  double difference_position = sqrt(pow((delta_r / image_with_scales[scale_index].height), 2.0) + pow((delta_c / image_with_scales[scale_index].width), 2.0)) / sqrt(2.0);
  double differecne_value = difference_color / (1 + SALIENCY_C * difference_position);
  saliency_map_with_scales[scale_index][r][c] += differecne_value;
  }
  }
  }
  }
  if (neighbor_count) {
  saliency_map_with_scales[scale_index][r][c] = 1 - exp(-1 * saliency_map_with_scales[scale_index][r][c] / (double)neighbor_count);
  }
  }
  }
  }

  saliency_map.clear();
  saliency_map = std::vector<std::vector<double> >(image.height, std::vector<double>(image.width));

  for (int r = 0; r < image.height; ++r) {
  for (int c = 0; c < image.width; ++c) {
  int real_scales_count = 0;
  for (int i = 0; i < IMAGE_SCALES_COUNT; ++i) {
  double scale = IMAGE_SCALES[i];
  int resized_r = r * scale;
  int resized_c = c * scale;
  if (resized_r < saliency_map_with_scales[i].size() && resized_c < saliency_map_with_scales[i][resized_r].size()) {
  real_scales_count += 1;
  saliency_map[r][c] += saliency_map_with_scales[i][resized_r][resized_c];
  }
  }
  if (real_scales_count) {
  saliency_map[r][c] = saliency_map[r][c] / (double)real_scales_count;
  }
  }
  }*/


  Image<unsigned char> saliency_image(image.width, image.height);
  for (int r = 0; r < image.height; ++r) {
    for (int c = 0; c < image.width; ++c) {
      double saliency_at_pixel = saliency_map[r][c] / PIXEL_VALUE_NORMALIZED_CONSTANT;
      saliency_image.pixel[r][c].r = saliency_image.pixel[r][c].g = saliency_image.pixel[r][c].b = saliency_at_pixel;
    }
  }

  return saliency_image;

}
#endif