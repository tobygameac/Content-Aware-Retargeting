#ifndef SALIENCY_H_
#define SALIENCY_H_

#include "image.h"

#include <vector>

double CIE_F(double t) {
  return (t > 0.008856) ? pow(t, 1 / 3.0) : (7.787 * t + (16.0 / 116.0));
}

void CalculateContextAwareSaliencyMap(const Image<unsigned char> &image, std::vector<std::vector<double> > &saliency_map, double c, double K) {

  saliency_map.clear();
  saliency_map = std::vector<std::vector<double> >(image.height, std::vector<double>(image.width));

  const double RGBtoCIELAB_matrix[3][3] = {
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
      CIE_X[r][c] = image.pixel[r][c].r * RGBtoCIELAB_matrix[0][0] + image.pixel[r][c].g * RGBtoCIELAB_matrix[0][1] + image.pixel[r][c].b * RGBtoCIELAB_matrix[0][2];
      CIE_Y[r][c] = image.pixel[r][c].r * RGBtoCIELAB_matrix[1][0] + image.pixel[r][c].g * RGBtoCIELAB_matrix[1][1] + image.pixel[r][c].b * RGBtoCIELAB_matrix[1][2];
      CIE_Z[r][c] = image.pixel[r][c].r * RGBtoCIELAB_matrix[2][0] + image.pixel[r][c].g * RGBtoCIELAB_matrix[2][1] + image.pixel[r][c].b * RGBtoCIELAB_matrix[2][2];

      CIE_X[r][c] = CIE_X[r][c] * PIXEL_VALUE_NORMALIZED_CONSTANT;
      CIE_Y[r][c] = CIE_Y[r][c] * PIXEL_VALUE_NORMALIZED_CONSTANT;
      CIE_Z[r][c] = CIE_Z[r][c] * PIXEL_VALUE_NORMALIZED_CONSTANT;
    }
  }

  std::vector<std::vector<double> > CIE_L(image.height, std::vector<double>(image.width));
  std::vector<std::vector<double> > CIE_a(image.height, std::vector<double>(image.width));
  std::vector<std::vector<double> > CIE_b(image.height, std::vector<double>(image.width));

  double max_L_a_b = -2e9, min_L_a_b = 2e9;
  for (int r = 0; r < image.height; ++r) {
    for (int c = 0; c < image.width; ++c) {
      const double X_DIVIDE_BY_X_N = CIE_X[r][c] / CIE_X_N;
      const double Y_DIVIDE_BY_Y_N = CIE_Y[r][c] / CIE_Y_N;
      const double Z_DIVIDE_BY_Z_N = CIE_Z[r][c] / CIE_Z_N;
      if (Y_DIVIDE_BY_Y_N > 0.008856) {
        CIE_L[r][c] = 116 * pow(Y_DIVIDE_BY_Y_N, 1 / 3.0) - 16;
      } else {
        CIE_L[r][c] = 903.3 * Y_DIVIDE_BY_Y_N;
      }

      CIE_a[r][c] = 500.0 * (CIE_F(X_DIVIDE_BY_X_N) - CIE_F(Y_DIVIDE_BY_Y_N));
      CIE_b[r][c] = 500.0 * (CIE_F(Y_DIVIDE_BY_Y_N) - CIE_F(Z_DIVIDE_BY_Z_N));

      max_L_a_b = max_L_a_b > CIE_L[r][c] ? max_L_a_b : CIE_L[r][c];
      max_L_a_b = max_L_a_b > CIE_a[r][c] ? max_L_a_b : CIE_a[r][c];
      max_L_a_b = max_L_a_b > CIE_b[r][c] ? max_L_a_b : CIE_b[r][c];

      min_L_a_b = min_L_a_b < CIE_L[r][c] ? min_L_a_b : CIE_L[r][c];
      min_L_a_b = min_L_a_b < CIE_a[r][c] ? min_L_a_b : CIE_a[r][c];
      min_L_a_b = min_L_a_b < CIE_b[r][c] ? min_L_a_b : CIE_b[r][c];
    }
  }

  for (int r = 0; r < image.height; ++r) {
    for (int c = 0; c < image.width; ++c) {
      CIE_L[r][c] = (CIE_L[r][c] - min_L_a_b) * (max_L_a_b - min_L_a_b);
      CIE_a[r][c] = (CIE_a[r][c] - min_L_a_b) * (max_L_a_b - min_L_a_b);
      CIE_b[r][c] = (CIE_b[r][c] - min_L_a_b) * (max_L_a_b - min_L_a_b);
    }
  }

  const int NEAREST_PIXEL_MASK_SIZE = sqrt(K);

  std::vector<std::vector<double> > local_global_saliency_map(image.height, std::vector<double>(image.width));
  for (int r = 0; r < image.height; ++r) {
    for (int c = 0; c < image.width; ++c) {
      int neighbor_count = 0;
      for (int delta_r = -NEAREST_PIXEL_MASK_SIZE / 2; delta_r <= NEAREST_PIXEL_MASK_SIZE / 2; ++delta_r) {
        for (int delta_c = -NEAREST_PIXEL_MASK_SIZE / 2; delta_c <= NEAREST_PIXEL_MASK_SIZE / 2; ++delta_c) {
          int neighbor_r = r + delta_r;
          int neighbor_c = c + delta_c;
          if (neighbor_r >= 0 && neighbor_r < image.height && neighbor_c >= 0 && neighbor_c < image.width) {
            neighbor_count = neighbor_count + 1;
            double difference_color = sqrt(pow((CIE_L[r][c] - CIE_L[neighbor_r][neighbor_c]), 2) + pow((CIE_a[r][c] - CIE_a[neighbor_r][neighbor_c]), 2) + pow((CIE_b[r][c] - CIE_b[neighbor_r][neighbor_c]), 2));
            double difference_position = sqrt(delta_r * delta_r + delta_c * delta_c);
            double differecne_value = difference_color / (1 + c * difference_position);
            local_global_saliency_map[r][c] += differecne_value;
          }
        }
      }
      if (neighbor_count) {
        local_global_saliency_map[r][c] = 1 - exp(-1 * local_global_saliency_map[r][c] / (double)neighbor_count);
      }
    }
  }

  std::vector<std::vector<double> > multi_scale_saliency_map(image.height, std::vector<double>(image.width));
  for (int r = 0; r < image.height; ++r) {
    for (int c = 0; c < image.width; ++c) {
      int neighbor_count = 0;
      for (int delta_r = -NEAREST_PIXEL_MASK_SIZE / 2; delta_r <= NEAREST_PIXEL_MASK_SIZE / 2; ++delta_r) {
        for (int delta_c = -NEAREST_PIXEL_MASK_SIZE / 2; delta_c <= NEAREST_PIXEL_MASK_SIZE / 2; ++delta_c) {
          int neighbor_r = r + delta_r;
          int neighbor_c = c + delta_c;
          if (neighbor_r >= 0 && neighbor_r < image.height && neighbor_c >= 0 && neighbor_c < image.width) {
            neighbor_count = neighbor_count + 1;
            multi_scale_saliency_map[r][c] += local_global_saliency_map[neighbor_r][neighbor_c];
          }
        }
      }
      if (neighbor_count) {
        multi_scale_saliency_map[r][c] = multi_scale_saliency_map[r][c] / (double)neighbor_count;
      }
    }
  }

  Image<unsigned char> saliency_image(image.width, image.height);

  for (int r = 0; r < image.height; ++r) {
    for (int c = 0; c < image.width; ++c) {
      /*
      double d_foci = multi_scale_saliency_map[r][c];
      for (int delta_r = 0; delta_r < image.height; ++delta_r) {
        int neighbor_r = r + delta_r;
        if (neighbor_r >= 0 && neighbor_r < image.height && multi_scale_saliency_map[neighbor_r][c] >= 0.8) {
          d_foci = (delta_r / image.height) < d_foci ? (delta_r / image.height) : d_foci;
          break;
        }
      }
      for (int delta_r = 0; delta_r < image.height; ++delta_r) {
        int neighbor_r = r + delta_r;
        if (neighbor_r >= 0 && neighbor_r < image.height && multi_scale_saliency_map[neighbor_r][c] >= 0.8) {
          d_foci = (delta_r / image.height) < d_foci ? (delta_r / image.height) : d_foci;
          break;
        }
      }
      for (int delta_c = 0; delta_c < image.width; ++delta_c) {
        int neighbor_c = c + delta_c;
        if (neighbor_c >= 0 && neighbor_c < image.width && multi_scale_saliency_map[r][neighbor_c] >= 0.8) {
          d_foci = (delta_c / image.width) < d_foci ? (delta_c / image.width) : d_foci;
          break;
        }
      }
      for (int delta_c = 0; delta_c < image.width; ++delta_c) {
        int neighbor_c = c - delta_c;
        if (neighbor_c >= 0 && neighbor_c < image.width && multi_scale_saliency_map[r][neighbor_c] >= 0.8) {
          d_foci = (delta_c / image.width) < d_foci ? (delta_c / image.width) : d_foci;
          break;
        }
      }
      saliency_map[r][c] = multi_scale_saliency_map[r][c] - d_foci;
      */
      saliency_map[r][c] = multi_scale_saliency_map[r][c];
      saliency_image.pixel[r][c].r = saliency_image.pixel[r][c].g = saliency_image.pixel[r][c].b = saliency_map[r][c] / PIXEL_VALUE_NORMALIZED_CONSTANT;
    }
  }

  saliency_image.write_BMP_image("saliency.bmp");
}
#endif