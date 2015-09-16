#ifndef SEGMENTATION_H_
#define SEGMENTATION_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <algorithm>

#include "graph.h"
#include "disjoint_set.h"


typedef Graph2D<int> GraphType;

void BuildGraphFromImage(const cv::Mat &image, GraphType &G) {
  G.V.clear();
  G.E.clear();

  G.V.reserve(image.size().width * image.size().height);
  G.E.reserve(image.size().width * image.size().height * 4);

  const int DELTA_R[4] = {0, 1, 1, -1};
  const int DELTA_C[4] = {1, 0, 1, 1};

  for (int r = 0; r < image.size().height; ++r) {
    for (int c = 0; c < image.size().width; ++c) {
      G.V.push_back(std::pair<int, int>(c, r));

      int index = r * image.size().width + c;

      for (int direction = 0; direction < 4; ++direction) {
        int neighbor_r = r + DELTA_R[direction];
        int neighbor_c = c + DELTA_C[direction];
        if (neighbor_r >= 0 && neighbor_r < image.size().height && neighbor_c >= 0 && neighbor_c < image.size().width) {
          int neighbor_index = neighbor_r * image.size().width + neighbor_c;
          std::pair<int, int> e(index, neighbor_index);

          double w[3];
          for (size_t pixel_index = 0; pixel_index < 3; ++pixel_index) {
            w[pixel_index] = image.at<cv::Vec3b>(r, c).val[pixel_index] - image.at<cv::Vec3b>(neighbor_r, neighbor_c).val[pixel_index];
          }

          double edge_weight = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);

          G.E.push_back(Edge(e, edge_weight));
        }
      }
    }
  }
}

cv::Mat Segmentation(const cv::Mat &image, GraphType &G, std::vector<std::vector<int> > &group_of_pixel, const double k, const int min_patch_size, const double similar_color_patch_merge_threshold) {

  BuildGraphFromImage(image, G);

  sort(G.E.begin(), G.E.end());

  DisjointSet vertex_disjoint_set(G.V.size());

  std::vector<double> threshold(G.V.size());
  for (auto it = threshold.begin(); it != threshold.end(); ++it) {
    *it = 1 / k;
  }

  // Segmentation
  for (auto it = G.E.begin(); it != G.E.end(); ++it) {
    int group_of_x = vertex_disjoint_set.FindGroup(it->e.first);
    int group_of_y = vertex_disjoint_set.FindGroup(it->e.second);
    if (group_of_x == group_of_y) {
      continue;
    }
    if (it->w <= std::min(threshold[group_of_x], threshold[group_of_y])) {
      vertex_disjoint_set.UnionGroup(group_of_x, group_of_y);
      threshold[group_of_x] = it->w + (k / vertex_disjoint_set.SizeOfGroup(it->e.first));
    }
  }

  // Deal with the smaller set
  for (auto it = G.E.begin(); it != G.E.end(); ++it) {
    int group_of_x = vertex_disjoint_set.FindGroup(it->e.first);
    int group_of_y = vertex_disjoint_set.FindGroup(it->e.second);
    if (group_of_x == group_of_y) {
      continue;
    }
    if (min_patch_size > std::min(vertex_disjoint_set.SizeOfGroup(group_of_x), vertex_disjoint_set.SizeOfGroup(group_of_y))) {
      vertex_disjoint_set.UnionGroup(group_of_x, group_of_y);
    }
  }

  // Calculate the color of each group
  std::vector<cv::Vec3d> group_color(G.V.size());
  for (int r = 0; r < image.size().height; ++r) {
    for (int c = 0; c < image.size().width; ++c) {
      int index = r * image.size().width + c;
      int group = vertex_disjoint_set.FindGroup(index);
      int group_size = vertex_disjoint_set.SizeOfGroup(group);
      if (group_size) {
        for (size_t pixel_index = 0; pixel_index < 3; ++pixel_index) {
          group_color[group].val[pixel_index] += image.at<cv::Vec3b>(r, c).val[pixel_index] / (double)group_size;
        }
      }
    }
  }

  // Deal with the similar color set
  for (auto it = G.E.begin(); it != G.E.end(); ++it) {
    int group_of_x = vertex_disjoint_set.FindGroup(it->e.first);
    int group_of_y = vertex_disjoint_set.FindGroup(it->e.second);
    if (group_of_x == group_of_y) {
      continue;
    }
    double difference[3];
    for (size_t pixel_index = 0; pixel_index < 3; ++pixel_index) {
      difference[pixel_index] = group_color[group_of_x].val[pixel_index] - group_color[group_of_y].val[pixel_index];
    }
    double color_difference = sqrt(difference[0] * difference[0] + difference[1] * difference[1] + difference[2] * difference[2]);
    if (color_difference < similar_color_patch_merge_threshold) {
      vertex_disjoint_set.UnionGroup(group_of_x, group_of_y);
    }
  }

  // Calculate the color of each group again
  for (auto it = group_color.begin(); it != group_color.end(); ++it) {
    for (size_t pixel_index = 0; pixel_index < 3; ++pixel_index) {
      it->val[pixel_index] = 0;
    }
  }

  for (int r = 0; r < image.size().height; ++r) {
    for (int c = 0; c < image.size().width; ++c) {
      int index = r * image.size().width + c;
      int group = vertex_disjoint_set.FindGroup(index);
      int group_size = vertex_disjoint_set.SizeOfGroup(group);
      if (group_size) {
        for (size_t pixel_index = 0; pixel_index < 3; ++pixel_index) {
          group_color[group].val[pixel_index] += image.at<cv::Vec3b>(r, c).val[pixel_index] / (double)group_size;
        }
      }
    }
  }

  cv::Mat image_after_segmentation = image;

  // Write the pixel value
  for (int r = 0; r < image.size().height; ++r) {
    for (int c = 0; c < image.size().width; ++c) {
      int index = r * image.size().width + c;
      int group = vertex_disjoint_set.FindGroup(index);
      for (size_t pixel_index = 0; pixel_index < 3; ++pixel_index) {
        image_after_segmentation.at<cv::Vec3b>(r, c).val[pixel_index] = group_color[group].val[pixel_index];
      }
    }
  }

  group_of_pixel.clear();
  group_of_pixel = std::vector<std::vector<int> >(image.size().height);
  for (int r = 0; r < image.size().height; ++r) {
    group_of_pixel[r] = std::vector<int>(image.size().width);
    for (int c = 0; c < image.size().width; ++c) {
      int index = r * image.size().width + c;
      int group = vertex_disjoint_set.FindGroup(index);
      group_of_pixel[r][c] = group;
    }
  }

  return image_after_segmentation;
}

#endif