#include "segmentation.h"

void GaussianSmoothing(ImageType &image, double sigma) {
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

  image = new_image;
}

void BuildGraphFromImage(ImageType &image, GraphType &G) {
  G.V.clear();
  G.E.clear();

  G.V.reserve(image.width * image.height);
  G.E.reserve(image.width * image.height * 4);

  const int DELTA_R[4] = {0, 1, 1, -1};
  const int DELTA_C[4] = {1, 0, 1, 1};

  for (int r = 0; r < image.height; ++r) {
    for (int c = 0; c < image.width; ++c) {
      G.V.push_back(std::pair<int, int>(c, r));

      int index = r * image.width + c;

      for (int direction = 0; direction < 4; ++direction) {
        int neighbor_r = r + DELTA_R[direction];
        int neighbor_c = c + DELTA_C[direction];
        if (neighbor_r >= 0 && neighbor_r < image.height && neighbor_c >= 0 && neighbor_c < image.width) {
          int neighbor_index = neighbor_r * image.width + neighbor_c;
          std::pair<int, int> e(index, neighbor_index);

          double w1 = (double)image.pixel[r][c].r - (double)image.pixel[neighbor_r][neighbor_c].r;
          double w2 = (double)image.pixel[r][c].g - (double)image.pixel[neighbor_r][neighbor_c].g;
          double w3 = (double)image.pixel[r][c].b - (double)image.pixel[neighbor_r][neighbor_c].b;

          double w = sqrt(w1 * w1 + w2 * w2 + w3 * w3);

          G.E.push_back(Edge(e, w));
        }
      }
    }
  }
}

void Segmentation(ImageType &image, GraphType &G, std::vector<std::vector<int> > &group_of_pixel, double sigma, double k, int min_patch_size, double similar_color_patch_merge_threshold) {
  GaussianSmoothing(image, sigma);

  image.write_BMP_image("smooth.bmp");

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
    if (it->w <= min(threshold[group_of_x], threshold[group_of_y])) {
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
    if (min_patch_size > min(vertex_disjoint_set.SizeOfGroup(group_of_x), vertex_disjoint_set.SizeOfGroup(group_of_y))) {
      vertex_disjoint_set.UnionGroup(group_of_x, group_of_y);
    }
  }

  // Calculate the color of each group
  std::vector<Pixel<double> > group_color(G.V.size());
  for (int r = 0; r < image.height; ++r) {
    for (int c = 0; c < image.width; ++c) {
      int index = r * image.width + c;
      int group = vertex_disjoint_set.FindGroup(index);
      int group_size = vertex_disjoint_set.SizeOfGroup(group);
      if (group_size) {
        group_color[group].r += (double)image.pixel[r][c].r / group_size;
        group_color[group].g += (double)image.pixel[r][c].g / group_size;
        group_color[group].b += (double)image.pixel[r][c].b / group_size;
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
    double r_difference = (double)group_color[group_of_x].r - (double)group_color[group_of_y].r;
    double g_difference = (double)group_color[group_of_x].g - (double)group_color[group_of_y].g;
    double b_difference = (double)group_color[group_of_x].b - (double)group_color[group_of_y].b;
    double color_difference = sqrt(r_difference * r_difference + g_difference * g_difference + b_difference * b_difference);
    if (color_difference < similar_color_patch_merge_threshold) {
      vertex_disjoint_set.UnionGroup(group_of_x, group_of_y);
    }
  }

  // Calculate the color of each group again
  for (auto it = group_color.begin(); it != group_color.end(); ++it) {
    it->r = 0;
    it->g = 0;
    it->b = 0;
  }

  for (int r = 0; r < image.height; ++r) {
    for (int c = 0; c < image.width; ++c) {
      int index = r * image.width + c;
      int group = vertex_disjoint_set.FindGroup(index);
      int group_size = vertex_disjoint_set.SizeOfGroup(group);
      if (group_size) {
        group_color[group].r += (double)image.pixel[r][c].r / group_size;
        group_color[group].g += (double)image.pixel[r][c].g / group_size;
        group_color[group].b += (double)image.pixel[r][c].b / group_size;
      }
    }
  }

  // Write the pixel value
  for (int r = 0; r < image.height; ++r) {
    for (int c = 0; c < image.width; ++c) {
      int index = r * image.width + c;
      int group = vertex_disjoint_set.FindGroup(index);
      image.pixel[r][c].r = (unsigned char)group_color[group].r;
      image.pixel[r][c].g = (unsigned char)group_color[group].g;
      image.pixel[r][c].b = (unsigned char)group_color[group].b;
    }
  }

  group_of_pixel.clear();
  group_of_pixel = std::vector<std::vector<int> >(image.height);
  for (int r = 0; r < image.height; ++r) {
    group_of_pixel[r] = std::vector<int>(image.width);
    for (int c = 0; c < image.width; ++c) {
      int index = r * image.width + c;
      int group = vertex_disjoint_set.FindGroup(index);
      group_of_pixel[r][c] = group;
    }
  }
}
