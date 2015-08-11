#ifndef SEGMENTATION_H_
#define SEGMENTATION_H_

#include "bmp_reader.h"
#include "image.h"
#include "graph.h"
#include "disjoint_set.h"

#include <algorithm>

typedef Image<unsigned char> ImageType;
typedef Graph2D<int> GraphType;

void BuildGraphFromImage(const ImageType &image, GraphType &G);
ImageType Segmentation(const ImageType &image, GraphType &G, std::vector<std::vector<int> > &group_of_pixel, const double k, const int min_patch_size, const double similar_color_patch_merge_threshold);

#endif