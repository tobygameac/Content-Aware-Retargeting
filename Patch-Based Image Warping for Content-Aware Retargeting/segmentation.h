#ifndef SEGMENTATION_H_
#define SEGMENTATION_H_

#include "bmp_reader.h"
#include "image.h"
#include "graph.h"
#include "disjoint_set.h"

#include <algorithm>

typedef Image<unsigned char> ImageType;
typedef Graph2D<int> GraphType;

void GaussianSmoothing(ImageType &image, double sigma);
GraphType BuildGraphFromImage(ImageType &image);
void Segmentation(ImageType &image, GraphType &G, DisjointSet &vertex_disjoint_set, double sigma, double k, int min_patch_size, double similar_color_patch_merge_threshold);

#endif