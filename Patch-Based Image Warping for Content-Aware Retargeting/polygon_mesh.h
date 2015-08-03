#ifndef POLYGON_MESH_H_
#define POLYGON_MESH_H_

#include <vector>

template <class T>
class PolygonMesh {
 public:
  PolygonMesh() {}
  PolygonMesh(std::vector<int> vertex_index, std::vector<std::pair<T, T> > texture_coordinate)
    : vertex_index(vertex_index),
      texture_coordinate(texture_coordinate) {}

  std::vector<int> vertex_index;
  std::vector<std::pair<T, T> > texture_coordinate;
};

#endif
