#ifndef GRAPH_H_
#define GRAPH_H_

#include <vector>

class Edge {
public:
  Edge() {}
  Edge(std::pair<int, int> e, int w) : e(e), w(w) {}

  const bool operator <(const Edge &other) {
    if (w != other.w) { 
      return  w < other.w;
    }
    if ((e.first != other.e.first)) { 
      return  e.first < other.e.first;
    }
    return e.second < other.e.second;
  }

  std::pair<int, int> e;
  double w;
};

template <class T>
class Graph2D {
public:
  Graph2D() {}

  Graph2D(std::vector<std::pair<T, T> > V, std::vector<Edge> E) : V(V), E(E) {}

  std::vector<std::pair<T, T> > V;
  std::vector<Edge> E;
};

#endif
