#ifndef DISJOINT_SET_H_
#define DISJOINT_SET_H_

#include <vector>

class DisjointSet {
 public:
  DisjointSet() {
  }

  DisjointSet(int group_count) : group_count(group_count) {
    group_of_element_at.resize(group_count);
    group_size.resize(group_count);
    for (int i = 0; i < group_count; ++i) {
      group_of_element_at[i] = i;
      group_size[i] = 1;
    }
  }

  int GroupCount() {
    return group_count;
  }

  int SizeOfGroup(int x) {
    int group_of_x = FindGroup(x);
    return group_size[group_of_x];
  }

  int FindGroup(int x) {
    return (x == group_of_element_at[x]) ? x : (group_of_element_at[x] = FindGroup(group_of_element_at[x]));
  }

  void UnionGroup(int x, int y) {
    int group_of_x = FindGroup(x);
    int group_of_y = FindGroup(y);
    if (group_of_x == group_of_y) {
      return;
    }
    group_size[group_of_y] += group_size[group_of_x];
    group_size[group_of_x] = 0;
    group_of_element_at[x] = group_of_y;
  }

 private:
  int group_count;
  std::vector<int> group_of_element_at;
  std::vector<int> group_size;
};

#endif