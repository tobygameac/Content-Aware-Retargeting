#ifndef WARPING_H_
#define WARPING_H_

#define IL_STD

#include <ilcplex/ilocplex.h>
#include <ilconcert/iloexpression.h>

#include <vector>
#include <map>

#include "image.h"

typedef Image<unsigned char> ImageType;
typedef Graph2D<int> GraphType;

ImageType Warping(ImageType &image, GraphType &G, std::vector<std::vector<int> > &group_of_pixel, int target_image_width, int target_image_height, double mesh_width, double mesh_height/*, Saliency*/) {
  if (target_image_width <= 0 || target_image_height <= 0) {
    printf("Wrong target image size (%d x %d)\n", target_image_width, target_image_height);
    exit(-1);
  }
  ImageType result_image(target_image_width, target_image_height);

  // Build the vertex list of each patch
  std::map<int, std::vector<int> > vertex_index_list_of_patch;
  for (int vertex_index = 0; vertex_index < G.V.size(); ++vertex_index) {
    int group_of_vertex = group_of_pixel[G.V[vertex_index].second][G.V[vertex_index].first];
    vertex_index_list_of_patch[group_of_vertex].push_back(vertex_index);
  }

  // Build the edge list of each patch
  std::map<int, std::vector<int> > edge_index_list_of_patch;
  for (int edge_index = 0; edge_index < G.E.size(); ++edge_index) {
    int group_of_x = group_of_pixel[G.V[G.E[edge_index].e.first].second][G.V[G.E[edge_index].e.first].first];
    int group_of_y = group_of_pixel[G.V[G.E[edge_index].e.second].second][G.V[G.E[edge_index].e.second].first];
    if (group_of_x == group_of_y) {
      edge_index_list_of_patch[group_of_x].push_back(edge_index);
    } else {
      edge_index_list_of_patch[group_of_x].push_back(edge_index);
      edge_index_list_of_patch[group_of_y].push_back(edge_index);
    }
  }

  // Calculate the saliency value of each patch
  std::map<int, double> saliency_of_patch;
  srand(time(0));
  double min_saliency = 2e9, max_saliency = -2e9;
  for (int patch_index = 0; patch_index < vertex_index_list_of_patch.size(); ++patch_index) {
    if (edge_index_list_of_patch[patch_index].size()) {
      saliency_of_patch[patch_index] = 1 / (double)edge_index_list_of_patch[patch_index].size();
    }
    for (int vertex_index = 0; vertex_index < vertex_index_list_of_patch[patch_index].size(); ++vertex_index) {
      //saliency_of_patch[patch_index] += saliency value / (double)vertex_of_patch[patch_index].size();
      //saliency_of_patch[patch_index] = (rand() % 100 + 1) / 100.0;
      //saliency_of_patch[patch_index] = 1;
    }
    min_saliency = (min_saliency < saliency_of_patch[patch_index]) ? min_saliency : saliency_of_patch[patch_index];
    max_saliency = (max_saliency > saliency_of_patch[patch_index]) ? max_saliency : saliency_of_patch[patch_index];
  }

  // Normalize saliency values
  for (int patch_index = 0; patch_index < vertex_index_list_of_patch.size(); ++patch_index) {
    saliency_of_patch[patch_index] = (saliency_of_patch[patch_index] - min_saliency) / (max_saliency - min_saliency);
  }

  IloEnv env;

  IloModel model(env);
  IloNumVarArray x(env);
  IloRangeArray c(env);
  IloExpr expr(env);

  for (int vertex_index = 0; vertex_index < G.V.size(); ++vertex_index) {
    x.add(IloNumVar(env, -IloInfinity, IloInfinity));
    x.add(IloNumVar(env, -IloInfinity, IloInfinity));
  }

  for (int patch_index = 0; patch_index < edge_index_list_of_patch.size(); ++patch_index) {
    std::vector<int> &edge_index_list = edge_index_list_of_patch[patch_index];

    if (!edge_index_list.size()) {
      continue;
    }

    Edge representive_edge = G.E[edge_index_list[0]];
    double c_x = G.V[representive_edge.e.first].first - G.V[representive_edge.e.second].first;
    double c_y = G.V[representive_edge.e.first].second - G.V[representive_edge.e.second].second;

    double original_matrix_a = c_x;
    double original_matrix_b = c_y;
    double original_matrix_c = c_y;
    double original_matrix_d = -c_x;
    double matrix_rank = original_matrix_a * original_matrix_d - original_matrix_b * original_matrix_c;

    if (fabs(matrix_rank) <= 1e-9) {
      matrix_rank = 1e-9;
    }

    double matrix_a = original_matrix_d / matrix_rank;
    double matrix_b = -original_matrix_b / matrix_rank;
    double matrix_c = -original_matrix_c / matrix_rank;
    double matrix_d = original_matrix_a / matrix_rank;

    for (int edge_index = 0; edge_index < edge_index_list.size(); ++edge_index) {
      Edge edge = G.E[edge_index_list[edge_index]];
      double e_x = G.V[edge.e.first].first - G.V[edge.e.second].first;
      double e_y = G.V[edge.e.first].second - G.V[edge.e.second].second;

      double transformation_s = matrix_a * e_x + matrix_b * e_y;
      double transformation_r = matrix_c * e_x + matrix_d * e_y;

      expr += saliency_of_patch[patch_index] * IloPower((x[edge.e.first * 2] - x[edge.e.second * 2]) - (transformation_s * (x[representive_edge.e.first * 2] - x[representive_edge.e.second * 2]) + transformation_r * (x[representive_edge.e.first * 2 + 1] - x[representive_edge.e.second * 2 + 1])), 2);
      expr += saliency_of_patch[patch_index] * IloPower((x[edge.e.first * 2 + 1] - x[edge.e.second * 2 + 1]) - (-transformation_r * (x[representive_edge.e.first * 2] - x[representive_edge.e.second * 2]) + transformation_s * (x[representive_edge.e.first * 2 + 1] - x[representive_edge.e.second * 2 + 1])), 2);
    }
  }

  int mesh_column_count = image.width / mesh_width;
  int mesh_row_count = image.height / mesh_height;

  for (int vertex_index = 0; vertex_index < G.V.size(); ++vertex_index) {
    int left_neighbor_index = vertex_index - 1;
    int up_neighbor_index = vertex_index - mesh_column_count;

    if ((vertex_index % mesh_column_count) != 0) {
      expr += IloPower(x[vertex_index * 2 + 1] - x[left_neighbor_index * 2 + 1], 2);
    }

    if (up_neighbor_index >= 0) {
      expr += IloPower(x[vertex_index * 2] - x[up_neighbor_index * 2], 2);
    }
  }

  model.add(IloMinimize(env, expr));

  for (int row = 0; row < mesh_row_count; ++row) {
    int vertex_index = row * mesh_column_count;
    c.add(x[vertex_index * 2] == G.V[0].first);
    vertex_index = row * mesh_column_count + mesh_column_count - 1;
    c.add(x[vertex_index * 2] == target_image_width);
  }
  for (int column = 0; column < mesh_column_count; ++column) {
    int vertex_index = column;
    c.add(x[vertex_index * 2 + 1] == G.V[0].second);
    vertex_index = (mesh_row_count - 1) * mesh_column_count + column;
    c.add(x[vertex_index * 2 + 1] == target_image_height);
  }

  model.add(c);

  IloCplex cplex(model);

  if (!cplex.solve()) {
    puts("Failed to optimize.");
  }

  IloNumArray result(env);

  cplex.getValues(result, x);

  for (int vertex_index = 0; vertex_index < G.V.size(); ++vertex_index) {
    G.V[vertex_index].first = result[vertex_index * 2];
    G.V[vertex_index].second = result[vertex_index * 2 + 1];
  }

  // Linear
  /*
  IloEnv env;

  IloModel model(env);
  IloNumVarArray x(env);
  IloRangeArray c(env);

  for (int vertex_index = 0; vertex_index < G.V.size(); ++vertex_index) {
  x.add(IloNumVar(env, -IloInfinity, IloInfinity));
  x.add(IloNumVar(env, -IloInfinity, IloInfinity));
  }

  IloExpr expr(env);

  double x_gap = target_image_width / (double)image.width;
  double y_gap = target_image_height / (double)image.height;

  for (int vertex_index = 0; vertex_index < G.V.size(); ++vertex_index) {
  int left_neighbor_index = vertex_index - 1;
  int up_neighbor_index = vertex_index - image.width;

  if ((vertex_index % image.width) != 0) {
  expr += IloPower(x[vertex_index * 2] - x[left_neighbor_index * 2] - x_gap, 2);
  expr += IloPower(x[vertex_index * 2 + 1] - x[left_neighbor_index * 2 + 1], 2);
  }
  if (up_neighbor_index >= 0) {
  expr += IloPower(x[vertex_index * 2] - x[up_neighbor_index * 2], 2);
  expr += IloPower(x[vertex_index * 2 + 1] - x[up_neighbor_index * 2 + 1] - y_gap, 2);
  }
  }

  model.add(IloMinimize(env, expr));

  for (int row = 0; row < image.height; ++row) {
  int vertex_index = row * image.width;
  c.add(x[vertex_index * 2] == G.V[0].first);
  vertex_index = row * image.width + image.width - 1;
  c.add(x[vertex_index * 2] == target_image_width);
  }
  for (int column = 0; column < image.width; ++column) {
  int vertex_index = column;
  c.add(x[vertex_index * 2 + 1] == G.V[0].second);
  vertex_index = (image.height - 1) * image.width + column;
  c.add(x[vertex_index * 2 + 1] == target_image_height);
  }

  model.add(c);

  IloCplex cplex(model);

  if (!cplex.solve()) {
  puts("Failed to optimize.");
  }

  IloNumArray vals(env);
  IloNumArray result(env);

  cplex.getValues(result, x);

  for (int vertex_index = 0; vertex_index < G.V.size(); ++vertex_index) {
  G.V[vertex_index].first = result[vertex_index * 2];
  G.V[vertex_index].second = result[vertex_index * 2 + 1];
  }
  */

  return result_image;
}

#endif