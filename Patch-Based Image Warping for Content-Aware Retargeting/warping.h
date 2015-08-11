#ifndef WARPING_H_
#define WARPING_H_

#define IL_STD

#include <ilcplex/ilocplex.h>
#include <ilconcert/iloexpression.h>

#include <vector>
#include <map>

#include "graph.h"
#include "image.h"

typedef Image<unsigned char> ImageType;
typedef Graph2D<int> GraphType;

ImageType Warping(const ImageType &image, GraphType &G, const std::vector<std::vector<int> > &group_of_pixel, const std::vector<std::vector<double> > &saliency_map, const int target_image_width, const int target_image_height, const double mesh_width, const double mesh_height) {
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
  double min_saliency = 2e9, max_saliency = -2e9;
  for (int patch_index = 0; patch_index < vertex_index_list_of_patch.size(); ++patch_index) {
    for (int vertex_index = 0; vertex_index < vertex_index_list_of_patch[patch_index].size(); ++vertex_index) {
      int vertex_r = G.V[vertex_index_list_of_patch[patch_index][vertex_index]].second;
      int vertex_c = G.V[vertex_index_list_of_patch[patch_index][vertex_index]].first;
      saliency_of_patch[patch_index] += saliency_map[vertex_r][vertex_c] / (double)vertex_index_list_of_patch[patch_index].size();
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

  // Patch transformation constraint
  double alpha = 0.8;
  double width_ratio = target_image_width / (double)image.width;
  double height_ratio = target_image_height / (double)image.height;

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

      double t_s = matrix_a * e_x + matrix_b * e_y;
      double t_r = matrix_c * e_x + matrix_d * e_y;

      // DST
      expr += alpha * saliency_of_patch[patch_index] *
        IloPower((x[edge.e.first * 2] - x[edge.e.second * 2]) -
        (t_s * (x[representive_edge.e.first * 2] - x[representive_edge.e.second * 2]) + t_r * (x[representive_edge.e.first * 2 + 1] - x[representive_edge.e.second * 2 + 1])),
        2);
      expr += alpha * saliency_of_patch[patch_index] * 
        IloPower((x[edge.e.first * 2 + 1] - x[edge.e.second * 2 + 1]) -
        (-t_r * (x[representive_edge.e.first * 2] - x[representive_edge.e.second * 2]) + t_s * (x[representive_edge.e.first * 2 + 1] - x[representive_edge.e.second * 2 + 1])),
        2);

      // DLT
      expr += (1 - alpha) * (1 - saliency_of_patch[patch_index]) *
        IloPower((x[edge.e.first * 2] - x[edge.e.second * 2]) -
        width_ratio * (t_s * (x[representive_edge.e.first * 2] - x[representive_edge.e.second * 2]) + t_r * (x[representive_edge.e.first * 2 + 1] - x[representive_edge.e.second * 2 + 1])),
        2);
      expr += (1 - alpha) * (1 - saliency_of_patch[patch_index]) * 
        IloPower((x[edge.e.first * 2 + 1] - x[edge.e.second * 2 + 1]) -
        height_ratio * (-t_r *  (x[representive_edge.e.first * 2] - x[representive_edge.e.second * 2]) + t_s * (x[representive_edge.e.first * 2 + 1] - x[representive_edge.e.second * 2 + 1])),
        2);
    }
  }

  int mesh_column_count = image.width / mesh_width;
  int mesh_row_count = image.height / mesh_height;

  // Grid orientation constraint
  for (int edge_index = 0; edge_index < G.E.size(); ++edge_index) {
    int vertex_index_1 = G.E[edge_index].e.first;
    int vertex_index_2 = G.E[edge_index].e.second;
    int delta_x = abs(G.V[vertex_index_1].first - G.V[vertex_index_2].first);
    int delta_y = abs(G.V[vertex_index_1].second - G.V[vertex_index_2].second);
    if (delta_x > delta_y) { // Horizontal
      expr += 2 * IloPower(x[vertex_index_1 * 2 + 1] - x[vertex_index_2 * 2 + 1], 2);
    } else {
      expr += 2 * IloPower(x[vertex_index_1 * 2] - x[vertex_index_2 * 2], 2);
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
    puts("Failed to optimize the model.");
  }

  IloNumArray result(env);

  cplex.getValues(result, x);

  for (int vertex_index = 0; vertex_index < G.V.size(); ++vertex_index) {
    G.V[vertex_index].first = result[vertex_index * 2];
    G.V[vertex_index].second = result[vertex_index * 2 + 1];
  }

  return result_image;
}

#endif