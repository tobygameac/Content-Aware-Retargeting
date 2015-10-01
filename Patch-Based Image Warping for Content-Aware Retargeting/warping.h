#ifndef WARPING_H_
#define WARPING_H_

#define IL_STD

#include <ilcplex/ilocplex.h>
#include <ilconcert/iloexpression.h>

#include <vector>
#include <algorithm>

#include "graph.h"

typedef Graph2D<float> GraphType;

void PatchBasedWarping(const cv::Mat &image, GraphType &G, const std::vector<std::vector<int> > &group_of_pixel, const std::vector<double> &saliency_of_patch, const int target_image_width, const int target_image_height, const double mesh_width, const double mesh_height) {
  if (target_image_width <= 0 || target_image_height <= 0) {
    printf("Wrong target image size (%d x %d)\n", target_image_width, target_image_height);
    exit(-1);
  }

  // Build the edge list of each patch
  std::vector<std::vector<int> > edge_index_list_of_patch(image.size().width * image.size().height);
  for (size_t edge_index = 0; edge_index < G.E.size(); ++edge_index) {
    int x_vertex_index = G.E[edge_index].e.first;
    int y_vertex_index = G.E[edge_index].e.second;
    int group_of_x = group_of_pixel[G.V[x_vertex_index].second][G.V[x_vertex_index].first];
    int group_of_y = group_of_pixel[G.V[y_vertex_index].second][G.V[y_vertex_index].first];
    if (group_of_x == group_of_y) {
      edge_index_list_of_patch[group_of_x].push_back(edge_index);
    } else {
      edge_index_list_of_patch[group_of_x].push_back(edge_index);
      edge_index_list_of_patch[group_of_y].push_back(edge_index);
    }
  }

  IloEnv env;

  IloNumVarArray x(env);
  IloExpr expr(env);

  for (size_t vertex_index = 0; vertex_index < G.V.size(); ++vertex_index) {
    x.add(IloNumVar(env, -IloInfinity, IloInfinity));
    x.add(IloNumVar(env, -IloInfinity, IloInfinity));
  }

  const double DST_WEIGHT = 0.8;
  const double DLT_WEIGHT = 0.2;
  const double ORIENTATION_WEIGHT = 10.0;
  double width_ratio = target_image_width / (double)image.size().width;
  double height_ratio = target_image_height / (double)image.size().height;

  // Patch transformation constraint
  for (size_t patch_index = 0; patch_index < edge_index_list_of_patch.size(); ++patch_index) {
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

    for (size_t edge_index = 0; edge_index < edge_index_list.size(); ++edge_index) {
      Edge edge = G.E[edge_index_list[edge_index]];
      double e_x = G.V[edge.e.first].first - G.V[edge.e.second].first;
      double e_y = G.V[edge.e.first].second - G.V[edge.e.second].second;

      double t_s = matrix_a * e_x + matrix_b * e_y;
      double t_r = matrix_c * e_x + matrix_d * e_y;

      // DST
      expr += DST_WEIGHT * saliency_of_patch[patch_index] *
        IloPower((x[edge.e.first * 2] - x[edge.e.second * 2]) -
        (t_s * (x[representive_edge.e.first * 2] - x[representive_edge.e.second * 2]) + t_r * (x[representive_edge.e.first * 2 + 1] - x[representive_edge.e.second * 2 + 1])),
        2);
      expr += DST_WEIGHT * saliency_of_patch[patch_index] * 
        IloPower((x[edge.e.first * 2 + 1] - x[edge.e.second * 2 + 1]) -
        (-t_r * (x[representive_edge.e.first * 2] - x[representive_edge.e.second * 2]) + t_s * (x[representive_edge.e.first * 2 + 1] - x[representive_edge.e.second * 2 + 1])),
        2);

      // DLT
      expr += DLT_WEIGHT * (1 - saliency_of_patch[patch_index]) *
        IloPower((x[edge.e.first * 2] - x[edge.e.second * 2]) -
        width_ratio * (t_s * (x[representive_edge.e.first * 2] - x[representive_edge.e.second * 2]) + t_r * (x[representive_edge.e.first * 2 + 1] - x[representive_edge.e.second * 2 + 1])),
        2);
      expr += DLT_WEIGHT * (1 - saliency_of_patch[patch_index]) * 
        IloPower((x[edge.e.first * 2 + 1] - x[edge.e.second * 2 + 1]) -
        height_ratio * (-t_r *  (x[representive_edge.e.first * 2] - x[representive_edge.e.second * 2]) + t_s * (x[representive_edge.e.first * 2 + 1] - x[representive_edge.e.second * 2 + 1])),
        2);
    }
  }

  // Grid orientation constraint
  for (size_t edge_index = 0; edge_index < G.E.size(); ++edge_index) {
    int vertex_index_1 = G.E[edge_index].e.first;
    int vertex_index_2 = G.E[edge_index].e.second;
    float delta_x = abs(G.V[vertex_index_1].first - G.V[vertex_index_2].first);
    float delta_y = abs(G.V[vertex_index_1].second - G.V[vertex_index_2].second);
    if (delta_x > delta_y) { // Horizontal
      expr += ORIENTATION_WEIGHT * IloPower(x[vertex_index_1 * 2 + 1] - x[vertex_index_2 * 2 + 1], 2);
    } else {
      expr += ORIENTATION_WEIGHT * IloPower(x[vertex_index_1 * 2] - x[vertex_index_2 * 2], 2);
    }
  }

  IloModel model(env);

  model.add(IloMinimize(env, expr));

  IloRangeArray c(env);

  int mesh_column_count = (int)(image.size().width / mesh_width) + 1;
  int mesh_row_count = (int)(image.size().height / mesh_height) + 1;

  // Boundary constraint
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

  // Avoid flipping
  for (int row = 0; row < mesh_row_count; ++row) {
    for (int column = 1; column < mesh_column_count; ++column) {
      int vertex_index_right = row * mesh_column_count + column;
      int vertex_index_left = row * mesh_column_count + column - 1;
      c.add((x[vertex_index_right * 2] - x[vertex_index_left * 2]) >= 1e-4);
    }
  }

  for (int row = 1; row < mesh_row_count; ++row) {
    for (int column = 0; column < mesh_column_count; ++column) {
      int vertex_index_down = row * mesh_column_count + column;
      int vertex_index_up = (row - 1) * mesh_column_count + column;
      c.add((x[vertex_index_down * 2 + 1] - x[vertex_index_up * 2 + 1]) >= 1e-4);
    }
  }

  model.add(c);

  IloCplex cplex(model);

  cplex.setOut(env.getNullStream());

  if (!cplex.solve()) {
    puts("Failed to optimize the model.");
  }

  IloNumArray result(env);

  cplex.getValues(result, x);

  for (size_t vertex_index = 0; vertex_index < G.V.size(); ++vertex_index) {
    G.V[vertex_index].first = result[vertex_index * 2];
    G.V[vertex_index].second = result[vertex_index * 2 + 1];
  }

  model.end();
  cplex.end();
  env.end();
}

void FocusWarping(const cv::Mat &image, GraphType &G, const std::vector<std::vector<int> > &group_of_pixel, const std::vector<double> &saliency_of_patch, const int target_image_width, const int target_image_height, const double mesh_width, const double mesh_height, const double max_mesh_scale, const double focus_x, const double focus_y) {
  IloEnv env;

  IloNumVarArray x(env);
  IloExpr expr(env);

  for (size_t vertex_index = 0; vertex_index < G.V.size(); ++vertex_index) {
    x.add(IloNumVar(env, -IloInfinity, IloInfinity));
    x.add(IloNumVar(env, -IloInfinity, IloInfinity));
  }

  int mesh_column_count = (int)(image.size().width / mesh_width) + 1;
  int mesh_row_count = (int)(image.size().height / mesh_height) + 1;

  const double FOCUS_WEIGHT = 5.0;
  const double ORIENTATION_WEIGHT = 10.0;
  const double DISTORTION_WEIGHT = 5.0;

  for (size_t edge_index = 0; edge_index < G.E.size(); ++edge_index) {
    int vertex_index_1 = G.E[edge_index].e.first;
    int vertex_index_2 = G.E[edge_index].e.second;
    float delta_x = G.V[vertex_index_1].first - G.V[vertex_index_2].first;
    float delta_y = G.V[vertex_index_1].second - G.V[vertex_index_2].second;

    double distance_to_focus_point = 0;
    distance_to_focus_point += pow((G.V[vertex_index_1].first + G.V[vertex_index_2].first) / 2.0 - focus_x, 2.0);
    distance_to_focus_point += pow((G.V[vertex_index_1].second + G.V[vertex_index_2].second) / 2.0 - focus_y, 2.0);
    distance_to_focus_point = sqrt(distance_to_focus_point);

    // Normalize distance value to [0, 1]
    distance_to_focus_point /= sqrt(std::max(pow(focus_x, 2.0), pow(image.size().width - focus_x, 2.0)) + std::max(pow(focus_y, 2.0), pow(image.size().height - focus_y, 2.0)));

    double distance_weight = 1 - distance_to_focus_point;
    distance_weight = pow(distance_weight, 4.0);

    if (std::abs(delta_x) > std::abs(delta_y)) { // Horizontal
      expr += FOCUS_WEIGHT * distance_weight * IloPower((x[vertex_index_1 * 2] - x[vertex_index_2 * 2]) - max_mesh_scale * delta_x, 2);
    } else {
      expr += FOCUS_WEIGHT * distance_weight * IloPower((x[vertex_index_1 * 2 + 1] - x[vertex_index_2 * 2 + 1]) - max_mesh_scale * delta_y, 2);
    }
  }

  // Grid orientation & distortion constraint
  for (size_t edge_index = 0; edge_index < G.E.size(); ++edge_index) {
    int vertex_index_1 = G.E[edge_index].e.first;
    int vertex_index_2 = G.E[edge_index].e.second;
    float delta_x = G.V[vertex_index_1].first - G.V[vertex_index_2].first;
    float delta_y = G.V[vertex_index_1].second - G.V[vertex_index_2].second;
    if (std::abs(delta_x) > std::abs(delta_y)) { // Horizontal
      expr += ORIENTATION_WEIGHT * IloPower(x[vertex_index_1 * 2 + 1] - x[vertex_index_2 * 2 + 1], 2);
      expr += DISTORTION_WEIGHT * IloPower((x[vertex_index_1 * 2] - x[vertex_index_2 * 2]) - delta_x, 2);
    } else {
      expr += ORIENTATION_WEIGHT * IloPower(x[vertex_index_1 * 2] - x[vertex_index_2 * 2], 2);
      expr += DISTORTION_WEIGHT * IloPower((x[vertex_index_1 * 2 + 1] - x[vertex_index_2 * 2 + 1]) - delta_y, 2);
    }
  }


  IloModel model(env);

  model.add(IloMinimize(env, expr));

  IloRangeArray c(env);

  // Boundary constraint
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

  // Avoid flipping
  //for (int row = 0; row < mesh_row_count; ++row) {
  //  for (int column = 1; column < mesh_column_count; ++column) {
  //    int vertex_index_right = row * mesh_column_count + column;
  //    int vertex_index_left = row * mesh_column_count + column - 1;
  //    c.add((x[vertex_index_right * 2] - x[vertex_index_left * 2]) >= 1e-4);
  //  }
  //}

  //for (int row = 1; row < mesh_row_count; ++row) {
  //  for (int column = 0; column < mesh_column_count; ++column) {
  //    int vertex_index_down = row * mesh_column_count + column;
  //    int vertex_index_up = (row - 1) * mesh_column_count + column;
  //    c.add((x[vertex_index_down * 2 + 1] - x[vertex_index_up * 2 + 1]) >= 1e-4);
  //  }
  //}

  model.add(c);

  IloCplex cplex(model);

  cplex.setOut(env.getNullStream());

  if (!cplex.solve()) {
    puts("Failed to optimize the model.");
  }

  IloNumArray result(env);

  cplex.getValues(result, x);

  for (size_t vertex_index = 0; vertex_index < G.V.size(); ++vertex_index) {
    G.V[vertex_index].first = result[vertex_index * 2];
    G.V[vertex_index].second = result[vertex_index * 2 + 1];
  }

  model.end();
  cplex.end();
  env.end();
}

#endif