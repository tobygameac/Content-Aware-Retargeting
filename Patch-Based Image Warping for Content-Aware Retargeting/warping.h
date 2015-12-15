#pragma once

#define IL_STD

#include <ilcplex/ilocplex.h>
#include <ilconcert/iloexpression.h>

#include <vector>
#include <algorithm>
#include <set>
#include <map>

#include <opencv\cv.hpp>

#include "graph.h"

size_t Vec3bToValue(const cv::Vec3b &color) {
  return color.val[0] * 256 * 256 + color.val[1] * 256 + color.val[2];
}

double SignifanceColorToSaliencyValue(cv::Vec3b &signifance_color) {

  if (signifance_color[2] >= 255) {
    return (1.0 - (signifance_color[1] / 255.0)) / 3.0 + (2 / 3.0);
  }

  if (signifance_color[1] >= 255) {
    return (signifance_color[2] / 255.0) / 3.0 + (1 / 3.0);
  }

  return (signifance_color[1] / 255.0) / 3.0;
}

void ObjectPreservingVideoWarping(const std::string &segmentation_video_path, const std::string &significance_video_path, std::vector<Graph<glm::vec2> > &target_graphs, int target_video_width, const int target_video_height, const double mesh_width, const double mesh_height) {
  if (target_video_width <= 0 || target_video_height <= 0) {
    printf("Wrong target video size (%d x %d)\n", target_video_width, target_video_height);
    return;
  }

  if (!std::fstream(significance_video_path).good()) {
    std::cout << "Significance video file not found.\n";
    return;
  }

  const double DST_WEIGHT = 0.7;
  const double DLT_WEIGHT = 0.3;
  const double ORIENTATION_WEIGHT = 1.0;

  const double OBJECT_COHERENCE_WEIGHT = 0.7;
  const double LINE_COHERENCE_WEIGHT = 0.3;

  IloEnv env;

  IloNumVarArray x(env);
  IloExpr expr(env);

  IloRangeArray hard_constraint(env);

  for (size_t t = 0; t < target_graphs.size(); ++t) {
    for (size_t vertex_index = 0; vertex_index < target_graphs[t].vertices_.size(); ++vertex_index) {
      x.add(IloNumVar(env, -IloInfinity, IloInfinity));
      x.add(IloNumVar(env, -IloInfinity, IloInfinity));
    }
  }

  cv::VideoCapture segmentation_video_capture;
  segmentation_video_capture.open(segmentation_video_path);
  cv::VideoCapture significance_video_capture;
  significance_video_capture.open(significance_video_path);

  // Line coherence
  for (size_t t = 1; t < target_graphs.size(); ++t) {
    size_t variable_index_offset = t * target_graphs[t].vertices_.size() * 2;
    for (size_t vertex_index = 0; vertex_index < target_graphs[t].vertices_.size(); ++vertex_index) {
      expr += LINE_COHERENCE_WEIGHT * IloPower(x[vertex_index * 2 + variable_index_offset] - x[vertex_index * 2 - target_graphs[t].vertices_.size() * 2 + variable_index_offset], 2.0);
      expr += LINE_COHERENCE_WEIGHT * IloPower(x[vertex_index * 2 + 1 + variable_index_offset] - x[vertex_index * 2 + 1 - target_graphs[t].vertices_.size() * 2 + variable_index_offset], 2.0);
    }
  }

  std::map<size_t, Edge> object_representive_edge;
  std::map<size_t, double> object_saliency;

  struct VariableEdgeIndices {
    size_t x1, y1, x2, y2;
    bool is_horizontal;
  };

  std::map<size_t, std::vector<VariableEdgeIndices> > edge_variable_index_list_of_object_in_first_appear_frame;

  for (size_t t = 0; t < target_graphs.size(); ++t) {

    cv::Mat segmentation_video_frame;
    if (!segmentation_video_capture.read(segmentation_video_frame)) {
      break;
    }

    cv::Mat significance_video_frame;
    if (!significance_video_capture.read(significance_video_frame)) {
      break;
    }

    Graph<glm::vec2> &target_graph = target_graphs[t];

    const double WIDTH_RATIO = target_video_width / (double)segmentation_video_frame.size().width;
    const double HEIGHT_RATIO = target_video_height / (double)segmentation_video_frame.size().height;

    size_t mesh_column_count = (size_t)(segmentation_video_frame.size().width / mesh_width) + 1;
    size_t mesh_row_count = (size_t)(segmentation_video_frame.size().height / mesh_height) + 1;

    size_t variable_index_offset = t * target_graph.vertices_.size() * 2;

    // Boundary constraint
    for (size_t row = 0; row < mesh_row_count; ++row) {
      size_t vertex_index = row * mesh_column_count;
      hard_constraint.add(x[vertex_index * 2 + variable_index_offset] == target_graph.vertices_[0].x);

      vertex_index = row * mesh_column_count + mesh_column_count - 1;
      hard_constraint.add(x[vertex_index * 2 + variable_index_offset] == target_video_width);
    }

    for (size_t column = 0; column < mesh_column_count; ++column) {
      size_t vertex_index = column;
      hard_constraint.add(x[vertex_index * 2 + 1 + variable_index_offset] == target_graph.vertices_[0].y);

      vertex_index = (mesh_row_count - 1) * mesh_column_count + column;
      hard_constraint.add(x[vertex_index * 2 + 1 + variable_index_offset] == target_video_height);
    }

    // Avoid flipping
    for (size_t row = 0; row < mesh_row_count; ++row) {
      for (size_t column = 1; column < mesh_column_count; ++column) {
        size_t vertex_index_right = row * mesh_column_count + column;
        size_t vertex_index_left = row * mesh_column_count + column - 1;
        //hard_constraint.add((x[vertex_index_right * 2 + variable_index_offset] - x[vertex_index_left * 2 + variable_index_offset]) >= 1e-4);
      }
    }

    for (size_t row = 1; row < mesh_row_count; ++row) {
      for (size_t column = 0; column < mesh_column_count; ++column) {
        size_t vertex_index_down = row * mesh_column_count + column;
        size_t vertex_index_up = (row - 1) * mesh_column_count + column;
        //hard_constraint.add((x[vertex_index_down * 2 + 1 + variable_index_offset] - x[vertex_index_up * 2 + 1 + variable_index_offset]) >= 1e-4);
      }
    }

    // For the boundary pixel
    cv::resize(segmentation_video_frame, segmentation_video_frame, segmentation_video_frame.size() + cv::Size(1, 1));
    cv::resize(significance_video_frame, significance_video_frame, significance_video_frame.size() + cv::Size(1, 1));

    std::map<size_t, std::vector<size_t> > edge_index_list_of_object;

    for (size_t edge_index = 0; edge_index < target_graph.edges_.size(); ++edge_index) {
      size_t vertex_index1 = target_graph.edges_[edge_index].edge_indices_pair_.first;
      size_t vertex_index2 = target_graph.edges_[edge_index].edge_indices_pair_.second;
      size_t group_of_x = Vec3bToValue(segmentation_video_frame.at<cv::Vec3b>(target_graph.vertices_[vertex_index1].y, target_graph.vertices_[vertex_index1].x));
      size_t group_of_y = Vec3bToValue(segmentation_video_frame.at<cv::Vec3b>(target_graph.vertices_[vertex_index2].y, target_graph.vertices_[vertex_index2].x));
      if (group_of_x == group_of_y) {
        edge_index_list_of_object[group_of_x].push_back(edge_index);
      } else {
        edge_index_list_of_object[group_of_x].push_back(edge_index);
        edge_index_list_of_object[group_of_y].push_back(edge_index);
      }

      if (object_representive_edge.find(group_of_x) == object_representive_edge.end()) {
        object_representive_edge[group_of_x] = target_graph.edges_[edge_index];

        object_saliency[group_of_x] = SignifanceColorToSaliencyValue(significance_video_frame.at<cv::Vec3b>(target_graph.vertices_[object_representive_edge[group_of_x].edge_indices_pair_.first].y, target_graph.vertices_[object_representive_edge[group_of_x].edge_indices_pair_.first].x));
        object_saliency[group_of_x] += SignifanceColorToSaliencyValue(significance_video_frame.at<cv::Vec3b>(target_graph.vertices_[object_representive_edge[group_of_x].edge_indices_pair_.second].y, target_graph.vertices_[object_representive_edge[group_of_x].edge_indices_pair_.second].x));
        object_saliency[group_of_x] /= 2.0;
      }

      if (object_representive_edge.find(group_of_y) == object_representive_edge.end()) {
        object_representive_edge[group_of_y] = target_graph.edges_[edge_index];

        object_saliency[group_of_y] = SignifanceColorToSaliencyValue(significance_video_frame.at<cv::Vec3b>(target_graph.vertices_[object_representive_edge[group_of_y].edge_indices_pair_.first].y, target_graph.vertices_[object_representive_edge[group_of_y].edge_indices_pair_.first].x));
        object_saliency[group_of_y] += SignifanceColorToSaliencyValue(significance_video_frame.at<cv::Vec3b>(target_graph.vertices_[object_representive_edge[group_of_y].edge_indices_pair_.second].y, target_graph.vertices_[object_representive_edge[group_of_y].edge_indices_pair_.second].x));
        object_saliency[group_of_y] /= 2.0;
      }
    }

    // Average deformation
    for (auto edge_index_list_of_object_iterator = edge_index_list_of_object.begin(); edge_index_list_of_object_iterator != edge_index_list_of_object.end(); ++edge_index_list_of_object_iterator) {
      std::size_t object_index = edge_index_list_of_object_iterator->first;
      const std::vector<size_t> &edge_index_list = edge_index_list_of_object_iterator->second;

      if (edge_variable_index_list_of_object_in_first_appear_frame.find(object_index) == edge_variable_index_list_of_object_in_first_appear_frame.end()) {
        for (const auto &edge_index : edge_index_list) {
          const Edge &edge = target_graph.edges_[edge_index];

          size_t vertex_index1 = edge.edge_indices_pair_.first;
          size_t vertex_index2 = edge.edge_indices_pair_.second;

          VariableEdgeIndices variable_edge_indices;
          variable_edge_indices.x1 = vertex_index1 * 2 + variable_index_offset;
          variable_edge_indices.y1 = vertex_index1 * 2 + 1 + variable_index_offset;

          variable_edge_indices.x2 = vertex_index2 * 2 + variable_index_offset;
          variable_edge_indices.y2 = vertex_index2 * 2 + 1 + variable_index_offset;

          float delta_x = target_graph.vertices_[vertex_index1].x - target_graph.vertices_[vertex_index2].x;
          float delta_y = target_graph.vertices_[vertex_index1].y - target_graph.vertices_[vertex_index2].y;
          variable_edge_indices.is_horizontal = std::abs(delta_x) > std::abs(delta_y);

          edge_variable_index_list_of_object_in_first_appear_frame[object_index].push_back(variable_edge_indices);
        }
      }
    }

    for (auto edge_index_list_of_object_iterator = edge_index_list_of_object.begin(); edge_index_list_of_object_iterator != edge_index_list_of_object.end(); ++edge_index_list_of_object_iterator) {
      std::size_t object_index = edge_index_list_of_object_iterator->first;
      const std::vector<size_t> &edge_index_list = edge_index_list_of_object_iterator->second;

      for (const auto &edge_index : edge_index_list) {
        const Edge &edge = target_graph.edges_[edge_index];

        size_t vertex_index1 = edge.edge_indices_pair_.first;
        size_t vertex_index2 = edge.edge_indices_pair_.second;

        float delta_x = target_graph.vertices_[vertex_index1].x - target_graph.vertices_[vertex_index2].x;
        float delta_y = target_graph.vertices_[vertex_index1].y - target_graph.vertices_[vertex_index2].y;
        bool is_horizontal = std::abs(delta_x) > std::abs(delta_y);

        size_t edge_count = edge_variable_index_list_of_object_in_first_appear_frame[object_index].size();

        for (const VariableEdgeIndices &variable_edge_indices : edge_variable_index_list_of_object_in_first_appear_frame[object_index]) {
          if (is_horizontal == variable_edge_indices.is_horizontal) {
            expr += OBJECT_COHERENCE_WEIGHT * edge_count * 0.5 * IloPower((x[vertex_index2 * 2 + variable_index_offset] - x[vertex_index1 * 2 + variable_index_offset]) - (x[variable_edge_indices.x2] - x[variable_edge_indices.x1]), 2.0);
            expr += OBJECT_COHERENCE_WEIGHT * edge_count * 0.5 * IloPower((x[vertex_index2 * 2 + 1 + variable_index_offset] - x[vertex_index1 * 2 + 1 + variable_index_offset]) - (x[variable_edge_indices.y2] - x[variable_edge_indices.y1]), 2.0);
          }
        }
      }
    }

    // Object transformation constraint
    for (auto edge_index_list_of_object_iterator = edge_index_list_of_object.begin(); edge_index_list_of_object_iterator != edge_index_list_of_object.end(); ++edge_index_list_of_object_iterator) {

      std::size_t object_index = edge_index_list_of_object_iterator->first;
      const std::vector<size_t> &edge_index_list = edge_index_list_of_object_iterator->second;

      if (!edge_index_list.size()) {
        continue;
      }

      // Not global representive edge for each object
      //Edge representive_edge = target_graph.edges_[edge_index_list[0]];
      const Edge &representive_edge = object_representive_edge[object_index];

      double c_x = target_graph.vertices_[representive_edge.edge_indices_pair_.first].x - target_graph.vertices_[representive_edge.edge_indices_pair_.second].x;
      double c_y = target_graph.vertices_[representive_edge.edge_indices_pair_.first].y - target_graph.vertices_[representive_edge.edge_indices_pair_.second].y;

      double saliency_of_object = object_saliency[object_index];

      double original_matrix_a = c_x;
      double original_matrix_b = c_y;
      double original_matrix_c = c_y;
      double original_matrix_d = -c_x;

      double matrix_rank = original_matrix_a * original_matrix_d - original_matrix_b * original_matrix_c;

      if (fabs(matrix_rank) <= 1e-9) {
        matrix_rank = (matrix_rank > 0 ? 1 : -1) * 1e-9;
      }

      double matrix_a = original_matrix_d / matrix_rank;
      double matrix_b = -original_matrix_b / matrix_rank;
      double matrix_c = -original_matrix_c / matrix_rank;
      double matrix_d = original_matrix_a / matrix_rank;

      for (const auto &edge_index : edge_index_list) {
        const Edge &edge = target_graph.edges_[edge_index];
        double e_x = target_graph.vertices_[edge.edge_indices_pair_.first].x - target_graph.vertices_[edge.edge_indices_pair_.second].x;
        double e_y = target_graph.vertices_[edge.edge_indices_pair_.first].y - target_graph.vertices_[edge.edge_indices_pair_.second].y;

        double t_s = matrix_a * e_x + matrix_b * e_y;
        double t_r = matrix_c * e_x + matrix_d * e_y;

        // DST
        expr += DST_WEIGHT * saliency_of_object *
          IloPower((x[edge.edge_indices_pair_.first * 2 + variable_index_offset] - x[edge.edge_indices_pair_.second * 2 + variable_index_offset]) -
            (t_s * (x[representive_edge.edge_indices_pair_.first * 2 + variable_index_offset] - x[representive_edge.edge_indices_pair_.second * 2 + variable_index_offset]) + t_r * (x[representive_edge.edge_indices_pair_.first * 2 + 1 + variable_index_offset] - x[representive_edge.edge_indices_pair_.second * 2 + 1 + variable_index_offset])),
            2);
        expr += DST_WEIGHT * saliency_of_object *
          IloPower((x[edge.edge_indices_pair_.first * 2 + 1 + variable_index_offset] - x[edge.edge_indices_pair_.second * 2 + 1 + variable_index_offset]) -
            (-t_r * (x[representive_edge.edge_indices_pair_.first * 2 + variable_index_offset] - x[representive_edge.edge_indices_pair_.second * 2 + variable_index_offset]) + t_s * (x[representive_edge.edge_indices_pair_.first * 2 + 1 + variable_index_offset] - x[representive_edge.edge_indices_pair_.second * 2 + 1 + variable_index_offset])),
            2);

        // DLT
        expr += DLT_WEIGHT * (1 - saliency_of_object) *
          IloPower((x[edge.edge_indices_pair_.first * 2 + variable_index_offset] - x[edge.edge_indices_pair_.second * 2 + variable_index_offset]) -
            WIDTH_RATIO * (t_s * (x[representive_edge.edge_indices_pair_.first * 2 + variable_index_offset] - x[representive_edge.edge_indices_pair_.second * 2 + variable_index_offset]) + t_r * (x[representive_edge.edge_indices_pair_.first * 2 + 1 + variable_index_offset] - x[representive_edge.edge_indices_pair_.second * 2 + 1 + variable_index_offset])),
            2);
        expr += DLT_WEIGHT * (1 - saliency_of_object) *
          IloPower((x[edge.edge_indices_pair_.first * 2 + 1 + variable_index_offset] - x[edge.edge_indices_pair_.second * 2 + 1 + variable_index_offset]) -
            HEIGHT_RATIO * (-t_r *  (x[representive_edge.edge_indices_pair_.first * 2 + variable_index_offset] - x[representive_edge.edge_indices_pair_.second * 2 + variable_index_offset]) + t_s * (x[representive_edge.edge_indices_pair_.first * 2 + 1 + variable_index_offset] - x[representive_edge.edge_indices_pair_.second * 2 + 1 + variable_index_offset])),
            2);
      }
    }

    // Grid orientation constraint
    for (const auto &edge : target_graph.edges_) {
      size_t vertex_index_1 = edge.edge_indices_pair_.first;
      size_t vertex_index_2 = edge.edge_indices_pair_.second;
      float delta_x = target_graph.vertices_[vertex_index_1].x - target_graph.vertices_[vertex_index_2].x;
      float delta_y = target_graph.vertices_[vertex_index_1].y - target_graph.vertices_[vertex_index_2].y;
      if (std::abs(delta_x) > std::abs(delta_y)) { // Horizontal
        expr += ORIENTATION_WEIGHT * IloPower(x[vertex_index_1 * 2 + 1 + variable_index_offset] - x[vertex_index_2 * 2 + 1 + variable_index_offset], 2);
      } else {
        expr += ORIENTATION_WEIGHT * IloPower(x[vertex_index_1 * 2 + variable_index_offset] - x[vertex_index_2 * 2 + variable_index_offset], 2);
      }
    }
  }

  IloModel model(env);

  model.add(IloMinimize(env, expr));

  model.add(hard_constraint);

  IloCplex cplex(model);

  //cplex.setOut(env.getNullStream());

  if (!cplex.solve()) {
    puts("Failed to optimize the model.");
  }

  std::cout << "Done.\n";

  IloNumArray result(env);

  //cplex.getValues(result, x);

  for (size_t t = 0; t < target_graphs.size(); ++t) {
    Graph<glm::vec2> &target_graph = target_graphs[t];
    size_t variable_index_offset = t * target_graph.vertices_.size() * 2;
    for (size_t vertex_index = 0; vertex_index < target_graph.vertices_.size(); ++vertex_index) {
      //target_graph.vertices_[vertex_index].x = result[vertex_index * 2 + variable_index_offset];
      //target_graph.vertices_[vertex_index].y = result[vertex_index * 2 + 1 + variable_index_offset];

      //std::cout << t << " : (" << target_graph.vertices_[vertex_index].x << ", " << target_graph.vertices_[vertex_index].y << ") -> (" << cplex.getValue(x[vertex_index * 2 + variable_index_offset]) << ", " << cplex.getValue(x[vertex_index * 2 + 1 + variable_index_offset]) << ")\n";

      target_graph.vertices_[vertex_index].x = cplex.getValue(x[vertex_index * 2 + variable_index_offset]);
      target_graph.vertices_[vertex_index].y = cplex.getValue(x[vertex_index * 2 + 1 + variable_index_offset]);
    }
  }

  model.end();
  cplex.end();
  env.end();
}

void PatchBasedWarping(const cv::Mat &image, Graph<glm::vec2> &target_graph, const std::vector<std::vector<int> > &group_of_pixel, const std::vector<double> &saliency_of_patch, const int target_image_width, const int target_image_height, const double mesh_width, const double mesh_height) {
  if (target_image_width <= 0 || target_image_height <= 0) {
    printf("Wrong target image size (%d x %d)\n", target_image_width, target_image_height);
    return;
  }

  // Build the edge list of each patch
  std::vector<std::vector<size_t> > edge_index_list_of_patch(image.size().width * image.size().height);
  for (size_t edge_index = 0; edge_index < target_graph.edges_.size(); ++edge_index) {
    size_t x_vertex_index = target_graph.edges_[edge_index].edge_indices_pair_.first;
    size_t y_vertex_index = target_graph.edges_[edge_index].edge_indices_pair_.second;
    int group_of_x = group_of_pixel[target_graph.vertices_[x_vertex_index].y][target_graph.vertices_[x_vertex_index].x];
    int group_of_y = group_of_pixel[target_graph.vertices_[y_vertex_index].y][target_graph.vertices_[y_vertex_index].x];
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

  for (size_t vertex_index = 0; vertex_index < target_graph.vertices_.size(); ++vertex_index) {
    x.add(IloNumVar(env, -IloInfinity, IloInfinity));
    x.add(IloNumVar(env, -IloInfinity, IloInfinity));
  }

  const double DST_WEIGHT = 0.8;
  const double DLT_WEIGHT = 0.2;
  const double ORIENTATION_WEIGHT = 10.0;

  const double WIDTH_RATIO = target_image_width / (double)image.size().width;
  const double HEIGHT_RATIO = target_image_height / (double)image.size().height;

  // Patch transformation constraint
  for (size_t patch_index = 0; patch_index < edge_index_list_of_patch.size(); ++patch_index) {
    std::vector<size_t> &edge_index_list = edge_index_list_of_patch[patch_index];

    if (!edge_index_list.size()) {
      continue;
    }

    const Edge &representive_edge = target_graph.edges_[edge_index_list[0]];
    double c_x = target_graph.vertices_[representive_edge.edge_indices_pair_.first].x - target_graph.vertices_[representive_edge.edge_indices_pair_.second].x;
    double c_y = target_graph.vertices_[representive_edge.edge_indices_pair_.first].y - target_graph.vertices_[representive_edge.edge_indices_pair_.second].y;

    double original_matrix_a = c_x;
    double original_matrix_b = c_y;
    double original_matrix_c = c_y;
    double original_matrix_d = -c_x;

    double matrix_rank = original_matrix_a * original_matrix_d - original_matrix_b * original_matrix_c;

    if (fabs(matrix_rank) <= 1e-9) {
      matrix_rank = (matrix_rank > 0 ? 1 : -1) * 1e-9;
    }

    double matrix_a = original_matrix_d / matrix_rank;
    double matrix_b = -original_matrix_b / matrix_rank;
    double matrix_c = -original_matrix_c / matrix_rank;
    double matrix_d = original_matrix_a / matrix_rank;

    for (const auto &edge_index : edge_index_list) {
      const Edge &edge = target_graph.edges_[edge_index];
      double e_x = target_graph.vertices_[edge.edge_indices_pair_.first].x - target_graph.vertices_[edge.edge_indices_pair_.second].x;
      double e_y = target_graph.vertices_[edge.edge_indices_pair_.first].y - target_graph.vertices_[edge.edge_indices_pair_.second].y;

      double t_s = matrix_a * e_x + matrix_b * e_y;
      double t_r = matrix_c * e_x + matrix_d * e_y;

      // DST
      expr += DST_WEIGHT * saliency_of_patch[patch_index] *
        IloPower((x[edge.edge_indices_pair_.first * 2] - x[edge.edge_indices_pair_.second * 2]) -
          (t_s * (x[representive_edge.edge_indices_pair_.first * 2] - x[representive_edge.edge_indices_pair_.second * 2]) + t_r * (x[representive_edge.edge_indices_pair_.first * 2 + 1] - x[representive_edge.edge_indices_pair_.second * 2 + 1])),
          2);
      expr += DST_WEIGHT * saliency_of_patch[patch_index] *
        IloPower((x[edge.edge_indices_pair_.first * 2 + 1] - x[edge.edge_indices_pair_.second * 2 + 1]) -
          (-t_r * (x[representive_edge.edge_indices_pair_.first * 2] - x[representive_edge.edge_indices_pair_.second * 2]) + t_s * (x[representive_edge.edge_indices_pair_.first * 2 + 1] - x[representive_edge.edge_indices_pair_.second * 2 + 1])),
          2);

      // DLT
      expr += DLT_WEIGHT * (1 - saliency_of_patch[patch_index]) *
        IloPower((x[edge.edge_indices_pair_.first * 2] - x[edge.edge_indices_pair_.second * 2]) -
          WIDTH_RATIO * (t_s * (x[representive_edge.edge_indices_pair_.first * 2] - x[representive_edge.edge_indices_pair_.second * 2]) + t_r * (x[representive_edge.edge_indices_pair_.first * 2 + 1] - x[representive_edge.edge_indices_pair_.second * 2 + 1])),
          2);
      expr += DLT_WEIGHT * (1 - saliency_of_patch[patch_index]) *
        IloPower((x[edge.edge_indices_pair_.first * 2 + 1] - x[edge.edge_indices_pair_.second * 2 + 1]) -
          HEIGHT_RATIO * (-t_r *  (x[representive_edge.edge_indices_pair_.first * 2] - x[representive_edge.edge_indices_pair_.second * 2]) + t_s * (x[representive_edge.edge_indices_pair_.first * 2 + 1] - x[representive_edge.edge_indices_pair_.second * 2 + 1])),
          2);
    }
  }

  // Grid orientation constraint
  for (const auto &edge : target_graph.edges_) {
    int vertex_index_1 = edge.edge_indices_pair_.first;
    int vertex_index_2 = edge.edge_indices_pair_.second;
    float delta_x = target_graph.vertices_[vertex_index_1].x - target_graph.vertices_[vertex_index_2].x;
    float delta_y = target_graph.vertices_[vertex_index_1].y - target_graph.vertices_[vertex_index_2].y;
    if (std::abs(delta_x) > std::abs(delta_y)) { // Horizontal
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
  for (size_t row = 0; row < mesh_row_count; ++row) {
    size_t vertex_index = row * mesh_column_count;
    c.add(x[vertex_index * 2] == target_graph.vertices_[0].x);

    vertex_index = row * mesh_column_count + mesh_column_count - 1;
    c.add(x[vertex_index * 2] == target_image_width);
  }

  for (size_t column = 0; column < mesh_column_count; ++column) {
    size_t vertex_index = column;
    c.add(x[vertex_index * 2 + 1] == target_graph.vertices_[0].y);

    vertex_index = (mesh_row_count - 1) * mesh_column_count + column;
    c.add(x[vertex_index * 2 + 1] == target_image_height);
  }

  // Avoid flipping
  for (size_t row = 0; row < mesh_row_count; ++row) {
    for (size_t column = 1; column < mesh_column_count; ++column) {
      size_t vertex_index_right = row * mesh_column_count + column;
      size_t vertex_index_left = row * mesh_column_count + column - 1;
      c.add((x[vertex_index_right * 2] - x[vertex_index_left * 2]) >= 1e-4);
    }
  }

  for (int row = 1; row < mesh_row_count; ++row) {
    for (int column = 0; column < mesh_column_count; ++column) {
      size_t vertex_index_down = row * mesh_column_count + column;
      size_t vertex_index_up = (row - 1) * mesh_column_count + column;
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

  for (size_t vertex_index = 0; vertex_index < target_graph.vertices_.size(); ++vertex_index) {
    target_graph.vertices_[vertex_index].x = result[vertex_index * 2];
    target_graph.vertices_[vertex_index].y = result[vertex_index * 2 + 1];
  }

  model.end();
  cplex.end();
  env.end();
}

void FocusWarping(const cv::Mat &image, Graph<glm::vec2> &target_graph, const std::vector<std::vector<int> > &group_of_pixel, const std::vector<double> &saliency_of_patch, const int target_image_width, const int target_image_height, const double mesh_width, const double mesh_height, const double max_mesh_scale, const double focus_x, const double focus_y) {
  if (target_image_width <= 0 || target_image_height <= 0) {
    printf("Wrong target image size (%d x %d)\n", target_image_width, target_image_height);
    return;
  }

  IloEnv env;

  IloNumVarArray x(env);
  IloExpr expr(env);

  for (size_t vertex_index = 0; vertex_index < target_graph.vertices_.size(); ++vertex_index) {
    x.add(IloNumVar(env, -IloInfinity, IloInfinity));
    x.add(IloNumVar(env, -IloInfinity, IloInfinity));
  }

  int mesh_column_count = (int)(image.size().width / mesh_width) + 1;
  int mesh_row_count = (int)(image.size().height / mesh_height) + 1;

  const double FOCUS_WEIGHT = 5.0;
  const double ORIENTATION_WEIGHT = 10.0;
  const double DISTORTION_WEIGHT = 5.0;

  for (size_t edge_index = 0; edge_index < target_graph.edges_.size(); ++edge_index) {
    int vertex_index_1 = target_graph.edges_[edge_index].edge_indices_pair_.first;
    int vertex_index_2 = target_graph.edges_[edge_index].edge_indices_pair_.second;
    float delta_x = target_graph.vertices_[vertex_index_1].x - target_graph.vertices_[vertex_index_2].x;
    float delta_y = target_graph.vertices_[vertex_index_1].y - target_graph.vertices_[vertex_index_2].y;

    double distance_to_focus_point = 0;
    distance_to_focus_point += pow((target_graph.vertices_[vertex_index_1].x + target_graph.vertices_[vertex_index_2].x) / 2.0 - focus_x, 2.0);
    distance_to_focus_point += pow((target_graph.vertices_[vertex_index_1].y + target_graph.vertices_[vertex_index_2].y) / 2.0 - focus_y, 2.0);
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
  for (const auto &edge : target_graph.edges_) {
    int vertex_index_1 = edge.edge_indices_pair_.first;
    int vertex_index_2 = edge.edge_indices_pair_.second;
    float delta_x = target_graph.vertices_[vertex_index_1].x - target_graph.vertices_[vertex_index_2].x;
    float delta_y = target_graph.vertices_[vertex_index_1].y - target_graph.vertices_[vertex_index_2].y;
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
    c.add(x[vertex_index * 2] == target_graph.vertices_[0].x);

    vertex_index = row * mesh_column_count + mesh_column_count - 1;
    c.add(x[vertex_index * 2] == target_image_width);
  }

  for (int column = 0; column < mesh_column_count; ++column) {
    int vertex_index = column;
    c.add(x[vertex_index * 2 + 1] == target_graph.vertices_[0].y);

    vertex_index = (mesh_row_count - 1) * mesh_column_count + column;
    c.add(x[vertex_index * 2 + 1] == target_image_height);
  }

  // Avoid flipping
  //for (int row = 0; row < mesh_row_count; ++row) {
  //  for (int column = 1; column < mesh_column_count; ++column) {
  //    int vertex_index_right = row * mesh_column_count + column;
  //    int vertex_index_left = row * mesh_column_count + column - 1;
  //    hard_constraint.add((x[vertex_index_right * 2] - x[vertex_index_left * 2]) >= 1e-4);
  //  }
  //}

  //for (int row = 1; row < mesh_row_count; ++row) {
  //  for (int column = 0; column < mesh_column_count; ++column) {
  //    int vertex_index_down = row * mesh_column_count + column;
  //    int vertex_index_up = (row - 1) * mesh_column_count + column;
  //    hard_constraint.add((x[vertex_index_down * 2 + 1] - x[vertex_index_up * 2 + 1]) >= 1e-4);
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

  for (size_t vertex_index = 0; vertex_index < target_graph.vertices_.size(); ++vertex_index) {
    target_graph.vertices_[vertex_index].x = result[vertex_index * 2];
    target_graph.vertices_[vertex_index].y = result[vertex_index * 2 + 1];
  }

  model.end();
  cplex.end();
  env.end();
}