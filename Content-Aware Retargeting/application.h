#ifndef APPLICATION_H_
#define APPLICATION_H_

#include <windows.h>
#include <wingdi.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <vector>

#include "saliency.h"
#include "segmentation.h"
#include "warping.h"

class Application {
  typedef Graph2D<float> GraphType;
  typedef std::pair<float, float> FloatPair;

public:

  Application(std::string input_file_name) :
    window_name("(1) : image (2) : patche-based (3) : focus (`) : toggle / hidde mesh (+), (-) : adjust size of mesh (4)(5)(6) : other images") {
  }

  void Run();

private:

  void Reshape(GLFWwindow *window, int w, int h);
  void Keyboard(GLFWwindow *window, int key, int scancode, int action, int mods);
  void Mouse(GLFWwindow *window, int button, int action, int mods);
  void Motion(GLFWwindow *window, double x, double y);
  void Scroll(GLFWwindow *window, double x_offset, double y_offset);

  void RenderGL();
  void ChangeGLTexture(const cv::Mat &cv_image);
  void ChangeGLTexture(void *texture_pointer, int width, int height);
  void SaveScreen(const std::string &filename);
  void RecordScreen(const std::string &filename, const int fps);

  void ContentAwareRetargeting(const int target_image_width, const int target_image_height, const double mesh_width, const double mesh_height);
  cv::Vec3b SaliencyValueToSignifanceColor(double saliency_value);

  int GetMouseNearestVertexIndex(const std::vector<FloatPair> &vertex_list, const FloatPair &target, float &nearest_distance);

  void BuildQuadMeshWithGraph(GraphType &G, double mesh_width, double mesh_height);
  void BuildTriangleMesh();
  void ReBuildMesh();
  void DrawPolygonMesh(const std::vector<PolygonMesh<float> > &mesh_list, const std::vector<FloatPair> &vertex_list);
  void DrawImage();

  void ReadImage(const std::string &filename);
  void Initial();
  void Exit();

  enum ProgramMode {
    VIEWING_IMAGE,
    PATCH_BASED_WARPING,
    FOCUS_WARPING,
    VIEWING_TRIANGLE_MESH
  };

  ProgramMode program_mode;

  std::string input_file_name;
  const std::string window_name;

  int target_image_width;
  int target_image_height;

  const float MESH_LINE_WIDTH;
  const float MESH_POINT_SIZE;

  const int MIN_MESH_SIZE;
  const int MAX_MESH_SIZE;
  const int MESH_SIZE_GAP;
  int current_mesh_size;
  double focus_mesh_scale;
  double focus_x;
  double focus_y;

  bool is_viewing_mesh;
  bool is_viewing_mesh_point;

  std::vector<PolygonMesh<float> > quad_mesh_list;
  std::vector<FloatPair> quad_mesh_vertex_list;
  int selected_quad_mesh_vertex_index;

  std::vector<PolygonMesh<float> > triangle_mesh_list;
  std::vector<FloatPair> triangle_mesh_vertex_list;
  int selected_triangle_mesh_vertex_index;

  cv::Mat image;
  cv::Mat image_after_smoothing;
  cv::Mat image_after_segmentation;
  cv::Mat saliency_image;
  cv::Mat significance_image;

  cv::Mat image_for_gl_texture;

  GraphType image_graph;
  std::vector<std::vector<int> > group_of_pixel;
  std::vector<std::vector<double> > saliency_map;
  std::vector<double> saliency_of_patch;
  std::vector<double> saliency_of_mesh_vertex;

  bool data_for_image_warping_were_generated;
  bool is_recording_screen;

  std::vector<cv::Mat> recorded_images;
  GLFWwindow *window;

  float eye_x_offset;
  float eye_y_offset;
  float eye_z_offset;

  const float EYE_TRANSLATION_OFFSET_GAP;
};

void Application::BuildQuadMeshWithGraph(GraphType &G, double mesh_width, double mesh_height) {
  G = GraphType();

  int mesh_column_count = (int)(image.size().width / mesh_width) + 1;
  int mesh_row_count = (int)(image.size().height / mesh_height) + 1;

  float real_mesh_width = image.size().width / (float)(mesh_column_count - 1);
  float real_mesh_height = image.size().height / (float)(mesh_row_count - 1);

  quad_mesh_vertex_list.clear();

  for (int r = 0; r < mesh_row_count; ++r) {
    for (int c = 0; c < mesh_column_count; ++c) {
      quad_mesh_vertex_list.push_back(FloatPair(c * real_mesh_width, r * real_mesh_height));
      G.vertices_.push_back(quad_mesh_vertex_list.back());
    }
  }

  quad_mesh_list.clear();

  for (int r = 0; r < mesh_row_count - 1; ++r) {
    for (int c = 0; c < mesh_column_count - 1; ++c) {
      std::vector<int> vertex_index;
      std::vector<FloatPair> texture_coordinate;

      int base_index = r * (mesh_column_count)+c;
      vertex_index.push_back(base_index);
      vertex_index.push_back(base_index + mesh_column_count);
      vertex_index.push_back(base_index + mesh_column_count + 1);
      vertex_index.push_back(base_index + 1);

      if (!c) {
        G.edges_.push_back(Edge(std::pair<int, int>(vertex_index[0], vertex_index[1])));
      }
      G.edges_.push_back(Edge(std::pair<int, int>(vertex_index[1], vertex_index[2])));
      G.edges_.push_back(Edge(std::pair<int, int>(vertex_index[2], vertex_index[3])));
      if (!r) {
        G.edges_.push_back(Edge(std::pair<int, int>(vertex_index[3], vertex_index[0])));
      }

      for (const auto &index : vertex_index) {
        FloatPair mesh_vertex = quad_mesh_vertex_list[index];
        texture_coordinate.push_back(FloatPair(mesh_vertex.first / image.size().width, mesh_vertex.second / image.size().height));
      }

      quad_mesh_list.push_back(PolygonMesh<float>(vertex_index, texture_coordinate));
    }
  }
}

void Application::ReBuildMesh() {
  if (current_mesh_size < MIN_MESH_SIZE || current_mesh_size > MAX_MESH_SIZE) {
    current_mesh_size = std::max(MIN_MESH_SIZE, current_mesh_size);
    current_mesh_size = std::min(MAX_MESH_SIZE, current_mesh_size);
    return;
  }
  BuildTriangleMesh();

  ContentAwareRetargeting(target_image_width, target_image_height, current_mesh_size, current_mesh_size);
}

void Application::DrawPolygonMesh(const std::vector<PolygonMesh<float> > &mesh_list, const std::vector<FloatPair> &vertex_list) {
  for (const auto &mesh : mesh_list) {

    int vertex_count = mesh.vertex_index.size();

    if (is_viewing_mesh) {
      // line

      glLineWidth(MESH_LINE_WIDTH);
      glBegin(GL_LINE_STRIP);

      double vertex_saliency = saliency_of_mesh_vertex[mesh.vertex_index[0]];
      cv::Vec3b siginfance_color = SaliencyValueToSignifanceColor(vertex_saliency);
      glColor3f(siginfance_color[2] / 255.0, siginfance_color[1] / 255.0, siginfance_color[0] / 255.0);

      for (int j = 0; j < vertex_count + 1; ++j) {
        int vertex_index = mesh.vertex_index[j % vertex_count];
        glVertex3f(vertex_list[vertex_index].first, vertex_list[vertex_index].second, 0);
      }

      glEnd();

      if (is_viewing_mesh_point) {
        // point
        glColor4f(0.0, 0.0, 1.0, 1.0);

        glPointSize(MESH_POINT_SIZE);
        glBegin(GL_POINTS);

        for (int j = 0; j < vertex_count; ++j) {
          int vertex_index = mesh.vertex_index[j % vertex_count];
          glVertex3f(vertex_list[vertex_index].first, vertex_list[vertex_index].second, 0);
        }

        glEnd();
      }
    }

    // texture

    glEnable(GL_TEXTURE_2D);

    //glColor4f(1.0, 1.0, 1.0, is_viewing_mesh ? 0.75 : 1);
    glColor4f(1.0, 1.0, 1.0, 1.0);

    glBegin(GL_POLYGON);

    for (int j = 0; j < vertex_count; ++j) {
      FloatPair texture_coordinate = mesh.texture_coordinate[j];
      glTexCoord2f(texture_coordinate.first, texture_coordinate.second);

      int vertex_index = mesh.vertex_index[j % vertex_count];
      glVertex3f(vertex_list[vertex_index].first, vertex_list[vertex_index].second, 0);
    }

    glEnd();

    glDisable(GL_TEXTURE_2D);
  }
}

void Application::DrawImage() {
  glEnable(GL_TEXTURE_2D);

  glColor4f(1.0, 1.0, 1.0, 1.0);

  std::vector<FloatPair> vertex_list;
  vertex_list.push_back(FloatPair(0.0, 0.0));;
  vertex_list.push_back(FloatPair(0.0, image.size().height));
  vertex_list.push_back(FloatPair(image.size().width, image.size().height));
  vertex_list.push_back(FloatPair(image.size().width, 0.0));

  glBegin(GL_POLYGON);
  for (const auto &vertex : vertex_list) {
    glTexCoord2f(vertex.first / image.size().width, vertex.second / image.size().height);
    glVertex3f(vertex.first, vertex.second, 0);
  }
  glEnd();

  glDisable(GL_TEXTURE_2D);
}

int Application::GetMouseNearestVertexIndex(const std::vector<FloatPair> &vertex_list, const FloatPair &target, float &nearest_distance) {
  nearest_distance = 2e9;
  int selected_index = -1;
  int vertex_index = 0;
  for (const auto &vertex : vertex_list) {
    float distance = pow((vertex.first - target.first), 2) + pow((vertex.second - target.second), 2);
    if (distance < nearest_distance) {
      nearest_distance = distance;
      selected_index = vertex_index;
    }
    ++vertex_index;
  }
  return selected_index;
}

void Application::RenderGL() {
  //glClearColor(1.0, 1.0, 1.0, 1.0);
  //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  //glMatrixMode(GL_MODELVIEW);
  //glLoadIdentity();

  //int window_width, window_height;
  //glfwGetWindowSize(window, &window_width, &window_height);
  //gluLookAt(window_width / 2.0 + eye_x_offset, window_height / 2.0 + eye_y_offset, 0 + eye_z_offset + 1e-6,
  //  window_width / 2.0 + eye_x_offset, window_height / 2.0 + eye_y_offset, 0,
  //  0, 1, 0
  //  );

  //gluLookAt(0, 0, eye_z_offset + 1e-6,
  //  0, 0, 0,
  //  0, 1, 0
  //  );

  //switch (program_mode) {
  //case VIEWING_IMAGE:
  //  DrawImage();
  //  break;
  //case PATCH_BASED_WARPING:
  //case FOCUS_WARPING:
  //  DrawPolygonMesh(quad_mesh_list, quad_mesh_vertex_list);
  //  break;
  //case VIEWING_TRIANGLE_MESH:
  //  DrawPolygonMesh(triangle_mesh_list, triangle_mesh_vertex_list);
  //  break;
  //}

}

//void Application::Keyboard(GLFWwindow *window, int key, int scancode, int action, int mods) {
//  if (action == GLFW_PRESS) {
//
//    if (key == GLFW_KEY_1 || key == GLFW_KEY_2 || key == GLFW_KEY_3) {
//      ChangeGLTexture(image);
//    }
//    switch (key) {
//    case GLFW_KEY_F1:
//      printf("Input image file name : ");
//      char buffer[1 << 10];
//      scanf("%s", buffer);
//      input_file_name = std::string(buffer);
//      ReadImage(input_file_name);
//      ContentAwareRetargeting(target_image_width, target_image_height, current_mesh_size, current_mesh_size);
//      break;
//    case GLFW_KEY_GRAVE_ACCENT:
//      is_viewing_mesh = !is_viewing_mesh;
//      break;
//    case GLFW_KEY_1:
//      program_mode = VIEWING_IMAGE;
//      break;
//    case GLFW_KEY_2:
//      program_mode = PATCH_BASED_WARPING;
//      selected_quad_mesh_vertex_index = -1;
//      break;
//    case GLFW_KEY_3:
//      program_mode = FOCUS_WARPING;
//      selected_quad_mesh_vertex_index = -1;
//      break;
//    case GLFW_KEY_4:
//      program_mode = VIEWING_IMAGE;
//      ChangeGLTexture(image_after_segmentation);
//      break;
//    case GLFW_KEY_5:
//      program_mode = VIEWING_IMAGE;
//      ChangeGLTexture(saliency_image);
//      break;
//    case GLFW_KEY_6:
//      program_mode = VIEWING_IMAGE;
//      ChangeGLTexture(significance_image);
//      break;
//    case GLFW_KEY_7:
//      program_mode = VIEWING_TRIANGLE_MESH;
//      selected_triangle_mesh_vertex_index = -1;
//      break;
//    case GLFW_KEY_0:
//      is_viewing_mesh_point = !is_viewing_mesh_point;
//      break;
//    case GLFW_KEY_P:
//      SaveScreen("warping_" + input_file_name);
//      break;
//    case GLFW_KEY_R:
//      if (!is_recording_screen) {
//        recorded_images.clear();
//      } else {
//        RecordScreen("warping.avi", 30);
//      }
//      is_recording_screen = !is_recording_screen;
//      break;
//    case GLFW_KEY_W:
//      ContentAwareRetargeting(target_image_width, target_image_height, current_mesh_size, current_mesh_size);
//      break;
//    case GLFW_KEY_KP_ADD:
//      if (program_mode == PATCH_BASED_WARPING || program_mode == FOCUS_WARPING || program_mode == VIEWING_TRIANGLE_MESH) {
//        current_mesh_size += MESH_SIZE_GAP;
//        ReBuildMesh();
//      }
//      break;
//    case GLFW_KEY_KP_SUBTRACT:
//      if (program_mode == PATCH_BASED_WARPING || program_mode == FOCUS_WARPING || program_mode == VIEWING_TRIANGLE_MESH) {
//        current_mesh_size -= MESH_SIZE_GAP;
//        ReBuildMesh();
//      }
//      break;
//    case GLFW_KEY_UP:
//      eye_y_offset += EYE_TRANSLATION_OFFSET_GAP;
//      break;
//    case GLFW_KEY_DOWN:
//      eye_y_offset -= EYE_TRANSLATION_OFFSET_GAP;
//      break;
//    case GLFW_KEY_LEFT:
//      eye_x_offset -= EYE_TRANSLATION_OFFSET_GAP;
//      break;
//    case GLFW_KEY_RIGHT:
//      eye_x_offset += EYE_TRANSLATION_OFFSET_GAP;
//      break;
//    case GLFW_KEY_ESCAPE:
//      Exit();
//      break;
//    }
//
//    if (program_mode == PATCH_BASED_WARPING || program_mode == FOCUS_WARPING) {
//      Reshape(window, target_image_width, target_image_height);
//    } else {
//      Reshape(window, image.size().width, image.size().height);
//    }
//  }
//}

void Application::Reshape(GLFWwindow *window, int w, int h) {
  h = h < 1 ? 1 : h;

  glViewport(0, 0, w, h);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  //glOrtho(0, w, 0, h, 1.0, 10000.0);
  gluPerspective(45.0, w / (double)h, 1.0, 10000.0);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  char screen_size_str[20] = {};
  if (program_mode == PATCH_BASED_WARPING || program_mode == FOCUS_WARPING) {
    target_image_width = w;
    target_image_height = h;
  } else {
    w = image.size().width;
    h = image.size().height;
  }

  sprintf(screen_size_str, "(%04d x %04d)", w, h);
  glfwSetWindowSize(window, w, h);
  glfwSetWindowTitle(window, (std::string(screen_size_str) + window_name).c_str());
}

; void Application::Mouse(GLFWwindow *window, int button, int action, int mods) {
  if (action == GLFW_PRESS) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
      //float world_x = x;
      //float world_y = image.size().height - y;

      //FloatPair target(world_x, world_y);
      //float nearest_distance;
      //if (program_mode == PATCH_BASED_WARPING) {
      //  selected_quad_mesh_vertex_index = GetMouseNearestVertexIndex(quad_mesh_vertex_list, target, nearest_distance);
      //  if (nearest_distance > MESH_POINT_SIZE * 10) {
      //    selected_quad_mesh_vertex_index = -1;
      //  }
      //} else if (program_mode == VIEWING_TRIANGLE_MESH) {
      //  selected_triangle_mesh_vertex_index = GetMouseNearestVertexIndex(triangle_mesh_vertex_list, target, nearest_distance);
      //  if (nearest_distance > MESH_POINT_SIZE * 10) {
      //    selected_triangle_mesh_vertex_index = -1;
      //  }
      //}
    }
  }
}

void Application::Motion(GLFWwindow *window, double x, double y) {
  if (program_mode == FOCUS_WARPING) {
    double new_focus_x = x;
    double new_focus_y = image.size().height - y;
    if (std::abs(new_focus_x - focus_x) >= 1 || std::abs(new_focus_y - focus_y) >= 1) {
      ContentAwareRetargeting(target_image_width, target_image_height, current_mesh_size, current_mesh_size);
    }
    focus_x = new_focus_x;
    focus_y = new_focus_y;
  }

  //if (program_mode == PATCH_BASED_WARPING) {
  //  if (selected_quad_mesh_vertex_index != -1) {
  //    quad_mesh_vertex_list[selected_quad_mesh_vertex_index].first = x;
  //    quad_mesh_vertex_list[selected_quad_mesh_vertex_index].second = y;
  //  }
  //} else if (program_mode == VIEWING_TRIANGLE_MESH) {
  //  if (selected_triangle_mesh_vertex_index != -1) {
  //    triangle_mesh_vertex_list[selected_triangle_mesh_vertex_index].first = x;
  //    triangle_mesh_vertex_list[selected_triangle_mesh_vertex_index].second = y;
  //  }
  //}
}

void Application::Scroll(GLFWwindow *window, double x_offset, double y_offset) {
  if (program_mode == FOCUS_WARPING) {
    focus_mesh_scale -= y_offset * 0.25;
  } else {
    eye_z_offset -= y_offset * EYE_TRANSLATION_OFFSET_GAP;
  }
}

void Application::ChangeGLTexture(const cv::Mat &cv_image) {
  cv::cvtColor(cv_image, image_for_gl_texture, CV_BGR2RGB);
  cv::flip(image_for_gl_texture, image_for_gl_texture, 0);
  ChangeGLTexture(image_for_gl_texture.data, image.size().width, image.size().height);
}

void Application::ChangeGLTexture(void *texture_pointer, int width, int height) {
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_pointer);
}

void Application::ContentAwareRetargeting(const int target_image_width, const int target_image_height, const double mesh_width, const double mesh_height) {
  if (!data_for_image_warping_were_generated) {
    puts("Start : Gaussian smoothing");
    const double SMOOTH_SIGMA = 0.8;
    const cv::Size K_SIZE(3, 3);
    cv::GaussianBlur(image, image_after_smoothing, K_SIZE, SMOOTH_SIGMA);
    cv::imwrite("smooth_" + input_file_name, image_after_smoothing);
    puts("Done : Gaussian smoothing");

    puts("Start : Image segmentation");
    const double SEGMENTATION_K = (image.size().width + image.size().height) / 1.75;
    const double SEGMENTATION_MIN_PATCH_SIZE = (image.size().width * image.size().height) * 0.0001;
    const double SEGMENTATION_SIMILAR_COLOR_MERGE_THRESHOLD = 20;

    image_after_segmentation = Segmentation(image_after_smoothing, image_graph, group_of_pixel, SEGMENTATION_K, SEGMENTATION_MIN_PATCH_SIZE, SEGMENTATION_SIMILAR_COLOR_MERGE_THRESHOLD);
    cv::imwrite("segmentation_" + input_file_name, image_after_segmentation);
    puts("Done : Image segmentation");

    puts("Start : Image saliency calculation");
    const double SALIENCY_C = 3;
    const double SALIENCY_K = 64;
    saliency_image = CalculateContextAwareSaliencyMapWithMatlabProgram(image, saliency_map, "run_saliency.exe", input_file_name, "saliency_" + input_file_name);
    cv::imwrite("saliency_" + input_file_name, saliency_image);

    // Calculate the saliency value of each patch
    for (int r = 0; r < image.size().height; ++r) {
      saliency_map[r].push_back(saliency_map[r].back());
      group_of_pixel[r].push_back(group_of_pixel[r].back());
    }
    saliency_map.push_back(saliency_map.back());
    group_of_pixel.push_back(group_of_pixel.back());

    saliency_of_patch = std::vector<double>((image.size().width + 1) * (image.size().height + 1));
    std::vector<int> group_count((image.size().width + 1) * (image.size().height + 1));
    for (int r = 0; r < image.size().height + 1; ++r) {
      for (int c = 0; c < image.size().width + 1; ++c) {
        ++group_count[group_of_pixel[r][c]];
        saliency_of_patch[group_of_pixel[r][c]] += saliency_map[r][c];
      }
    }

    double min_saliency = 2e9, max_saliency = -2e9;
    for (size_t patch_index = 0; patch_index < saliency_of_patch.size(); ++patch_index) {
      if (group_count[patch_index]) {
        saliency_of_patch[patch_index] /= (double)group_count[patch_index];
      }
      min_saliency = std::min(min_saliency, saliency_of_patch[patch_index]);
      max_saliency = std::max(max_saliency, saliency_of_patch[patch_index]);
    }

    // Normalize saliency values
    for (size_t patch_index = 0; patch_index < saliency_of_patch.size(); ++patch_index) {
      saliency_of_patch[patch_index] = (saliency_of_patch[patch_index] - min_saliency) / (max_saliency - min_saliency);
    }

    significance_image = cv::Mat(image.size(), image.type());
    for (int r = 0; r < image.size().height; ++r) {
      for (int c = 0; c < image.size().width; ++c) {
        double vertex_saliency = saliency_of_patch[group_of_pixel[r][c]];
        significance_image.at<cv::Vec3b>(r, c) = SaliencyValueToSignifanceColor(vertex_saliency);
      }
    }
    cv::imwrite("significance_" + input_file_name, significance_image);

    puts("Done : Image saliency calculation");

    data_for_image_warping_were_generated = true;
  }

  puts("Start : Build mesh and graph");
  BuildQuadMeshWithGraph(image_graph, mesh_width, mesh_height);
  puts("Done : Build mesh and graph");

  if (program_mode == PATCH_BASED_WARPING) {
    puts("Start : Patch based warping");
    PatchBasedWarping(image, image_graph, group_of_pixel, saliency_of_patch, target_image_width, target_image_height, mesh_width, mesh_height);
    puts("Done : Patch based warping");
    printf("New image size : %d %d\n", target_image_width, target_image_height);
  } else if (program_mode == FOCUS_WARPING) {
    puts("Start : Focus warping");
    FocusWarping(image, image_graph, group_of_pixel, saliency_of_patch, target_image_width, target_image_height, mesh_width, mesh_height, focus_mesh_scale, focus_x, focus_y);
    puts("Done : Focus warping");
    printf("New image size : %d %d\n", target_image_width, target_image_height);
  }

  saliency_of_mesh_vertex.clear();
  saliency_of_mesh_vertex = std::vector<double>(image_graph.vertices_.size());

  for (size_t vertex_index = 0; vertex_index < image_graph.vertices_.size(); ++vertex_index) {
    float original_x = quad_mesh_vertex_list[vertex_index].first;
    float original_y = quad_mesh_vertex_list[vertex_index].second;
    saliency_of_mesh_vertex[vertex_index] = saliency_of_patch[group_of_pixel[original_y][original_x]];
    quad_mesh_vertex_list[vertex_index].first = image_graph.vertices_[vertex_index].first;
    quad_mesh_vertex_list[vertex_index].second = image_graph.vertices_[vertex_index].second;
  }

  selected_quad_mesh_vertex_index = -1;
  Reshape(window, target_image_width, target_image_height);

  double cotanget_of_half_of_fovy = 1.0 / tan(22.5 * acos(-1.0) / 180.0);
  eye_z_offset = cotanget_of_half_of_fovy * (target_image_height / 2.0);
}

cv::Vec3b Application::SaliencyValueToSignifanceColor(double saliency_value) {
  cv::Vec3b signifance_color(0, 0, 0);

  if (saliency_value > 1) {
    signifance_color[2] = 1;
  }

  if (saliency_value < 0) {
    signifance_color[0] = 1;
  }

  if (saliency_value < (1 / 3.0)) {
    signifance_color[1] = (saliency_value * 3.0) * 255;
    signifance_color[0] = (1 - saliency_value * 3.0) * 255;
  } else if (saliency_value < (2 / 3.0)) {
    signifance_color[2] = ((saliency_value - (1 / 3.0)) * 3.0) * 255;
    signifance_color[1] = 1.0 * 255;
  } else if (saliency_value <= 1) {
    signifance_color[2] = 1.0 * 255;
    signifance_color[1] = (1.0 - (saliency_value - (2 / 3.0)) * 3.0) * 255;
  }

  return signifance_color;
}

void Application::SaveScreen(const std::string &filename) {
  unsigned char *screen_image_data = new unsigned char[3 * target_image_width * target_image_height];
  glReadPixels(0, 0, target_image_width, target_image_height, GL_RGB, GL_UNSIGNED_BYTE, screen_image_data);
  cv::Mat screen_image(target_image_height, target_image_width, image.type(), screen_image_data);
  cv::cvtColor(screen_image, screen_image, CV_BGR2RGB);
  cv::flip(screen_image, screen_image, 0);
  cv::imwrite(filename, screen_image);
  delete[]screen_image_data;
  printf("Screen saved : %s\n", filename.c_str());
}

void Application::RecordScreen(const std::string &filename, const int fps) {
  if (!recorded_images.size()) {
    return;
  }
  cv::VideoWriter cv_video_writer;
  cv_video_writer.open(filename.c_str(), CV_FOURCC('M', 'J', 'P', 'G'), fps, recorded_images[0].size(), true);
  for (const auto &screen_image : recorded_images) {
    cv_video_writer.write(screen_image);
  }
  printf("Video saved : %s\n", filename.c_str());
}

void Application::Initial() {
}

void Application::ReadImage(const std::string &filename) {
  if (!std::fstream(filename).good()) {
    printf("image %s not found.\n", filename.c_str());
    return;
  }
  printf("Start : Read image %s\n", filename.c_str());
  image = cv::imread(filename);
  printf("Input image(%s) size : (%d x %d)\n", filename.c_str(), image.size().width, image.size().height);
  printf("Done : Read image %s\n", filename.c_str());

  Reshape(window, image.size().width, image.size().height);

  double cotanget_of_half_of_fovy = 1.0 / tan(22.5 * acos(-1.0) / 180.0);
  eye_z_offset = cotanget_of_half_of_fovy * (image.size().height / 2.0);

  ChangeGLTexture(image);

  focus_x = image.size().width / 2.0;
  focus_y = image.size().height / 2.0;

  BuildTriangleMesh();

  program_mode = VIEWING_IMAGE;

  target_image_width = image.size().width;
  target_image_height = image.size().height;

  data_for_image_warping_were_generated = false;
}

void Application::Run() {
  Initial();

  ReadImage(input_file_name);

  ContentAwareRetargeting(target_image_width, target_image_height, current_mesh_size, current_mesh_size);

  while (1) {
    RenderGL();
    if (is_recording_screen) {
      const std::string temp_frame_image_name = "warping_" + input_file_name;
      SaveScreen(temp_frame_image_name);
      recorded_images.push_back(cv::imread(temp_frame_image_name));
    }
  }

  Exit();
}

void Application::Exit() {
  exit(0);
}

#endif