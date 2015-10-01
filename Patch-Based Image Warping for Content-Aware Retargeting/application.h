#ifndef APPLICATION_H_
#define APPLICATION_H_

#include <windows.h>
#include <wingdi.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <functional>
#include <vector>

#include "polygon_mesh.h"
#include "saliency.h"
#include "segmentation.h"
#include "triangle.h"
#include "warping.h"

class Application {
  typedef Graph2D<float> GraphType;
  typedef std::pair<float, float> FloatPair;

public:

  Application(std::string input_file_name) :
    window_name("(1) : image (2) : patche-based (3) : focus (`) : toggle / hidde mesh (+), (-) : adjust size of mesh (4)(5)(6) : other images"),
    MESH_LINE_WIDTH(4.5),
    MESH_POINT_SIZE(7.5),

    MIN_MESH_SIZE(10),
    MAX_MESH_SIZE(120),
    MESH_SIZE_GAP(10),

    EYE_TRANSLATION_OFFSET_GAP(25.0),

    input_file_name(input_file_name),


    current_mesh_size(70),
    focus_mesh_scale(3.5),

    is_viewing_mesh(true),
    is_viewing_mesh_point(false),

    data_for_warping_were_generated(false),

    eye_x_offset(0),
    eye_y_offset(0),
    eye_z_offset(0) {
  }

  void Initial();
  void Run();
  void Exit();

private:

  void Reshape(GLFWwindow *window, int w, int h);
  void Keyboard(GLFWwindow *window, int key, int scancode, int action, int mods);
  void Mouse(GLFWwindow *window, int button, int action, int mods);
  void Motion(GLFWwindow *window, double x, double y);
  void Scroll(GLFWwindow *window, double x_offset, double y_offset);

  void PatchBasedImageWarpingForContentAwareRetargeting(const int target_image_width, const int target_image_height, const double mesh_width, const double mesh_height);
  void SaveScreen(const std::string &filename);

  void RenderGL();

  void ChangeGLTexture(void *texture_pointer, int width, int height);

  int GetMouseNearestVertexIndex(const std::vector<FloatPair> &vertex_list, const FloatPair &target, float &nearest_distance);

  void BuildQuadMeshWithGraph(GraphType &G, double mesh_width, double mesh_height);
  void BuildTriangleMesh();
  void ReBuildMesh();
  void DrawPolygonMesh(const std::vector<PolygonMesh<float> > &mesh_list, const std::vector<FloatPair> &vertex_list);
  void DrawImage();

  enum ProgramMode {
    VIEWING_IMAGE,
    PATCH_BASED_WARPING,
    FOCUS_WARPING,
    VIEWING_TRIANGLE_MESH
  };

  ProgramMode program_mode;

  std::string input_file_name;
  const std::string window_name;

  int target_image_width, target_image_height;

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

  bool data_for_warping_were_generated;

  GLFWwindow *window;

  float eye_x_offset;
  float eye_y_offset;
  float eye_z_offset;

  const float EYE_TRANSLATION_OFFSET_GAP;

};

void Application::BuildQuadMeshWithGraph(GraphType &G, double mesh_width, double mesh_height) {
  G.V.clear();
  G.E.clear();

  int mesh_column_count = (int)(image.size().width / mesh_width) + 1;
  int mesh_row_count = (int)(image.size().height / mesh_height) + 1;

  float real_mesh_width = image.size().width / (float)(mesh_column_count - 1);
  float real_mesh_height = image.size().height / (float)(mesh_row_count - 1);

  quad_mesh_vertex_list.clear();

  for (int r = 0; r < mesh_row_count; ++r) {
    for (int c = 0; c < mesh_column_count; ++c) {
      quad_mesh_vertex_list.push_back(FloatPair(c * real_mesh_width, r * real_mesh_height));
      G.V.push_back(quad_mesh_vertex_list.back());
    }
  }

  quad_mesh_list.clear();

  for (int r = 0; r < mesh_row_count - 1; ++r) {
    for (int c = 0; c < mesh_column_count - 1; ++c) {
      std::vector<int> vertex_index;
      std::vector<FloatPair> texture_coordinate;

      int base_index = r * (mesh_column_count) + c;
      vertex_index.push_back(base_index);
      vertex_index.push_back(base_index + mesh_column_count);
      vertex_index.push_back(base_index + mesh_column_count + 1);
      vertex_index.push_back(base_index + 1);

      if (!c) {
        G.E.push_back(Edge(std::pair<int, int>(vertex_index[0], vertex_index[1])));
      }
      G.E.push_back(Edge(std::pair<int, int>(vertex_index[1], vertex_index[2])));
      G.E.push_back(Edge(std::pair<int, int>(vertex_index[2], vertex_index[3])));
      if (!r) {
        G.E.push_back(Edge(std::pair<int, int>(vertex_index[3], vertex_index[0])));
      }

      for (const auto &index : vertex_index) {
        FloatPair mesh_vertex = quad_mesh_vertex_list[index];
        texture_coordinate.push_back(FloatPair(mesh_vertex.first / image.size().width, mesh_vertex.second / image.size().height));
      }

      quad_mesh_list.push_back(PolygonMesh<float>(vertex_index, texture_coordinate));
    }
  }
}

void Application::BuildTriangleMesh() {
  triangulateio in, mid, out, vorout;

  in.numberofpoints = 4;
  in.numberofpointattributes = 1;
  in.pointlist = (REAL *)malloc(in.numberofpoints * 2 * sizeof(REAL));
  in.pointlist[0] = 0;
  in.pointlist[1] = 0;

  in.pointlist[2] = 0;
  in.pointlist[3] = image.size().height;

  in.pointlist[4] = image.size().width;
  in.pointlist[5] = image.size().height;

  in.pointlist[6] = image.size().width;
  in.pointlist[7] = 0;

  in.pointattributelist = (REAL *)malloc(in.numberofpoints * in.numberofpointattributes * sizeof(REAL));
  in.pointattributelist[0] = 0.0;
  in.pointattributelist[1] = image.size().width;
  in.pointattributelist[2] = image.size().width + image.size().height;
  in.pointattributelist[3] = image.size().height;

  in.pointmarkerlist = (int *)malloc(in.numberofpoints * sizeof(int));
  in.pointmarkerlist[0] = 0;
  in.pointmarkerlist[1] = 2;
  in.pointmarkerlist[2] = 0;
  in.pointmarkerlist[3] = 0;

  in.numberofsegments = 0;

  in.numberofholes = 0;

  in.numberofregions = 1;
  in.regionlist = (REAL *)malloc(in.numberofregions * 4 * sizeof(REAL));
  in.regionlist[0] = 0.5;
  in.regionlist[1] = 5.0;
  in.regionlist[2] = 7.0;
  in.regionlist[3] = 0.1;

  mid.pointlist = (REAL *) NULL;            /* Not needed if -N switch used. */
  /* Not needed if -N switch used or number of point attributes is zero: */
  mid.pointattributelist = (REAL *) NULL;
  mid.pointmarkerlist = (int *) NULL; /* Not needed if -N or -B switch used. */
  mid.trianglelist = (int *) NULL;          /* Not needed if -E switch used. */
  /* Not needed if -E switch used or number of triangle attributes is zero: */
  mid.triangleattributelist = (REAL *) NULL;
  mid.neighborlist = (int *) NULL;         /* Needed only if -n switch used. */
  /* Needed only if segments are output (-p or -c) and -P not used: */
  mid.segmentlist = (int *) NULL;
  /* Needed only if segments are output (-p or -c) and -P and -B not used: */
  mid.segmentmarkerlist = (int *) NULL;
  mid.edgelist = (int *) NULL;             /* Needed only if -e switch used. */
  mid.edgemarkerlist = (int *) NULL;   /* Needed if -e used and -B not used. */

  vorout.pointlist = (REAL *) NULL;        /* Needed only if -v switch used. */
  /* Needed only if -v switch used and number of attributes is not zero: */
  vorout.pointattributelist = (REAL *) NULL;
  vorout.edgelist = (int *) NULL;          /* Needed only if -v switch used. */
  vorout.normlist = (REAL *) NULL;         /* Needed only if -v switch used. */

  triangulate("pczAevn -Q", &in, &mid, &vorout);

  mid.trianglearealist = (REAL *)malloc(mid.numberoftriangles * sizeof(REAL));
  mid.trianglearealist[0] = 3.0;
  mid.trianglearealist[1] = 1.0;

  out.pointlist = (REAL *)NULL;
  out.pointattributelist = (REAL *)NULL;
  out.trianglelist = (int *)NULL;
  out.triangleattributelist = (REAL *)NULL;

  char triangulate_instruction[100] = "";
  sprintf(triangulate_instruction, "pq30ra%dzBP -Q", current_mesh_size * 200);
  triangulate(triangulate_instruction, &mid, &out, NULL);

  selected_triangle_mesh_vertex_index = -1;

  triangle_mesh_vertex_list.clear();

  for (int i = 0; i < out.numberofpoints; ++i) {
    triangle_mesh_vertex_list.push_back(FloatPair(out.pointlist[i * 2], out.pointlist[i * 2 + 1]));
  }

  triangle_mesh_list.clear();

  for (int i = 0; i < out.numberoftriangles; ++i) {
    std::vector<int> vertex_index;
    std::vector<FloatPair> texture_coordinate;
    for (int j = 0; j < out.numberofcorners; ++j) {
      int target_vertex_index = out.trianglelist[i * out.numberofcorners + j];
      vertex_index.push_back(target_vertex_index);
      FloatPair mesh_vertex = triangle_mesh_vertex_list[target_vertex_index];
      texture_coordinate.push_back(FloatPair(mesh_vertex.first / image.size().width, mesh_vertex.second / image.size().height));
    }
    triangle_mesh_list.push_back(PolygonMesh<float>(vertex_index, texture_coordinate));
  }

  free(in.pointlist);
  free(in.pointattributelist);
  free(in.pointmarkerlist);
  free(in.regionlist);
  free(mid.pointlist);
  free(mid.pointattributelist);
  free(mid.pointmarkerlist);
  free(mid.trianglelist);
  free(mid.triangleattributelist);
  free(mid.trianglearealist);
  free(mid.neighborlist);
  free(mid.segmentlist);
  free(mid.segmentmarkerlist);
  free(mid.edgelist);
  free(mid.edgemarkerlist);
  free(vorout.pointlist);
  free(vorout.pointattributelist);
  free(vorout.edgelist);
  free(vorout.normlist);
  free(out.pointlist);
  free(out.pointattributelist);
  free(out.trianglelist);
  free(out.triangleattributelist);
}

void Application::ReBuildMesh() {
  if (current_mesh_size < MIN_MESH_SIZE || current_mesh_size > MAX_MESH_SIZE) {
    current_mesh_size = std::max(MIN_MESH_SIZE, current_mesh_size);
    current_mesh_size = std::min(MAX_MESH_SIZE, current_mesh_size);
    return;
  }
  BuildTriangleMesh();

  PatchBasedImageWarpingForContentAwareRetargeting(target_image_width, target_image_height, current_mesh_size, current_mesh_size);
}

void Application::DrawPolygonMesh(const std::vector<PolygonMesh<float> > &mesh_list, const std::vector<FloatPair> &vertex_list) {
  for (const auto &mesh : mesh_list) {

    int vertex_count = mesh.vertex_index.size();

    if (is_viewing_mesh) {
      // line

      glLineWidth(MESH_LINE_WIDTH);
      glBegin(GL_LINE_STRIP);

      double vertex_saliency = saliency_of_mesh_vertex[mesh.vertex_index[0]];
      if (vertex_saliency < (1 / 3.0)) {
        glColor3f(0, vertex_saliency * 3.0, 1 - vertex_saliency * 3.0);
      } else if (vertex_saliency < (2 / 3.0)) {
        glColor3f((vertex_saliency - (1 / 3.0)) * 3.0, 1.0, 0);
      } else {
        glColor3f(1.0, 1.0 - (vertex_saliency - (2 / 3.0)) * 3.0, 0);
      }
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

    glNormal3f(0, 0, 1);

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
  glNormal3f(0, 0, 1);
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
  glClearColor(1.0, 1.0, 1.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  int window_width, window_height;
  glfwGetWindowSize(window, &window_width, &window_height);
  gluLookAt(window_width / 2.0 + eye_x_offset, window_height / 2.0 + eye_y_offset, 0 + eye_z_offset + 1e-6,
    window_width / 2.0 + eye_x_offset, window_height / 2.0 + eye_y_offset, 0,
    0, 1, 0
    );

  //gluLookAt(0, 0, eye_z_offset + 1e-6,
  //  0, 0, 0,
  //  0, 1, 0
  //  );

  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_BLEND);

  switch (program_mode) {
  case VIEWING_IMAGE:
    DrawImage();
    break;
  case PATCH_BASED_WARPING:
  case FOCUS_WARPING:
    DrawPolygonMesh(quad_mesh_list, quad_mesh_vertex_list);
    break;
  case VIEWING_TRIANGLE_MESH:
    DrawPolygonMesh(triangle_mesh_list, triangle_mesh_vertex_list);
    break;
  }

  glfwSwapBuffers(window);
}

void Application::Keyboard(GLFWwindow *window, int key, int scancode, int action, int mods) {
  if (action == GLFW_PRESS) {
    if (key == GLFW_KEY_1 || key == GLFW_KEY_2 || key == GLFW_KEY_3) {
      cv::cvtColor(image, image_for_gl_texture, CV_BGR2RGB);
      cv::flip(image_for_gl_texture, image_for_gl_texture, 0);
      ChangeGLTexture(image_for_gl_texture.data, image.size().width, image.size().height);
    }
    switch (key) {
    case GLFW_KEY_GRAVE_ACCENT:
      is_viewing_mesh = !is_viewing_mesh;
      break;
    case GLFW_KEY_1: 
      program_mode = VIEWING_IMAGE;
      break;
    case GLFW_KEY_2: 
      program_mode = PATCH_BASED_WARPING;
      selected_quad_mesh_vertex_index = -1;
      break;
    case GLFW_KEY_3:
      program_mode = FOCUS_WARPING;
      selected_quad_mesh_vertex_index = -1;
      break;
    case GLFW_KEY_4:
      program_mode = VIEWING_IMAGE;
      cv::cvtColor(image_after_segmentation, image_for_gl_texture, CV_BGR2RGB);
      cv::flip(image_for_gl_texture, image_for_gl_texture, 0);
      ChangeGLTexture(image_for_gl_texture.data, image.size().width, image.size().height);
      break;
    case GLFW_KEY_5:
      program_mode = VIEWING_IMAGE;
      cv::cvtColor(saliency_image, image_for_gl_texture, CV_BGR2RGB);
      cv::flip(image_for_gl_texture, image_for_gl_texture, 0);
      ChangeGLTexture(image_for_gl_texture.data, image.size().width, image.size().height);
      break;
    case GLFW_KEY_6:
      program_mode = VIEWING_IMAGE;
      cv::cvtColor(significance_image, image_for_gl_texture, CV_BGR2RGB);
      cv::flip(image_for_gl_texture, image_for_gl_texture, 0);
      ChangeGLTexture(image_for_gl_texture.data, image.size().width, image.size().height);
      break;
    case GLFW_KEY_7: 
      program_mode = VIEWING_TRIANGLE_MESH;
      selected_triangle_mesh_vertex_index = -1;
      break;
    case GLFW_KEY_0: 
      is_viewing_mesh_point = !is_viewing_mesh_point;
      break;
    case GLFW_KEY_P:
      SaveScreen("warping_" + input_file_name);
      break;
    case GLFW_KEY_W:
      PatchBasedImageWarpingForContentAwareRetargeting(target_image_width, target_image_height, current_mesh_size, current_mesh_size);
      break;
    case GLFW_KEY_KP_ADD: 
      if (program_mode == PATCH_BASED_WARPING || program_mode == FOCUS_WARPING || program_mode == VIEWING_TRIANGLE_MESH) {
        current_mesh_size += MESH_SIZE_GAP;
        ReBuildMesh();
      }
      break;
    case GLFW_KEY_KP_SUBTRACT:
      if (program_mode == PATCH_BASED_WARPING || program_mode == FOCUS_WARPING || program_mode == VIEWING_TRIANGLE_MESH) {
        current_mesh_size -= MESH_SIZE_GAP;
        ReBuildMesh();
      }
      break;
    case GLFW_KEY_UP:
      eye_y_offset += EYE_TRANSLATION_OFFSET_GAP;
      break;
    case GLFW_KEY_DOWN:
      eye_y_offset -= EYE_TRANSLATION_OFFSET_GAP;
      break;
    case GLFW_KEY_LEFT:
      eye_x_offset -= EYE_TRANSLATION_OFFSET_GAP;
      break;
    case GLFW_KEY_RIGHT:
      eye_x_offset += EYE_TRANSLATION_OFFSET_GAP;
      break;
    case GLFW_KEY_ESCAPE:
      Exit();
      break;
    }

    if (program_mode == PATCH_BASED_WARPING || program_mode == FOCUS_WARPING) {
      Reshape(window, target_image_width, target_image_height);
    } else {
      Reshape(window, image.size().width, image.size().height);
    }
  }
}

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

void Application::Mouse(GLFWwindow *window, int button, int action, int mods) {
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
    if (std::abs(new_focus_x - focus_x) >= 1e-3 || std::abs(new_focus_y - focus_y) >= 1e-3) {
      PatchBasedImageWarpingForContentAwareRetargeting(target_image_width, target_image_height, current_mesh_size, current_mesh_size);
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

void Application::ChangeGLTexture(void *texture_pointer, int width, int height) {
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, texture_pointer);
}

void Application::PatchBasedImageWarpingForContentAwareRetargeting(const int target_image_width, const int target_image_height, const double mesh_width, const double mesh_height) {
  if (!data_for_warping_were_generated) {
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

    saliency_of_patch.clear();
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
        if (vertex_saliency < (1 / 3.0)) {
          significance_image.at<cv::Vec3b>(r, c).val[1] = (vertex_saliency * 3.0) * 255;
          significance_image.at<cv::Vec3b>(r, c).val[0] = (1 - vertex_saliency * 3.0) * 255;
        } else if (vertex_saliency < (2 / 3.0)) {
          significance_image.at<cv::Vec3b>(r, c).val[2] = ((vertex_saliency - (1 / 3.0)) * 3.0) * 255;
          significance_image.at<cv::Vec3b>(r, c).val[1] = 1.0 * 255;
        } else {
          significance_image.at<cv::Vec3b>(r, c).val[2] = 1.0 * 255;
          significance_image.at<cv::Vec3b>(r, c).val[1] = (1.0 - (vertex_saliency - (2 / 3.0)) * 3.0) * 255;
        }
      }
    }
    cv::imwrite("significance_" + input_file_name, significance_image);

    puts("Done : Image saliency calculation");

    data_for_warping_were_generated = true;
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
  saliency_of_mesh_vertex = std::vector<double>(image_graph.V.size());

  for (size_t vertex_index = 0; vertex_index < image_graph.V.size(); ++vertex_index) {
    float original_x = quad_mesh_vertex_list[vertex_index].first;
    float original_y = quad_mesh_vertex_list[vertex_index].second;
    saliency_of_mesh_vertex[vertex_index] = saliency_of_patch[group_of_pixel[original_y][original_x]];
    quad_mesh_vertex_list[vertex_index].first = image_graph.V[vertex_index].first;
    quad_mesh_vertex_list[vertex_index].second = image_graph.V[vertex_index].second;
  }

  selected_quad_mesh_vertex_index = -1;
  Reshape(window, target_image_width, target_image_height);

  double cotanget_of_half_of_fovy = 1.0 / tan(22.5 * acos(-1.0) / 180.0);
  eye_z_offset = cotanget_of_half_of_fovy * (target_image_height / 2.0);
}

void Application::SaveScreen(const std::string &filename) {
  printf("Screen saved : %s\n", filename);
  unsigned char *image_after_warping_data = new unsigned char[3 * target_image_width * target_image_height];
  glReadPixels(0, 0, target_image_width, target_image_height, GL_RGB, GL_UNSIGNED_BYTE, image_after_warping_data);
  cv::Mat image_after_warping(target_image_height, target_image_width, image.type(), image_after_warping_data);
  cv::cvtColor(image_after_warping, image_after_warping, CV_BGR2RGB);
  cv::flip(image_after_warping, image_after_warping, 0);
  cv::imwrite(filename, image_after_warping);
  delete []image_after_warping_data;
}

void Application::Initial() {
  if (!glfwInit ()) {
    exit(EXIT_FAILURE);
  }

  printf("Start : Read image %s\n", input_file_name.c_str());
  image = cv::imread(input_file_name);
  printf("Input image(%s) size : (%d x %d)\n", input_file_name.c_str(), image.size().width, image.size().height);
  printf("Done : Read image %s\n", input_file_name.c_str());

  window = glfwCreateWindow(image.size().width, image.size().height, window_name.c_str(), NULL, NULL);
  if (!window) {
    glfwTerminate();
    exit(EXIT_FAILURE);
  }

  glewExperimental = GL_TRUE;
  glewInit();

  BuildTriangleMesh();

  glfwSetWindowUserPointer(window, this);

  auto reshape_callback = [](GLFWwindow *window, int w, int h) {
    static_cast<Application *>(glfwGetWindowUserPointer(window))->Reshape(window, w, h);
  };

  auto keyboard_callback = [](GLFWwindow *window, int key, int scancode, int action, int mods) {
    static_cast<Application *>(glfwGetWindowUserPointer(window))->Keyboard(window, key, scancode, action, mods);
  };

  auto mouse_callback = [](GLFWwindow *window, int button, int action, int mods) {
    static_cast<Application *>(glfwGetWindowUserPointer(window))->Mouse(window, button, action, mods);
  };

  auto motion_callback = [](GLFWwindow *window, double x, double y) {
    static_cast<Application *>(glfwGetWindowUserPointer(window))->Motion(window, x, y);
  };

  auto scroll_callback = [](GLFWwindow *window, double x_offset, double y_offset) {
    static_cast<Application *>(glfwGetWindowUserPointer(window))->Scroll(window, x_offset, y_offset);
  };


  glfwSetFramebufferSizeCallback(window, reshape_callback);
  glfwSetKeyCallback(window, keyboard_callback);
  glfwSetMouseButtonCallback(window, mouse_callback);
  glfwSetCursorPosCallback(window, motion_callback);
  glfwSetScrollCallback(window, scroll_callback);

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  double cotanget_of_half_of_fovy = 1.0 / tan(22.5 * acos(-1.0) / 180.0);
  eye_z_offset = cotanget_of_half_of_fovy * (image.size().height / 2.0);

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  cv::cvtColor(image, image_for_gl_texture, CV_BGR2RGB);
  cv::flip(image_for_gl_texture, image_for_gl_texture, 0);
  ChangeGLTexture(image_for_gl_texture.data, image.size().width, image.size().height);

  focus_x = image.size().width / 2.0;
  focus_y = image.size().height / 2.0;

  program_mode = VIEWING_IMAGE;
}

void Application::Run() {
  target_image_width = image.size().width;
  target_image_height = image.size().height;
  PatchBasedImageWarpingForContentAwareRetargeting(target_image_width, target_image_height, current_mesh_size, current_mesh_size);

  while (!glfwWindowShouldClose(window)) {
    RenderGL();
    glfwPollEvents();
  }
}

void Application::Exit() {
  glfwTerminate();
  exit(0);
}

#endif