#define REAL double

#include <windows.h>
#include <wingdi.h>

#include <GL/glut.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <vector>

#include "bmp_reader.h"
#include "image.h"
#include "polygon_mesh.h"
#include "saliency.h"
#include "segmentation.h"
#include "smooth.h"
#include "triangle.h"
#include "warping.h"

typedef Image<unsigned char> ImageType;
typedef Graph2D<int> GraphType;

typedef std::pair<float, float> FloatPair;

char *input_file_name = "input3.bmp";
const char *window_name = "Mesh Generator (1) : image (2) : quad mesh (3) : triangle mesh (+), (-) : adjust size of mesh (4) : display / hide mesh";

int target_image_width, target_image_height;

void Initial();
void PatchBasedImageWarpingForContentAwareRetargeting(const int target_image_width, const int target_image_height, const double mesh_width, const double mesh_height);
void StartOpenGL();
void Exit();

void Display();
void Reshape(int w, int h);
void Keyboard(unsigned char key, int x, int y);
void Mouse(int button, int state, int x, int y);
void Motion(int x, int y);

void ChangeGLTexture(void *texture_pointer, int width, int height);

int GetMouseNearestVertexIndex(std::vector<FloatPair> &vertex_list, FloatPair &target, float &nearest_distance);

void BuildQuadMesh();
void BuildQuadMeshWithGraph(GraphType &G, double mesh_width, double mesh_height);
void BuildTriangleMesh();
void ReBuildMesh();
void DrawPolygonMesh(std::vector<PolygonMesh<float> > &mesh_list, std::vector<FloatPair> &vertex_list);
void DrawImage();

enum ProgramMode {
  VIEWING_IMAGE,
  VIEWING_QUAD_MESH,
  VIEWING_TRIANGLE_MESH
};

ProgramMode program_mode;

const float MESH_LINE_WIDTH = 1.5;
const float MESH_POINT_SIZE = 7.5;

const int MIN_MESH_SIZE = 10;
const int MAX_MESH_SIZE = 120;
const int MESH_SIZE_GAP = 5;
int current_mesh_size = 70;

bool is_viewing_mesh = true;
bool is_viewing_mesh_point = false;

std::vector<PolygonMesh<float> > quad_mesh_list;
std::vector<FloatPair> quad_mesh_vertex_list;
int selected_quad_mesh_vertex_index;

std::vector<PolygonMesh<float> > triangle_mesh_list;
std::vector<FloatPair> triangle_mesh_vertex_list;
int selected_triangle_mesh_vertex_index;

ImageType image;

int window_number;

float eye_z_translation = 0;

int main(int argc, char **argv) {

  if (argc == 2) {
    strcpy(input_file_name, argv[1]);
  }

  Initial();

  printf("Input image(%s) size : (%d x %d)\n", input_file_name, image.width, image.height);
  printf("Please input target image width : ");
  scanf("%d", &target_image_width);
  printf("Please input target image height : ");
  scanf("%d", &target_image_height);
  printf("Warping image from (%d x %d) to (%d x %d)\n", image.width, image.height, target_image_width, target_image_height);

  PatchBasedImageWarpingForContentAwareRetargeting(target_image_width, target_image_height, 10, 10);

  StartOpenGL();
  Exit();

  return 0;
}

void BuildQuadMesh() {
  int mesh_column_count = image.width / current_mesh_size;
  int mesh_row_count = image.height / current_mesh_size;

  float real_mesh_width = image.width / (float)mesh_column_count;
  float real_mesh_height = image.height / (float)mesh_column_count;

  quad_mesh_vertex_list.clear();

  for (int r = 0; r < mesh_row_count + 1; ++r) {
    for (int c = 0; c < mesh_column_count + 1; ++c) {
      quad_mesh_vertex_list.push_back(FloatPair(c * real_mesh_width, r * real_mesh_height));
    }
  }

  quad_mesh_list.clear();

  for (int r = 0; r < mesh_row_count; ++r) {
    for (int c = 0; c < mesh_column_count; ++c) {
      std::vector<int> vertex_index;
      std::vector<FloatPair> texture_coordinate;

      int base_index = r * (mesh_column_count + 1) + c;
      vertex_index.push_back(base_index);
      vertex_index.push_back(base_index + mesh_column_count + 1);
      vertex_index.push_back(base_index + mesh_column_count + 2);
      vertex_index.push_back(base_index + 1);

      for (auto it = vertex_index.begin(); it != vertex_index.end(); ++it) {
        FloatPair mesh_vertex = quad_mesh_vertex_list[*it];
        texture_coordinate.push_back(FloatPair(mesh_vertex.first / image.width, mesh_vertex.second / image.height));
      }

      quad_mesh_list.push_back(PolygonMesh<float>(vertex_index, texture_coordinate));
    }
  }
}

void BuildQuadMeshWithGraph(GraphType &G, double mesh_width, double mesh_height) {
  G.V.clear();
  G.E.clear();

  int mesh_column_count = image.width / mesh_width;
  int mesh_row_count = image.height / mesh_height;

  float real_mesh_width = image.width / (float)mesh_column_count;
  float real_mesh_height = image.height / (float)mesh_row_count;

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

      G.E.push_back(Edge(std::pair<int, int>(base_index, base_index + mesh_column_count)));
      G.E.push_back(Edge(std::pair<int, int>(base_index + mesh_column_count, base_index + mesh_column_count + 1)));
      G.E.push_back(Edge(std::pair<int, int>(base_index + mesh_column_count + 1, base_index + 1)));
      G.E.push_back(Edge(std::pair<int, int>(base_index, base_index + 1)));

      for (auto it = vertex_index.begin(); it != vertex_index.end(); ++it) {
        FloatPair mesh_vertex = quad_mesh_vertex_list[*it];
        texture_coordinate.push_back(FloatPair(mesh_vertex.first / image.width, mesh_vertex.second / image.height));
      }

      quad_mesh_list.push_back(PolygonMesh<float>(vertex_index, texture_coordinate));
    }
  }
}

void BuildTriangleMesh() {
  triangulateio in, mid, out, vorout;

  in.numberofpoints = 4;
  in.numberofpointattributes = 1;
  in.pointlist = (REAL *)malloc(in.numberofpoints * 2 * sizeof(REAL));
  in.pointlist[0] = 0;
  in.pointlist[1] = 0;

  in.pointlist[2] = 0;
  in.pointlist[3] = image.height;

  in.pointlist[4] = image.width;
  in.pointlist[5] = image.height;

  in.pointlist[6] = image.width;
  in.pointlist[7] = 0;

  in.pointattributelist = (REAL *)malloc(in.numberofpoints * in.numberofpointattributes * sizeof(REAL));
  in.pointattributelist[0] = 0.0;
  in.pointattributelist[1] = image.width;
  in.pointattributelist[2] = image.width + image.height;
  in.pointattributelist[3] = image.height;

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
      texture_coordinate.push_back(FloatPair(mesh_vertex.first / image.width, mesh_vertex.second / image.height));
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

void ReBuildMesh() {
  if (current_mesh_size < MIN_MESH_SIZE || current_mesh_size > MAX_MESH_SIZE) {
    current_mesh_size = max(MIN_MESH_SIZE, current_mesh_size);
    current_mesh_size = min(MAX_MESH_SIZE, current_mesh_size);
    return;
  }
  BuildQuadMesh();
  BuildTriangleMesh();
}

void DrawPolygonMesh(std::vector<PolygonMesh<float> > &mesh_list, std::vector<FloatPair> &vertex_list) {
  for (auto it = mesh_list.begin(); it != mesh_list.end(); ++it) {

    int vertex_count = it->vertex_index.size();

    if (is_viewing_mesh) {
      // line
      glColor4f(1.0, 0.0, 0.0, 1.0);

      glLineWidth(MESH_LINE_WIDTH);
      glBegin(GL_LINE_STRIP);

      for (int j = 0; j < vertex_count + 1; ++j) {
        int vertex_index = it->vertex_index[j % vertex_count];
        glVertex3f(vertex_list[vertex_index].first, vertex_list[vertex_index].second, 0);
      }

      glEnd();

      if (is_viewing_mesh_point) {
        // point
        glColor4f(0.0, 0.0, 1.0, 1.0);

        glPointSize(MESH_POINT_SIZE);
        glBegin(GL_POINTS);

        for (int j = 0; j < vertex_count; ++j) {
          int vertex_index = it->vertex_index[j % vertex_count];
          glVertex3f(vertex_list[vertex_index].first, vertex_list[vertex_index].second, 0);
        }

        glEnd();
      }
    }

    // texture

    glEnable(GL_TEXTURE_2D);

    glColor4f(1.0, 1.0, 1.0, is_viewing_mesh ? 0.75 : 1);

    glBegin(GL_POLYGON);

    glNormal3f(0, 0, 1);

    for (int j = 0; j < vertex_count; ++j) {
      FloatPair texture_coordinate = it->texture_coordinate[j];
      glTexCoord2f(texture_coordinate.first, texture_coordinate.second);

      int vertex_index = it->vertex_index[j % vertex_count];
      glVertex3f(vertex_list[vertex_index].first, vertex_list[vertex_index].second, 0);
    }

    glEnd();

    glDisable(GL_TEXTURE_2D);
  }
}

void DrawImage() {
  glEnable(GL_TEXTURE_2D);

  glColor4f(1.0, 1.0, 1.0, 1.0);

  std::vector<FloatPair> vertex_list;
  vertex_list.push_back(FloatPair(0.0, 0.0));
  vertex_list.push_back(FloatPair(0.0, image.height));
  vertex_list.push_back(FloatPair(image.width, image.height));
  vertex_list.push_back(FloatPair(image.width, 0.0));

  glBegin(GL_POLYGON);
  glNormal3f(0, 0, 1);
  for (auto it = vertex_list.begin(); it != vertex_list.end(); ++it) {
    glTexCoord2f(it->first / image.width, it->second / image.height);
    glVertex3f(it->first, it->second, 0);
  }
  glEnd();

  glDisable(GL_TEXTURE_2D);
}

int GetMouseNearestVertexIndex(std::vector<FloatPair> &vertex_list, FloatPair &target, float &nearest_distance) {
  nearest_distance = 2e9;
  int selected_index = -1;
  for (auto it = vertex_list.begin(); it != vertex_list.end(); ++it) {
    float distance = pow((it->first - target.first), 2) + pow((it->second - target.second), 2);
    if (distance < nearest_distance) {
      nearest_distance = distance;
      selected_index = it - vertex_list.begin();
    }
  }
  return selected_index;
}

void Display() {
  glClearColor(1.0, 1.0, 1.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  /*
  gluLookAt(image.width / 2.0, image.height / 2.0, eye_z_translation + 1e-6,
  image.width / 2.0, image.height / 2.0, 0,
  0, 1, 0
  );
  */
  gluLookAt(0, 0, eye_z_translation + 1e-6,
    0, 0, 0,
    0, 1, 0
    );

  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_BLEND);

  switch (program_mode) {
  case VIEWING_IMAGE:
    DrawImage();
    break;
  case VIEWING_QUAD_MESH:
    DrawPolygonMesh(quad_mesh_list, quad_mesh_vertex_list);
    break;
  case VIEWING_TRIANGLE_MESH:
    DrawPolygonMesh(triangle_mesh_list, triangle_mesh_vertex_list);
    break;
  }

  glutSwapBuffers();
}

void Keyboard(unsigned char key, int x, int y) {
  switch (key) {
  case '1': 
    program_mode = VIEWING_IMAGE;
    break;
  case '2': 
    program_mode = VIEWING_QUAD_MESH;
    selected_quad_mesh_vertex_index = -1;
    break;
  case '3': 
    program_mode = VIEWING_TRIANGLE_MESH;
    selected_triangle_mesh_vertex_index = -1;
    break;
  case '4': 
    is_viewing_mesh = !is_viewing_mesh;
    break;
  case '5': 
    is_viewing_mesh_point = !is_viewing_mesh_point;
    break;
  case '+': 
    if (program_mode == VIEWING_QUAD_MESH || program_mode == VIEWING_TRIANGLE_MESH) {
      current_mesh_size += MESH_SIZE_GAP;
      ReBuildMesh();
    }
    break;
  case '-':
    if (program_mode == VIEWING_QUAD_MESH || program_mode == VIEWING_TRIANGLE_MESH) {
      current_mesh_size -= MESH_SIZE_GAP;
      ReBuildMesh();
    }
    break;
  case 27: // ESC
    Exit();
    break;
  }
  if (program_mode == VIEWING_QUAD_MESH) {
    Reshape(target_image_width, target_image_height);
  } else {
    Reshape(image.width, image.height);
  }
  glutPostRedisplay();
}

void Reshape(int w, int h) {
  h = h < 1 ? 1 : h;

  glViewport(0, 0, w, h);

  float rate = w / (float)h;

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  //gluPerspective(45, rate, 1.0, 10000.0);
  glOrtho(0, w, 0, h, 1.0, 10000.0);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  if (program_mode == VIEWING_QUAD_MESH) {
    glutReshapeWindow(target_image_width, target_image_height);
  } else {
    glutReshapeWindow(image.width, image.height);
  }
}

void Mouse(int button, int state, int x, int y) {
  if (!state) {
    float world_x = x;
    float world_y = image.height - y;

    FloatPair target(world_x, world_y);
    float nearest_distance;
    if (program_mode == VIEWING_QUAD_MESH) {
      selected_quad_mesh_vertex_index = GetMouseNearestVertexIndex(quad_mesh_vertex_list, target, nearest_distance);
      if (nearest_distance > MESH_POINT_SIZE * 10) {
        selected_quad_mesh_vertex_index = -1;
      }
    } else if (program_mode == VIEWING_TRIANGLE_MESH) {
      selected_triangle_mesh_vertex_index = GetMouseNearestVertexIndex(triangle_mesh_vertex_list, target, nearest_distance);
      if (nearest_distance > MESH_POINT_SIZE * 10) {
        selected_triangle_mesh_vertex_index = -1;
      }
    }
  }

}

void Motion(int x, int y) {
  float world_x = x;
  float world_y = image.height - y;

  if (program_mode == VIEWING_QUAD_MESH) {
    if (selected_quad_mesh_vertex_index != -1) {
      quad_mesh_vertex_list[selected_quad_mesh_vertex_index].first = world_x;
      quad_mesh_vertex_list[selected_quad_mesh_vertex_index].second = world_y;
    }
  } else if (program_mode == VIEWING_TRIANGLE_MESH) {
    if (selected_triangle_mesh_vertex_index != -1) {
      triangle_mesh_vertex_list[selected_triangle_mesh_vertex_index].first = world_x;
      triangle_mesh_vertex_list[selected_triangle_mesh_vertex_index].second = world_y;
    }
  }
  glutPostRedisplay();
}

void ChangeGLTexture(void *texture_pointer, int width, int height) {
  glTexImage2D(GL_TEXTURE_2D, 0, 3, width, height, 0, GL_RGB,GL_UNSIGNED_BYTE, texture_pointer);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
}

void Initial() {
  printf("Start : Read image %s\n", input_file_name);
  BITMAPINFO image_bitmap_info;
  BMPImagePointer image_pointer = LoadBitmapFile(input_file_name, &image_bitmap_info);
  image = ImageType(image_bitmap_info.bmiHeader.biWidth, image_bitmap_info.bmiHeader.biHeight, image_pointer);
  printf("Done : Read image %s\n", input_file_name);

  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  glutInitWindowSize(image.width, image.height);
  window_number = glutCreateWindow(window_name);

  ChangeGLTexture(image_pointer, image.width, image.height);

  BuildQuadMesh();
  BuildTriangleMesh();

  glutReshapeFunc(Reshape);
  glutKeyboardFunc(Keyboard);
  glutMouseFunc(Mouse);
  glutMotionFunc(Motion);
  glutDisplayFunc(Display);

  double cotanget_of_half_of_fovy = 1.0 / tan(22.5 * acos(-1.0) / 180.0);
  eye_z_translation = cotanget_of_half_of_fovy * (image.height / 2.0);
}

void PatchBasedImageWarpingForContentAwareRetargeting(const int target_image_width, const int target_image_height, const double mesh_width, const double mesh_height) {

  GraphType G;
  std::vector<std::vector<int> > group_of_pixel;

  puts("Start : Gaussian smoothing");
  const double SMOOTH_SIGMA = 0.8;
  ImageType image_after_smoothing = GaussianSmoothing(image, SMOOTH_SIGMA);
  image_after_smoothing.WirteBMPImage("smooth.bmp");
  puts("Done : Gaussian smoothing");

  puts("Start : Image segmentation");
  const double SEGMENTATION_K = (image.width + image.height) / 1.75;
  const double SEGMENTATION_MIN_PATCH_SIZE = (image.width * image.height) * 0.0001;
  const double SEGMENTATION_SIMILAR_COLOR_MERGE_THRESHOLD = 20;
  ImageType image_after_segmentation = Segmentation(image_after_smoothing, G, group_of_pixel, SEGMENTATION_K, SEGMENTATION_MIN_PATCH_SIZE, SEGMENTATION_SIMILAR_COLOR_MERGE_THRESHOLD);
  image_after_segmentation.WirteBMPImage("segmentation.bmp");
  puts("Done : Image segmentation");

  puts("Start : Build mesh and graph");
  BuildQuadMeshWithGraph(G, mesh_width, mesh_height);
  puts("Done : Build mesh and graph");

  puts("Start : Image saliency calculation");
  const double SALIENCY_C = 3;
  const double SALIENCY_K = 64;
  std::vector<std::vector<double> > saliency_map;
  ImageType saliency_image = CalculateContextAwareSaliencyMap(image, saliency_map, SALIENCY_C, SALIENCY_K);
  saliency_image.WirteBMPImage("saliency.bmp");
  puts("Done : Image saliency calculation");

  puts("Start : Image warping");
  ImageType image_after_warping = Warping(image, G, group_of_pixel, saliency_map, target_image_width, target_image_height, mesh_width, mesh_height/*, Saliency*/);
  image_after_warping.WirteBMPImage("warping.bmp");
  puts("Done : Image warping");

  for (int vertex_index = 0; vertex_index < G.V.size(); ++vertex_index) {
    //printf("(%f %f) -> (%d %d)\n", quad_mesh_vertex_list[vertex_index].first, quad_mesh_vertex_list[vertex_index].second, G.V[vertex_index].first, G.V[vertex_index].second);
    quad_mesh_vertex_list[vertex_index].first = G.V[vertex_index].first;
    quad_mesh_vertex_list[vertex_index].second = G.V[vertex_index].second;
  }
}

void StartOpenGL() {
  glutMainLoop();
}

void Exit() {
  glutDestroyWindow(window_number);
  exit(0);
}