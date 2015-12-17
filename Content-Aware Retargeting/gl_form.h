#pragma once

#include <windows.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <map>

#include <GL\glew.h>
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <opencv\cv.hpp>

#include "gl_mesh.h"
#include "gl_texture.h"
#include "saliency.h"
#include "segmentation.h"
#include "video_segmentation.h"
#include "warping.h"

#include <msclr\marshal_cppstd.h>
#using <mscorlib.dll>

namespace ContentAwareRetargeting {

  enum ProgramStatus {
    VIEWING_RESULT,
  };

  ProgramStatus program_status;

  const std::string DEFAULT_VERTEX_SHADER_FILE_PATH = "..\\shader\\vertex_shader.glsl";
  const std::string DEFAULT_FRAGMENT_SHADER_FILE_PATH = "..\\shader\\fragment_shader.glsl";

  const std::string SHADER_ATTRIBUTE_VERTEX_POSITION_NAME = "vertex_position";
  const std::string SHADER_ATTRIBUTE_VERTEX_COLOR_NAME = "vertex_color";
  const std::string SHADER_ATTRIBUTE_VERTEX_UV_NAME = "vertex_uv";

  const std::string SHADER_UNIFORM_MODELVIEW_MATRIX_NAME = "modelview_matrix";
  const std::string SHADER_UNIFORM_VIEW_MATRIX_NAME = "view_matrix";
  const std::string SHADER_UNIFORM_PROJECTION_MATRIX_NAME = "projection_matrix";
  const std::string SHADER_UNIFORM_TEXTURE_NAME = "texture";
  const std::string SHADER_UNIFORM_TEXTURE_FLAG_NAME = "texture_flag";

  GLint shader_program_id;
  GLint shader_attribute_vertex_position_id;
  GLint shader_attribute_vertex_color_id;
  GLint shader_attribute_vertex_uv_id;
  GLint shader_uniform_modelview_matrix_id;
  GLint shader_uniform_view_matrix_id;
  GLint shader_uniform_projection_matrix_id;
  GLint shader_uniform_texture_id;
  GLint shader_uniform_texture_flag_id;

  const float FOVY = 45.0f;

  glm::vec3 eye_position(0.0, 0.0, 0.0);
  glm::vec3 look_at_position(0.0, 0.0, 0.0);

  const float EYE_POSITION_SCALE_PER_SCROLLING = 0.9f;
  float eye_position_scale = 1.0;

  bool is_dragging_panel;

  const int MIN_PANEL_WIDTH = 10;
  const int MIN_PANEL_HEIGHT = 10;
  const int MAX_PANEL_WIDTH = 1 << 12;
  const int MAX_PANEL_HEIGHT = 1 << 12;

  std::chrono::steady_clock::time_point last_clock = std::chrono::steady_clock::now();

  std::string source_image_file_directory;
  std::string source_image_file_name;

  cv::Mat source_image;
  cv::Mat source_image_after_smoothing;
  cv::Mat source_image_after_segmentation;
  cv::Mat saliency_map_of_source_image;
  cv::Mat significance_map_of_source_image;

  std::string source_video_file_directory;
  std::string source_video_file_name;

  std::vector<cv::Mat> source_video_frames;
  std::vector<cv::Mat> source_video_frames_after_segmentation;

  double source_video_fps;
  int source_video_fourcc;

  GLMesh gl_panel_image_mesh;

  const float GRID_LINE_WIDTH = 4.5;
  const float GRID_POINT_SIZE = 7.5;

  const float MIN_GRID_SIZE = 10;
  const float MAX_GRID_SIZE = 120;

  const float GRID_SIZE_GAP = 10;

  float current_grid_size = 4;

  float current_focus_grid_scale = 3.5;

  bool is_viewing_mesh = true;
  bool is_viewing_mesh_point = false;

  Graph<glm::vec2> image_graph;
  std::vector<std::vector<int> > group_of_pixel;
  std::vector<std::vector<double> > saliency_map;
  std::vector<double> saliency_of_patch;
  std::vector<double> saliency_of_mesh_vertex;

  std::map<size_t, double> saliency_of_object;

  bool data_for_image_warping_were_generated = false;

  bool is_recording_screen = false;

  using namespace System::Runtime::InteropServices;

  [DllImport("opengl32.dll")]
  extern HDC wglSwapBuffers(HDC hdc);

  using namespace System;
  using namespace System::ComponentModel;
  using namespace System::Collections;
  using namespace System::Windows::Forms;
  using namespace System::Data;
  using namespace System::Drawing;

  /// <summary>
  /// Summary for GLForm
  /// </summary>
  public ref class GLForm : public System::Windows::Forms::Form {

  public:

    GLForm();

  private:

    static HWND hwnd;
    static HDC hdc;
    static HGLRC hrc;

    static bool ParseFileIntoString(const std::string &file_path, std::string &file_string);

    void InitializeOpenGL();

    void RenderGLPanel();

    cv::Mat GLScreenToMat();

    void SaveGLScreen(const std::string &file_path);

    void ChangeGLPanelSize(int new_panel_width, int new_panel_height);

    void BuildGridMeshAndGraphForImage(const cv::Mat &image, GLMesh &target_mesh, Graph<glm::vec2> &G, float grid_size);

    void ContentAwareRetargeting(const int target_image_width, const int target_image_height, const float grid_size);

    void ContentAwareVideoRetargetingUsingObjectPreservingWarping(const int target_video_width, const int target_video_height, const float grid_size);

    cv::Vec3b SaliencyValueToSignifanceColor(double saliency_value);

    size_t Vec3bToValue(const cv::Vec3b &color);

    void OnButtonsClick(System::Object ^sender, System::EventArgs ^e);

    void OnMouseDown(System::Object ^sender, System::Windows::Forms::MouseEventArgs ^e);

    void OnMouseUp(System::Object ^sender, System::Windows::Forms::MouseEventArgs ^e);

    void OnMouseMove(System::Object ^sender, System::Windows::Forms::MouseEventArgs ^e);

    void OnKeyDown(System::Object ^sender, System::Windows::Forms::KeyEventArgs ^e);

  protected:
    /// <summary>
    /// Clean up any resources being used.
    /// </summary>
    ~GLForm() {
      if (components) {
        delete components;
      }
    }

  private:

    System::Windows::Forms::Panel ^gl_panel_;
    System::ComponentModel::Container ^components;
    System::Windows::Forms::MenuStrip ^menu_strip_;
    System::Windows::Forms::ToolStripMenuItem ^file_tool_strip_menu_item_;
    System::Windows::Forms::ToolStripMenuItem ^mode_tool_strip_menu_item_;
    System::Windows::Forms::ToolStripMenuItem ^view_tool_strip_menu_item_;
    System::Windows::Forms::ToolStripMenuItem ^open_image_file_tool_strip_menu_item_;
    System::Windows::Forms::ToolStripMenuItem ^open_video_file_tool_strip_menu_item_;
    System::Windows::Forms::ToolStripMenuItem ^save_screen_tool_strip_menu_item_;
    System::Windows::Forms::ToolStripMenuItem ^record_screen_tool_strip_menu_item_;


#pragma region Windows Form Designer generated code
    /// <summary>
    /// Required method for Designer support - do not modify
    /// the contents of this method with the code editor.
    /// </summary>
    void InitializeComponent(void) {
      this->gl_panel_ = (gcnew System::Windows::Forms::Panel());
      this->menu_strip_ = (gcnew System::Windows::Forms::MenuStrip());
      this->file_tool_strip_menu_item_ = (gcnew System::Windows::Forms::ToolStripMenuItem());
      this->open_image_file_tool_strip_menu_item_ = (gcnew System::Windows::Forms::ToolStripMenuItem());
      this->save_screen_tool_strip_menu_item_ = (gcnew System::Windows::Forms::ToolStripMenuItem());
      this->record_screen_tool_strip_menu_item_ = (gcnew System::Windows::Forms::ToolStripMenuItem());
      this->mode_tool_strip_menu_item_ = (gcnew System::Windows::Forms::ToolStripMenuItem());
      this->view_tool_strip_menu_item_ = (gcnew System::Windows::Forms::ToolStripMenuItem());
      this->open_video_file_tool_strip_menu_item_ = (gcnew System::Windows::Forms::ToolStripMenuItem());
      this->menu_strip_->SuspendLayout();
      this->SuspendLayout();
      // 
      // gl_panel_
      // 
      this->gl_panel_->BorderStyle = System::Windows::Forms::BorderStyle::Fixed3D;
      this->gl_panel_->Cursor = System::Windows::Forms::Cursors::Default;
      this->gl_panel_->Location = System::Drawing::Point(20, 20);
      this->gl_panel_->Name = L"gl_panel_";
      this->gl_panel_->Size = System::Drawing::Size(1024, 768);
      this->gl_panel_->TabIndex = 0;
      // 
      // menu_strip_
      // 
      this->menu_strip_->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(3) {
        this->file_tool_strip_menu_item_,
          this->mode_tool_strip_menu_item_, this->view_tool_strip_menu_item_
      });
      this->menu_strip_->Location = System::Drawing::Point(0, 0);
      this->menu_strip_->Name = L"menu_strip_";
      this->menu_strip_->Size = System::Drawing::Size(1264, 24);
      this->menu_strip_->TabIndex = 1;
      this->menu_strip_->Text = L"menuStrip1";
      // 
      // file_tool_strip_menu_item_
      // 
      this->file_tool_strip_menu_item_->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(4) {
        this->open_image_file_tool_strip_menu_item_,
          this->open_video_file_tool_strip_menu_item_, this->save_screen_tool_strip_menu_item_, this->record_screen_tool_strip_menu_item_
      });
      this->file_tool_strip_menu_item_->Name = L"file_tool_strip_menu_item_";
      this->file_tool_strip_menu_item_->Size = System::Drawing::Size(37, 20);
      this->file_tool_strip_menu_item_->Text = L"File";
      // 
      // open_image_file_tool_strip_menu_item_
      // 
      this->open_image_file_tool_strip_menu_item_->Name = L"open_image_file_tool_strip_menu_item_";
      this->open_image_file_tool_strip_menu_item_->Size = System::Drawing::Size(160, 22);
      this->open_image_file_tool_strip_menu_item_->Text = L"Open Image File";
      // 
      // save_screen_tool_strip_menu_item_
      // 
      this->save_screen_tool_strip_menu_item_->Name = L"save_screen_tool_strip_menu_item_";
      this->save_screen_tool_strip_menu_item_->Size = System::Drawing::Size(160, 22);
      this->save_screen_tool_strip_menu_item_->Text = L"Save Screen";
      // 
      // record_screen_tool_strip_menu_item_
      // 
      this->record_screen_tool_strip_menu_item_->Name = L"record_screen_tool_strip_menu_item_";
      this->record_screen_tool_strip_menu_item_->Size = System::Drawing::Size(160, 22);
      this->record_screen_tool_strip_menu_item_->Text = L"Record Screen";
      // 
      // mode_tool_strip_menu_item_
      // 
      this->mode_tool_strip_menu_item_->Name = L"mode_tool_strip_menu_item_";
      this->mode_tool_strip_menu_item_->Size = System::Drawing::Size(50, 20);
      this->mode_tool_strip_menu_item_->Text = L"Mode";
      // 
      // view_tool_strip_menu_item_
      // 
      this->view_tool_strip_menu_item_->Name = L"view_tool_strip_menu_item_";
      this->view_tool_strip_menu_item_->Size = System::Drawing::Size(44, 20);
      this->view_tool_strip_menu_item_->Text = L"View";
      // 
      // open_video_file_tool_strip_menu_item_
      // 
      this->open_video_file_tool_strip_menu_item_->Name = L"open_video_file_tool_strip_menu_item_";
      this->open_video_file_tool_strip_menu_item_->Size = System::Drawing::Size(160, 22);
      this->open_video_file_tool_strip_menu_item_->Text = L"Open Video File";
      // 
      // GLForm
      // 
      this->AutoScaleDimensions = System::Drawing::SizeF(6, 13);
      this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
      this->ClientSize = System::Drawing::Size(1264, 762);
      this->Controls->Add(this->gl_panel_);
      this->Controls->Add(this->menu_strip_);
      this->MainMenuStrip = this->menu_strip_;
      this->Name = L"GLForm";
      this->Text = L"Content-Aware Retargeting";
      this->menu_strip_->ResumeLayout(false);
      this->menu_strip_->PerformLayout();
      this->ResumeLayout(false);
      this->PerformLayout();

    }
#pragma endregion
  };
}
