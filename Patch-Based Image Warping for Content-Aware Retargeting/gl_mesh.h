#pragma once

#include <cstdlib>

#include <memory>
#include <string>
#include <vector>

#include <GL\glew.h>
#include <glm\glm.hpp>
#include <glm\gtc\matrix_transform.hpp>
#include <glm\gtc\type_ptr.hpp>
#include <opencv\cv.hpp>

namespace PatchBasedImageWarpingForContentAwareRetargeting {

  extern GLint shader_program_id;
  extern GLint shader_attribute_vertex_position_id;
  extern GLint shader_attribute_vertex_color_id;
  extern GLint shader_attribute_vertex_uv_id;
  extern GLint shader_uniform_modelview_matrix_id;
  extern GLint shader_uniform_view_matrix_id;
  extern GLint shader_uniform_projection_matrix_id;
  extern GLint shader_uniform_texture_id;
  extern GLint shader_uniform_texture_flag_id;

  class GLMesh {

  public:

    GLMesh() : vbo_vertices_(0), vbo_colors_(0), vbo_uvs_(0), vertices_type(GL_TRIANGLES), local_modelview_matrix_(glm::mat4(1.0)), texture_id_(0), texture_flag_(false) {
    }

    void Translate(const glm::vec3 &translation_vector) {
      local_modelview_matrix_ = glm::translate(local_modelview_matrix_, translation_vector);
    }

    void Upload() {
      if (vertices_.size()) {
        glGenBuffers(1, &vbo_vertices_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices_);
        glBufferData(GL_ARRAY_BUFFER, vertices_.size() * sizeof(vertices_[0]), vertices_.data(), GL_STATIC_DRAW);
      }

      if (colors_.size()) {
        glGenBuffers(1, &vbo_colors_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_colors_);
        glBufferData(GL_ARRAY_BUFFER, colors_.size() * sizeof(colors_[0]), colors_.data(), GL_STATIC_DRAW);
      }

      if (uvs_.size()) {
        glGenBuffers(1, &vbo_uvs_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_uvs_);
        glBufferData(GL_ARRAY_BUFFER, uvs_.size() * sizeof(uvs_[0]), uvs_.data(), GL_STATIC_DRAW);
      }

      texture_flag_ = uvs_.size();
    }

    void Clear() {
      vertices_.clear();
      colors_.clear();
      uvs_.clear();
      local_modelview_matrix_ = glm::mat4(1.0);
      texture_id_ = 0;

      vbo_vertices_ = vbo_colors_ = vbo_uvs_ = 0;

      Upload();
    }

    void Draw() {
      Draw(glm::mat4(1.0f));
    }

    void Draw(const glm::mat4 &parent_modelview_matrix) {

      if (vbo_vertices_) {
        glEnableVertexAttribArray(shader_attribute_vertex_position_id);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices_);
        glVertexAttribPointer(shader_attribute_vertex_position_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
      }

      if (vbo_colors_) {
        glEnableVertexAttribArray(shader_attribute_vertex_color_id);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_colors_);
        glVertexAttribPointer(shader_attribute_vertex_color_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
      }

      if (vbo_uvs_) {
        glEnableVertexAttribArray(shader_attribute_vertex_uv_id);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_uvs_);
        glVertexAttribPointer(shader_attribute_vertex_uv_id, 2, GL_FLOAT, GL_FALSE, 0, 0);
      }

      if (texture_flag_) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_id_);
        glUniform1i(shader_uniform_texture_id, 0);
      }

      glUniform1f(shader_uniform_texture_flag_id, texture_flag_ ? 1.0 : 0.0);

      glm::mat4 modelview_matrix = parent_modelview_matrix * local_modelview_matrix_;

      glUniformMatrix4fv(shader_uniform_modelview_matrix_id, 1, GL_FALSE, glm::value_ptr(modelview_matrix));

      glDrawArrays(vertices_type, 0, vertices_.size());

      if (vbo_vertices_) {
        glDisableVertexAttribArray(shader_attribute_vertex_position_id);
      }

      if (vbo_colors_) {
        glDisableVertexAttribArray(shader_attribute_vertex_color_id);
      }

      if (vbo_uvs_) {
        glDisableVertexAttribArray(shader_attribute_vertex_uv_id);
      }

    }

    std::vector<glm::vec3> vertices_;
    std::vector<glm::vec3> colors_;
    std::vector<glm::vec2> uvs_;

    GLuint texture_id_;
    GLuint vertices_type;

  private:

    GLuint vbo_vertices_;
    GLuint vbo_colors_;
    GLuint vbo_uvs_;

    bool texture_flag_;

    glm::mat4 local_modelview_matrix_;
  };

}