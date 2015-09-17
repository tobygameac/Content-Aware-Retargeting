#ifndef SALIENCY_H_
#define SALIENCY_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

cv::Mat CalculateContextAwareSaliencyMapWithMatlabProgram(const cv::Mat &image, std::vector<std::vector<double> > &saliency_map, const std::string &matlab_program_name, const std::string &input_image_name, const std::string &output_image_name) {
  std::string run_saliency_program_command = matlab_program_name;
  run_saliency_program_command += std::string(" ");
  run_saliency_program_command += input_image_name;
  run_saliency_program_command += std::string(" ");
  run_saliency_program_command += output_image_name;

  if (std::fstream(output_image_name).good()) {
    std::cout << "The saliency image was already generated.\n";
  } else {
    std::cout << run_saliency_program_command + "\n";
    system(run_saliency_program_command.c_str());
  }

  cv::Mat saliency_image = cv::imread(output_image_name);
  cv::resize(saliency_image, saliency_image, image.size());

  saliency_map.clear();
  saliency_map = std::vector<std::vector<double> >(image.size().height, std::vector<double>(image.size().width));

  for (int r = 0; r < image.size().height; ++r) {
    for (int c = 0; c < image.size().width; ++c) {
      for (size_t pixel_index = 0; pixel_index < 3; ++pixel_index) {
        saliency_map[r][c] += saliency_image.at<cv::Vec3b>(r, c).val[pixel_index];
      }
      saliency_map[r][c] /= 3;
      saliency_map[r][c] /= 255.0;
    }
  }

  return saliency_image;
}
#endif