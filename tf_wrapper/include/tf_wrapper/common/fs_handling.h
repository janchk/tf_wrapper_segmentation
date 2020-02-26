//
// Created by jakhremchik
//

#ifndef TF_WRAPPER_SEGMENTATION_FS_HANDLING_H
#define TF_WRAPPER_SEGMENTATION_FS_HANDLING_H

#include "fstream"
#include "opencv/cv.h"
#include "opencv/cv.hpp"
#include "string"
#include "vector"

#include "rapidjson/document.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

#include "csv/csv.h"

#define EXPERIMENTAL
#ifdef EXPERIMENTAL
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif
namespace fs_img {
struct image_data_struct {
  cv::Size orig_size;
  cv::Mat img_data;
};

image_data_struct read_img(const std::string &im_filename, cv::Size &size);

std::vector<std::string> list_imgs(const std::string &dir_path);
} // namespace fs_img

class DataHandling {
public:
  DataHandling() { config_path = ""; }
  virtual ~DataHandling() = default;

  struct image_data_struct {
    cv::Size orig_size;
    cv::Mat img_data;
  };

  struct config_data {
    cv::Size input_size;
    std::string input_node;
    std::string output_node;
    std::string pb_path;
    std::string colors_path;
  };

  // struct with all config data
  config_data config;
  std::vector<std::array<int, 3>> colors;
  std::string config_path;

  bool set_config_path(std::string path);

  bool load_config();

  bool load_colors();

  cv::Size get_config_input_size();

  std::string get_config_input_node();

  std::string get_config_output_node();

  std::string get_config_pb_path();

  std::string get_config_colors_path();

  std::vector<std::array<int, 3>> get_colors();

  bool set_config_input_size(const cv::Size &size);

  bool set_config_input_node(const std::string &input_node);

  bool set_config_output_node(const std::string &output_node);

  bool set_config_pb_path(const std::string &pb_path);

  bool set_config_colors_path(const std::string &colors_path);

protected:
  std::fstream config_datafile;

  std::vector<std::pair<cv::Mat, std::string>> imgs_and_paths;

  bool open_config();
};

#endif // TF_WRAPPER_SEGMENTATION_FS_HANDLING_H
