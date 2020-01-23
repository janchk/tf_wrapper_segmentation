//
// Created by jakhremchik
//

#ifndef TF_WRAPPER_SEGMENTATION_FS_HANDLING_H
#define TF_WRAPPER_SEGMENTATION_FS_HANDLING_H

#include "opencv/cv.h"
#include "opencv/cv.hpp"
#include "vector"
#include "string"
#include "fstream"


#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

#include "csv/csv.h"

#include "../tensorflow_auxiliary.h"

#define EXPERIMENTAL
#ifdef EXPERIMENTAL
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif
namespace fs_img
{
    struct image_data_struct
    {
        cv::Size orig_size;
        cv::Mat img_data;
    };

    image_data_struct read_img(const std::string &im_filename, cv::Size &size);

    std::vector<std::string> list_imgs(const std::string & dir_path);
}

class DataHandling
{
public:
    DataHandling()
    {
        config_path = "config.json";
    }
    virtual ~DataHandling() = default;

    struct image_data_struct
    {
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

    bool load_config();
    bool load_colors();


protected:
    std::fstream config_datafile;

    std::vector<std::pair<cv::Mat, std::string>> imgs_and_paths;

    bool open_config();
};



#endif //TF_WRAPPER_SEGMENTATION_FS_HANDLING_H
