//
// Created by jakhremchik
//

#ifndef TF_WRAPPER_SEGMENTATION_INTERFACES_H
#define TF_WRAPPER_SEGMENTATION_INTERFACES_H

#include <memory>
#include <vector>
#include <opencv2/core/mat.hpp>

class SegmentatorInterface
{
public:

    virtual bool setGpuNumberPreferred(int value) = 0;

    virtual bool setSegmentationColors(std::vector<std::array<int, 3>> colors) = 0;

    virtual bool set_input_output(std::vector<std::string> in_nodes,
                                  std::vector<std::string> out_nodes) = 0;

    virtual bool load(const std::string &filename,
                      const std::string &inputNodeName) = 0;

    virtual bool clearData() = 0;

    virtual std::string inference(const std::vector<cv::Mat> &imgs) = 0;

    virtual std::string getVisibleDevices() = 0;

    virtual std::vector<cv::Mat> getOutputSegmentationIndices() = 0;

    virtual std::vector<cv::Mat> getOutputSegmentationColored() = 0;
};

class DBInterface
{
public:

    virtual bool set_config_path(std::string path) = 0;

    virtual bool load_config() = 0;

    virtual bool load_colors() = 0;

    virtual cv::Size get_config_input_size() = 0;

    virtual std::string get_config_input_node() = 0;

    virtual std::string get_config_output_node() = 0;

    virtual std::string get_config_pb_path() = 0;

    virtual std::string get_config_colors_path() = 0;

    virtual std::vector<std::array<int, 3>> get_colors() = 0;

    virtual bool set_config_input_size(const cv::Size& size) = 0;

    virtual bool set_config_input_node(const std::string& input_node) = 0;

    virtual bool set_config_output_node(const std::string& output_node) = 0;

    virtual bool set_config_pb_path(const std::string& pb_path) = 0;

    virtual bool set_config_colors_path(const std::string& colors_path) = 0;

};

#endif //TF_WRAPPER_SEGMENTATION_INTERFACES_H
