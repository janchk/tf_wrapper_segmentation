//
// Created by jakhremchik
//

#include "wrapper_base.h"

#include <utility>

bool WrapperBase::set_images(const std::vector<std::string>& imgs_paths) {
    for (const auto &img_path : imgs_paths) {
        this->_imgs.emplace_back(fs_img::read_img(img_path, db_handler->config.input_size));
    }
    return true;
}

bool WrapperBase::process_images() {
    this->inference_handler->load(db_handler->config.pb_path, db_handler->config.input_node);
    this->inference_handler->inference(this->_imgs);

    return true;
}

WrapperBase::~WrapperBase() {
    common_ops::delete_safe( inference_handler);
    common_ops::delete_safe(db_handler);
}

std::vector<cv::Mat> WrapperBase::get_indices() {
    std::vector<cv::Mat> indices = this->inference_handler->getOutputSegmentationIndices();
    return indices;
}

std::vector<cv::Mat> WrapperBase::get_colored() {
    std::vector<cv::Mat> colored_indices = this->inference_handler->getOutputSegmentationColored();
    return colored_indices;
}


