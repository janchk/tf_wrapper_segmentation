//
// Created by jakhremchik
//

#include "wrapper_base.h"

#include <utility>

bool WrapperBase::set_images(const std::vector<std::string>& imgs_paths) {
    fs_img::image_data_struct cur_img;
    for (const auto &img_path : imgs_paths) {
        cur_img = fs_img::read_img(img_path, db_handler->config.input_size);
        this->_imgs.emplace_back(std::move(cur_img.img_data));
        this->_img_orig_size.emplace_back(std::move(cur_img.orig_size));
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

std::vector<cv::Mat> WrapperBase::get_indices(bool resized) {
    std::vector<cv::Mat> indices = this->inference_handler->getOutputSegmentationIndices();
    if (resized) {
        for (auto i = 0; i != indices.size(); ++i) {
            cv::resize(indices[i], indices[i], this->_img_orig_size[i], 0, 0, cv::INTER_LINEAR);
            cv::cvtColor(indices[i], indices[i], cv::COLOR_BGR2RGB);
        }
    }
    return indices;
}

std::vector<cv::Mat> WrapperBase::get_colored(bool resized) {
    db_handler->load_colors();
    inference_handler->setSegmentationColors(db_handler->colors);
    std::vector<cv::Mat> colored_indices = this->inference_handler->getOutputSegmentationColored();
    if (resized) {
        for (auto i = 0; i != colored_indices.size(); ++i) {
            cv::resize(colored_indices[i], colored_indices[i], this->_img_orig_size[i], 0, 0, cv::INTER_LINEAR);
            cv::cvtColor(colored_indices[i], colored_indices[i], cv::COLOR_BGR2RGB);
        }
    }
    return colored_indices;
}


