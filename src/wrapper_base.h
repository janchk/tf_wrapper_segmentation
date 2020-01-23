//
// Created by jakhremchik
//

#ifndef TF_WRAPPER_SEGMENTATION_WRAPPER_BASE_H
#define TF_WRAPPER_SEGMENTATION_WRAPPER_BASE_H

#include "common/fs_handling.h"
#include "tensorflow_segmentator.h"
#include "common/common_ops.h"

class WrapperBase
{
public:

    WrapperBase();

    ~WrapperBase();

    bool set_images(const std::vector<std::string>& imgs_paths); // opt for future_batch

    bool process_images();

    std::vector<cv::Mat> get_indices(bool resized=true);

    std::vector<cv::Mat> get_colored(bool resized=true);

protected:
    std::vector<cv::Size> _img_orig_size;
    std::vector<cv::Mat> _imgs;
    std::vector<cv::Mat> _result;
    DataHandling *db_handler;
    TensorFlowSegmentator *inference_handler;


};

#endif //TF_WRAPPER_SEGMENTATION_WRAPPER_BASE_H
