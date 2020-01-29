//
// Created by jakhremchik
//

#ifndef TF_WRAPPER_SEGMENTATION_WRAPPER_BASE_H
#define TF_WRAPPER_SEGMENTATION_WRAPPER_BASE_H

#include "common/fs_handling.h"
#include "tensorflow_segmentator.h"
#include "common/common_ops.h"

// TODO Make WrapperBase abstract

class WrapperBase
{
public:

    WrapperBase();

    ~WrapperBase();

    bool set_images(const std::vector<std::string>& imgs_paths); // opt for future_batch

    bool process_images();

    bool configure_wrapper( const cv::Size& input_size,
                            const std::string& colors_path,
                            const std::string& pb_path,
                            const std::string& input_node,
                            const std::string& output_node);

    std::vector<cv::Mat> get_indices(bool resized=true);

    std::vector<cv::Mat> get_colored(bool resized=true);

protected:
    bool _is_configured = false;
    cv::Size _img_des_size;
    std::vector<cv::Size> _img_orig_size;
    std::vector<cv::Mat> _imgs;
    std::vector<cv::Mat> _result;
    DataHandling *db_handler;
    TensorFlowSegmentator *inference_handler;


};

#endif //TF_WRAPPER_SEGMENTATION_WRAPPER_BASE_H
