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

    WrapperBase()
    {
        db_handler = new DataHandling();
        db_handler->config_path = "config.json";
        db_handler->load_config();

        inference_handler = new TensorFlowSegmentator();

        //TODO this is kinda implicit. Why is converting string to vec.
        inference_handler->set_input_output({db_handler->config.input_node}, {db_handler->config.output_node});


    }
    ~WrapperBase();

    bool set_images(const std::vector<std::string>& imgs_paths);

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
