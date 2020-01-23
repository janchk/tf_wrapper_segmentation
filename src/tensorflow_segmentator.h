//
// Created by jakhremchik
//

#ifndef TF_WRAPPER_SEGMENTATION_TENSORFLOW_SEGMENTATOR_H
#define TF_WRAPPER_SEGMENTATION_TENSORFLOW_SEGMENTATOR_H

#include "tensorflow_wrapper_core.h"
#include "tensorflow_base.h"
#include "opencv2/imgproc/imgproc.hpp"

#include <cmath>
#include "tensorflow_wrapper_core.h"
#include "tensorflow_base.h"

class TensorFlowSegmentator : public TensorflowWrapperCore
{
public:
    TensorFlowSegmentator() {
        this->_colors = {};

    };
    virtual  ~TensorFlowSegmentator() = default;

    bool set_input_output(std::vector<std::string> in_nodes, std::vector<std::string> out_nodes);

    std::string inference(const std::vector<cv::Mat> &imgs) override;

    std::vector<cv::Mat> getOutputSegmentationIndices();

    std::vector<cv::Mat> getOutputSegmentationColored();

    bool setSegmentationColors(std::vector<std::array<int, 3>> colors);

    bool clearData();

    bool normalize_image(cv::Mat &img);

    std::vector<cv::Mat> convertTensorToMat(const tensorflow::Tensor& tensor);

protected:

//    std::vector<std::vector<int>> _colors;
    std::vector<std::array<int, 3>> _colors;
    tensorflow::Status _status;
    tensorflow::Tensor _input_tensor;

    std::vector<cv::Mat> _imgs;
    std::vector<tensorflow::Tensor> _out_tensors_vector;
    std::vector<cv::Mat> _indices;
    std::vector<std::string> _input_node_names;
    std::vector<std::string> _output_node_names;



};

#endif //TF_WRAPPER_SEGMENTATION_TENSORFLOW_SEGMENTATOR_H
