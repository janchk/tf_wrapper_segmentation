//
// Created by jakhremchik
//

#ifndef TF_WRAPPER_SEGMENTATION_TENSORFLOW_SEGMENTATOR_H
#define TF_WRAPPER_SEGMENTATION_TENSORFLOW_SEGMENTATOR_H

#include "opencv2/imgproc/imgproc.hpp"
#include "tensorflow_base.h"
#include "tensorflow_wrapper_core.h"

#include "tensorflow_base.h"
#include "tensorflow_wrapper_core.h"
#include <cmath>

class TensorFlowSegmentator : public TensorflowWrapperCore {
public:
  TensorFlowSegmentator() { _colors = {}; };
  ~TensorFlowSegmentator() override = default;

  virtual bool setInputOutput(std::vector<std::string> in_nodes,
                              std::vector<std::string> out_nodes);

  std::string inference(const std::vector<cv::Mat> &imgs) override;

  virtual std::vector<cv::Mat> getOutputSegmentationIndices();

  virtual std::vector<cv::Mat> getOutputSegmentationColored();

  virtual bool setSegmentationColors(std::vector<std::array<int, 3>> colors);

  virtual bool clearData();

  virtual bool setGpuNumberPreferred(int value);

  //    bool normalize_image(cv::Mat &img);

protected:
  std::vector<cv::Mat> convertTensorToMat(const tensorflow::Tensor &tensor);

  std::vector<std::array<int, 3>> _colors;
  tensorflow::Status _status;
  tensorflow::Tensor _input_tensor;

  std::vector<cv::Mat> _imgs;
  std::vector<tensorflow::Tensor> _out_tensors_vector;
  std::vector<cv::Mat> _indices;
  std::vector<std::string> _input_node_names;
  std::vector<std::string> _output_node_names;
};

#endif // TF_WRAPPER_SEGMENTATION_TENSORFLOW_SEGMENTATOR_H
