//
// Created by jakhremchik
//

#ifndef TF_WRAPPER_SEGMENTATION_WRAPPER_BASE_H
#define TF_WRAPPER_SEGMENTATION_WRAPPER_BASE_H

#include "interfaces.h"

class SegmentationWrapperBase {
public:
  SegmentationWrapperBase();

  ~SegmentationWrapperBase() = default;

  /// \brief Method for setting number of GPU to make calculations on.
  /// \param value
  bool set_gpu(int value);

  /// \brief
  /// \param imgs_paths
  /// \return
  bool set_images(
      const std::vector<std::string> &imgs_paths); // opt for future_batch

  /// \brief
  /// \return
  bool process_images();

  /// \brief Method for configuring wrapper in case that config need to be
  /// loaded from file
  /// \param config_path is a path to .json file with config to
  /// wrapper
  /// \return
  bool load_config(std::string config_path); // ="config.json"

  /// \brief Method for configuring wrapper by setting all values explicitly
  /// \param input_size is a size for input images. Need to be set according to
  /// network architecture that are using.
  /// \param colors_path is a path to
  /// classes according colors of dataset that network is pretrained on.
  /// \param pb_path is a path for pretrained protobuf file
  /// \param input_node is a name of input node
  /// \param output_node is a name of output node
  /// \return
  bool configure_wrapper(const cv::Size &input_size,
                         const std::string &colors_path,
                         const std::string &pb_path,
                         const std::string &input_node,
                         const std::string &output_node);

  /// \brief
  /// \param resized
  /// \return
  std::vector<cv::Mat> get_indices(bool resized); // =true

  /// \brief
  /// \param resized
  /// \return
  std::vector<cv::Mat> get_colored(bool resized); // =true

  /// \brief Method for getting all visible devices that can handle computations
  /// \return
  std::string get_devices();

protected:
  bool is_configured = false;
  cv::Size img_des_size;
  std::vector<cv::Size> img_orig_size;
  std::vector<cv::Mat> imgs;
  std::vector<cv::Mat> result;
  std::unique_ptr<SegmentatorInterface> inference_handler;
  std::unique_ptr<DBInterface> db_handler;
};

#endif // TF_WRAPPER_SEGMENTATION_WRAPPER_BASE_H
