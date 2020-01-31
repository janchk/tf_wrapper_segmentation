//
// Created by jakhremchik
//

#include "tensorflow_segmentator.h"

#include <utility>

//TODO Main objective is ti remember about Batch

std::vector<cv::Mat> TensorFlowSegmentator::getOutputSegmentationIndices() {
    std::vector<cv::Mat> _out_data;
    if (_indices.empty()) {
        for (const auto &output: _out_tensors_vector) {
            _out_data = std::move(TensorFlowSegmentator::convertTensorToMat(output));
            _indices.insert(_indices.begin(), _out_data.begin(),  _out_data.end());
//            this->_indices.emplace_back(TensorFlowSegmentator::convertTensorToMat(output));
        }
    } else
        _indices = {};
    return _indices;

}

std::vector<cv::Mat> TensorFlowSegmentator::getOutputSegmentationColored() {
    std::vector<cv::Mat> _out_data;
    if (_indices.empty() && !_colors.empty()) {
        for(const auto &output : _out_tensors_vector) {
            _out_data = std::move(TensorFlowSegmentator::convertTensorToMat(output));
            _indices.insert(_indices.begin(), _out_data.begin(),  _out_data.end());
        }
    } else {
        std::cerr << "Colors not set" << std::endl;
        _indices = {};
    }
    return _indices;
}

//bool TensorFlowSegmentator::normalize_image(cv::Mat &img) {
//    double  min, max;
//    cv::Scalar data = img.at<cv::Vec3b>(0,0);
//    cv::minMaxLoc(img, &min, &max);
//    img.convertTo(img, CV_32F, 1, 0); //TODO normalize it in a right way
//    img = ((img - cv::Scalar(min, min, min)) / (max - min));
//    img = (img * 2) - cv::Scalar(1);
//
//    return true;
//}

std::vector<cv::Mat> TensorFlowSegmentator::convertTensorToMat(const tensorflow::Tensor &tensor) {

    if ( isLoaded() && !_output_tensors.empty()) {
//        const auto &temp_tensor = tensor.tensor<tensorflow::int64, 4>();
        const auto &temp_tensor = tensor.tensor<tensorflow::int64, 3>();
        const auto &dims = tensor.shape();
        std::vector<cv::Mat> imgs(size_t(dims.dim_size(0)));

        for (size_t example = 0; example < imgs.size(); ++example) {
#if 0
            imgs[example] = cv::Mat(cv::Size_<int64>(dims.dim_size(1), dims.dim_size(2)), colors.size() ? CV_8UC3 : CV_8UC1);
#else
            imgs[example] = cv::Mat(cv::Size_<int64>(dims.dim_size(1), dims.dim_size(2)), CV_8UC3);
#endif
            if (_colors.empty()) {
#if 0
                imgs[example].forEach<uchar>([&](uchar& pixel, const int position[]) -> void {
                    pixel = uchar(temp_tensor(long(example), position[0], position[1], 0));
                });
#else
                imgs[example].forEach<cv::Vec3b>([&](cv::Vec3b &pixel, const int position[]) -> void {
//                    auto clrs = uchar(temp_tensor(long(example), position[0], position[1], 0));
                    auto clrs = uchar(temp_tensor(long(example), position[0], position[1]));
                    pixel = cv::Vec3b(cv::Vec3i{clrs, clrs, clrs});
                });
#endif
            } else
                imgs[example].forEach<cv::Vec3b>([&](cv::Vec3b &pixel, const int position[]) -> void {
//                    auto clrs(this->_colors[size_t(temp_tensor(long(example), position[0], position[1], 0))]);
                    auto clrs(_colors[size_t(temp_tensor(long(example), position[0], position[1]))]);
                    pixel = cv::Vec3b(cv::Vec3i{clrs[0], clrs[1], clrs[2]});
                });
        }

        return imgs;
    } else
        return {};
}

std::string TensorFlowSegmentator::inference(const std::vector<cv::Mat> &imgs) {
    using namespace tensorflow;
    PROFILE_BLOCK("inference time");

//    for (const cv::Mat &img : imgs) {
//        if(!normalize_image(const_cast<cv::Mat &>(img))){
//            return "Fail to normalize images";
//        }
//    }
//    this->_input_tensor = wrapper_legacy::convertMatToTensor<tensorflow::DT_UINT8>(imgs, 256, 256, 3, false, {0, 0, 0});
    if (!tf_aux::convertMatToTensor_v2(imgs, _input_tensor)){
        return "Fail to convert Mat to Tensor";
    }

    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {{_input_node_names[0], _input_tensor}};

    _status = _session->Run(inputs, _output_node_names, {}, &_output_tensors);

    /// _output_tensors is a vector of tensors where each tensor represent every possible output from net
    /// taking 0'th out we targeting tensor that contains output indices that we need

    _out_tensors_vector.emplace_back(std::move(_output_tensors[0]));
    TF_CHECK_OK(_status);
//    tf_aux::DebugOutput("NETWORK_STATUS", _status.ToString());
    return _status.ToString();
}


bool TensorFlowSegmentator::setSegmentationColors(std::vector<std::array<int, 3>> colors) {
    _colors = std::move(colors);

    return true;
}

bool TensorFlowSegmentator::set_input_output(std::vector<std::string> in_nodes, std::vector<std::string> out_nodes) {
    _input_node_names = std::move(in_nodes);
    _output_node_names = std::move(out_nodes);

    return true;
}

bool TensorFlowSegmentator::clearData() {
    if (!_out_tensors_vector.empty())
        _out_tensors_vector.clear();
    if (!_indices.empty())
        _indices.clear();

    return true;
}
