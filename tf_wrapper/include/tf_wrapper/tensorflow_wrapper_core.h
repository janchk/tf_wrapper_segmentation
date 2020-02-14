//
// Created by jakhremchik
//

#ifndef TF_WRAPPER_SEGMENTATION_TENSORFLOW_WRAPPER_CORE_H
#define TF_WRAPPER_SEGMENTATION_TENSORFLOW_WRAPPER_CORE_H

#include "tensorflow_base.h"
#include "tensorflow_auxiliary.h"

#include <string>
#include <vector>

class TensorflowWrapperCore
{
public:

    enum INPUT_TYPE {
        DT_FLOAT,
        DT_UINT8
    };

    TensorflowWrapperCore() = default;
    virtual ~TensorflowWrapperCore() = default;

    TensorflowWrapperCore(const TensorflowWrapperCore&) = delete;
    TensorflowWrapperCore(TensorflowWrapperCore&& that);

    virtual bool load(const std::string& filename, const std::string &inputNodeName);

    virtual inline std::string inference(const std::vector<cv::Mat> &imgs);

    virtual inline bool isLoaded() const { return _is_loaded; }

    virtual void clearSession();

    virtual inline std::string getName() const { return _name; }
    virtual void setName(const std::string& name);


//    virtual void warmUp(int batchSize) = 0;

    std::string getPath() const;

    bool getAggressiveOptimizationGPUEnabled() const;
    void setAggressiveOptimizationGPUEnabled(bool enabled);

    bool getAllowSoftPlacement() const;
    void setAllowSoftPlacement(bool allowSoftPlacement);

    bool getCpuOnly() const;
    void setCpuOnly(bool cpu_only);

    bool getAggressiveOptimizationCPUEnabled() const;
    ///
    /// \brief setAgressiveOptimizationCPUEnabled JIT optimizations for CPU. Only for CPU Only mode.
    ///
    void setAggressiveOptimizationCPUEnabled(bool enabled);


    int getGpuNumber() const;
    // If -1 may use all visible GPUs. Otherwise that GPU number that was set. Override with default device in the model
    void setGpuNumber(int value);

    double getGpuMemoryFraction() const;
    void setGpuMemoryFraction(double gpu_memory_fraction);

    std::string getVisibleDevices() const;
    void setVisibleDevices(const std::string &visible_devices);

    bool getAllowGrowth() const;
    void setAllowGrowth(bool allow_growth);


protected:

//    virtual void clearModel() = 0;
//    virtual void clearSession() = 0;

    void getInputNodeNameFromGraphIfPossible(const std::string &inputNodeName);

    tensorflow::Status _status;

    ///_______________________________________

    ///values for inference
    ///Inputs and outputs should be reassigned accordinly
    std::vector<std::string> _input_node_names;
    std::vector<std::string> _output_node_names;

    ///_______________________________________

    std::vector<tensorflow::Tensor> _output_tensors;

    void parseName(const std::string& filename);
    tensorflow::SessionOptions configureSession();
    void configureGraph();

    ///
    /// \brief getTensorFromGraph Method for extracting tensors from graph. For usage, model must be loaded and Session must be active.
    /// \param tensor_name Name in the Graph
    /// \return Empty Tensor if failed, otherwise extructed Tensor
    ///
    tensorflow::Tensor getTensorFromGraph(const std::string& tensor_name);

//    using ConvertFunctionType = decltype(&(wrapper_legacy::convertMatToTensor<tensorflow::DT_FLOAT>));

//    ConvertFunctionType getConvertFunction(INPUT_TYPE type) {
//        if (type == INPUT_TYPE::DT_FLOAT) {
//            return wrapper_legacy::convertMatToTensor<tensorflow::DT_FLOAT>;
//        }
///Actually we don't need support for int operations because we don't have strong hardware limits.
//        else if (type == INPUT_TYPE::DT_UINT8) {
//            return tf_aux::convertMatToTensor<tensorflow::DT_UINT8>;
//        }
//        else throw std::invalid_argument("not implemented");
//    }


    bool _is_loaded = false;
    bool _agres_optim_enabled = false;

    /// Mostly that nedeed for XLA, because it's not possible to enable XLA for CPU on session level
    /// But is possible manually. Works only if cpu only mode.
    bool _agres_optim_cpu_enabled = false;

    bool _allow_soft_placement = true;
    bool _cpu_only = false;

    std::string _name = "UnknownModel";
    std::string _path = "";
    std::string _visible_devices = "";

    tensorflow::GraphDef _graph_def;
    tensorflow::Session* _session = nullptr;

    int _gpu_number = -1;
    double _gpu_memory_fraction = 0.;
    bool _allow_growth = true;

};

#endif //TF_WRAPPER_SEGMENTATION_TENSORFLOW_WRAPPER_CORE_H
