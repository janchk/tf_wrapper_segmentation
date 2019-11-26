//
// Created by jakhremchik
//

#include "tensorflow_wrapper_core.h"

TensorflowWrapperCore::TensorflowWrapperCore(TensorflowWrapperCore &&that) {
    this->_session = that._session;
    this->_is_loaded = that._is_loaded;

    this->_name = std::move(that._name);
    this->_path = std::move(that._path);
    this->_graph_def = std::move(that._graph_def);

}

void TensorflowWrapperCore::clearSession() {
    _output_tensors.clear();
}

// TODO Disable graph optimization. We assume that graph already optimized.
tensorflow::SessionOptions TensorflowWrapperCore::configureSession(){
    using namespace tensorflow;

    SessionOptions opts;
    opts.config.set_allow_soft_placement(_allow_soft_placement);

    GPUOptions *gpu_options = new GPUOptions;
#ifdef TFDEBUG
    //opts.config.set_log_device_placement(true);
#endif
    if (_cpu_only) {
        auto device_map = opts.config.mutable_device_count();
        if (device_map) {
            tf_aux::DebugOutput("Warning", "Disabling GPU!!!");
            (*device_map)["GPU"] = 0;
        }
    } else {
        if (!_visible_devices.empty()) {
            gpu_options->set_visible_device_list(_visible_devices);
        }
        gpu_options->set_per_process_gpu_memory_fraction(_gpu_memory_fraction);
        gpu_options->set_allow_growth(_allow_growth);
    }

    GraphOptions *graph_opts = new GraphOptions;
    /// TODO: Needs tests, maybe not all options is ok
    OptimizerOptions *optim_opts = new OptimizerOptions;
    // OptimizerOptions_GlobalJitLevel_ON_2 turn on compilation, with higher values being
    // more aggressive.  Higher values may reduce opportunities for parallelism
    // and may use more memory.  (At present, there is no distinction, but this
    // is expected to change.)

    //TODO think about jit
//    optim_opts->set_global_jit_level( (_agres_optim_enabled ? OptimizerOptions_GlobalJitLevel_ON_2
//                                                    : OptimizerOptions_GlobalJitLevel_OFF) );
    optim_opts->set_do_common_subexpression_elimination(_agres_optim_enabled ? true : false);
    optim_opts->set_do_constant_folding(_agres_optim_enabled ? true : false);
    optim_opts->set_do_function_inlining(_agres_optim_enabled ? true : false);
//
    graph_opts->set_allocated_optimizer_options(optim_opts);
//
    opts.config.set_allocated_graph_options(graph_opts);
    opts.config.set_allocated_gpu_options(gpu_options);

    return opts;
}

//TODO Think about graph configuration
void TensorflowWrapperCore::configureGraph(){
    using namespace tensorflow;
    if(_cpu_only && _agres_optim_cpu_enabled)
        graph::SetDefaultDevice("/job:localhost/replica:0/task:0/device:XLA_CPU:0", &_graph_def);
    if(!_cpu_only && _gpu_number >= 0)
//        graph::SetDefaultDevice("/job:localhost/replica:0/task:0/device:GPU:" + std::to_string(_gpu_number), &_graph_def);
        graph::SetDefaultDevice("/device:GPU:" + std::to_string(_gpu_number), &_graph_def);
}

bool TensorflowWrapperCore::load(const std::string &filename, const std::string &inputNodeName) {
    using namespace tensorflow;

    // Configuration for session
    SessionOptions opts = configureSession();
    if (_session) {
        _session->Close();
        delete _session;
        _session = nullptr;
    }
    Status status = NewSession(opts, &_session);
    if (!status.ok()) {
        tf_aux::DebugOutput("tf error: ", status.ToString());

        return _is_loaded = false;
    }
    status = ReadBinaryProto(Env::Default(), filename, &_graph_def);
    if (!status.ok()) {
        tf_aux::DebugOutput("tf error: ", status.ToString());

        return _is_loaded = false;
    }
    configureGraph();
    status = _session->Create(_graph_def);
    if (!status.ok()) {
        tf_aux::DebugOutput("tf error: ", status.ToString());
        return _is_loaded = false;
    } else {
        tf_aux::DebugOutput("WRAPPER_STATUS", "Graph successfully loaded!");
    }
    parseName(filename);

    getInputNodeNameFromGraphIfPossible(inputNodeName);

    _path = filename;
    return _is_loaded = true;
}

std::string TensorflowWrapperCore::inference(const std::vector<cv::Mat> &imgs){
    using namespace tensorflow;

    Tensor input = getConvertFunction(INPUT_TYPE::DT_FLOAT)(imgs, _input_height, _input_width, _input_depth, _convert_to_float, _mean);

    std::vector<int> in_tensor_shape = tf_aux::get_tensor_shape(input);
    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {{_input_node_names[0], input}};
    std::cout << _input_node_names[0] << std::endl;
    _status = _session->Run(inputs, _output_node_names, {}, &_output_tensors);
    std::cerr << "NETWORK_STATUS: " << _status << std::endl;
    return _status.ToString();
}

//std::string TensorflowWrapperCore
void TensorflowWrapperCore::setName(const std::string& name){
    _name = name;
}

void TensorflowWrapperCore::parseName(const std::string &filename){
    auto last_slash = filename.rfind("/");
    if (last_slash == std::string::npos) {
        last_slash = 0;
    }

    auto last_dot = filename.rfind(".");
    if (last_dot == std::string::npos) {
        _name = "UnknownModel";
        return;
    }

    if (last_slash > last_dot) {
        _name = "UnknownModel";
        return;
    }

    _name = filename.substr(last_slash + 1, (last_dot - last_slash) - 1);
}

void TensorflowWrapperCore::getInputNodeNameFromGraphIfPossible(const std::string &inputNodeName){
    using namespace tensorflow;

    const Tensor& names_tensor = getTensorFromGraph(inputNodeName);
    if (names_tensor.NumElements() == 1) {
        const auto& names_mapped = names_tensor.tensor<std::string, 1>();
//#ifdef TFDEBUG
        std::cerr << "Input node name:\n------------------" << std::endl;
//#endif
        _input_node_names[0] = names_mapped(0);
//#ifdef TFDEBUG
        std::cerr << names_mapped(0) << std::endl;
//#endif

//#ifdef TFDEBUG
        std::cerr << "------------------\nInput node name loaded" << std::endl;
//#endif
    }
}

tensorflow::Tensor TensorflowWrapperCore::getTensorFromGraph(const std::string &tensor_name){
    using namespace tensorflow;

    if (tensor_name.empty()) {
        return Tensor();
    }

    if (!_is_loaded) {
        return Tensor();
    }

    tensorflow::Status status;
    std::vector<tensorflow::Tensor> tensors;

    status = _session->Run({}, {tensor_name}, {}, &tensors);

    tf_aux::DebugOutput("Sucessfully run graph! Status is: ", status.ToString());

    if (!status.ok()) {
        return Tensor();
    }

    return tensors[0];
}

bool TensorflowWrapperCore::getAllowGrowth() const{
    return _allow_growth;
}

void TensorflowWrapperCore::setAllowGrowth(bool allow_growth){
    _allow_growth = allow_growth;
}

std::string TensorflowWrapperCore::getVisibleDevices() const{
    return _visible_devices;
}

void TensorflowWrapperCore::setVisibleDevices(const std::string &visible_devices){
    _visible_devices = visible_devices;
}

double TensorflowWrapperCore::getGpuMemoryFraction() const{
    return _gpu_memory_fraction;
}

void TensorflowWrapperCore::setGpuMemoryFraction(double gpu_memory_fraction){
    if (gpu_memory_fraction > 1.0)
        gpu_memory_fraction = 1.0;
    if (gpu_memory_fraction < 0.0)
        gpu_memory_fraction = 0.1;

    _gpu_memory_fraction = gpu_memory_fraction;
}

int TensorflowWrapperCore::getGpuNumber() const{
    return _gpu_number;
}

void TensorflowWrapperCore::setGpuNumber(int value){
    _gpu_number = value;
}

bool TensorflowWrapperCore::getAggressiveOptimizationCPUEnabled() const{
    return _agres_optim_cpu_enabled;
}

void TensorflowWrapperCore::setAggressiveOptimizationCPUEnabled(bool enabled){
    _agres_optim_cpu_enabled = enabled;
}

bool TensorflowWrapperCore::getCpuOnly() const{
    return _cpu_only;
}

void TensorflowWrapperCore::setCpuOnly(bool cpu_only){
    _cpu_only = cpu_only;
}

bool TensorflowWrapperCore::getAllowSoftPlacement() const{
    return _allow_soft_placement;
}

void TensorflowWrapperCore::setAllowSoftPlacement(bool allowSoftPlacement){
    _allow_soft_placement = allowSoftPlacement;
}

bool TensorflowWrapperCore::getAggressiveOptimizationGPUEnabled() const{
    return _agres_optim_enabled;
}

void TensorflowWrapperCore::setAggressiveOptimizationGPUEnabled(bool enabled){
    _agres_optim_enabled = enabled;
}

std::string TensorflowWrapperCore::getPath() const{
    return _path;
}


