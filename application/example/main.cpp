#include <iostream>
#include <vector>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
//#include <tensorflow_auxiliary.h>
#include "main.h"


char *getCmdOption(char **begin, char **end, const std::string &option) {
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return nullptr;
}

bool cmdOptionExists(char **begin, char **end, const std::string &option) {
    return std::find(begin, end, option) != end;
}


std::string parseCommandLine(int argc, char *argv[], const std::string& c) {
    std::string ret;
    if (cmdOptionExists(argv, argv + argc, c)) {
        char *filename = getCmdOption(argv, argv + argc, c);
        ret = std::string(filename);
    } else {
        std::cout << "Use -img $path to image$ -colored $true/false$"
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return ret;
}


int main(int argc, char *argv[]) {

    std::string const inFileName = parseCommandLine(argc, argv, std::string("-img"));
    std::string const is_colored = parseCommandLine(argc, argv, std::string("-colored"));

    std::vector<cv::Mat> output_indices;

    SegmentationWrapperBase seg_wrapper;

//    seg_wrapper.load_config("config.json");

    seg_wrapper.configure_wrapper(cv::Size(256, 256),
                                    "/home/jakhremchik/CLionProjects/TF_WRAPPER_SEGMENTATION/classes.csv",
                                    "/home/jakhremchik/Downloads/train_fine/frozen_inference_graph.pb",
                                    "ImageTensor:0",
                                    "SemanticPredictions:0");

    seg_wrapper.set_images({inFileName, inFileName, inFileName});

//    PROFILE_BLOCK("process images");
    if(!seg_wrapper.process_images())
        std::cerr << "Failed to process images" << std::endl;
    if ("true" == is_colored)
        output_indices = seg_wrapper.get_colored(true);
    else if ("false" == is_colored)
        output_indices = seg_wrapper.get_indices(true);
    else {
        std::cout << "Option not recognized" << std::endl;
        return 1;
    }
    for (unsigned long i=0; i < output_indices.size(); ++i) {
        cv::imwrite(cv::format("out_%i.png", i), output_indices[i]);
    }


    std::cout << "Wrapper finished successfully" << std::endl;
    return 0;
}