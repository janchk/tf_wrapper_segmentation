#include <iostream>
#include "wrapper_base.h"


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

    auto *seg_wrapper = new WrapperBase();

    seg_wrapper->set_image(inFileName);
    if(!seg_wrapper->process_images())
        std::cerr << "Failed to process images" << std::endl;

    if ("true" == is_colored)
        output_indices = seg_wrapper->get_colored();
    else if ("false" == is_colored)
        output_indices = seg_wrapper->get_indices();
    else {
        std::cout << "Option not recognized" << std::endl;
        return 1;
    }

    cv::imwrite("out.png", output_indices);

    common_ops::delete_safe(seg_wrapper);

    std::cout << "Wrapper finished successfully" << std::endl;
    return 0;
}