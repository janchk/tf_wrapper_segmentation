//
// Created by jakhremchik
//

#include "fs_handling.h"
#include <utility>



fs_img::image_data_struct fs_img::read_img(const std::string &im_filename, cv::Size &size ) {
    fs_img::image_data_struct out_data;
    out_data.img_data = cv::imread(im_filename, cv::IMREAD_COLOR);
    out_data.orig_size = out_data.img_data.size();
//    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    tf_aux::fastResizeIfPossible(out_data.img_data, const_cast<cv::Mat *>(&out_data.img_data), size);

    return out_data;
}

bool path_is_img(std::string path){
    auto  extension = path.substr(path.find_last_of('.') + 1);
    return extension == "jpg" || extension == "JPG";

}

std::vector<std::string> fs_img::list_imgs(const std::string &dir_path) {
    std::vector<std::string> vector_of_data;
    for (const auto &entry : fs::recursive_directory_iterator(dir_path)) {
        if (fs::is_regular_file(entry) && path_is_img(entry.path()))
            vector_of_data.emplace_back(entry.path());
    }
    return vector_of_data;
}

bool DataHandling::open_config() {
    config_datafile.open(config_path, std::ios::in | std::ios::app);
    return true;
}



bool DataHandling::load_config() {
    using namespace rapidjson;
    Document doc;
    std::string line;

    open_config();

    if (config_datafile.is_open()) {
        std::getline(config_datafile, line);
        doc.Parse(line.c_str());
        if (doc.IsObject()) {

            rapidjson::Value &input_size = doc["input_size"];
            rapidjson::Value &input_node = doc["input_node"];
            rapidjson::Value &output_node = doc["output_node"];
            rapidjson::Value &pb_path = doc["pb_path"];
            rapidjson::Value &colors_path = doc["colors_path"];

            config.input_node = input_node.GetString();
            config.output_node = output_node.GetString();
            config.pb_path = pb_path.GetString();
            config.colors_path = colors_path.GetString();
            config.input_size.height = input_size.GetArray()[0].GetInt();
            config.input_size.width = input_size.GetArray()[1].GetInt();

            return true;
        } else
            return false;
    } else {
        return false;
    }

}

bool DataHandling::load_colors() {
    io::CSVReader<4> in(config.colors_path);
    in.read_header(io::ignore_extra_column, "name", "r", "g", "b");
    std::string name; int r; int g; int b;
    while(in.read_row(name, r, g, b)) {
        std::array<int, 3> color = {r, g, b};
        colors.emplace_back(color);
//        std::reverse(std::begin(this->colors), std::end(this->colors));
//        this->colors.
//        std::cout << name << r << g << b << std::endl;
    }

    return true;
}



