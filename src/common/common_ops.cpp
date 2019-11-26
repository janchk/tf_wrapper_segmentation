//
// Created by jakhremchik on 08.11.2019.
//

#include "common_ops.h"

std::string common_ops::extract_class(const std::string &filepath) {
   std::string slash_delim = "/"; //TODO another delimiter for windows
   std::string token;
   std::string train_identifier = "series/";
   std::string test_identifier = "queries/";
   std::string classname_delim = "__";

   size_t pos_begin = filepath.find(test_identifier);
   size_t pos_end = filepath.find(classname_delim);

   if (std::string::npos == pos_begin) { // if not test directory
       pos_begin = filepath.find(train_identifier); // find train directory identifier
       if (std::string::npos == pos_end) {
           classname_delim = "/";
           token = filepath.substr(pos_begin + train_identifier.size(), std::string::npos);
           pos_end = token.find(classname_delim);
           token = token.substr(0, pos_end);
       } else {
           pos_end = filepath.find(classname_delim);
           pos_begin = pos_begin + train_identifier.size();
           token = filepath.substr(pos_begin, pos_end - pos_begin);
       }

   } else {
       if (std::string::npos == pos_end) {
           classname_delim = "/";
           token = filepath.substr(pos_begin + test_identifier.size(), std::string::npos);
           pos_end = token.find(classname_delim);
           token = token.substr(0, pos_end);
       } else {
           pos_end = filepath.find(classname_delim);
           pos_begin = pos_begin + test_identifier.size();
           token = filepath.substr(pos_begin, pos_end - pos_begin);
       }
   }
   return token;
}


