cmake_minimum_required(VERSION 3.10)
project(tf_wrapper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules")

find_package(TensorFlow REQUIRED)
find_package(Eigen3 REQUIRED)
#find_package(Protobuf REQUIRED)
find_package(absl REQUIRED)
find_package(OpenCV REQUIRED)
#find_package(Glog REQUIRED)

add_library(tf_wrapper src/tensorflow_base.cpp
  src/tensorflow_wrapper_core.cpp
  src/tensorflow_segmentator.cpp
  src/tensorflow_auxiliary.cpp
  src/wrapper_base.cpp
  src/common/fs_handling.cpp
  src/common/common_ops.cpp
  )

target_include_directories(tf_wrapper PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_include_directories(tf_wrapper PRIVATE ${TensorFlow_INCLUDE_DIRS} ${Protobuf_INCLUDE_DIRS})
target_link_libraries(tf_wrapper PRIVATE stdc++fs ${OpenCV_LIBS}  Eigen3::Eigen ${TensorFlow_LIBRARIES} ${Protobuf_LIBRARIES} absl::base absl::synchronization absl::strings ${GLOG_LIBRARIES})