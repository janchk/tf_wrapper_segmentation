cmake_minimum_required(VERSION 3.10)
project(${PROJECT_NAME})

#set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH
    ${PROJECT_SOURCE_DIR}/../cmake
    ${CMAKE_MODULE_PATH})

find_package(TensorFlow CONFIG REQUIRED)
#message("${CMAKE_MODULE_PATH}")
find_package( OpenCV REQUIRED )

add_library(${PROJECT_NAME} STATIC)

target_include_directories(${PROJECT_NAME}
        PUBLIC
        ~/.local/include
        ${PROJECT_SOURCE_DIR}/../include/
        )

target_sources(${PROJECT_NAME}
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/tensorflow_base.h
        ${CMAKE_CURRENT_LIST_DIR}/tensorflow_wrapper_core.h
        ${CMAKE_CURRENT_LIST_DIR}/tensorflow_segmentator.h
        ${CMAKE_CURRENT_LIST_DIR}/tensorflow_auxiliary.h
        ${CMAKE_CURRENT_LIST_DIR}/wrapper_legacy.h
        ${CMAKE_CURRENT_LIST_DIR}/wrapper_base.h
        ${CMAKE_CURRENT_LIST_DIR}/common/fs_handling.h
        ${CMAKE_CURRENT_LIST_DIR}/common/common_ops.h
        PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/tensorflow_base.cpp
        ${CMAKE_CURRENT_LIST_DIR}/tensorflow_wrapper_core.cpp
        ${CMAKE_CURRENT_LIST_DIR}/tensorflow_segmentator.cpp
        ${CMAKE_CURRENT_LIST_DIR}/tensorflow_auxiliary.cpp
        ${CMAKE_CURRENT_LIST_DIR}/wrapper_legacy.cpp
        ${CMAKE_CURRENT_LIST_DIR}/wrapper_base.cpp
        ${CMAKE_CURRENT_LIST_DIR}/common/fs_handling.cpp
        ${CMAKE_CURRENT_LIST_DIR}/common/common_ops.cpp

)

# message("${TensorFlow_INCLUDE_DIRS}")

#find_package(Protobuf REQUIRED)
message(TF LIBRARIES ${TensorFlow_LIBRARIES})

#set(PROJECT_LIBRARIES ${TensorFlow_LIBRARIES} ${PROTOBUF_LIBRARIES})

target_include_directories(${PROJECT_NAME} PRIVATE)
target_link_libraries(${PROJECT_NAME} stdc++fs TensorFlow::TensorFlow ${OpenCV_LIBS})




find_package(CUDA)
if(CUDA_FOUND)
  target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
endif()

#message("${TensorFlow_LIBRARIES}")
# message("${TensorFlow_INCLUDE_DIRS}")
