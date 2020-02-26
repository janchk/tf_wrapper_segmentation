project(TF_WRAPPER_SEGMENTATION_EXAMPLE)

add_executable(TF_SEGMENTATOR main.cpp)

configure_file(../../resources/config.json
               ${CMAKE_BINARY_DIR}/application/example COPYONLY)

target_link_libraries(TF_SEGMENTATOR tf_wrapper)
target_include_directories(TF_SEGMENTATOR
                           PUBLIC ${TF_WRAPPER_SEGMENTATION_SOURCE_DIR})
