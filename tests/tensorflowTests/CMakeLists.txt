set(MODULE_NAME tensorflowTests)
set(SRC_FILES
        ${CMAKE_CURRENT_LIST_DIR}/testTensorflowCorrectness.cpp
        )

add_executable(${MODULE_NAME} ${SRC_FILES})

target_link_libraries(${MODULE_NAME} gtest gtest_main ${TensorFlow_LIBRARIES} tf_wrapper)

target_include_directories(${MODULE_NAME} PUBLIC
        ${TF_WRAPPER_SEGMENTATION_SOURCE_DIR} ${TensorFlow_INCLUDE_DIRS})

add_test(
        NAME
        ${MODULE_NAME}
        COMMAND
        ./${MODULE_NAME}
)