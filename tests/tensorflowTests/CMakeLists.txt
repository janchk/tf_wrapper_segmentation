set(MODULE_NAME tensorflowTests)
set (SRC_FILES
        ${CMAKE_CURRENT_LIST_DIR}/testTensorflowCorrectness.cpp
        )

add_executable(${MODULE_NAME} ${SRC_FILES})



#file(MAKE_DIRECTORY ${TEST_DATA_PATH_QUERIES})


target_link_libraries(${MODULE_NAME} gtest gtest_main TF_WRAPPER_SEGMENTATION)

target_include_directories(${MODULE_NAME} PUBLIC
                            ${TF_WRAPPER_SEGMENTATION_SOURCE_DIR})

add_test(
        NAME
            ${MODULE_NAME}
        COMMAND
            ./${MODULE_NAME}
)