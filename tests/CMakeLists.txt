message(STATUS "Including GTest on Tests build")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR} ${CMAKE_MODULE_PATH}../cmake)

find_package(TensorFlow CONFIG REQUIRED)

add_subdirectory(tensorflowTests)
