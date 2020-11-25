cmake_minimum_required(VERSION 3.9.0)
project(CambriconSDK)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/cnrt.cmake)
include(cmake/cndev.cmake)
include(cmake/cnml.cmake)
list(APPEND CAMBRICON_LIBS libcnrt libcndev libcnml)
set(CAMBRICON_LIBS "${CAMBRICON_LIBS}")
set(CAMBRICON_INCLUDE_DIR "${CNML_INCLUDE_DIR}")

find_package(PythonLibs ${PYTHON_VERSION_STRING} REQUIRED)

find_package(SWIG REQUIRED)
set(SDK_OUTPUT_LIB_NAME cambriconLib)
set(SWIG_SDK_SRC swig/cambricon.i)
set(CMAKE_SWIG_FLAGS -Wall -threads -py3 -modern -DSWIGWORDSIZE64 -I${CAMBRICON_INCLUDE_DIR})

include(UseSWIG)
set_property(SOURCE ${SWIG_SDK_SRC} PROPERTY CPLUSPLUS ON)
swig_add_library(${SDK_OUTPUT_LIB_NAME} LANGUAGE python SOURCES ${SWIG_SDK_SRC})

set_target_properties(_${SDK_OUTPUT_LIB_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${_sdk_name})
target_include_directories(_${SDK_OUTPUT_LIB_NAME} PRIVATE ${PYTHON_INCLUDE_DIRS} ${CAMBRICON_INCLUDE_DIR} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(_${SDK_OUTPUT_LIB_NAME} ${PYTHON_LIBRARIES} ${CAMBRICON_LIBS})

if (TARGET _${SDK_OUTPUT_LIB_NAME})
    add_custom_target(
        develop
        COMMAND ${CMAKE_COMMAND} -E create_symlink
          ${CMAKE_CURRENT_BINARY_DIR}/$<TARGET_FILE_NAME:_${SDK_OUTPUT_LIB_NAME}>
          ${CMAKE_CURRENT_SOURCE_DIR}/lib/cnlib/$<TARGET_FILE_NAME:_${SDK_OUTPUT_LIB_NAME}>
        COMMAND ${CMAKE_COMMAND} -E create_symlink
          ${CMAKE_CURRENT_BINARY_DIR}/${SDK_OUTPUT_LIB_NAME}.py
          ${CMAKE_CURRENT_SOURCE_DIR}/lib/cnlib/${SDK_OUTPUT_LIB_NAME}.py
        DEPENDS _${SDK_OUTPUT_LIB_NAME}
    )
endif()
