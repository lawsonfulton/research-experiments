cmake_minimum_required (VERSION 3.5)

project(BucklingBar)

set(CMAKE_CXX_STANDARD 14)

#Output directories
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib )

#grab important packages
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#Gauss
set(Gauss_ROOT_DIR "/home/lawson/Workspace/GAUSS/")
find_package(Gauss REQUIRED)

#get the source code files
file(GLOB HEADER_CODE ${GBucklingBar_SOURCE_DIR}/*.h)
file(GLOB SRC_CODE ${BucklingBar_SOURCE_DIR}/*.cpp)

include_directories(${Gauss_INCLUDE_DIRS})

#this call is necessary if any targets are GUI applications
#initUI() 

#Example codes
link_directories(${Gauss_LIB_DIR_DEBUG})
add_executable(BucklingBar ${BucklingBar_SOURCE_DIR}/main.cpp)
target_link_libraries(BucklingBar ${Gauss_LIBS} ${UI_LIBS})

