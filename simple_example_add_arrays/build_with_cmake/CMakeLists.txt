#------------------------------------------------------------------------------
# - CMakeLists.txt 
#
# - Assuming that we build in the bin (sub)directory,
#   it executes as follows:
#   cmake -DProject=add ..
#   make 
#------------------------------------------------------------------------------
#
# - Define CMake requirements and override make rules as needed
#
cmake_minimum_required(VERSION 3.11)

#------------------------------------------------------------------------------
# - Project definition
#------------------------------------------------------------------------------
project(${Project})

SET(PROJECT_TOP_DIR "./../../")
#------------------------------------------------------------------------------
# - Find packages : CUDA
#------------------------------------------------------------------------------
FIND_PACKAGE(CUDA REQUIRED)

#------------------------------------------------------------------------------
# - cuda SDK and libs 
#------------------------------------------------------------------------------
SET(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-10.0")

#------------------------------------------------------------------------------
# - Build Options: Release (default --optimize 3) or Debug (-G -lineinfo)
#------------------------------------------------------------------------------
SET(CMAKE_BUILD_TYPE Release)
#
# NOTE (JVY): If nvcc flags are NOT used (nvcc default), executable can be
#             fully profiled with nvprof, including CUDA kernels
#             If flags --use_fast_math and --ptxas-options=-v are added, 
#             nvprof is still able to profile CUDA kernels
#             If flag --arch=sm_50 is added, says that kernels are not profiled
#             The --arch (-arch) is the shortcut for --gpu-architechture;
#             see details at the following link:
# https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation
# 
# ---> SET(CUDA_NVCC_FLAGS -arch=sm_50 --use_fast_math --ptxas-options=-v)
SET(CUDA_NVCC_FLAGS --use_fast_math --ptxas-options=-v)

INCLUDE_DIRECTORIES( ${CUDA_TOOLKIT_ROOT_DIR}/include )
LINK_DIRECTORIES( ${CUDA_TOOLKIT_ROOT_DIR}/lib64 )

#------------------------------------------------------------------------------
# - executable and link 
#------------------------------------------------------------------------------
cuda_add_executable ( ${Project}
                      ${Project}.cu
)

target_link_libraries(${Project} ${CUDA_LIBRARIES} )

#------------------------------------------------------------------------------
# End of CMakeLists.txt
#------------------------------------------------------------------------------
