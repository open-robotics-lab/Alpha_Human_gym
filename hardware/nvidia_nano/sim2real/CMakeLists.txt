cmake_minimum_required(VERSION 3.25)
project(sim2real)

set(CMAKE_CXX_STANDARD 17)

# 找到 CUDA
#set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.4/bin/nvcc")
#set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.4)
#set(CUDA_INCLUDE_DIRS "/usr/local/cuda-12.4/include")
#set(CUDA_LIBRARY_DIRS "/usr/local/cuda-12.4/lib64")
#find_package(CUDA REQUIRED)
# 找到 CUDA
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-10.2/bin/nvcc")
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.2)
set(CUDA_INCLUDE_DIRS "/usr/local/cuda-10.2/include")
set(CUDA_LIBRARY_DIRS "/usr/local/cuda-10.2/lib64")
set(cuBLAS_LIBRARY_DIRS "/usr/lib/aarch64-linux-gnu")
find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
#include_directories(${cuBLAS_LIBRARY_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})


include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

# 设置 LibTorch 的路径
set(Torch_DIR "/home/nvidia/rlgpu/lib/python3.6/site-packages/torch")
set(CMAKE_PREFIX_PATH "/home/nvidia/rlgpu/lib/python3.6/site-packages/")  # 将此路径替换为您解压 LibTorch 的路径
find_package(Torch REQUIRED)
include_directories(${Torch_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/lcm_types)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(sim2real_h1 src/sim2real_h1.cpp)
target_link_libraries(sim2real_h1 ${TORCH_LIBRARIES})