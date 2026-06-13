cmake_minimum_required(VERSION 3.18)

project(gsplatcu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find by `python -c "import torch; print(torch.utils.cmake_prefix_path)"`
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../torch/.venv/lib/python3.14/site-packages/torch/share/cmake")
find_package(Torch REQUIRED)


add_library(gsplatcu STATIC
        gsplatcu/gausplat.cu
        gsplatcu/kernel.cu
        gsplatcu/additional.cu
)

set_target_properties(gsplatcu PROPERTIES CUDA_ARCHITECTURES "86")

target_include_directories(gsplatcu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gsplatcu PUBLIC ${TORCH_LIBRARIES})
set_target_properties(gsplatcu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
