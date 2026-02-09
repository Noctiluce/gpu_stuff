cmake_minimum_required(VERSION 3.18)


set(PROJECT_NAME "Vector_And_Matrices_Operations")

project(${PROJECT_NAME} LANGUAGES CXX CUDA)


add_executable(${PROJECT_NAME} main.cu)


set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


#set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#target_compile_options(${PROJECT_NAME} PRIVATE
#    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
#    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -Wall -Xcompiler -Wextra>
#)
