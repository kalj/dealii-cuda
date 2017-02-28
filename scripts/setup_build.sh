#!/bin/bash
#
# @(#)setup_build.sh
# @author Karl Ljungkvist <k.ljungkvist@gmail.com>

ADDITIONAL_FLAGS=$@

mkdir build_debug
cd build_debug
cmake  -D CMAKE_CXX_FLAGS="--std=c++11 -march=native ${ADDITIONAL_FLAGS}" -DCMAKE_BUILD_TYPE=Debug .. || exit 1
make -j matrix_free_gpu_lib || exit 1
cd ..
cp -r matrix_free_gpu  build_debug


mkdir build_release
cd build_release
cmake  -D CMAKE_CXX_FLAGS="--std=c++11 -march=native ${ADDITIONAL_FLAGS}" -DCMAKE_BUILD_TYPE=Release .. || exit 1
make -j matrix_free_gpu_lib  || exit 1
cd ..
cp -r matrix_free_gpu  build_release
