#!/bin/bash
#
# @(#)setup_build.sh
# @author Karl Ljungkvist <k.ljungkvist@gmail.com>

ADDITIONAL_FLAGS=$@

rm -rf build_debug
rm -rf build_release

mkdir build_debug
mkdir build_release

(
cd build_debug
cmake  -D CMAKE_CXX_FLAGS="--std=c++11 -march=native ${ADDITIONAL_FLAGS}" -DCMAKE_BUILD_TYPE=Debug .. || exit 1
make -j matrix_free_gpu_lib || exit 1
cp -r ../matrix_free_gpu .
) &


(
cd build_release
cmake  -D CMAKE_CXX_FLAGS="--std=c++11 -march=native ${ADDITIONAL_FLAGS}" -DCMAKE_BUILD_TYPE=Release .. || exit 1
make -j matrix_free_gpu_lib  || exit 1
cp -r ../matrix_free_gpu .
) &

wait
