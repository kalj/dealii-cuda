CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

PROJECT(poisson)

FIND_PACKAGE(CUDA REQUIRED)

# deal.II package for deal.II support
FIND_PACKAGE(deal.II 8.0 REQUIRED
  HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )

IF (deal.II_FOUND)
    INCLUDE_DIRECTORIES(${DEAL_II_INCLUDE_DIRS})
    IF (${CMAKE_BUILD_TYPE} STREQUAL "DEBUG")
        LINK_DIRECTORIES(${DEAL_II_LIBRARIES_DEBUG})
        SET(LIBRARIES
            ${LIBRARIES}
            ${DEAL_II_LIBRARIES_DEBUG}
            )
    ELSE()
        LINK_DIRECTORIES(${DEAL_II_LIBRARIES_RELEASE})
        SET(LIBRARIES
            ${LIBRARIES}
            ${DEAL_II_LIBRARIES_RELEASE}
            )
        ENDIF()
    MESSAGE("-- DEAL.II include files found at ${DEAL_II_INCLUDE_DIRS}")
ELSE()
    MESSAGE("-- DEAL.II package not found")
ENDIF()

SITE_NAME(HOSTNAME)

MESSAGE("-- Setting up for Compute Capability 3.5")
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_35)

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++11 -lineinfo)

IF(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -G)
ENDIF()

INCLUDE_DIRECTORIES(matrix_free_gpu)

CUDA_ADD_EXECUTABLE(poisson poisson.cu matrix_free_gpu/gpu_vec.cu matrix_free_gpu/coloring.cc)
TARGET_LINK_LIBRARIES(poisson ${LIBRARIES})

CUDA_ADD_EXECUTABLE(bmop bmop.cu matrix_free_gpu/gpu_vec.cu matrix_free_gpu/coloring.cc)
TARGET_LINK_LIBRARIES(bmop ${LIBRARIES})

CUDA_ADD_EXECUTABLE(test_hanging_nodes_gpu test_hanging_nodes_gpu.cu  matrix_free_gpu/coloring.cc)
TARGET_LINK_LIBRARIES(test_hanging_nodes_gpu ${LIBRARIES})

CUDA_ADD_EXECUTABLE(test_hanging_node_interpolation test_hanging_node_interpolation.cu)
TARGET_LINK_LIBRARIES(test_hanging_node_interpolation ${LIBRARIES})

CUDA_ADD_EXECUTABLE(test_dot test_dot.cpp matrix_free_gpu/gpu_vec.cu matrix_free_gpu/timing.cu)
TARGET_LINK_LIBRARIES(test_dot ${LIBRARIES})
