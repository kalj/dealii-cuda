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

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lineinfo)

IF(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -G -g )
ENDIF()

INCLUDE_DIRECTORIES(matrix_free_gpu)

# tests

CUDA_ADD_EXECUTABLE(test_spm test_spm.cu matrix_free_gpu/cuda_sparse_matrix.cu matrix_free_gpu/gpu_vec.cu)
TARGET_LINK_LIBRARIES(test_spm ${CUDA_cusparse_LIBRARY} ${LIBRARIES})

CUDA_ADD_EXECUTABLE(test_mg_transfer test_mg_transfer.cc matrix_free_gpu/gpu_vec.cu matrix_free_gpu/mg_transfer_matrix_free_gpu.cu)
TARGET_LINK_LIBRARIES(test_mg_transfer ${LIBRARIES})

CUDA_ADD_EXECUTABLE(test_hanging_nodes_gpu test_hanging_nodes_gpu.cu  matrix_free_gpu/coloring.cc)
TARGET_LINK_LIBRARIES(test_hanging_nodes_gpu ${LIBRARIES})

CUDA_ADD_EXECUTABLE(test_hanging_node_interpolation test_hanging_node_interpolation.cu)
TARGET_LINK_LIBRARIES(test_hanging_node_interpolation ${LIBRARIES})

CUDA_ADD_EXECUTABLE(test_dot test_dot.cpp matrix_free_gpu/gpu_vec.cu matrix_free_gpu/timing.cu)
TARGET_LINK_LIBRARIES(test_dot ${LIBRARIES})

CUDA_ADD_EXECUTABLE(test_laplace_op poisson_common.cc test_laplace_op.cu laplace_operator_cpu.cc matrix_free_gpu/gpu_vec.cu matrix_free_gpu/coloring.cc )
TARGET_LINK_LIBRARIES(test_laplace_op ${LIBRARIES})

# solvers

CUDA_ADD_EXECUTABLE(poisson_spm poisson_spm.cu poisson_common.cc matrix_free_gpu/cuda_sparse_matrix.cu matrix_free_gpu/gpu_vec.cu)
TARGET_LINK_LIBRARIES(poisson_spm ${CUDA_cusparse_LIBRARY} ${LIBRARIES})

CUDA_ADD_EXECUTABLE(poisson poisson.cu poisson_common.cc matrix_free_gpu/gpu_vec.cu matrix_free_gpu/coloring.cc)
TARGET_LINK_LIBRARIES(poisson ${LIBRARIES})

CUDA_ADD_EXECUTABLE(poisson_mg poisson_mg.cu poisson_common.cc matrix_free_gpu/gpu_vec.cu matrix_free_gpu/mg_transfer_matrix_free_gpu.cu matrix_free_gpu/coloring.cc )
TARGET_LINK_LIBRARIES(poisson_mg ${LIBRARIES})

CUDA_ADD_EXECUTABLE(poisson_mg_cpu poisson_mg_cpu.cc poisson_common.cc laplace_operator_cpu.cc)
TARGET_LINK_LIBRARIES(poisson_mg_cpu ${LIBRARIES})

# benchmarks

CUDA_ADD_EXECUTABLE(bmop bmop.cu poisson_common.cc matrix_free_gpu/gpu_vec.cu matrix_free_gpu/coloring.cc)
TARGET_LINK_LIBRARIES(bmop ${LIBRARIES})

CUDA_ADD_EXECUTABLE(bmop_spm bmop_spm.cu poisson_common.cc matrix_free_gpu/cuda_sparse_matrix.cu matrix_free_gpu/gpu_vec.cu)
TARGET_LINK_LIBRARIES(bmop_spm ${CUDA_cusparse_LIBRARY} ${LIBRARIES})

CUDA_ADD_EXECUTABLE(bmop-cpu bmop-cpu.cc laplace_operator_cpu.cc)
TARGET_LINK_LIBRARIES(bmop-cpu ${LIBRARIES})
