
#if(True)
#set(QDLDL_FLOAT_TYPE "float")
#else()
#  set(QDLDL_FLOAT_TYPE "double")
#endif()
#
#if(True)
#set(QDLDL_INT_TYPE "long long")
#else()
#  set(QDLDL_INT_TYPE "int")
#endif()
#
#boolean type is always unsigned char
#for now, since _Bool does not exist in
#C89 and we want to avoid interoperability
#problems when calling QDLDL from C++
#set(QDLDL_BOOL_TYPE "unsigned char")
#configure_file(${PROJECT_SOURCE_DIR}/include/proxsuite/linalg/qdldl/qdldl_types.h.in
#               ${PROJECT_SOURCE_DIR}/include/proxsuite/linalg/qdldl/qdldl_types.h
#               NEWLINE_STYLE LF)

configure_file(${PROJECT_SOURCE_DIR}/include/proxsuite/linalg/qdldl/qdldl_types.h.in
               ${PROJECT_SOURCE_DIR}/include/proxsuite/linalg/qdldl/qdldl_types.h
               NEWLINE_STYLE LF)
set(
    amd_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_aat.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_control.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_defaults.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_info.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_order.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_post_tree.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_postorder.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_preprocess.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd_valid.c
    ${CMAKE_CURRENT_SOURCE_DIR}/SuiteSparse_config.c
)

set(qdldl_interface_src
    ${CMAKE_CURRENT_SOURCE_DIR}/lin_alg.c
    ${CMAKE_CURRENT_SOURCE_DIR}/lin_sys.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cs.c
    ${CMAKE_CURRENT_SOURCE_DIR}/kkt.c
    ${CMAKE_CURRENT_SOURCE_DIR}/qdldl_interface.c
    ${CMAKE_CURRENT_SOURCE_DIR}/qdldl.c
)

# Create object library for linear system solver interface
add_library(linsys_qdldl SHARED ${qdldl_interface_src} ${amd_sources})
