
# Minimum version required

# # Project name
# project (linsys_qdldl)

# # Set the output folder where your program will be created
# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/out)
# set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/out)

# Add qdldl
add_subdirectory(qdldl_sources)

# Use long integers for indexing
# set(DLONG ON CACHE BOOL "Use long integers (64bit) for indexing")
# if (NOT (CMAKE_SIZEOF_VOID_P EQUAL 8))
# 	message(STATUS "Disabling long integers (64bit) on 32bit machine")
# 	set(DLONG OFF)
#   endif()
# message(STATUS "Long integers (64bit) are ${DLONG}")

# if(DLONG)
#   set(QDLDL_INT_TYPE "long long")
# 	set(QDLDL_INT_TYPE_MAX "LLONG_MAX")
# else()
# 	set(QDLDL_INT_TYPE "int")
# 	set(QDLDL_INT_TYPE_MAX "INT_MAX")
# endif()

if(NOT DEFINED EMBEDDED)
set(
    amd_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/include/amd_internal.h
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/include/amd.h
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/include/SuiteSparse_config.h
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_1.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_2.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_aat.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_control.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_defaults.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_info.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_order.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_post_tree.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_postorder.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_preprocess.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/amd_valid.c
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/src/SuiteSparse_config.c
)
endif()


set(qdldl_interface_includes
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/amd/include
    ${CMAKE_CURRENT_SOURCE_DIR}/qdldl_sources
)

set(qdldl_interface_src
    ${amd_sources}
    ${CMAKE_CURRENT_SOURCE_DIR}/types.h
    ${CMAKE_CURRENT_SOURCE_DIR}/lin_alg.h
    ${CMAKE_CURRENT_SOURCE_DIR}/lin_alg.c

    ${CMAKE_CURRENT_SOURCE_DIR}/lin_sys.h
    ${CMAKE_CURRENT_SOURCE_DIR}/lin_sys.c

    ${CMAKE_CURRENT_SOURCE_DIR}/cs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/cs.c

    ${CMAKE_CURRENT_SOURCE_DIR}/kkt.h
    ${CMAKE_CURRENT_SOURCE_DIR}/kkt.c

    ${CMAKE_CURRENT_SOURCE_DIR}/glob_opts.h
    ${CMAKE_CURRENT_SOURCE_DIR}/constants.h

    ${CMAKE_CURRENT_SOURCE_DIR}/qdldl_interface.h
    ${CMAKE_CURRENT_SOURCE_DIR}/qdldl_interface.c
)

# Create object library for linear system solver interface
add_library(linsys_qdldl SHARED ${qdldl_interface_src})
# Declare include directories for the cmake exported target
# target_include_directories(linsys_qdldl 
# 	PUBLIC "$<BUILD_INTERFACE:${qdldl_interface_includes}>"
# 	"$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}/linsys_qdldl>")

#target_include_directories(linsys_qdldl PUBLIC ${qdldl_interface_includes} ${PROJECT_SOURCE_DIR})
target_link_libraries(linsys_qdldl PUBLIC qdldl)

install(
  TARGETS linsys_qdldl
  EXPORT ${TARGETS_EXPORT_NAME}
  DESTINATION ${${PYWRAP}_INSTALL_DIR})

# install(FILES ${qdldl_interface_src} DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/linsys_qdldl")

# # Install linsys_qdldl object library
# install(TARGETS linsys_qdldl
# 	EXPORT  ${PROJECT_NAME}
# 	LIBRARY       DESTINATION "${CMAKE_INSTALL_LIBDIR}"
# 	ARCHIVE       DESTINATION "${CMAKE_INSTALL_LIBDIR}"
# 	RUNTIME       DESTINATION "${CMAKE_INSTALL_BINDIR}")