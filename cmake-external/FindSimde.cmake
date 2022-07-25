find_path(SIMDE_INCLUDE_DIR simde/simde-math.h)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(simde
    FOUND_VAR
    	SIMDE_FOUND
    REQUIRED_VARS
	    SIMDE_INCLUDE_DIR
)

if(SIMDE_FOUND)
    add_library(simde INTERFACE IMPORTED)
    set_target_properties(simde PROPERTIES
	    INTERFACE_INCLUDE_DIRECTORIES "${SIMDE_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(SIMDE_INCLUDE_DIR)

include(FeatureSummary)
set_package_properties(simde PROPERTIES
    DESCRIPTION "Implementations of SIMD instruction sets for systems which don't natively support them."
    URL "https://github.com/simd-everywhere/simde"
)

