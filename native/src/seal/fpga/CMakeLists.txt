# native/src/seal/fpga/CMakeLists.txt

# Add HAL source files to the parent scope's SEAL_SOURCE_FILES variable
set(SEAL_SOURCE_FILES
    ${SEAL_SOURCE_FILES}
    ${CMAKE_CURRENT_LIST_DIR}/fpga.cpp
    ${CMAKE_CURRENT_LIST_DIR}/fpga_encoder.cpp
    # Add other .cpp files here
 PARENT_SCOPE)

# No target_include_directories needed here for the 'seal' target.
# Include paths are typically managed by the parent CMakeLists.txt
# that defines the 'seal' target.