cmake_minimum_required (VERSION 3.13)

#####################################################################
#                  Start of configuration section                   #
#####################################################################

# Set the build to always be local
set(SE_BUILD_LOCAL ON CACHE BOOL "Build for native instead of device" FORCE)

# Set the compilers to Intel OneAPI
set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)
set(FPGA_DEVICE "Agilex7")
set(EMULATOR_TARGET fpga_emu)
set(REPORT_TARGET report)
set(TARGET_NAME sealembedded)
set(REPORT_OUTPUT_NAME ${TARGET_NAME}.${REPORT_TARGET})

# Enable C99 support for C files
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Rno-debug-disables-optimization")

# Keep the emulator flag for the main build
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFPGA_EMULATOR -fsycl -std=c++17 -Rno-debug-disables-optimization")

# Define the hardware flags separately - used only for report generation
set(HARDWARE_FLAGS "-DFPGA_HARDWARE -fintelfpga")
set(REPORT_LINK_FLAGS "-Xshardware -Xstarget=${FPGA_DEVICE} -fsycl-link=early")

project(SEAL_EMBEDDED VERSION 1.1.0 LANGUAGES C CXX ASM)

#################
# Configuration #
#################

# Always build position-independent-code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set the directory for SEAL-Embedded Adapter outputs
set(SE_ADAPTER_FILE_OUTPUT_DIR ${CMAKE_CURRENT_LIST_DIR}/adapter_output_data
    CACHE STRING "Adapter file directory" FORCE)
message(STATUS "Adapter file directory: ${SE_ADAPTER_FILE_OUTPUT_DIR}")

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug" "MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "Build type (CMAKE_BUILD_TYPE): ${CMAKE_BUILD_TYPE}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  include(CheckCXXCompilerFlag)
  function(se_enable_cxx_compiler_flag_if_supported flag)
    string(FIND "${CMAKE_CXX_FLAGS}" "${flag}" flag_already_set)
    if(flag_already_set EQUAL -1)
      message(STATUS "Adding CXX compiler flag: ${flag} ...")
      check_cxx_compiler_flag("${flag}" flag_supported)
      if(flag_supported)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
      endif()
      unset(flag_supported CACHE)
    endif()
  endfunction()
  if(NOT MSVC AND SEAL_DEBUG)
    se_enable_cxx_compiler_flag_if_supported("-Wall")
    se_enable_cxx_compiler_flag_if_supported("-Wextra")
    se_enable_cxx_compiler_flag_if_supported("-Wconversion")
    se_enable_cxx_compiler_flag_if_supported("-Wshadow")
    se_enable_cxx_compiler_flag_if_supported("-pedantic")
  endif()
endif()

# Build executables or library
if(NOT SE_BUILD_TYPE)
  set(SE_BUILD_TYPE "Tests" CACHE STRING "SEAL-Embedded build type" FORCE)
  set_property(CACHE SE_BUILD_TYPE PROPERTY STRINGS "Tests" "Bench" "Lib")
endif()
message(STATUS "Build type (SE_BUILD_TYPE): ${SE_BUILD_TYPE}")

set(SE_ADAPTER_FILE_OUTPUT_DIR_DEVICE adapter_output_data
    CACHE STRING "Adapter file directory (build)" FORCE)

set_property(CACHE SE_BUILD_TYPE PROPERTY STRINGS "Lib" "Bench" "Tests")
message(STATUS "BUILD LOCAL (SE_BUILD_LOCAL): ${SE_BUILD_LOCAL}")
message(STATUS "CMAKE build type (CMAKE_BUILD_TYPE): ${CMAKE_BUILD_TYPE}")
message(STATUS "SEAL-Embedded build type (SE_BUILD_TYPE): ${SE_BUILD_TYPE}")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_RUNTIME_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_CURRENT_BINARY_DIR}/bin)

set(SE_LIB_SOURCE_FILES "")
set(SE_BENCH_SOURCE_FILES "")
set(SE_TESTS_SOURCE_FILES "")

add_subdirectory(lib)
add_subdirectory(bench)
add_subdirectory(test)

set(CMAKE_ASM_FLAGS "${CMAKE_C_FLAGS} -x assembler-with-cpp")

# Set up the library
add_library(seal_embedded STATIC ${SE_LIB_SOURCE_FILES})
target_include_directories(seal_embedded PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/lib>)
target_include_directories(seal_embedded PUBLIC ${SE_ADAPTER_FILE_OUTPUT_DIR})
target_link_libraries(seal_embedded PRIVATE m)
set_target_properties(seal_embedded PROPERTIES VERSION ${SEAL_EMBEDDED_VERSION})
set_target_properties(seal_embedded PROPERTIES 
  sealembedded seal_embedded-${SEAL_EMBEDDED_VERSION_MAJOR}.${SEAL_EMBEDDED_VERSION_MINOR})

# Build executables based on SE_BUILD_TYPE.
if(SE_BUILD_TYPE STREQUAL "Bench")
  add_executable(seal_embedded_bench ${SE_BENCH_SOURCE_FILES})
  target_link_libraries(seal_embedded_bench PRIVATE m gcc_s c seal_embedded)
elseif(SE_BUILD_TYPE STREQUAL "Tests")
  # Build the test binary normally for the emulator (with emulator flags)
  add_executable(seal_embedded_tests ${SE_TESTS_SOURCE_FILES})
  target_link_libraries(seal_embedded_tests PRIVATE m gcc_s c seal_embedded)
  
  # Create a script focusing only on the SYCL_ckks_sym.cpp file
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/generate_report.sh "#!/bin/bash
echo \"Generating FPGA hardware report for SYCL_ckks_sym.cpp...\"

# The specific file containing all FPGA kernels
KERNEL_FILE=\"${CMAKE_CURRENT_LIST_DIR}/lib/SYCL_ckks_sym.cpp\"

# Include directories needed for compilation
INCLUDE_DIRS=\"-I${CMAKE_CURRENT_LIST_DIR}/lib -I${SE_ADAPTER_FILE_OUTPUT_DIR}\"

# Compile with hardware flags to get a report
echo \"Compiling with OneAPI for FPGA hardware report...\"
icpx -fsycl ${HARDWARE_FLAGS} ${REPORT_LINK_FLAGS} ${INCLUDE_DIRS} \\
     -o \"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${REPORT_OUTPUT_NAME}\" \\
     \"$KERNEL_FILE\"

# Check if the report generation succeeded
if [ $? -eq 0 ]; then
  echo \"Hardware report generated successfully\"
  echo \"Report file: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${REPORT_OUTPUT_NAME}\"
  
  # Check if report contains kernel information
  KERNEL_COUNT=\$(grep -c \"Kernel\" \"${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${REPORT_OUTPUT_NAME}\" || echo 0)
  if [ \$KERNEL_COUNT -gt 0 ]; then
    echo \"Success: Report contains \$KERNEL_COUNT kernel(s)\"
  else
    echo \"Warning: No kernel information found in the report\"
  fi
else
  echo \"Hardware report generation failed with exit code $?\"
  echo \"Check the error messages above for details\"
fi
")

  # Make the script executable
  file(CHMOD ${CMAKE_CURRENT_BINARY_DIR}/generate_report.sh
       PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  
  # Add a custom target to run the script
  add_custom_target(
    fpga_report
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/generate_report.sh
    COMMENT "Generating FPGA hardware report for SYCL_ckks_sym.cpp"
    VERBATIM
  )
  
  # Add a simple post-build step with report information
  add_custom_command(TARGET seal_embedded_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Main build successful. To generate the FPGA hardware report, run: cmake --build build --target fpga_report"
  )
else()
  message(STATUS "No executable will be built")
endif()

# Set defines for library
add_definitions(-DSE_DATA_PATH="${SE_ADAPTER_FILE_OUTPUT_DIR_DEVICE}")
string(LENGTH "${SE_ADAPTER_FILE_OUTPUT_DIR_DEVICE}" SE_ADAPTER_FILE_OUTPUT_DIR_DEVICE_LEN)
add_definitions(-DSE_DATA_PATH_LEN=${SE_ADAPTER_FILE_OUTPUT_DIR_DEVICE_LEN})

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_definitions(-DSE_UNDEF_ASSERT_FORCE)
endif()