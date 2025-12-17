cmake_minimum_required (VERSION 3.13)

#####################################################################
#                  Start of configuration section                   #
#####################################################################

set(SE_BUILD_LOCAL ON CACHE BOOL "Build for native instead of device" FORCE)
set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)

set(FPGA_DEVICE "Agilex7")
set(TARGET_NAME sealembedded)
set(REPORT_ARTIFACT_NAME ${TARGET_NAME}_report_artifact)

set(BASE_C_FLAGS "-std=c99 -Rno-debug-disables-optimization")
set(BASE_CXX_FLAGS "-fsycl -std=c++17 -Rno-debug-disables-optimization")

# --- Report-specific Flags ---
set(REPORT_COMPILE_FLAGS "-fintelfpga")
set(REPORT_LINK_FLAGS "-Xshardware -Xstarget=${FPGA_DEVICE} -fsycl-link=early")

set(EMULATOR_COMPILE_FLAGS "-fintelfpga -DFPGA_EMULATOR")

project(SEAL_EMBEDDED VERSION 1.1.0 LANGUAGES C CXX ASM)

#################
# Configuration #
#################

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(SE_ADAPTER_FILE_OUTPUT_DIR ${CMAKE_CURRENT_LIST_DIR}/adapter_output_data CACHE STRING "Adapter file directory" FORCE)
message(STATUS "Adapter file directory: ${SE_ADAPTER_FILE_OUTPUT_DIR}")

set(CMAKE_BUILD_TYPE_LIST Debug Release MinSizeRel RelWithDebInfo Report)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CMAKE_BUILD_TYPE_LIST})
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type (choose from ${CMAKE_BUILD_TYPE_LIST})" FORCE)
endif()
message(STATUS "Build type (CMAKE_BUILD_TYPE): ${CMAKE_BUILD_TYPE}")

set(CMAKE_C_FLAGS "${BASE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${BASE_CXX_FLAGS}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} ${EMULATOR_COMPILE_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} ${EMULATOR_COMPILE_FLAGS}")
set(CMAKE_CXX_FLAGS_REPORT "${CMAKE_CXX_FLAGS} ${REPORT_COMPILE_FLAGS} -DFPGA_HARDWARE")
set(CMAKE_EXE_LINKER_FLAGS_REPORT "${REPORT_LINK_FLAGS}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  include(CheckCXXCompilerFlag)
  function(se_enable_cxx_compiler_flag_if_supported flag)
    string(FIND "${CMAKE_CXX_FLAGS_DEBUG}" "${flag}" flag_already_set)
    if(flag_already_set EQUAL -1)
      check_cxx_compiler_flag("${BASE_CXX_FLAGS} ${flag}" flag_supported)
      if(flag_supported)
        message(STATUS "Adding CXX compiler flag for Debug: ${flag} ...")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${flag}" PARENT_SCOPE)
      endif()
      unset(flag_supported CACHE)
    endif()
  endfunction()
  if(NOT MSVC AND DEFINED SEAL_DEBUG AND SEAL_DEBUG)
    se_enable_cxx_compiler_flag_if_supported("-Wall")
    se_enable_cxx_compiler_flag_if_supported("-Wextra")
    se_enable_cxx_compiler_flag_if_supported("-Wconversion")
    se_enable_cxx_compiler_flag_if_supported("-Wshadow")
    se_enable_cxx_compiler_flag_if_supported("-pedantic")
  endif()
endif()

if(NOT SE_BUILD_TYPE)
  set(SE_BUILD_TYPE "Tests" CACHE STRING "SEAL-Embedded build type (Tests, Lib)" FORCE)
endif()
set_property(CACHE SE_BUILD_TYPE PROPERTY STRINGS "Lib" "Tests")
message(STATUS "Build type (SE_BUILD_TYPE): ${SE_BUILD_TYPE}")

set(SE_ADAPTER_FILE_OUTPUT_DIR_DEVICE adapter_output_data CACHE STRING "Adapter file directory (build)" FORCE)
message(STATUS "BUILD LOCAL (SE_BUILD_LOCAL): ${SE_BUILD_LOCAL}")
message(STATUS "SEAL-Embedded build type (SE_BUILD_TYPE): ${SE_BUILD_TYPE}")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${CMAKE_BINARY_DIR}/bin)

set(SE_LIB_SOURCE_FILES "")
set(SE_TESTS_SOURCE_FILES "")

add_subdirectory(lib)
add_subdirectory(test)

set(CMAKE_ASM_FLAGS "${CMAKE_C_FLAGS} -x assembler-with-cpp")

add_library(${TARGET_NAME} STATIC ${SE_LIB_SOURCE_FILES})
target_include_directories(${TARGET_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/lib>)
target_include_directories(${TARGET_NAME} PUBLIC ${SE_ADAPTER_FILE_OUTPUT_DIR})
target_link_libraries(${TARGET_NAME} PRIVATE m)

if(NOT DEFINED SEAL_EMBEDDED_VERSION)
  set(SEAL_EMBEDDED_VERSION "1.1.0")
endif()
set_target_properties(${TARGET_NAME} PROPERTIES VERSION ${SEAL_EMBEDDED_VERSION})
set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${TARGET_NAME})

if(SE_BUILD_TYPE STREQUAL "Tests")
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/main.c")
        add_executable(seal_embedded_tests ${SE_TESTS_SOURCE_FILES})
        target_link_libraries(seal_embedded_tests PRIVATE m gcc_s c ${TARGET_NAME})
    else()
        message(WARNING "Test sources not found, skipping test executable.")
    endif()
else() # SE_BUILD_TYPE is "Lib"
    message(STATUS "Only the static library will be built by default (SE_BUILD_TYPE is Lib).")
endif()

# --- Define Custom Target for Report File Generation ---
if(TARGET seal_embedded_tests OR SE_BUILD_TYPE STREQUAL "Lib")
    add_custom_target(fpga_report_file
      COMMAND ${CMAKE_CXX_COMPILER}
              # Essential flags only:
              -v
              -fsycl
              -fintelfpga
              -Xshardware             
              -Xstarget=${FPGA_DEVICE} 
              -fsycl-link=early       
              ../lib/SYCL_ckks_sym.cpp
      COMMENT "Compiling SYCL file to generate FPGA hardware report (minimal flags)"
      VERBATIM
      # Run command from build directory
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
endif()

# --- Add Post-Build Message ---
if (TARGET seal_embedded_tests)
    add_custom_command(TARGET seal_embedded_tests POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Build successful for config ${CMAKE_BUILD_TYPE}. To generate report files, run: cmake --build ${CMAKE_BINARY_DIR} --target fpga_report_file"
    )
endif()
if(SE_BUILD_TYPE STREQUAL "Lib")
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Library build successful. To build test exe, change SE_BUILD_TYPE. To generate report files, run: cmake --build ${CMAKE_BINARY_DIR} --target fpga_report_file"
    )
endif()

add_compile_definitions(SE_DATA_PATH="${SE_ADAPTER_FILE_OUTPUT_DIR_DEVICE}")
string(LENGTH "${SE_ADAPTER_FILE_OUTPUT_DIR_DEVICE}" SE_ADAPTER_FILE_OUTPUT_DIR_DEVICE_LEN)
add_compile_definitions(SE_DATA_PATH_LEN=${SE_ADAPTER_FILE_OUTPUT_DIR_DEVICE_LEN})
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_definitions(SE_UNDEF_ASSERT_FORCE)
endif()