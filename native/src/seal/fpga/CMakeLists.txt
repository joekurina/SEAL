# native/src/seal/fpga/CMakeLists.txt

# Option to enable FPGA support (requires Intel oneAPI/SYCL compiler)
option(SEAL_USE_FPGA "Enable FPGA acceleration support" OFF)

if(SEAL_USE_FPGA)
    message(STATUS "FPGA support enabled")

    # Check for Intel oneAPI compiler (icpx)
    if(NOT (CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
        message(WARNING "FPGA support requires Intel oneAPI compiler (icpx). Current compiler: ${CMAKE_CXX_COMPILER_ID}")
        message(WARNING "Disabling FPGA support.")
        set(SEAL_USE_FPGA OFF)
    endif()
endif()

# Add FPGA source files to the parent scope's SEAL_SOURCE_FILES variable
if(SEAL_USE_FPGA)
    set(SEAL_SOURCE_FILES
        ${SEAL_SOURCE_FILES}
        ${CMAKE_CURRENT_LIST_DIR}/Src/fpga_ckks_context.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Src/fpga_ckks_encoder.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Src/fpga_dwt.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Src/fpga_ntt.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Src/fpga_encrypt.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Src/fpga_ckks_encryptor.cpp
        PARENT_SCOPE)
else()
    set(SEAL_SOURCE_FILES
        ${SEAL_SOURCE_FILES}
        ${CMAKE_CURRENT_LIST_DIR}/Src/fpga_ckks_context.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Src/fpga_ckks_encoder.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Src/fpga_dwt.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Src/fpga_ntt.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Src/fpga_encrypt.cpp
        ${CMAKE_CURRENT_LIST_DIR}/Src/fpga_ckks_encryptor.cpp
        PARENT_SCOPE)
endif()

# Export FPGA-specific configuration to parent scope
if(SEAL_USE_FPGA)
    # Export include directories for SYCL headers
    set(SEAL_FPGA_INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/Inc
        PARENT_SCOPE)



    # FPGA build mode selection
    option(FPGA_EMULATOR "Build for FPGA emulator (CPU-based, no hardware needed)" ON)
    option(FPGA_SIMULATOR "Build for FPGA simulator (detailed RTL simulation)" OFF)
    option(FPGA_HARDWARE "Build for FPGA hardware" OFF)

    # Prepare SYCL compile options
    set(SEAL_FPGA_COMPILE_OPTIONS -fsycl -fintelfpga)
    set(SEAL_FPGA_LINK_OPTIONS -fsycl -fintelfpga)

    if(FPGA_HARDWARE)
        message(STATUS "Building FPGA code for hardware target")
        list(APPEND SEAL_FPGA_COMPILE_OPTIONS -Xshardware)
        list(APPEND SEAL_FPGA_LINK_OPTIONS -Xshardware)
        # Hardware compilation can take hours, warn user
        message(WARNING "FPGA hardware compilation can take several hours!")
    elseif(FPGA_SIMULATOR)
        message(STATUS "Building FPGA code for simulator target")
        list(APPEND SEAL_FPGA_COMPILE_OPTIONS -Xssimulation)
        list(APPEND SEAL_FPGA_LINK_OPTIONS -Xssimulation)
    else()
        # Default to emulator mode
        message(STATUS "Building FPGA code for emulator target (default)")
        # Emulator is default, no additional flags needed
    endif()

    # Export SYCL flags to parent scope
    set(SEAL_FPGA_COMPILE_OPTIONS ${SEAL_FPGA_COMPILE_OPTIONS} PARENT_SCOPE)
    set(SEAL_FPGA_LINK_OPTIONS ${SEAL_FPGA_LINK_OPTIONS} PARENT_SCOPE)

    # Export preprocessor definition
    set(SEAL_FPGA_DEFINITIONS SEAL_USE_FPGA PARENT_SCOPE)
endif()