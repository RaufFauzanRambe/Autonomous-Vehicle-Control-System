# =============================================================================
# Autonomous Vehicle Control System - CMakeLists.txt
# =============================================================================
# CMake build configuration for the C++ core, Python bindings,
# ROS2 integration, and testing infrastructure.
# =============================================================================

cmake_minimum_required(VERSION 3.20)
project(autonomous_vehicle_control_system VERSION 0.1.0 LANGUAGES CXX)

# ---- C++ Standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Build Type ----
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- Compiler Flags ----
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")

# ---- Options ----
option(BUILD_TESTING "Build unit tests" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings with pybind11" OFF)
option(BUILD_BENCHMARKS "Build benchmark executables" OFF)
option(USE_CUDA "Enable CUDA support for perception" OFF)
option(USE_TBB "Enable Intel TBB for parallel algorithms" OFF)

# ---- Dependencies ----

# Eigen3
find_package(Eigen3 3.4 REQUIRED)

# Threads
find_package(Threads REQUIRED)

# OpenCV (optional)
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    add_definitions(-DAVCS_USE_OPENCV)
endif()

# CUDA (optional)
if(USE_CUDA)
    find_package(CUDA 11.8 REQUIRED)
    add_definitions(-DAVCS_USE_CUDA)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17")
endif()

# ---- ROS2 Integration ----
find_package(ament_cmake QUIET)
if(ament_cmake_FOUND)
    message(STATUS "Building with ROS2 support")
    find_package(rclcpp REQUIRED)
    find_package(std_msgs REQUIRED)
    find_package(geometry_msgs REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(nav_msgs REQUIRED)
    find_package(tf2 REQUIRED)
    find_package(tf2_ros REQUIRED)
    find_package(tf2_geometry_msgs REQUIRED)
    find_package(cv_bridge REQUIRED)
    add_definitions(-DAVCS_USE_ROS2)
endif()

# ---- Python Bindings ----
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        find_program(PYBIND11_CONFIG pybind11-config)
        if(PYBIND11_CONFIG)
            execute_process(
                COMMAND ${PYBIND11_CONFIG} --cmakedir
                OUTPUT_VARIABLE pybind11_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            find_package(pybind11 REQUIRED)
        else()
            message(WARNING "pybind11 not found. Python bindings will not be built.")
            set(BUILD_PYTHON_BINDINGS OFF)
        endif()
    endif()
endif()

# ---- Source Files ----

# Core library sources
set(AVCS_CORE_SOURCES
    src/main.cpp
    src/sensor_fusion.cpp
    src/localization.cpp
    src/slam_system.cpp
    src/lidar_processing.cpp
    src/radar_processing.cpp
    src/gps_processing.cpp
    src/imu_processing.cpp
    src/real_time_engine.cpp
    src/trajectory_tracking.cpp
    src/vehicle_state_estimator.cpp
    src/autonomous_core.cpp
)

set(AVCS_CORE_HEADERS
    src/sensor_fusion.hpp
    src/localization.hpp
    src/slam_system.hpp
    src/trajectory_tracking.hpp
    src/vehicle_state_estimator.hpp
    src/autonomous_core.hpp
)

# ---- Core Library ----
add_library(avcs_core STATIC ${AVCS_CORE_SOURCES})

target_include_directories(avcs_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(avcs_core
    PUBLIC
        Eigen3::Eigen
        Threads::Threads
    PRIVATE
        ${OpenCV_LIBS}
)

target_compile_definitions(avcs_core
    PRIVATE
        AVCS_VERSION="${PROJECT_VERSION}"
)

# ---- CUDA Support ----
if(USE_CUDA)
    target_include_directories(avcs_core PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(avcs_core PRIVATE ${CUDA_LIBRARIES})
endif()

# ---- ROS2 Node ----
if(ament_cmake_FOUND)
    add_executable(avcs_ros2_node src/main.cpp)
    target_link_libraries(avcs_ros2_node
        avcs_core
        ${rclcpp_TARGETS}
    )
    ament_target_dependencies(avcs_ros2_node
        rclcpp std_msgs geometry_msgs sensor_msgs nav_msgs
        tf2 tf2_ros tf2_geometry_msgs cv_bridge
    )
endif()

# ---- Standalone Executable ----
add_executable(avcs_core_bin src/main.cpp)
target_link_libraries(avcs_core_bin PRIVATE avcs_core)
set_target_properties(avcs_core_bin PROPERTIES OUTPUT_NAME avcs_core)

# ---- Python Bindings ----
if(BUILD_PYTHON_BINDINGS)
    pybind11_add_module(_cpp_core src/python_bindings.cpp)
    target_link_libraries(_cpp_core PRIVATE avcs_core)
    install(TARGETS _cpp_core LIBRARY DESTINATION avcs)
endif()

# ---- Testing ----
if(BUILD_TESTING)
    enable_testing()
    find_package(GTest QUIET)

    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    # Test executable
    file(GLOB_RECURSE TEST_SOURCES test/cpp/*.cpp)
    if(TEST_SOURCES)
        add_executable(avcs_tests ${TEST_SOURCES})
        target_link_libraries(avcs_tests
            PRIVATE
                avcs_core
                GTest::gtest
                GTest::gtest_main
                GTest::gmock
        )
        include(GoogleTest)
        gtest_discover_tests(avcs_tests)
    endif()
endif()

# ---- Benchmarks ----
if(BUILD_BENCHMARKS)
    find_package(benchmark QUIET)
    if(benchmark_FOUND)
        add_executable(avcs_benchmarks test/benchmarks/benchmark_main.cpp)
        target_link_libraries(avcs_benchmarks PRIVATE avcs_core benchmark::benchmark)
    endif()
endif()

# ---- Install Rules ----
install(TARGETS avcs_core avcs_core_bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${AVCS_CORE_HEADERS}
    DESTINATION include/avcs
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

install(DIRECTORY config/
    DESTINATION config
    FILES_MATCHING PATTERN "*.yaml" PATTERN "*.yml"
)

# ---- ROS2 Package ----
if(ament_cmake_FOUND)
    ament_package()
endif()

# ---- Summary ----
message(STATUS "")
message(STATUS "=== AVCS Build Configuration ===")
message(STATUS "  Version:            ${PROJECT_VERSION}")
message(STATUS "  Build Type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:       ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Testing:      ${BUILD_TESTING}")
message(STATUS "  Python Bindings:    ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  CUDA Support:       ${USE_CUDA}")
message(STATUS "  ROS2 Support:       ${ament_cmake_FOUND}")
message(STATUS "  OpenCV:             ${OpenCV_FOUND}")
message(STATUS "  Install Prefix:     ${CMAKE_INSTALL_PREFIX}")
message(STATUS "================================")
message(STATUS "")
