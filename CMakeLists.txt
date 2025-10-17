cmake_minimum_required(VERSION 3.16)
project(TenSure VERSION 1.0 LANGUAGES CXX)

# ------------------------------
# C++ standard
# ------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------
# Paths to external libraries
# ------------------------------
set(TACO_DIR ${CMAKE_SOURCE_DIR}/external/taco)
set(JSON_DIR ${CMAKE_SOURCE_DIR}/external/nlohmann_json)

# ------------------------------
# Custom target to build externals
# ------------------------------
add_custom_target(build_externals
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/build_externals.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building external dependencies (TACO)..."
)

# ------------------------------
# Include directories
# ------------------------------
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${TACO_DIR}/include
    ${JSON_DIR}/include
)

# ------------------------------
# Source files
# ------------------------------
file(GLOB_RECURSE SRC_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)

# ------------------------------
# Add executable
# ------------------------------
add_executable(${PROJECT_NAME} ${SRC_FILES})

# ------------------------------
# Link TACO library
# ------------------------------
# Assumes your build_externals.sh builds TACO into external/taco/build/lib/libtaco.so
target_link_libraries(${PROJECT_NAME} PRIVATE ${TACO_DIR}/build/lib/libtaco.so)

# ------------------------------
# Make sure your app depends on externals
# ------------------------------
add_dependencies(${PROJECT_NAME} build_externals)

# ------------------------------
# Optional: verbose build output
# ------------------------------
set(CMAKE_VERBOSE_MAKEFILE ON)
