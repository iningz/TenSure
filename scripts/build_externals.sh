#!/bin/bash
set -e

TACO_LIB="external/taco/build/lib/libtaco.so"

if [ ! -f "$TACO_LIB" ]; then
    echo "Building TACO..."
    mkdir -p external/taco/build
    cd external/taco/build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j8
    cd ../../../
fi

# nlohmann_json is header-only, so nothing to build
echo "All externals are ready"