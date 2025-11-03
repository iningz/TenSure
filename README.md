## TenSure - Sparse Tensor Compiler Fuzzer

Build Process
1. Make ./scripts/build_externals.sh executable

```bash
chmod +x ./scripts/build_externals.sh
```

2. Build the external libraries for dependent modules.

```bash
./scripts/build_externals.sh
```

3. Build the fuzzer

```bash
mkdir build && cd build
cmake ..
make -j8
```

4. Run the fuzzer

```bash
./TenSure
```



mkdir build2 && cd build2
cmake .. -DBUILD_TACO=ON
make -j$(nproc)
./TenSure --backend ./libtaco_wrapper.so