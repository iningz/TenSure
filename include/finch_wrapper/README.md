Build backend-specific shared library: `libfinch_wrapper`:

```bash
cmake .. -DBUILD_FINCH=ON
```

```bash
make
```

Launch TenSure using Finch backend (`julia` needs to be in `PATH`):

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

```bash
./TenSure --backend ./libfinch_wrapper.<so/dylib>
```