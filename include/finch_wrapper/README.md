Build backend-specific shared library: `libfinch_wrapper`:

```bash
cmake .. -DBUILD_<BACKEND_NAME_IN_UPPERCASE>=ON
```

```bach
make
```

Launch TenSure using Finch backend (`julia` needs to be in `PATH`):

```bash
./TenSure --backend ./libfinch_wrapper.<so/dylib>
```
