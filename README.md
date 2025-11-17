## TenSure - Sparse Tensor Compiler Fuzzer

### Build Process

```bash
mkdir build && cd build
cmake .. -DBUILD_TACO=ON
make -j$(nproc)
./TenSure --backend ./libtaco_wrapper.so
```

### Running the Fuzzer

TenSure is compute-heavy, so you should run it inside a cgroup (for safety) and inside a tmux session (so it never dies when SSH disconnects).

1. Create a cgroup (one time only)

```bash
sudo cgcreate -g memory,cpu:kfuzz
```
This creates a control group named kfuzz under both memory and cpu controllers.
All processes executed inside this group will have limited resources.

```bash
sudo cgset -r memory.max=24G kfuzz
```
This caps the maximum RAM the fuzzer can use.
If the process exceeds this limit → only the fuzzer dies, not your entire machine.

```bash
sudo cgset -r cpu.max="800000 1000000" kfuzz
```
This limits the fuzzer to use 80% of a CPU core, preventing it from hogging the machine.

2. Start a new `tmux` session.

```bash
tmux new -s tensure
```

3. Start the fuzzer inside the cgroup

```bash
sudo cgexec -g memory,cpu:kfuzz ./TenSure --backend ./libtaco_wrapper.so
```

4. Detach the session (fuzzer keeps running)

> `Ctrl+b` then `d`

You can reattach to the session anytime using:
```bash
tmux attach -t tensure
```

Few `tmux` commands:
```bash
tmux ls
tmux kill-session -t tensure
```