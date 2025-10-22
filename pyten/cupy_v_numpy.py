# file: cupy_vs_numpy.py
import time
import numpy as np
import cupy as cp

def run_numpy(n=20_000_000):
    x = np.random.rand(n).astype(np.float32)
    y = np.random.rand(n).astype(np.float32)
    start = time.time()
    z = np.sqrt(x * y + 1.2345).astype(np.float32)
    return time.time() - start, z.mean()

def run_cupy(n=20_000_000):
    x = cp.random.rand(n, dtype=cp.float32)
    y = cp.random.rand(n, dtype=cp.float32)
    cp.cuda.runtime.deviceSynchronize()
    start = time.time()
    z = cp.sqrt(x * y + 1.2345).astype(cp.float32)
    cp.cuda.runtime.deviceSynchronize()
    return time.time() - start, cp.asnumpy(z.mean()).item()

if __name__ == "__main__":
    n = 10_000_000  # Start smaller if memory is tight
    dt_np, mn_np = run_numpy(n)
    dt_cp, mn_cp = run_cupy(n)
    print(f"NumPy: {dt_np:.3f}s, mean={mn_np:.6f}")
    print(f"CuPy : {dt_cp:.3f}s, mean={mn_cp:.6f}")
    print("Speedup (CuPy/NumPy):", dt_np / dt_cp if dt_cp > 0 else "inf")
    print("GPU device:", cp.cuda.runtime.getDeviceProperties(0)['name'].decode())
