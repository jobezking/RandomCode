# file: torch_cuda_benchmark.py
import time
import torch

def info():
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA:", torch.version.cuda)
        print("cuDNN:", torch.backends.cudnn.version())

def benchmark(n=4096, iters=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    a = torch.randn((n, n), device=device)
    b = torch.randn((n, n), device=device)
    # Warmup
    for _ in range(3):
        _ = a @ b
    torch.cuda.synchronize() if device == 'cuda' else None

    start = time.time()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize() if device == 'cuda' else None
    dt = time.time() - start
    print(f"Device: {device} | size={n} | iters={iters} | total={dt:.3f}s | avg={dt/iters:.3f}s")

if __name__ == "__main__":
    info()
    benchmark()
