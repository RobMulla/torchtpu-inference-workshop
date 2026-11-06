# example2_compile.py under torch.profiler: the same model and loop with the
# torch.compile line, warmed up first (the compile happens there), then five
# steps traced (CPU and TPU) to traces/compiled for TensorBoard's trace
# viewer (the TorchTPU docs, "Performance profiling with XProf"). Run:
# python profile_compile.py
import os
import time

import torch
from torch import nn
from torch.tpu.profiler import TpuProfilerConfig

device = "tpu"
trace_dir = os.environ.get("TRACES", "traces") + "/compiled"
steps = 5

model = nn.Sequential(*[
    nn.Sequential(nn.LayerNorm(1024),
                  nn.Linear(1024, 4096), nn.GELU(),
                  nn.Linear(4096, 1024))
    for _ in range(12)
]).to(device)
x = torch.randn(16, 512, 1024, device=device)
model = torch.compile(model, backend="tpu")

with torch.no_grad():
    for i in range(1, 21):  # warm-up (the compile), not traced
        y = model(x)
        torch.accelerator.synchronize()

    config = TpuProfilerConfig(host_tracer_level=2, device_tracer_level=1,
                               run_dir=trace_dir)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.TPU],
        experimental_config=config,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
    ) as prof:
        for i in range(1, steps + 1):
            t0 = time.perf_counter()
            y = model(x)
            torch.accelerator.synchronize()  # wait
            ms = (time.perf_counter() - t0) * 1000
            print(f"compiled step {i}: {ms:.1f} ms")
            prof.step()

print(y.device, tuple(y.shape))
print(f"trace: {trace_dir}")
for root, _, files in os.walk(trace_dir):
    for f in sorted(files):
        p = os.path.join(root, f)
        print(f"  {os.path.relpath(p, trace_dir)} ({os.path.getsize(p) / 1024:.0f} KB)")
