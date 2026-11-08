# Workshop notebooks: Inference with TorchTPU on one chip

The take-home notebooks of the TorchTPU DevLabs workshop *Inference with TorchTPU*
(Nov 9, 2026). The live demos run on Cloud TPU VMs (one chip for demo 1; four chips and
`Qwen/Qwen3-32B` for demos 2 and 3); attendees have no TPU VM, so these are the same three
demos reworked for **one chip and a small model**, meant for Colab's TPU runtime. They are
numbered like the demos: demo 1 is in part 1 of the session, demos 2 and 3 are in part 3
(part 2 has no live demo). Each badge opens the notebook in Colab from this repository;
pick the TPU runtime.

| Notebook | Open | Demo, part | What it does |
| --- | --- | --- | --- |
| `01_torchtpu_one_chip.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobMulla/torchtpu-inference-workshop/blob/main/notebooks/01_torchtpu_one_chip.ipynb) | Demo 1; part 1, TorchTPU: the TPU is a PyTorch device | The device line (`import torch` is enough), an MLP in eager mode, the same with DeferAndFuse (`torch.tpu.set_eager_mode`), the same under `torch.compile`; the three runs under the standard profiler and one step of each drawn on one axis in the notebook (and how to open them in TensorBoard). Ends with a link to the TorchTPU docs' own Colab. |
| `02_sglang_torchtpu.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobMulla/torchtpu-inference-workshop/blob/main/notebooks/02_sglang_torchtpu.ipynb) | Demo 2; part 3, vLLM and SGLang on TorchTPU | Install SGLang-TorchTPU into a Python 3.12 environment of its own; `python -m sglang.launch_server --device tpu ... --enable-torch-compile --enable-cache-report` on one chip in the background; wait for `/health`; `tpu-info`; one request; a burst of sixteen with tokens/s; the prefix cache (one long prompt asked about twice, printing first-token time and cached tokens); stop. |
| `03_vllm_torchtpu.ipynb` | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobMulla/torchtpu-inference-workshop/blob/main/notebooks/03_vllm_torchtpu.ipynb) | Demo 3; part 3, vLLM and SGLang on TorchTPU | The same shape with vLLM-TorchTPU: `vllm serve Qwen/Qwen3-0.6B ... --enable-prompt-tokens-details`, the same requests, the same prefix-cache cell; stop. |

**Where they were tested.** Every notebook was executed top to bottom on a **Cloud TPU
v6e-1** (one Trillium chip, 32 GB HBM) with `jupyter nbconvert --execute`, from a Python
3.13 kernel like Colab's (Nov 8); the clean, un-executed copies are what is committed.
Colab's TPU runtime is a different chip, so timings and memory differ. Open points are
listed at the end of this file.

**What is in the markdown.** A few of the session's slides are embedded in the notebooks
as cell attachments (the three modes and the three traces in `01`; the KV cache, paged
attention and prefix caching in `02`; the two numbers, continuous batching and prefix
caching in `03`), so a reader coming from the room finds the same pictures. The whole deck
is `../slides/` (one PDF).

## Running them

- **The serving notebooks make the server its own Python 3.12 environment** (`venv-sglang`,
  `venv-vllm`, next to the notebook, made with `uv`) and start the server from it; the
  kernel is only the client (`openai`, `requests`) and never imports torch. The reason: the
  engines' wheels are built for Python 3.12 (`vllm-torchtpu-core` on PyPI says
  `Requires-Python >=3.12,<3.13`) and pin their own torch, `torch_tpu`, `libtpu` and jax,
  while Colab's kernel is Python 3.13 with its own packages. Installed into the kernel
  itself, pip on Colab resolves `vllm-torchtpu` to the empty `0.0.0` placeholder wheel and
  `import vllm` fails. The first run downloads about two gigabytes per environment.
- One chip, one engine: run the stop cell of `02` before starting `03` (or the other way
  round), or restart the runtime. In Colab, do not run `01` first in the same runtime: its
  kernel holds the chip. On a TPU VM, the server's process group is what the stop cell
  signals; if a kernel dies with a server up, `pkill -INT -f
  "vllm.entrypoints|sglang.launch_server"` frees the chip.
- `01` relies on TorchTPU's autoload (`import torch` registers the device; there is no
  `import torch_tpu`, as in the live demo's scripts) and switches to DeferAndFuse with the
  context manager `torch.tpu.set_eager_mode(torch.tpu.EagerMode.DEFER_AND_FUSE)`, the
  in-process form of the live demo's `TORCH_TPU_DEFER_AND_FUSE=1`.
- The serving notebooks' install lines pass `--index-strategy unsafe-best-match` to `uv`:
  `uv` otherwise stops at the first index that has a package, the PyTorch CPU index has an
  old `requests`, and `vllm-torchtpu` then resolves to the empty `0.0.0` placeholder (it
  did, on Nov 8; the cell now also asserts the install is real).
- The install cells are idempotent and print versions: if the package or the environment
  is already there, nothing is installed. The lines are the engines' public README lines
  (Oct 26 heads): `pip install --extra-index-url https://download.pytorch.org/whl/cpu
  vllm-torchtpu` and the two `sglang-torchtpu` release wheels with the same extra index,
  run by `uv pip install --python <venv>`. `01` uses `pip install torch_tpu==0.2.0
  --extra-index-url https://download.pytorch.org/whl/cpu` from the TorchTPU README, in the
  kernel.
- The model, `Qwen/Qwen3-0.6B`, is ungated and named by both engines' own lists
  (SGLang-TorchTPU `SUPPORT_MATRIX.md`, its CI model for the prefix-cache and torch.compile
  runs; vLLM-TorchTPU `docs/user_guide/supported_models.md`). `MODEL` is one variable in
  each serving notebook.
- On a VM, to re-run the test: `jupyter nbconvert --to notebook --execute
  --ExecutePreprocessor.timeout=1800 <notebook>` from a kernel with `jupyter`, `nbconvert`,
  `ipykernel`, `pip` and (for `01`) `matplotlib`, with `HF_HOME` pointing at a disk with
  room for the weights.

## Rules the notebooks follow

Plain tone; the machine named by type only (never a hostname or project); every claim
traced to the engine's README, docs or source; no engine compared with the other and
no performance claim in prose. The numbers the cells print are that run's own, and
the markdown says so; none is written down here.

## Sources

- `github.com/google-pytorch/torch_tpu` (71a863941): README "Install via pip",
  `docs/performance_profiling_with_xprof.md`.
- `github.com/vllm-project/vllm-torchtpu` (cdf1c14d8): README "Install",
  `docs/deployment/v6e_recipes.md`, `docs/user_guide/supported_models.md`; PyPI
  `vllm-torchtpu` and `vllm-torchtpu-core` (0.29.0 ran on Nov 2; 0.32.0, released that
  evening, installed with `../demos/setup/install_vllm.sh` and served on a v6e-1 on Nov 4;
  `vllm-torchtpu-core` still says `Requires-Python >=3.12,<3.13`).
- `github.com/sgl-project/sglang-torchtpu` (d7197b91f3): README "Install" and "Launch a server",
  `SUPPORT_MATRIX.md`, `benchmark/tpu/common_env.sh`; the `tpu-v0.1.0` release wheels'
  metadata (Nov 2).
- The live demos' own files, in `../demos/`: `01_one_chip/` (the examples and the
  `profiler.py`), `02_sglang/serve.sh` and `ask.py`, `03_vllm/serve.sh` and `burst.py`.

## Open points

- Colab's TPU runtime is a v5e. Both engines' default attention kernel takes the same
  branch for v5e and v6e, and a v5e chip has 16 GB of HBM to the v6e's 32 GB, so the
  same `serve` lines reserve half the memory; `Qwen/Qwen3-0.6B` fits either way.
  SGLang-TorchTPU's tuned block-size tables have v6e entries only (a v5e falls back to
  defaults), and its device-name mapping for Pallas was not exercised on a v5e.
- vLLM-TorchTPU reads the chip type from `TPU_ACCELERATOR_TYPE` or the VM's GCE metadata;
  where neither is available, `03` sets the variable from the chip `tpu-info` sees
  (`v5litepod-1`, `v6e-1`, ...), so the kernels get the right geometry.
