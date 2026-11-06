# Inference with TorchTPU

The material from the **Inference with TorchTPU** workshop at TorchTPU DevLabs
(Nov 9, 2026, 30 minutes): the files behind each live demo, three notebooks that do the
same on one chip, and the slides.

TorchTPU makes a Cloud TPU a PyTorch device. The workshop is three parts and three live
demos:

1. **TorchTPU: the TPU is a PyTorch device.** Set the device, debug in eager mode, ship
   with `torch.compile`. Demo 1: one script on one chip, in eager mode, then with
   `TORCH_TPU_DEFER_AND_FUSE=1`, then under `torch.compile`.
2. **Inference 101: what decides serving speed.** Attention and the KV cache, prefill and
   decode, HBM capacity and bandwidth. No live demo.
3. **vLLM and SGLang on TorchTPU.** The engines install and run the way you know; the TPU
   work is underneath, in kernels they ship. Demo 2: SGLang, one long prompt asked about
   twice (the prefix cache). Demo 3: vLLM, one request, then sixty-four at once (continuous
   batching).

## What is here

| Folder | What | Needs |
|---|---|---|
| [`demos/01_one_chip/`](demos/01_one_chip/) | Demo 1: one MLP, on the CPU, on the TPU in eager mode, and under `torch.compile`; the TPU runs again under the profiler for TensorBoard's trace viewer. Demo 1 also runs the same script with `TORCH_TPU_DEFER_AND_FUSE=1` (`profile_defer.py` profiles that run) | a TPU VM, one chip |
| [`demos/02_sglang/`](demos/02_sglang/) | Demo 2: SGLang serving Qwen3-32B across four chips; one long prompt asked about twice, showing the prefix cache | a TPU VM, four chips |
| [`demos/03_vllm/`](demos/03_vllm/) | Demo 3: vLLM serving the same model the same way; one request, then sixty-four at once | a TPU VM, four chips |
| [`demos/04_kernel/`](demos/04_kernel/) | Extra, not shown in the session: RMSNorm in plain PyTorch ops, then as a Pallas kernel called from PyTorch, both profiled | a TPU VM, one chip |
| [`demos/setup/`](demos/setup/) | Install scripts for the three environments (TorchTPU; SGLang; vLLM), from the projects' own READMEs | |
| [`notebooks/`](notebooks/) | The three demos on **one chip and a small model** (`Qwen/Qwen3-0.6B`), numbered like the demos, for Colab's TPU runtime or a single-chip VM; badges below | Colab |
| [`slides/`](slides/) | The deck as a PDF (one page per slide, as shown) | |

## Notebooks

Each opens in Colab from this repository (pick the TPU runtime); see
[`notebooks/README.md`](notebooks/README.md) for how they run.

- Demo 1, TorchTPU on one chip: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobMulla/torchtpu-inference-workshop/blob/main/notebooks/01_torchtpu_one_chip.ipynb) [`notebooks/01_torchtpu_one_chip.ipynb`](notebooks/01_torchtpu_one_chip.ipynb)
- Demo 2, SGLang on TorchTPU: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobMulla/torchtpu-inference-workshop/blob/main/notebooks/02_sglang_torchtpu.ipynb) [`notebooks/02_sglang_torchtpu.ipynb`](notebooks/02_sglang_torchtpu.ipynb)
- Demo 3, vLLM on TorchTPU: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RobMulla/torchtpu-inference-workshop/blob/main/notebooks/03_vllm_torchtpu.ipynb) [`notebooks/03_vllm_torchtpu.ipynb`](notebooks/03_vllm_torchtpu.ipynb)

## Running the demos on a TPU VM

Create a Cloud TPU VM (the demos were written on a v6e-1 for demo 1 and a v6e-4 for demos
2 and 3; see the [Cloud TPU documentation](https://cloud.google.com/tpu/docs)), clone this
repository on it, and:

```bash
# Demo 1: one chip
bash demos/setup/install_torchtpu.sh && source ~/venv-torchtpu/bin/activate
cd demos/01_one_chip && python example1_eager.py
TORCH_TPU_DEFER_AND_FUSE=1 python example1_eager.py        # the same script, DeferAndFuse on
python example2_compile.py
python profile_eager.py && python profile_defer.py && python profile_compile.py   # traces/ for TensorBoard
tensorboard --logdir traces --port 6006                   # Profile tab, Trace Viewer

# Demo 2: SGLang on four chips (its own venv)
bash demos/setup/install_sglang.sh && source ~/venv-sglang/bin/activate
cd demos/02_sglang && bash serve.sh &                      # first start downloads the model
cp "$VIRTUAL_ENV"/lib/python3.*/site-packages/torch/nn/modules/transformer.py .  # ask.py's long prompt
python ask.py                                              # once the server answers on :30000

# Demo 3: vLLM on four chips (its own venv; stop SGLang first, one engine per machine)
bash demos/setup/install_vllm.sh && source ~/venv-vllm/bin/activate
cd demos/03_vllm && bash serve.sh &
python burst.py 1 && python burst.py 64                    # once the server answers on :8000

# Extra: the kernel demo, in the TorchTPU venv on one chip
cd demos/04_kernel && python rmsnorm_eager.py && python rmsnorm_pallas.py
```

Every number the scripts print is that run's own. The model, `Qwen/Qwen3-32B`, is about
64 GB in bfloat16, which is why it takes four chips; the notebooks use `Qwen/Qwen3-0.6B`
on one.

## Sources

- TorchTPU: [github.com/google-pytorch/torch_tpu](https://github.com/google-pytorch/torch_tpu)
  (README, `docs/performance_profiling_with_xprof.md`, `docs/custom_kernels.md`).
- vLLM on TPU: [github.com/vllm-project/vllm-torchtpu](https://github.com/vllm-project/vllm-torchtpu).
- SGLang on TPU: [github.com/sgl-project/sglang-torchtpu](https://github.com/sgl-project/sglang-torchtpu).

## License

Apache 2.0, see [LICENSE](LICENSE).
