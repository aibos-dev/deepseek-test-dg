## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/aibos-dev/deepseek-test-dg.git
cd deepseek-test-dg
```

2. Build and start the development container:
```bash
docker compose up -d --build
docker exec -it deepseek-test-container bash
```

3. Login to required services:
```bash
huggingface-cli login
wandb login
```

The container comes with all required dependencies pre-installed:
- vLLM with CUDA support
- Project dependencies
- Properly configured NVIDIA JIT linker path

Note: If you need to reinstall or update dependencies manually, you can still run:
```bash
uv pip install vllm>=0.7.0
uv pip install -e ".[dev]"
```

## Training models

We support training models with either DDP or DeepSpeed (ZeRO-2 and ZeRO-3). To switch between methods, simply change the path to the `accelerate` YAML config in `configs`.

> [!NOTE]
> The training commands below are configured for a node of 8 x H100s (80GB). For different hardware and topologies, you may need to tune the batch size and number of gradient accumulation steps.

### SFT

To run SFT on a dataset distilled from DeepSeek-R1 with reasoning traces such as [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k), run:

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py --config recipes/qwen/Qwen2.5-1.5B-Instruct/sft/config_full.yaml
```
