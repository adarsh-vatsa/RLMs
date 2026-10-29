# One-time setup

Do these steps once, on the Jarvis login node. If a step's check already
passes, skip the step. Each suite page also has one data step of its own.

## 1. Get the code

```bash
ssh edogu@jarvis.stevens.edu
cd ~
git clone https://github.com/adarsh-vatsa/RLMs.git adarsh-rlms   # only if ~/adarsh-rlms does not exist yet
git -C adarsh-rlms fetch origin
git -C adarsh-rlms checkout shared-execution-linux-runners
git -C adarsh-rlms pull
```

All later commands run from `~`, the directory that contains `adarsh-rlms`.

## 2. Shell settings

Jobs write their logs, and the executor's address, to `PROJECT_LOG_DIR`. Set it
in your shell profile so every terminal has it:

```bash
echo 'export PROJECT_LOG_DIR="$HOME/adarsh-rlms-logs"' >> ~/.bashrc
source ~/.bashrc
mkdir -p "$PROJECT_LOG_DIR"
```

Next, check which Slurm accounts you belong to:

```bash
sacctmgr show assoc user=$USER format=account%30
```

If more than one account is listed, choose the one for this project. Then add
it to the profile so every job is charged to it:

```bash
echo 'export SBATCH_ACCOUNT=<account>' >> ~/.bashrc
source ~/.bashrc
```

## 3. Install uv

Check: `uv --version` prints a version.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

## 4. vLLM environment (serves the model)

Check: this prints `0.19.1`.

```bash
"$HOME/.venvs/adarsh-vllm/bin/python" -c "import vllm; print(vllm.__version__)"
```

Install:

```bash
[ -d "$HOME/.venvs/adarsh-vllm" ] || uv venv "$HOME/.venvs/adarsh-vllm" --python 3.12 --seed
uv pip install --python "$HOME/.venvs/adarsh-vllm/bin/python" "vllm==0.19.1" --torch-backend=cu128
"$HOME/.venvs/adarsh-vllm/bin/python" -c "import vllm; print(vllm.__version__)"
```

`cu128` matches Jarvis's CUDA 12 drivers. Without it, the default CUDA 13 build
fails to load on Jarvis with `libcudart.so.13` errors.

## 5. Client environment (runs the benchmarks)

Check: this prints a version with no import error.

```bash
adarsh-rlms/.venv/bin/python -c "import torch, transformers, faiss, dotenv, numpy; print(torch.__version__)"
```

Install. Running the install again on an existing environment is safe.

```bash
cd adarsh-rlms
[ -d .venv ] || uv venv .venv --python 3.13
uv pip install --python .venv/bin/python --torch-backend=cu128 transformers torch faiss-cpu python-dotenv numpy
cd ..
```

Check that a GPU job can use the GPU. This submits a short job without an
executor:

```bash
CLIENT_CMD='.venv/bin/python -c "import faiss, transformers, torch; print(torch.cuda.is_available())"' \
  bash adarsh-rlms/jarvis/run.sh submit client-gpu
```

When `squeue -u "$USER"` no longer lists the job, the last line of
`$PROJECT_LOG_DIR/rlms-client-gpu-<job id>.out` should be `True`. If it is
`False`, reinstall PyTorch and check again:

```bash
uv pip install --python adarsh-rlms/.venv/bin/python --torch-backend=cu128 --reinstall-package torch torch
```

## 6. Keys

Open `adarsh-rlms/.env` in an editor, for example `nano adarsh-rlms/.env`, and
add:

```text
HF_TOKEN=<your Hugging Face token>
OPENROUTER_API_KEY=<your OpenRouter key>
```

- `HF_TOKEN` lets jobs download the model and tokenizers.
- `OPENROUTER_API_KEY` is used only by the final AA-LCR regrade with GPT-5.6
  Luna.

Git ignores `.env`. Make it readable only by you, then check that both keys are
there; this prints `2` and does not print the keys:

```bash
chmod 600 adarsh-rlms/.env
grep -cE '^(HF_TOKEN|OPENROUTER_API_KEY)=' adarsh-rlms/.env
```

## Next

Prepare each benchmark's data, as described in its page:

- [LONGBENCH_V2.md](LONGBENCH_V2.md#before-the-first-suite-get-the-data):
  download `data.json`.
- [AA_LCR.md](AA_LCR.md#before-the-first-suite-prepare-the-data): run the
  preparation script.
- [MRCR_V2.md](MRCR_V2.md): nothing to do; the data comes with the code.

Then start the executor as described in [EXECUTOR.md](EXECUTOR.md).
