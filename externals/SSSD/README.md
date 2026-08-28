# SGLang + SSSD

This is a fork of [SGLang](https://github.com/sgl-project/sglang) that integrates **Simply-Scalable Speculative Decoding (SSSD)** — a retrieval-based speculative decoding method (**ACL 2026**).

The SSSD data structures and algorithms and its standalone evaluation code live in a separate repository and are pulled in here as a git submodule:

➡️ **[huawei-csl/sssd_speculator](https://github.com/huawei-csl/sssd_speculator)** — the actual SSSD speculator implementation.

This repo contains only the SGLang-side integration: the runtime hooks, the speculative worker integration, and the benchmark scripts needed to run and evaluate SSSD end-to-end inside SGLang.

## Branches

| Branch | Purpose |
| --- | --- |
| `main` | Latest SSSD integration, rebased on top of current upstream SGLang. |
| `upstream-main` | Clean snapshot of [sgl-project/sglang](https://github.com/sgl-project/sglang) `main` at the time of the last pull — no SSSD changes. Useful as a reference point for diffing the integration. |
| `sssd-v0.5.3.post3-acl26` | Frozen branch used for the ACL 2026 paper experiments (based on SGLang v0.5.3.post3). Check this out to reproduce the numbers reported in the paper; see its own `README.md` for the full setup and benchmark instructions. |

## Getting started

Clone the repository with submodules and install both packages:

```bash
git clone --recurse-submodules https://github.com/huawei-csl/sglang-sssd.git
cd sglang-sssd
pip install -e "python"
(cd sssd_speculator && pip install -e . --config-settings editable_mode=compat)
```

For full installation options (Docker images, prebuilt wheels, hardware-specific builds, etc.) refer to the upstream [SGLang repository](https://github.com/sgl-project/sglang) and its [documentation](https://docs.sglang.io/). See the [sssd_speculator README](https://github.com/huawei-csl/sssd_speculator) for datastore construction and speculator-specific documentation.

For end-to-end examples of launching SGLang with SSSD and running the evaluation pipelines, see the scripts in [`speculative_bench_scripts/`](speculative_bench_scripts/).

## Reproducing the ACL 2026 paper

Check out the `sssd-v0.5.3.post3-acl26` branch and follow the instructions in its README — it pins the exact SGLang version, datasets, and scripts used in the paper.

## Citation

If you use SSSD in your research, please cite our paper:

```bibtex
@misc{marzollo2026sssdsimplyscalablespeculativedecoding,
      title={SSSD: Simply-Scalable Speculative Decoding},
      author={Michele Marzollo and Jiawei Zhuang and Niklas Roemer and Niklas Zwingenberger and Lorenz K. Muller and Lukas Cavigelli},
      year={2026},
      eprint={2411.05894},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.05894},
}
```

## About SGLang

SGLang is a high-performance serving framework for large language models and multimodal models, maintained by the [LMSYS](https://lmsys.org/about/) organization at [sgl-project/sglang](https://github.com/sgl-project/sglang). All credit for the underlying runtime goes to the upstream project and its contributors.
