#!/usr/bin/env python3
"""Probe GPU riêng Phase A: attention theo query, không đo pipeline A+B.

MODAL_PROFILE=phuclk30 modal run --profile phuclk30 scripts/modal_phase_a_probe.py
"""
from __future__ import annotations
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = Path('/workspace/fast_infer_text_sum')
LOCAL_SOURCE = ROOT / 'outputs/phase_a_logic_audit_20261018/source_prompts'
VOLUME = modal.Volume.from_name('fast-infer-text-sum-cache')
IMAGE = (modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch==2.11.0', 'transformers==5.12.1', 'accelerate==1.14.0',
                 'numpy==2.2.6', 'einops==0.8.2')
    .add_local_dir(ROOT / 'scripts/common', str(REMOTE / 'scripts/common'), ignore=['__pycache__', '*.pyc'])
    .add_local_dir(ROOT / 'src/TrainingFree', str(REMOTE / 'src/TrainingFree'), ignore=['__pycache__', '*.pyc'])
    .add_local_dir(ROOT / 'externals/dflash', str(REMOTE / 'externals/dflash'), ignore=['__pycache__', '*.pyc', '.git'])
    .add_local_dir(LOCAL_SOURCE, '/bootstrap/phase_a_source')
    .add_local_file(ROOT / 'scripts/probe_dflash_attention.py', str(REMOTE / 'scripts/probe_dflash_attention.py'))
    .add_local_file(ROOT / 'scripts/probe_dflash_target_comparison.py', str(REMOTE / 'scripts/probe_dflash_target_comparison.py'))
    .add_local_file(ROOT / 'scripts/run_candidate_benchmarks.py', str(REMOTE / 'scripts/run_candidate_benchmarks.py')))
app = modal.App('phase-a-query-logic-probe')


@app.function(image=IMAGE, gpu='L40S', cpu=8, memory=49152, timeout=1800,
              volumes={'/mnt/fast-infer': VOLUME})
def probe(run_id: str, samples: int, max_rounds: int):
    import shutil
    os.environ['HF_HOME'] = '/mnt/fast-infer/hf'
    from huggingface_hub import snapshot_download
    source = Path('/tmp/phase_a_source')
    shutil.copytree('/bootstrap/phase_a_source', source, dirs_exist_ok=True)
    manifest = json.loads((source / 'manifest.json').read_text())
    for field, repo in [('target', 'Qwen/Qwen3-4B'), ('draft', 'z-lab/Qwen3-4B-DFlash-b16')]:
        manifest[field] = snapshot_download(repo, cache_dir='/mnt/fast-infer/hf/hub')
    (source / 'manifest.json').write_text(json.dumps(manifest))
    VOLUME.commit()
    output = Path('/mnt/fast-infer/outputs/phase_a') / run_id
    if output.exists():
        raise FileExistsError(f'Run đã tồn tại: {output}')
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    subprocess.run([sys.executable, '-u', str(REMOTE / 'scripts/probe_dflash_target_comparison.py'),
        '--source-run', str(source), '--output', str(output), '--caps', '8192,16384',
        '--samples', str(samples), '--max-rounds', str(max_rounds), '--capture-parent-attention',
        '--target-layers', '0,12,14,16,18,20,27,35', '--save-all-rounds'],
        cwd=REMOTE, env=env, check=True)
    subprocess.run([sys.executable, str(REMOTE / 'scripts/run_candidate_benchmarks.py'),
        '--target-dir', str(output / 'vector_snapshots'), '--output-dir', str(output / 'evaluation'),
        '--all-rounds', '--budgets', '256,512,1024,2048', '--chunk-sizes', '1,128'],
        cwd=REMOTE, env=env, check=True)
    VOLUME.commit()
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w:gz', compresslevel=1) as archive:
        for path in output.rglob('*'):
            if path.is_file() and 'parent_transition_snapshots' not in path.parts:
                archive.add(path, arcname=str(path.relative_to(output)))
    return data.getvalue()


@app.local_entrypoint()
def main(run_id: str = 'phase-a-query-audit-20261018', samples: int = 2, max_rounds: int = 8):
    if os.environ.get('MODAL_PROFILE') != 'phuclk30':
        raise ValueError('Cần MODAL_PROFILE=phuclk30 và modal run --profile phuclk30')
    if not 1 <= samples <= 2 or not 2 <= max_rounds <= 16 or Path(run_id).name != run_id:
        raise ValueError('Probe giới hạn 1–2 samples, 2–16 rounds và run ID đơn')
    payload = probe.remote(run_id, samples, max_rounds)
    output = ROOT / 'outputs/phase_a_logic_audit_20261018' / run_id
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
        archive.extractall(output, filter='data')
    print(f'Phase A artifacts: {output}')
