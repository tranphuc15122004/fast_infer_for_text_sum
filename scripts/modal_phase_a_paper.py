#!/usr/bin/env python3
"""Thực nghiệm và tái tạo bằng chứng Phase A trên Modal phuclk30."""
from __future__ import annotations
import io,json,os,subprocess,sys,tarfile
from pathlib import Path
import modal
ROOT=Path(__file__).resolve().parents[1];REMOTE=Path('/workspace/fast_infer_text_sum')
VOLUME=modal.Volume.from_name('fast-infer-text-sum-cache')
IMAGE=(modal.Image.debian_slim(python_version='3.12')
 .pip_install('torch==2.11.0','transformers==5.12.1','accelerate==1.14.0','numpy==2.2.6','einops==0.8.2')
 .add_local_dir(ROOT/'scripts/common',str(REMOTE/'scripts/common'),ignore=['__pycache__','*.pyc'])
 .add_local_dir(ROOT/'src/TrainingFree',str(REMOTE/'src/TrainingFree'),ignore=['__pycache__','*.pyc'])
 .add_local_dir(ROOT/'externals/dflash',str(REMOTE/'externals/dflash'),ignore=['__pycache__','*.pyc','.git'])
 .add_local_dir(ROOT/'outputs/phase_a_paper_20261019/source_prompts','/bootstrap/phase_a_source'))
for name in ('probe_dflash_attention.py','probe_dflash_target_comparison.py','run_candidate_benchmarks.py','run_phase_a_paper_evaluation.py','diagnose_phase_a_parity.py'):
 IMAGE=IMAGE.add_local_file(ROOT/'scripts'/name,str(REMOTE/'scripts'/name))
app=modal.App('phase-a-paper-corrected')

@app.function(image=IMAGE,gpu='L40S',cpu=8,memory=49152,timeout=7200,volumes={'/mnt/fast-infer':VOLUME})
def run(run_id:str,mode:str,samples:int,rounds:int,repetitions:int):
 import shutil
 from huggingface_hub import snapshot_download
 source=Path('/tmp/source');shutil.copytree('/bootstrap/phase_a_source',source,dirs_exist_ok=True)
 manifest=json.loads((source/'manifest.json').read_text())
 for field,repo in [('target','Qwen/Qwen3-4B'),('draft','z-lab/Qwen3-4B-DFlash-b16')]:
  manifest[field]=snapshot_download(repo,cache_dir='/mnt/fast-infer/hf/hub',local_files_only=True)
 (source/'manifest.json').write_text(json.dumps(manifest));VOLUME.commit()
 out=Path('/mnt/fast-infer/outputs/phase_a_paper')/run_id/mode
 if out.exists():raise FileExistsError(out)
 env=dict(os.environ,HF_HOME='/mnt/fast-infer/hf',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',OPENBLAS_NUM_THREADS='1')
 if mode=='coverage':
  subprocess.run([sys.executable,'-u',str(REMOTE/'scripts/probe_dflash_target_comparison.py'),
   '--source-run',str(source),'--output',str(out),'--caps','3072,5120,8192,16384',
   '--samples',str(samples),'--max-rounds',str(rounds),'--capture-parent-attention',
   '--target-layers','0,12,14,16,18,20,27,35','--save-all-rounds'],cwd=REMOTE,env=env,check=True)
  VOLUME.commit()
  subprocess.run([sys.executable,'-u',str(REMOTE/'scripts/run_phase_a_paper_evaluation.py'),'evaluate',
   '--source',str(out),'--output',str(out/'evaluation')],cwd=REMOTE,env=env,check=True)
  # Every bounded replay is checked as a full generated prefix, never counted as a full-summary match.
  import numpy as np
  checks=[]
  for i in range(samples):
   for cap in (3072,5120,8192,16384):
    actual=np.load(out/f'sample_{i:02d}/context_{cap}_output_ids.npy').reshape(-1)
    reference=np.load(source/f'sample_{i:02d}/context_{cap}_output_ids.npy').reshape(-1)
    checks.append({'sample_index':i,'context_cap':cap,'tokens':len(actual),'prefix_exact':bool(np.array_equal(actual,reference[:len(actual)]))})
  (out/'prefix_checks.json').write_text(json.dumps(checks,indent=2))
 elif mode=='parity':
  subprocess.run([sys.executable,'-u',str(REMOTE/'scripts/diagnose_phase_a_parity.py'),
   '--source',str(source),'--runtime',str(out.parent/'runtime/report.json'),'--output',str(out)],cwd=REMOTE,env=env,check=True)
 else:
  subprocess.run([sys.executable,'-u',str(REMOTE/'scripts/run_phase_a_paper_evaluation.py'),'runtime',
   '--source',str(source),'--output',str(out),'--samples',str(samples),'--repetitions',str(repetitions)],cwd=REMOTE,env=env,check=True)
 VOLUME.commit()
 data=io.BytesIO()
 with tarfile.open(fileobj=data,mode='w:gz',compresslevel=1) as archive:
  for path in out.rglob('*'):
   if path.is_file() and path.suffix in ('.json','.csv','.jsonl','.npy') and path.name!='attention_comparison.jsonl':
    archive.add(path,arcname=str(path.relative_to(out)))
 return data.getvalue()

@app.local_entrypoint()
def main(run_id:str='phase-a-paper-20261019',mode:str='coverage',samples:int=10,rounds:int=24,repetitions:int=3):
 if os.environ.get('MODAL_PROFILE')!='phuclk30':raise ValueError('MODAL_PROFILE=phuclk30 bắt buộc')
 if mode not in ('coverage','runtime','parity') or Path(run_id).name!=run_id or not 1<=samples<=10 or not 2<=rounds<=64 or not 1<=repetitions<=5:raise ValueError('Invalid protocol')
 payload=run.remote(run_id,mode,samples,rounds,repetitions)
 out=ROOT/'outputs/phase_a_paper_20261019'/run_id/mode;out.mkdir(parents=True,exist_ok=False)
 with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:archive.extractall(out,filter='data')
 print(f'Artifacts: {out}')
