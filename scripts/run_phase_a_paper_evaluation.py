#!/usr/bin/env python3
"""Đo riêng Phase A; streaming snapshot, bootstrap theo document, không giả lập GPU."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'externals/dflash')]
from common.io_util import JsonlWriter
from run_candidate_benchmarks import load_snapshot
from TrainingFree.context_interventions.selector_evaluation import (
    CANDIDATES, normalized, reference_attention, selector_scores, select_chunks, spatial_continuity)
BUDGETS = (256,512,1024,2048,4096)
CHUNKS = (1,16,32,64,128,256)
FIELDS = ('context_cap','budget','chunk_size','method','candidate','evaluation_scope')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def bootstrap(values):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261019)
    means = a[rng.integers(len(a),size=(2000,len(a)))].mean(1)
    return float(a.mean()), *map(float,np.quantile(means,[.025,.975]))


def evaluate(run, output):
    output.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((run/'manifest.json').read_text())
    paths = sorted((run/'vector_snapshots').glob('*_round_*.npz'))
    # File names sort by document, cap, round. Only two consecutive snapshots live in RAM.
    groups = defaultdict(list)
    for p in paths:
        prefix = p.name.split('_round_')[0]; groups[prefix].append(p)
    accum = defaultdict(lambda:np.zeros(5,dtype=np.float64))
    sources = []; layer_ids = None; pairs = 0
    for name, files in sorted(groups.items()):
        current = load_snapshot(files[0],manifest)
        for i,path in enumerate(files):
            if i: current = following
            following = load_snapshot(files[i+1],manifest) if i+1<len(files) else None
            sources.append({'file':path.name,'sha256':sha(path)})
            ids = list(map(int,current['target_layer_ids']))
            if ids != [0,12,14,16,18,20,27,35]: raise ValueError(f'Unexpected layers: {ids}')
            layer_ids = ids
            n = current['prefix_length']; scores, missing = selector_scores(current,n)
            if missing: raise ValueError(missing)
            selections = []
            for budget in BUDGETS:
                if budget>n: continue
                for method,vector in scores.items():
                    for chunk in CHUNKS:
                        idx = select_chunks(vector,budget,chunk_size=chunk,sink=16,recent=32)
                        selections.append((budget,method,chunk,idx,spatial_continuity(idx)))
            refs = [('same_round_diagnostic',current)]
            if following is not None:
                if following['round'] != current['round']+1: raise ValueError('Round gap')
                refs.append(('next_round',following)); pairs+=1
            for scope,ref in refs:
                target_refs = {k:reference_attention(ref,layers,n) for k,layers in CANDIDATES.items()}
                draft_ref = np.mean([normalized(row[:n]) for row in ref['draft']],axis=0)
                for budget,method,chunk,idx,continuity in selections:
                    draft_cov = float(draft_ref[idx].sum())
                    for candidate,gt in target_refs.items():
                        key=(current['sample_id'],current['context_cap'],budget,chunk,method,candidate,scope)
                        accum[key] += [float(gt[idx].sum()),draft_cov,continuity,len(idx),1]
        print(f'[evaluate] {name}: {len(files)} snapshots',flush=True)
    document_rows=[]
    for key,v in sorted(accum.items()):
        document_rows.append(dict(zip(('sample_id',)+FIELDS,key)) | {
            'coverage':v[0]/v[4],'draft_coverage':v[1]/v[4],
            'spatial_continuity':v[2]/v[4],'selected_tokens':v[3]/v[4],'rounds':int(v[4])})
    by_group=defaultdict(list)
    for r in document_rows: by_group[tuple(r[f] for f in FIELDS)].append(r)
    results=[]
    for key,rows in sorted(by_group.items()):
        record=dict(zip(FIELDS,key))|{'documents':len(rows),'snapshots':sum(r['rounds'] for r in rows)}
        for metric in ('coverage','draft_coverage','spatial_continuity','selected_tokens'):
            mean,low,high=bootstrap([r[metric] for r in rows])
            record.update({f'mean_{metric}':mean,f'{metric}_ci_low':low,f'{metric}_ci_high':high})
        results.append(record)
    report={'schema_version':'phase-a.paper.v1','status':'ok','phase':'A','pipeline_ab':False,
        'unique_documents':len({r['sample_id'] for r in document_rows}), 'unique_snapshots':len(paths),
        'next_round_pairs':pairs,'layers':layer_ids,'budgets':BUDGETS,'chunk_sizes':CHUNKS,
        'reference':'mean-head mean q[0:15] target attention conditional on previous visible prefix',
        'draft_reference':'mean-layer mean-head mean proposal-query draft attention on previous visible prefix',
        'aggregation':'macro mean over documents after mean over rounds; percentile document bootstrap 2000, seed20261019',
        'protection':{'sink':16,'recent':32},'source_manifest':manifest,'source_files':sources,
        'code_sha256':sha(__file__),'metric_code_sha256':sha(ROOT/'src/TrainingFree/context_interventions/selector_evaluation.py'),
        'results':results}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    write_csv(output/'document_metrics.csv',document_rows);write_csv(output/'summary.csv',results)
    w=JsonlWriter(output/'document_metrics.jsonl')
    for r in document_rows:w.add({'type':'phase_a_document_metric',**r})
    w.finalize({k:v for k,v in report.items() if k not in ('results','source_files','source_manifest')})
    return report


def runtime(source, output, samples=10, repetitions=3, max_new_tokens=64):
    import gc
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from dflash.model import DFlashDraftModel
    from TrainingFree.context_adaptive.config import AdaptiveConfig
    from TrainingFree.context_adaptive.generation import generate_adaptive,generate_target_only
    from TrainingFree.context_adaptive.types import PromptLayout
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'manifest.json').read_text())
    tokenizer=AutoTokenizer.from_pretrained(manifest['target'],local_files_only=True)
    target=AutoModelForCausalLM.from_pretrained(manifest['target'],dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    draft=DFlashDraftModel.from_pretrained(manifest['draft'],dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).to('cuda').eval()
    variants=[('dense','recent_only','full'),('parent_L18','target_parent',2048),
        ('max_L18_accepted','multi_query_max_pooling',2048),
        ('mean_L18_accepted','multi_query_mean_pooling',2048),
        ('ensemble_L18_L35_accepted','duo_ensemble',2048)]
    records=[]; refs=[]; writer=JsonlWriter(output/'requests.jsonl')
    rng=np.random.default_rng(20261019)
    with torch.inference_mode():
        for sample_index in range(samples):
            for cap in (8192,16384):
                ids_np=np.load(source/f'sample_{sample_index:02d}/context_{cap}_input_ids.npy')
                ids=torch.from_numpy(ids_np.astype(np.int64)).reshape(1,-1).to('cuda');n=ids.shape[1]
                sid=manifest['samples'][sample_index]['sample_id']
                layout=PromptLayout(n,tuple(range(n)),tuple(range(16))+tuple(range(n-32,n)),
                    tuple(tuple(range(j,min(n,j+128))) for j in range(0,n,128)),
                    hashlib.sha256(ids_np.tobytes()).hexdigest(),True,dataset='gov_report',source_group_id=sid)
                reference=generate_target_only(target,ids,max_new_tokens=max_new_tokens,temperature=0.)
                reference_ids=reference.output_ids.detach().cpu().reshape(-1).tolist()
                refs.append({'sample_id':sid,'context_cap':cap,'input_sha256':sha(source/f'sample_{sample_index:02d}/context_{cap}_input_ids.npy'),'output_ids':reference_ids})
                del reference
                # Warm each execution path outside measured repetitions.
                for method,selector,budget in variants:
                    cfg=AdaptiveConfig(budgets=(2048,'full'),selector=selector,target_layer=18,duo_deep_layer=35,
                        source_chunk_size=128,source_anchors=0,recent_output=64,timing_mode='wall')
                    warm=generate_adaptive(target,draft,ids,tokenizer,layout,cfg,max_new_tokens=8,fixed_budget=budget,fixed_gamma=15,gamma_reference=15,variant='dflash_full_fixed' if budget=='full' else 'best_fixed_pair')
                    del warm
                for repetition in range(repetitions):
                    for vi in rng.permutation(len(variants)):
                        method,selector,budget=variants[vi]
                        cfg=AdaptiveConfig(budgets=(2048,'full'),selector=selector,target_layer=18,duo_deep_layer=35,
                            source_chunk_size=128,source_anchors=0,recent_output=64,timing_mode='wall')
                        torch.cuda.synchronize();start=time.perf_counter()
                        result=generate_adaptive(target,draft,ids,tokenizer,layout,cfg,max_new_tokens=max_new_tokens,
                            fixed_budget=budget,fixed_gamma=15,gamma_reference=15,temperature=0.,
                            variant='dflash_full_fixed' if budget=='full' else 'best_fixed_pair',
                            run_id='phase-a-paper',sample_id=sid,dataset='gov_report',repetition=repetition)
                        torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000
                        out=result.output_ids.detach().cpu().reshape(-1).tolist();count=result.counters['speculative_rounds']
                        row={'type':'phase_a_runtime','sample_id':sid,'context_cap':cap,'method':method,'repetition':repetition,
                            'budget':budget,'wall_ms':wall,'output_tokens':result.output_tokens,'tps':1000*result.output_tokens/wall,
                            'exact_match_ar':out==reference_ids,'output_ids':out,'counters':result.counters,'timings':result.timings,
                            'mean_candidates_accepted':result.counters['draft_tokens_accepted']/count if count else 0.,
                            'mean_commits':result.counters['mean_accept_length'],'status':result.status}
                        writer.add(row);records.append(row)
                        print(f'[runtime] {sample_index+1}/{samples} cap={cap} rep={repetition} {method} {wall:.1f}ms exactAR={row["exact_match_ar"]}',flush=True)
                        del result;gc.collect()
                torch.cuda.empty_cache()
    report={'schema_version':'phase-a.runtime.v1','phase':'A','pipeline_ab':False,'gpu':torch.cuda.get_device_name(0),
        'torch':torch.__version__,'transformers':transformers.__version__,'samples':samples,'repetitions':repetitions,
        'max_new_tokens':max_new_tokens,'chunk_size':128,'fixed_gamma':15,'budget':2048,
        'prompt_policy':'same saved token IDs as coverage; select over whole prompt; preserve first16,last32 initial prompt and all <=64 generated tokens',
        'timing':'wall clock around full generate_adaptive call, CUDA synchronize at boundaries; wall instrumentation for all variants',
        'order':'seeded random permutation per repetition, warmup each variant per prompt',
        'exact_match_ar_count':sum(r['exact_match_ar'] for r in records),'request_count':len(records),
        'code_sha256':sha(__file__),'generation_code_sha256':sha(ROOT/'src/TrainingFree/context_adaptive/generation.py'),
        'attention_code_sha256':sha(ROOT/'src/TrainingFree/context_interventions/attention.py'),
        'records':records,'references':refs}
    writer.finalize({k:v for k,v in report.items() if k not in ('records','references')})
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['evaluate','runtime'])
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=10);p.add_argument('--repetitions',type=int,default=3)
    a=p.parse_args()
    if a.mode=='evaluate':evaluate(a.source,a.output)
    else:runtime(a.source,a.output,a.samples,a.repetitions)
