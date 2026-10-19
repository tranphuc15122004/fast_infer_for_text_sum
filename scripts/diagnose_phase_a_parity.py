#!/usr/bin/env python3
"""Đối chiếu native DFlash và kiểm tra target teacher-forcing theo shape query."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'externals/dflash')]


def diagnose(source,runtime_path,output):
 import numpy as np
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache
 from dflash.model import DFlashDraftModel
 from TrainingFree.context_adaptive.generation import _eos_ids
 runtime=json.loads(runtime_path.read_text());manifest=json.loads((source/'manifest.json').read_text())
 target=AutoModelForCausalLM.from_pretrained(manifest['target'],dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
 draft=DFlashDraftModel.from_pretrained(manifest['draft'],dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True).to('cuda').eval()
 tokenizer=AutoTokenizer.from_pretrained(manifest['target'],local_files_only=True)
 lookup={s['sample_id']:i for i,s in enumerate(manifest['samples'])};examples=[];native_rows=[]
 def mismatch(a,b):
  common=min(len(a),len(b));first=next((i for i in range(common) if a[i]!=b[i]),common if len(a)!=len(b) else None)
  return {'exact':a==b,'first_difference':first,'matching_positions':sum(x==y for x,y in zip(a,b)),'lengths':[len(a),len(b)]}
 with torch.inference_mode():
  for ref in runtime['references']:
   sid,cap=ref['sample_id'],ref['context_cap'];sample=lookup[sid]
   dense=next(r for r in runtime['records'] if r['sample_id']==sid and r['context_cap']==cap and r['method']=='dense')
   if dense['exact_match_ar']:continue
   ids=torch.tensor(np.load(source/f'sample_{sample:02d}/context_{cap}_input_ids.npy').astype(np.int64),device='cuda').reshape(1,-1)
   native=draft.spec_generate(target=target,input_ids=ids,max_new_tokens=runtime['max_new_tokens'],stop_token_ids=sorted(_eos_ids(target)),temperature=0.)
   native_ids=native[0,ids.shape[1]:].detach().cpu().tolist()
   row={'sample_id':sid,'context_cap':cap,'native_output_ids':native_ids,'native_vs_ar':mismatch(native_ids,ref['output_ids']),'native_vs_adapter_dense':mismatch(native_ids,dense['output_ids'])}
   native_rows.append(row);print('[parity native]',sid,cap,row['native_vs_ar'],flush=True)
   # Same verified input history under q=1 and q=16. A differing argmax here is
   # independent of selector policy and cache cropping during speculation.
   if len(examples)<3:
    history=ref['output_ids'];n=ids.shape[1]
    def forced_logits(width):
     cache=DynamicCache();prefill=target(ids,position_ids=torch.arange(n,device='cuda').reshape(1,-1),past_key_values=cache,use_cache=True,logits_to_keep=1)
     rows=[prefill.logits[0,-1].float().cpu()];del prefill
     for start in range(0,len(history)-1,width):
      chunk=history[start:min(start+width,len(history)-1)]
      result=target(torch.tensor([chunk],device='cuda'),position_ids=torch.arange(n+start,n+start+len(chunk),device='cuda').reshape(1,-1),past_key_values=cache,use_cache=True)
      rows.extend(result.logits[0].float().cpu().unbind(0));del result
     del cache
     return torch.stack(rows)
    single=forced_logits(1);blocked=forced_logits(16);a=single.argmax(-1);b=blocked.argmax(-1)
    differences=[]
    for i in (a!=b).nonzero().reshape(-1).tolist():
     top=single[i].topk(2);topb=blocked[i].topk(2)
     differences.append({'output_index':i,'single_argmax':int(a[i]),'block_argmax':int(b[i]),'single_top2_values':top.values.tolist(),'block_top2_values':topb.values.tolist(),'single_top2_ids':top.indices.tolist(),'block_top2_ids':topb.indices.tolist(),'single_tokens':tokenizer.convert_ids_to_tokens(top.indices.tolist()),'block_tokens':tokenizer.convert_ids_to_tokens(topb.indices.tolist())})
    examples.append({'sample_id':sid,'context_cap':cap,'same_history':True,'q_widths':[1,16],'argmax_differences':differences,'max_absolute_logit_difference':float((single-blocked).abs().max()),'single_matches_ar_all':a.tolist()==history})
    print('[parity teacher forcing]',sid,cap,'different argmax',len(differences),flush=True)
   torch.cuda.empty_cache()
 report={'schema_version':'phase-a.parity-diagnostic.v1','gpu':torch.cuda.get_device_name(0),'dtype':'bfloat16','target_attention':'sdpa','native_rows':native_rows,'teacher_forcing':examples,'runtime_report_sha256':hashlib.sha256(runtime_path.read_bytes()).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'note':'Native mismatches and q-shape teacher forcing diagnose finite precision sensitivity; no claim that every output divergence has this cause.'}
 output.mkdir(parents=True,exist_ok=True);(output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 return report

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();diagnose(a.source,a.runtime,a.output)
