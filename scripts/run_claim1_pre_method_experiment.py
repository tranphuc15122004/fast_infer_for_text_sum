#!/usr/bin/env python3
"""Phase B: liên hệ dự báo của suffix trên trace lịch sử, không kiểm định nhân quả.

Top-N logits thiếu full logsumexp: chỉ dùng logit gap, không gán confidence/margin
full-vocab. Grouped OOF và bootstrap document; nhãn chỉ thuộc risk set.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common.io_util import JsonlWriter


def compute_prob_and_margin(logits, *, full_logsumexp=None):
    if full_logsumexp is None:
        raise ValueError('Full-vocabulary normalizer required for confidence/probability margin')
    values = np.sort(np.asarray(logits,dtype=float))[::-1]
    if len(values)<2 or not np.isfinite(values).all() or not np.isfinite(full_logsumexp) or full_logsumexp<values[0]:
        raise ValueError('Invalid logits/normalizer')
    probabilities = np.exp(values[:2]-full_logsumexp)
    return float(probabilities[0]),float(probabilities[0]-probabilities[1])


def load_traces(paths, *, include_interventions=False):
    blocks=defaultdict(dict)
    discarded=defaultdict(int)
    for path in paths:
        for line in path.open():
            row=json.loads(line)
            if row.get('status')!='ok' or 'candidate_logits' not in row:
                continue
            if not include_interventions and row.get('self_condition_stage','baseline')!='baseline':
                discarded['intervention_positions']+=1
                continue
            key=(str(path.resolve()),str(row['sample_id']),int(row['round_index']))
            pos=int(row['draft_position'])
            if pos in blocks[key]:
                raise ValueError(f'Duplicate position in round: {key}, {pos}')
            blocks[key][pos]=row
    records=[]; complete=0
    for key,block in blocks.items():
        if sorted(block)!=list(range(1,16)):
            discarded['incomplete_blocks']+=1
            continue
        complete+=1
        gaps=[]; labels=[]
        for j in range(1,16):
            row=block[j]
            logits=np.sort(np.asarray(row['candidate_logits'],dtype=float))[::-1]
            if len(logits)<2 or not np.isfinite(logits).all():
                raise ValueError('Top-two logits must be present and finite')
            gaps.append(float(logits[0]-logits[1]))
            labels.append(int(row['dflash_selected_token_id']==row['target_token_id']))
        # Success labels up to L and one first failure; suffix labels unobserved.
        first_failure=next((i for i,label in enumerate(labels) if not label),15)
        risk_end=min(first_failure+1,15)
        for index in range(risk_end):
            row=block[index+1]
            document=str(row.get('document_id') or row['sample_id'].split('::')[0])
            records.append({'type':'phase_b_observation','source_group_id':f"{row['dataset']}::{document}",
                'dataset':row['dataset'],'sample_id':row['sample_id'],'round':row['round_index'],
                'position':index+1,'label':labels[index],'logit_margin':gaps[index],
                # Ratio p2/p1 is identifiable from top-two logits without logZ.
                'suffix_top2_ratio_exposure':float(np.exp(-np.asarray(gaps[index+1:])).sum()/15.),
                'signal_scope':'top-two logit gap; not full-vocabulary confidence'})
    return records,{'complete_blocks':complete,'discarded':dict(discarded)}


def fit_logistic(x,y):
    center=x.mean(axis=0); scale=x.std(axis=0); scale[scale<1e-8]=1.
    design=np.column_stack([np.ones(len(x)),(x-center)/scale])
    ridge=1.
    def loss(beta):
        z=design@beta
        return float(np.logaddexp(0,z).sum()-y@z+.5*ridge*(beta@beta))
    def grad(beta):
        return design.T@(expit(design@beta)-y)+ridge*beta
    fit=minimize(loss,np.zeros(design.shape[1]),jac=grad,method='L-BFGS-B',options={'maxiter':500,'gtol':1e-7})
    if not fit.success:
        raise RuntimeError(f'Logistic fit failed: {fit.message}')
    return fit.x,center,scale


def grouped_evaluation(records, *, bootstrap=2000, seed=20261018):
    groups=sorted({r['source_group_id'] for r in records})
    if len(groups)<2:
        return {'status':'insufficient_documents','documents':len(groups),'causal_evidence':False}
    rng=np.random.default_rng(seed)
    groups=list(rng.permutation(groups)); folds={group:i%min(5,len(groups)) for i,group in enumerate(groups)}
    base=np.array([[r['logit_margin'],r['position']] for r in records],dtype=float)
    augmented=np.column_stack([base,[r['suffix_top2_ratio_exposure'] for r in records]])
    y=np.array([r['label'] for r in records],dtype=float)
    predictions={name:np.zeros(len(records)) for name in ('present_position','present_position_suffix')}
    for fold in sorted(set(folds.values())):
        test=np.array([folds[r['source_group_id']]==fold for r in records]); train=~test
        for name,x in [('present_position',base),('present_position_suffix',augmented)]:
            beta,center,scale=fit_logistic(x[train],y[train])
            design=np.column_stack([np.ones(test.sum()),(x[test]-center)/scale])
            predictions[name][test]=expit(design@beta)
    by_document=defaultdict(dict)
    result={}
    for name,probability in predictions.items():
        probability=np.clip(probability,1e-12,1-1e-12)
        nll=-(y*np.log(probability)+(1-y)*np.log1p(-probability))
        brier=(probability-y)**2
        for group in groups:
            mask=np.array([r['source_group_id']==group for r in records])
            by_document[group][name]={'nll':float(nll[mask].mean()),'brier':float(brier[mask].mean())}
        result[name]={'document_macro_nll':float(np.mean([v[name]['nll'] for v in by_document.values()])),
                      'document_macro_brier':float(np.mean([v[name]['brier'] for v in by_document.values()]))}
    difference=np.array([by_document[g]['present_position_suffix']['nll']-by_document[g]['present_position']['nll'] for g in groups])
    draws=[float(rng.choice(difference,size=len(groups),replace=True).mean()) for _ in range(bootstrap)]
    return {'status':'ok','documents':len(groups),'risk_set_positions':len(records),
        'grouped_folds':len(set(folds.values())),'models':result,'document_scores':dict(by_document),
        'suffix_minus_baseline_nll':float(difference.mean()),
        'difference_ci95_document_bootstrap':np.quantile(draws,[.025,.975]).tolist() if draws else None,
        'causal_evidence':False,'scope':'held-out predictive association on historical trace configuration; not new hazard/GPU performance',
        'survival_brier_note':'Reported Brier is conditional continuation-label Brier, not integrated survival Brier'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,action='append')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--include-interventions',action='store_true')
    args=parser.parse_args()
    paths=args.trace or [ROOT/f'outputs/dflash_residual/2026-09-07_causal_screening/self_conditioned/trace_{name}.jsonl' for name in ('govreport','cnn_dm','multi_news')]
    if not paths or any(not p.is_file() for p in paths):
        parser.error('Thiếu trace; không âm thầm bỏ dataset')
    if args.bootstrap<0:
        parser.error('bootstrap phải không âm')
    if (args.output_dir/'observations.jsonl').exists():
        parser.error('Dùng output directory mới để giữ provenance')
    records,coverage=load_traces(paths,include_interventions=args.include_interventions)
    report=grouped_evaluation(records,bootstrap=args.bootstrap)
    report.update(type='summary',phase='B_observational_only',pipeline_ab=False,**coverage,
        includes_interventions=args.include_interventions,
        sources=[{'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    writer=JsonlWriter(args.output_dir/'observations.jsonl')
    for record in records:writer.add(record)
    writer.finalize(report)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'document_scores','sources'}},ensure_ascii=False))


if __name__=='__main__':
    main()
