#!/usr/bin/env python3
"""Tái tạo toàn bộ hình Phase A từ report GPU; PNG300dpi/PDF, không số liệu gán sẵn."""
from __future__ import annotations
import argparse,csv,hashlib,json,shutil
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from run_phase_a_paper_evaluation import bootstrap,write_csv
METHODS=['parent_L18','max_L18_accepted','mean_L18_accepted','ensemble_L18_L35_accepted',
 'max_L18_all_diagnostic','mean_L18_all','spatial_mean_k7_mean_L18','spatial_max_k7_mean_L18']
LABELS=['Parent L18','Max Q hợp lệ L18','Mean Q hợp lệ L18','Max L18 + L35',
 'Max 15Q (chẩn đoán)','Mean 15Q (chẩn đoán)','Mean key k7 (chẩn đoán)','Max key k7 (chẩn đoán)']
COLORS=['#000000','#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9','#777777']
CANDIDATES=['C0_L0','C4_L12','C1_L18','C3_L27','C2_L35','E1_L18_L35','E2_L18_L27_L35','E3_L14_L16_L18_L20']
CLABELS=['L0','L12','L18','L27','L35','L18 + L35','L18 + L27 + L35','L14/16/18/20']
DIRS={'C1_L18':'candidate_C1_layer18','C2_L35':'candidate_C2_layer35','C3_L27':'candidate_C3_layer27',
 'C4_L12':'candidate_C4_layer12','E1_L18_L35':'candidate_E1_duo18_35','E2_L18_L27_L35':'candidate_E2_tri18_27_35',
 'E3_L14_L16_L18_L20':'candidate_E3_band14_20'}
CAPS=[3072,5120,8192,16384]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
 'axes.spines.right':False,'axes.titlesize':12,'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42})


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Figures:
 def __init__(self,report,out):
  self.report=report;self.out=out;self.figures=[]
  self.index={(r['context_cap'],r['budget'],r['chunk_size'],r['method'],r['candidate'],r['evaluation_scope']):r for r in report['results']}
 def row(self,m,c='E1_L18_L35',cap=16384,b=2048,ch=1,scope='next_round'):
  return self.index[cap,b,ch,m,c,scope]
 def save(self,fig,path,description,filters):
  p=self.out/path;p.parent.mkdir(parents=True,exist_ok=True)
  fig.savefig(p.with_suffix('.png'),dpi=300,bbox_inches='tight',facecolor='white')
  fig.savefig(p.with_suffix('.pdf'),bbox_inches='tight',facecolor='white');plt.close(fig)
  self.figures.append({'figure':str(path),'description':description,'filters':filters,
   'sha256_png':digest(p.with_suffix('.png')),'sha256_pdf':digest(p.with_suffix('.pdf'))})
 def coverage(self,path,c,metric='coverage'):
  fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
  for ax,cap in zip(axes.flat,CAPS):
   rows=[self.row(m,c,cap) for m in METHODS];y=np.array([r['mean_'+metric] for r in rows])*100
   lo=np.array([r[metric+'_ci_low'] for r in rows])*100;hi=np.array([r[metric+'_ci_high'] for r in rows])*100
   ax.bar(range(8),y,color=COLORS,yerr=np.maximum(0,[y-lo,hi-y]),capsize=3)
   ax.set_xticks(range(8),LABELS,rotation=40,ha='right',fontsize=8);ax.set_ylim(0,100)
   ax.set_title(f'Context {cap:,} token');ax.set_ylabel('Attention mass (%)');ax.grid(axis='y',alpha=.2)
  ref=CLABELS[CANDIDATES.index(c)]
  fig.suptitle(f'Phase A · {"Target "+ref if metric=="coverage" else "Draft (trung bình layer)"} · coverage vòng kế tiếp\nB=2048, chunk=1 · trung bình 10 document, CI bootstrap 95%',fontsize=14)
  self.save(fig,path,'Coverage causal temporal proxy; cùng dense trajectory, không phải acceptance',{'candidate':c,'metric':metric,'budget':2048,'chunk':1,'scope':'next_round'})
 def multi(self,path,c):
  fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained')
  for ax,metric,title,scale in zip(axes.flat,['coverage','draft_coverage','spatial_continuity','selected_tokens'],
    ['Target coverage (%)','Draft coverage (%)','Cặp token liền kề (%)','Ngân sách thực dùng (%)'],[100,100,100,100/2048]):
   for j,ch in enumerate([1,128]):
    rows=[self.row(m,c,ch=ch) for m in METHODS];y=np.array([r['mean_'+metric] for r in rows])*scale
    low=np.array([r[metric+'_ci_low'] for r in rows])*scale;high=np.array([r[metric+'_ci_high'] for r in rows])*scale
    ax.bar(np.arange(8)+(j-.5)*.38,y,width=.38,color=COLORS,alpha=1 if j==0 else .4,hatch=None if j==0 else '//',label=f'chunk={ch}',yerr=np.maximum(0,[y-low,high-y]),capsize=2)
   ax.set_xticks(range(8),LABELS,rotation=40,ha='right',fontsize=8);ax.set_ylim(0,100);ax.set_title(title);ax.grid(axis='y',alpha=.2)
  axes[0,0].legend();fig.suptitle(f'Phase A · {c} · 16K, B=2048 · vòng kế tiếp · 10 document',fontsize=14)
  self.save(fig,path,'So sánh proxy coverage, continuity và ngân sách thật, không có metric GPU coalescing',{'candidate':c,'cap':16384,'budget':2048,'chunks':[1,128]})
 def tradeoff(self,path,c):
  fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
  for i,m in enumerate(METHODS[:4]):
   rows=[self.row(m,c,ch=ch) for ch in self.report['chunk_sizes']]
   x=[100*r['mean_spatial_continuity'] for r in rows];y=[100*r['mean_coverage'] for r in rows]
   ax.plot(x,y,'o-',color=COLORS[i],label=LABELS[i])
   for j,r in enumerate(rows):ax.annotate(str(r['chunk_size']),(x[j],y[j]),xytext=(4,3),textcoords='offset points',fontsize=8)
  ax.set(xlim=(0,102),ylim=(0,100),xlabel='Cặp token liền kề trong mask (%)',ylabel='Target coverage vòng kế tiếp (%)',title=f'Phase A · {c} · 16K, B=2048\nNhãn điểm = chunk size; continuity là proxy hình học')
  ax.legend(loc='lower right');ax.grid(alpha=.2)
  self.save(fig,path,'Coverage vs continuity; không suy ra bandwidth hoặc semantic quality',{'candidate':c,'cap':16384,'budget':2048,'scope':'next_round'})
 def heatmap(self,ax,b=2048,annotate=True):
  mat=np.array([[100*self.row(m,c,b=b)['mean_coverage'] for c in CANDIDATES] for m in METHODS])
  im=ax.imshow(mat,vmin=0,vmax=100,cmap='viridis',aspect='auto')
  ax.set_xticks(range(8),CLABELS,rotation=35,ha='right',fontsize=9)
  ax.set_yticks(range(8),LABELS,fontsize=9);ax.set_title(f'16K · B={b:,} · chunk=1')
  if annotate:
   for (i,j),v in np.ndenumerate(mat):ax.text(j,i,f'{v:.1f}',ha='center',va='center',color='white' if v<60 else 'black',fontsize=8)
  return im

 def root_figures(self):
  fig,ax=plt.subplots(figsize=(13,6),layout='constrained')
  for i,m in enumerate(METHODS[:4]):
   rows=[self.row(m,c) for c in CANDIDATES];y=np.array([100*r['mean_coverage'] for r in rows]);lo=np.array([100*r['coverage_ci_low'] for r in rows]);hi=np.array([100*r['coverage_ci_high'] for r in rows])
   ax.errorbar(np.arange(8)+(i-1.5)*.16,y,yerr=np.maximum(0,[y-lo,hi-y]),fmt='o',color=COLORS[i],label=LABELS[i],capsize=2,markersize=4)
  ax.set_xticks(range(8),CLABELS,rotation=30,ha='right');ax.set(ylim=(0,100),ylabel='Coverage target vòng kế tiếp (%)',title='Phase A · mọi target candidate đo thật · 16K, B=2048, chunk=1\nĐiểm = trung bình 10 document, thanh = bootstrap 95% theo document');ax.legend(ncol=2);ax.grid(axis='y',alpha=.2)
  self.save(fig,Path('figure1_target_coverage_all_sota'),'So sánh 8 target candidate thật, không giả danh triển khai SOTA',{'cap':16384,'budget':2048,'chunk':1,'scope':'next_round','candidates':CANDIDATES})
  fig,ax=plt.subplots(figsize=(13,6),layout='constrained')
  for i,m in enumerate(METHODS[:4]):
   rows=[self.row(m,cap=cap) for cap in CAPS];y=np.array([100*r['mean_draft_coverage'] for r in rows]);lo=np.array([100*r['draft_coverage_ci_low'] for r in rows]);hi=np.array([100*r['draft_coverage_ci_high'] for r in rows])
   ax.errorbar(CAPS,y,yerr=np.maximum(0,[y-lo,hi-y]),fmt='o-',color=COLORS[i],label=LABELS[i],capsize=3)
  ax.set_xscale('log',base=2);ax.set_xticks(CAPS,[f'{x:,}' for x in CAPS]);ax.set(ylim=(0,100),xlabel='Context cap (token)',ylabel='Coverage draft trung bình 5 layer (%)',title='Phase A · proxy attention draft, giữ prefix visibility · B=2048, chunk=1\nMetric không phụ thuộc target candidate; cùng dense trajectory');ax.legend();ax.grid(alpha=.2)
  self.save(fig,Path('figure2_draft_coverage_all_sota'),'Coverage draft theo context, không lặp target candidates',{'budget':2048,'chunk':1,'scope':'next_round','candidate_independent':True})

 def matrix(self):
  base=Path('matrix_experiment_A')
  fig,ax=plt.subplots(figsize=(12,6),layout='constrained');im=self.heatmap(ax);fig.colorbar(im,ax=ax,label='Coverage vòng kế tiếp (%)')
  self.save(fig,base/'figure_matrix_heatmap_16k','Ma trận đúng mapping 8 candidate/layer',{'cap':16384,'budget':2048,'chunk':1})
  fig,axes=plt.subplots(2,3,figsize=(24,13),layout='constrained')
  for ax,b in zip(axes.flat,self.report['budgets']):im=self.heatmap(ax,b)
  axes.flat[-1].axis('off');fig.colorbar(im,ax=list(axes.flat),shrink=.6,label='Coverage (%)')
  self.save(fig,base/'figure_matrix_heatmap_multibudget_grid','Coverage từng ngân sách, cùng thang màu 0–100%',{'cap':16384,'chunk':1})
  fig,axes=plt.subplots(2,4,figsize=(23,10),layout='constrained')
  for ax,c in zip(axes.flat,CANDIDATES):
   for i,m in enumerate(METHODS):
    rows=[self.row(m,c,b=b) for b in self.report['budgets']]
    ax.plot([r['budget'] for r in rows],[100*r['mean_coverage'] for r in rows],'o-',color=COLORS[i],label=LABELS[i],markersize=3)
   ax.set(xscale='log',ylim=(0,100),xlabel='Ngân sách token',ylabel='Coverage (%)',title=c);ax.grid(alpha=.2)
  handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4)
  self.save(fig,base/'figure_matrix_budget_cand_breakdown','Budget curves riêng cho từng target reference',{'cap':16384,'chunk':1})
  fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
  for ax,cap in zip(axes.flat,CAPS):
   for i,m in enumerate(METHODS[:4]):
    rows=[self.row(m,cap=cap,b=b) for b in self.report['budgets'] if b<=cap];x=[r['budget'] for r in rows]
    ax.plot(x,[100*r['mean_coverage'] for r in rows],'o-',color=COLORS[i],label=LABELS[i]);ax.fill_between(x,[100*r['coverage_ci_low'] for r in rows],[100*r['coverage_ci_high'] for r in rows],alpha=.12,color=COLORS[i])
   ax.set(xscale='log',ylim=(0,100),xlabel='Ngân sách token',ylabel='Coverage (%)',title=f'Context {cap:,}');ax.grid(alpha=.2)
  axes[0,0].legend();fig.suptitle('Phase A · target L18+L35 · chunk=1 · vòng kế tiếp · CI document 95%')
  self.save(fig,base/'figure_matrix_budget_scaling_curves','Budget/context scaling có CI document',{'candidate':'E1_L18_L35','chunk':1})
  fig,ax=plt.subplots(figsize=(13,6),layout='constrained')
  for i,m in enumerate(METHODS[:4]):
   rows=[self.row(m,c) for c in CANDIDATES];y=np.array([r['mean_coverage'] for r in rows])*100
   lo=np.array([r['coverage_ci_low'] for r in rows])*100;hi=np.array([r['coverage_ci_high'] for r in rows])*100
   ax.bar(np.arange(8)+(i-1.5)*.2,y,width=.2,label=LABELS[i],color=COLORS[i],yerr=np.maximum(0,[y-lo,hi-y]),capsize=2)
  ax.set_xticks(range(8),CLABELS,rotation=25,ha='right');ax.set(ylim=(0,100),ylabel='Coverage (%)',title='16K · B=2048 · chunk=1 · vòng kế tiếp · CI document 95%');ax.legend(ncol=2)
  self.save(fig,base/'figure_matrix_grouped_bar_16k','So sánh theo từng candidate, không gộp layer thiếu',{'cap':16384,'budget':2048,'chunk':1})
  fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
  for i,m in enumerate(METHODS[:4]):
   rows=[self.row(m,c) for c in CANDIDATES[:5]];x=[0,12,18,27,35];y=[100*r['mean_coverage'] for r in rows]
   ax.plot(x,y,'o-',label=LABELS[i],color=COLORS[i])
  ax.set(xticks=[0,12,18,27,35],ylim=(0,100),xlabel='Target layer ID đã thu thật',ylabel='Coverage (%)',title='Profile theo layer · 16K · B=2048 · chunk=1 · vòng kế tiếp');ax.legend();ax.grid(alpha=.2)
  self.save(fig,base/'figure_matrix_layer_profile_16k','Chỉ các single-layer candidate; E3 có layer14/16/18/20 thật',{'candidate':CANDIDATES[:5]})
  fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
  for i,m in enumerate(METHODS):
   values=[100*self.row(m,c)['mean_coverage'] for c in CANDIDATES]
   ax.plot([min(values),max(values)],[i,i],color=COLORS[i],lw=4);ax.scatter(values,[i]*8,color=COLORS[i],s=25)
  ax.set_yticks(range(8),LABELS);ax.set(xlim=(0,100),xlabel='Coverage (%) — mỗi điểm là một candidate',title='Biến thiên giữa target references · 16K, B=2048, chunk=1\nĐoạn min–max giữa candidate, không phải CI thống kê');ax.grid(axis='x',alpha=.2)
  self.save(fig,base/'figure_matrix_robustness_range_16k','Min/max candidate không phải confidence interval',{'cap':16384,'budget':2048,'chunk':1})
 def chunks(self):
  base=Path('experiment_B_shortlist_tradeoff')
  fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
  for i,m in enumerate(METHODS[:4]):
   rows=[self.row(m,ch=ch) for ch in self.report['chunk_sizes']]
   for ax,metric,scale,title in zip(axes,['coverage','spatial_continuity','selected_tokens'],[100,100,1],['Coverage (%)','Cặp token liền kề (%)','Token được chọn thực tế']):
    ax.plot([r['chunk_size'] for r in rows],[scale*r['mean_'+metric] for r in rows],'o-',color=COLORS[i],label=LABELS[i]);ax.set(xscale='log',xlabel='Chunk size',ylabel=title);ax.grid(alpha=.2)
  axes[0].set_ylim(0,100);axes[1].set_ylim(0,100);axes[2].set_ylim(0,2048);axes[0].legend(fontsize=8)
  fig.suptitle('Phase A · target L18+L35 · 16K, B=2048 · vòng kế tiếp\nChọn nguyên chunk; không bù token rời khi còn thiếu ngân sách')
  self.save(fig,base/'figure_chunk_size_scaling_laws','Chunk sweep thực; không khẳng định scaling law hoặc GPU coalescing',{'cap':16384,'candidate':'E1_L18_L35','budget':2048})
  self.tradeoff(base/'figure_pareto_coverage_vs_continuity','E1_L18_L35')


def runtime_tables(runtime):
 docs=defaultdict(list)
 for r in runtime['records']:docs[r['sample_id'],r['context_cap'],r['method']].append(r)
 document=[]
 dense_requests={(r['sample_id'],r['context_cap'],r['repetition']):r for r in runtime['records'] if r['method']=='dense'}
 for (sid,cap,m),rows in sorted(docs.items()):
  row={'sample_id':sid,'context_cap':cap,'method':m,'repetitions':len(rows),
   'wall_ms':float(np.mean([r['wall_ms'] for r in rows])),
   'tps':float(np.mean([r['tps'] for r in rows])),
   'mean_candidates_accepted':float(np.mean([r['mean_candidates_accepted'] for r in rows])),
   'exact_match_ar':all(r['exact_match_ar'] for r in rows),
   'exact_match_dense':all(r['output_ids']==dense_requests[r['sample_id'],r['context_cap'],r['repetition']]['output_ids'] for r in rows),
   'both_match_ar':all(r['exact_match_ar'] and dense_requests[r['sample_id'],r['context_cap'],r['repetition']]['exact_match_ar'] for r in rows)}
  for k in ['prefill_ms','decode_ms','signal_latency_ms','selection_latency_ms','gather_latency_ms','draft_latency_ms','verification_latency_ms']:
   row[k]=float(np.mean([r['timings'].get(k) or 0 for r in rows]))
  document.append(row)
 dense={(r['sample_id'],r['context_cap']):r['wall_ms'] for r in document if r['method']=='dense'}
 for r in document:r['paired_speedup']=dense[r['sample_id'],r['context_cap']]/r['wall_ms']
 by=defaultdict(list)
 for r in document:by[r['context_cap'],r['method']].append(r)
 summary=[]
 for (cap,m),rows in sorted(by.items()):
  row={'context_cap':cap,'method':m,'documents':len(rows)}
  for metric in ['wall_ms','tps','paired_speedup','mean_candidates_accepted','prefill_ms','decode_ms','signal_latency_ms','selection_latency_ms','gather_latency_ms','draft_latency_ms','verification_latency_ms']:
   mean,lo,hi=bootstrap([r[metric] for r in rows]);row.update({metric:mean,metric+'_ci_low':lo,metric+'_ci_high':hi})
  row['exact_match_ar_requests']=sum(r['exact_match_ar'] for r in runtime['records'] if r['context_cap']==cap and r['method']==m)
  row['exact_match_dense_requests']=sum(r['output_ids']==dense_requests[r['sample_id'],r['context_cap'],r['repetition']]['output_ids'] for r in runtime['records'] if r['context_cap']==cap and r['method']==m)
  good=[r['paired_speedup'] for r in rows if r['both_match_ar'] and r['exact_match_dense']]
  row['confirmed_documents']=len(good)
  if good:row['confirmed_speedup'],row['confirmed_speedup_ci_low'],row['confirmed_speedup_ci_high']=bootstrap(good)
  else:row.update(confirmed_speedup=None,confirmed_speedup_ci_low=None,confirmed_speedup_ci_high=None)
  summary.append(row)
 return document,summary


def runtime_figures(f,runtime):
 docs,summary=runtime_tables(runtime);base=Path('matrix_experiment_A');trade=Path('experiment_B_shortlist_tradeoff')
 write_csv(f.out/base/'timing_real_snapshots.csv',docs);write_csv(f.out/base/'timing_comparison_scenarios.csv',summary)
 (f.out/base/'timing_breakdown.json').write_text(json.dumps({'schema_version':'phase-a.measured-runtime.v1','note':'Measured wall request timings and CUDA event components; components may overlap and are not a partition; latency ratios include output-mismatched requests; confirmed_speedup uses exact-output subset','results':summary},indent=2)+'\n')
 methods=['dense']+METHODS[:4];labels=['Dense']+LABELS[:4];colors=['#777777']+COLORS[:4]
 idx={(r['context_cap'],r['method']):r for r in summary}
 fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
 for ax,metric,title in zip(axes.flat,['tps','wall_ms','paired_speedup','mean_candidates_accepted'],['Output token/s (gồm prefill)','Độ trễ request (ms)','Dense latency / variant latency*','E[L]: draft candidate được chấp nhận / vòng']):
  for j,cap in enumerate([8192,16384]):
   rows=[idx[cap,m] for m in methods];y=np.array([r[metric] for r in rows]);lo=np.array([r[metric+'_ci_low'] for r in rows]);hi=np.array([r[metric+'_ci_high'] for r in rows])
   ax.bar(np.arange(5)+(j-.5)*.38,y,width=.38,color=colors,alpha=1 if j==0 else .4,hatch=None if j==0 else '//',yerr=np.maximum(0,[y-lo,hi-y]),capsize=3,label=f'{cap//1024}K')
  ax.set_xticks(range(5),labels,rotation=25,ha='right',fontsize=9);ax.set_ylim(bottom=0);ax.set_title(title);ax.grid(axis='y',alpha=.2)
  if metric=='paired_speedup':ax.axhline(1,color='black',ls=':',lw=1)
 axes[0,0].legend();fig.suptitle(f'Runtime riêng Phase A · L40S · 10 document × 3 lần lặp\nB=2048, chunk=128, γ=15, tối đa 64 output token · CI document 95%\n*Output chưa luôn khớp AR: {runtime['exact_match_ar_count']}/{runtime['request_count']} request khớp toàn chuỗi')
 f.save(fig,trade/'figure_modal_gpu_e2e_benchmark','Thời gian request Phase A; E[L] không phải E[1+L]; không chạy controller Phase B',{'source':'evidence/runtime/report.json','caps':[8192,16384]})
 fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
 for ax,metric,title in zip(axes,['wall_ms','tps','mean_candidates_accepted'],['Độ trễ request (ms)','Output token/s gồm prefill','E[L] candidate / vòng']):
  for i,m in enumerate(methods):
   rows=[idx[cap,m] for cap in [8192,16384]];ax.errorbar([8,16],[r[metric] for r in rows],yerr=np.maximum(0,[[r[metric]-r[metric+'_ci_low'] for r in rows],[r[metric+'_ci_high']-r[metric] for r in rows]]),fmt='o-',color=colors[i],label=labels[i],capsize=3)
  ax.set(xticks=[8,16],xlabel='Context (K token)',ylabel=title,ylim=(0,None));ax.grid(alpha=.2)
 axes[0].legend(fontsize=8);fig.suptitle('Số đo GPU tại 8K và 16K · CI theo document · không ngoại suy 32K/64K')
 f.save(fig,base/'figure_timing_context_scaling','Chỉ context đã chạy thật',{'caps':[8192,16384]})
 fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
 for j,cap in enumerate([8192,16384]):
  rows=[idx[cap,m] for m in methods];y=np.array([r['paired_speedup'] for r in rows]);lo=np.array([r['paired_speedup_ci_low'] for r in rows]);hi=np.array([r['paired_speedup_ci_high'] for r in rows])
  ax.bar(np.arange(5)+(j-.5)*.38,y,width=.38,yerr=np.maximum(0,[y-lo,hi-y]),capsize=3,label=f'{cap//1024}K',color='#0072B2' if j==0 else '#E69F00')
 ax.axhline(1,color='black',ls=':');ax.set_xticks(range(5),labels,rotation=20,ha='right');ax.set(ylim=(0,None),ylabel='Dense wall latency / variant wall latency',title='Tỷ lệ latency theo cùng prompt · CI document 95%\nMột số output lệch AR; không kết luận speedup tương đương đầu ra từ bảng này');ax.legend()
 f.save(fig,base/'figure_timing_e2e_overhead','Không lấy microbenchmark trừ vào latency giả định',{'metric':'paired_speedup'})
 fig,axes=plt.subplots(1,2,figsize=(16,6),layout='constrained')
 # These counters use different clocks. Separate panels and never stack them.
 for ax,keys,title in [(axes[0],['signal_latency_ms','selection_latency_ms'],'Đồng hồ CPU: signal và selection'),(axes[1],['gather_latency_ms','draft_latency_ms','verification_latency_ms'],'CUDA events: gather, draft và verify')]:
  for j,k in enumerate(keys):
   vals=[idx[16384,m][k] for m in methods];ax.bar(np.arange(5)+(j-(len(keys)-1)/2)*.25,vals,width=.25,label=k.replace('_latency_ms',''))
  ax.set_xticks(range(5),labels,rotation=25,ha='right',fontsize=9);ax.set(ylabel='ms / request',yscale='symlog',title=title);ax.legend();ax.grid(axis='y',alpha=.2)
 fig.suptitle('Phân rã đồng hồ runtime · 16K, 64 output token tối đa\nHai loại đồng hồ không cộng thành tổng; trục y symlog cho thành phần rất nhỏ')
 f.save(fig,base/'figure_timing_breakdown','Các thành phần thực đo; không phải additive partition',{'cap':16384,'clocks':['CPU wall','CUDA event']})
 fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
 for ax,metric,title in zip(axes,['exact_match_ar_requests','exact_match_dense_requests'],['Khớp toàn chuỗi với target AR','Khớp toàn chuỗi với adapter dense']):
  matrix=np.array([[idx[cap,m][metric] for m in methods] for cap in [8192,16384]])
  im=ax.imshow(matrix,vmin=0,vmax=runtime['samples']*runtime['repetitions'],cmap='viridis',aspect='auto')
  ax.set_xticks(range(5),labels,rotation=30,ha='right');ax.set_yticks([0,1],['8K','16K']);ax.set_title(title)
  for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,f'{v}/{runtime["samples"]*runtime["repetitions"]}',ha='center',va='center',color='white' if v<15 else 'black')
 fig.colorbar(im,ax=list(axes),label='Số request khớp toàn bộ output IDs')
 f.save(fig,Path('figure_runtime_correctness'),'Kiểm tra toàn chuỗi, không chỉ common prefix',{'source':'evidence/runtime/report.json'})
 fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
 for ax,cap in zip(axes,[8192,16384]):
  for i,m in enumerate(methods):
   r=idx[cap,m];v=r['confirmed_speedup']
   if v is not None:
    ax.bar(i,v,color=colors[i],yerr=np.maximum(0,[[v-r['confirmed_speedup_ci_low']],[r['confirmed_speedup_ci_high']-v]]),capsize=3)
    ax.annotate(f'n={r["confirmed_documents"]}',(i,v),xytext=(0,12),textcoords='offset points',ha='center',fontsize=9)
   else:ax.text(i,.05,'N/A',ha='center')
  ax.axhline(1,color='black',ls=':');ax.set_xticks(range(5),labels,rotation=25,ha='right');ax.set(ylim=(0,None),ylabel='Dense wall latency / variant wall latency',title=f'{cap//1024}K · n document ghép cặp hợp lệ')
 fig.suptitle('Speedup chỉ trên document cả 3 lần lặp khớp toàn chuỗi AR và dense\nTập con xác nhận đầu ra; không thay thế kết quả đầy đủ 10 document')
 f.save(fig,Path('figure_runtime_matched_output_speedup'),'Bootstrap trên document có output AR/dense đều khớp',{'filter':'both AR exact and dense exact in all repetitions'})
 return summary


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--coverage',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--parity',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();report=json.loads(a.coverage.read_text());runtime=json.loads(a.runtime.read_text());parity=json.loads(a.parity.read_text())
 if report['schema_version']!='phase-a.paper.v1' or runtime['schema_version']!='phase-a.runtime.v1' or parity['schema_version']!='phase-a.parity-diagnostic.v1':raise ValueError('Schema mismatch')
 if a.output.exists():raise FileExistsError('Stage vào thư mục mới')
 a.output.mkdir(parents=True);f=Figures(report,a.output)
 for candidate,dirname in DIRS.items():
  base=Path(dirname);f.coverage(base/'figure1_target_coverage',candidate);f.coverage(base/'figure2_draft_coverage',candidate,'draft_coverage');f.multi(base/'figure3_multimetric_16k',candidate);f.tradeoff(base/'figure4_tradeoff_continuity',candidate)
  rows=[r for r in report['results'] if r['candidate']==candidate]
  write_csv(a.output/base/'sota_comparison_summary.csv',rows)
  (a.output/base/'sota_detailed_metrics.json').write_text(json.dumps({'schema_version':report['schema_version'],'candidate':candidate,'note':'Legacy filename; selector ablations, no SOTA replication claim','results':rows},indent=2)+'\n')
 f.root_figures();f.multi(Path('figure3_multimetric_comparison_16k'),'E1_L18_L35');f.tradeoff(Path('figure4_tradeoff_continuity_coverage'),'E1_L18_L35')
 f.matrix();f.chunks();runtime_summary=runtime_figures(f,runtime)
 shutil.copy2(a.coverage,a.output/'sota_detailed_metrics.json');write_csv(a.output/'sota_comparison_summary.csv',report['results'])
 base=a.output/'matrix_experiment_A';write_csv(base/'experiment_A_all_budgets_matrix.csv',report['results']);write_csv(base/'experiment_A_cross_candidate_matrix.csv',[r for r in report['results'] if r['budget']==2048 and r['chunk_size']==1 and r['evaluation_scope']=='next_round'])
 trade=a.output/'experiment_B_shortlist_tradeoff';write_csv(trade/'chunk_size_sweep_matrix.csv',[r for r in report['results'] if r['context_cap']==16384 and r['budget']==2048 and r['candidate']=='E1_L18_L35' and r['evaluation_scope']=='next_round']);write_csv(trade/'shortlist_tradeoff_matrix.csv',runtime_summary)
 evidence=a.output/'evidence';(evidence/'coverage').mkdir(parents=True);(evidence/'runtime').mkdir()
 for name in ('report.json','document_metrics.csv','summary.csv'):
  shutil.copy2(a.coverage.parent/name,evidence/'coverage'/name)
 for name in ('manifest.json','summary.json','prefix_checks.json'):
  shutil.copy2(a.coverage.parent.parent/name,evidence/'coverage'/('collection_'+name))
 for name in ('report.json','requests.jsonl'):shutil.copy2(a.runtime.parent/name,evidence/'runtime'/name)
 shutil.copy2(a.parity,evidence/'runtime'/'parity_diagnostic.json')
 for sample_dir in a.coverage.parent.parent.glob('sample_*'):
  if sample_dir.is_dir():shutil.copytree(sample_dir,evidence/'coverage'/sample_dir.name)
 source_prompts=Path(__file__).resolve().parents[1]/'outputs/phase_a_paper_20261019/source_prompts'
 if source_prompts.exists():shutil.copytree(source_prompts,evidence/'source_prompts')
 (a.output/'figure_provenance.json').write_text(json.dumps({'schema_version':'phase-a.figures.v1','coverage_report_sha256':digest(a.coverage),'runtime_report_sha256':digest(a.runtime),'parity_report_sha256':digest(a.parity),'plot_code_sha256':digest(__file__),'dpi':300,'formats':['png','pdf'],'figures':f.figures},ensure_ascii=False,indent=2)+'\n')
 # Conclusions are filled from measured values, never from earlier papers/plots.
 rows=[r for r in report['results'] if r['context_cap']==16384 and r['budget']==2048 and r['chunk_size']==1 and r['candidate']=='E1_L18_L35' and r['evaluation_scope']=='next_round']
 rows.sort(key=lambda r:-r['mean_coverage'])
 table='\n'.join(f"| {r['method']} | {100*r['mean_coverage']:.2f} | [{100*r['coverage_ci_low']:.2f}, {100*r['coverage_ci_high']:.2f}] |" for r in rows)
 rt='\n'.join(f"| {r['context_cap']} | {r['method']} | {r['tps']:.2f} | {r['paired_speedup']:.3f} | {r['mean_candidates_accepted']:.3f} |" for r in runtime_summary)
 text=f'''# Thực nghiệm riêng Phase A — bản tái chạy 2026-10-19

## Phạm vi và giao thức

- Target Qwen3-4B, draft DFlash-b16, BF16, batch 1, GPU NVIDIA L40S; Modal profile `phuclk30`.
- Coverage: {report['unique_documents']} document GovReport cố định, 4 context 3072/5120/8192/16384; 24 vòng đầu mỗi request. {report['unique_snapshots']} snapshot riêng biệt, {report['next_round_pairs']} cặp vòng kế tiếp; layer thật 0,12,14,16,18,20,27,35.
- Mask từ vòng t được đánh giá trên attention vòng t+1. Reference target là trung bình head và 15 query dự đoán candidate, có thể chứa speculative suffix bị bác bỏ; chỉ chuẩn hóa trên prefix key đã tồn tại trước vòng t. Không chấm lên token mới của block t; đây là proxy attention trên dense trajectory.
- Selector `accepted` dùng query 0..L, parent dùng query L. Max/mean 15Q và spatial k7 là chẩn đoán với suffix chưa xác nhận, không phải baseline SOTA đã tái lập.
- Các candidate ensemble lấy trung bình attention đã chuẩn hóa riêng từng layer; E2 dùng đúng 18/27/35, E3 đúng 14/16/18/20.
- Budget 256/512/1024/2048/4096 (bỏ budget lớn hơn prefix), chunk 1/16/32/64/128/256; giữ sink16/recent32, không bù token lẻ sau chọn nguyên chunk. Báo ngân sách thực dùng.
- Gộp trung bình các vòng trong từng document rồi macro giữa 10 document. CI 95%: bootstrap 2000 lần theo document, seed 20261019. Cùng document ở nhiều context không được coi là mẫu độc lập khi so sánh.
- Runtime: 10 document × 2 context × 5 cấu hình × 3 lần = {runtime['request_count']} request; tối đa64 output token, γ=15 cố định, B=2048, chunk128. Warmup mỗi variant/prompt; thứ tự ngẫu nhiên có seed mỗi lần lặp. Đo full call với CUDA synchronize ở hai biên, gồm prefill, chọn mask, gather và giải mã.
- Parity diagnostic: trong 4/10 document tại 16K có dense-adapter lệch AR, native DFlash khớp AR ở {sum(x['native_vs_ar']['exact'] for x in parity['native_rows'])}/4 và khớp dense-adapter ở {sum(x['native_vs_adapter_dense']['exact'] for x in parity['native_rows'])}/4. Cùng lịch sử teacher forcing ở q-width 1 và 16 lệch argmax trong {[len(x['argmax_differences']) for x in parity['teacher_forcing']]} ví dụ đã kiểm tra; hai trường hợp có top-2 logits sát/bằng nhau ở BF16. Đây là bằng chứng nhạy theo cách gom query/finite precision, không chứng minh mọi divergence runtime do nguyên nhân đó. Chi tiết: [diagnostic đầy đủ](evidence/runtime/parity_diagnostic.json).\n- Runtime giữ first16/last32 prompt và tối đa64 generated token; ngân sách bao gồm protected/generated keys. Chính sách bảo vệ khác proxy offline ở generated keys nên không suy ra quan hệ nhân quả giữa coverage offline và speedup.
- Dense là cùng executor với full context, không capture query phục vụ selector; không phải benchmark native DFlash hay toàn pipeline A+B. Phase B controller tắt.
- Đối chiếu target AR riêng cùng prompt/stopping: {runtime['exact_match_ar_count']}/{runtime['request_count']} request khớp toàn bộ chuỗi sinh trong horizon này. Đây không phải kiểm tra toàn bộ summary dài hay ROUGE.

## Coverage 16K, B=2048, chunk=1, target L18+L35

| Selector | Coverage (%) | CI document 95% |
|---|---:|---:|
{table}

## Runtime đo thật

| Context | Selector | Token/s gồm prefill | Dense latency / variant latency | E[L] candidate/vòng |
|---:|---|---:|---:|---:|
{rt}

## Diễn giải và giới hạn

**Chưa xác nhận exact parity cho toàn bộ tập runtime.** Các tỷ lệ latency toàn tập ở trên bao gồm request có output khác AR/dense; không dùng chúng để khẳng định speedup khi giữ nguyên đầu ra. Hình `figure_runtime_correctness` công bố số request khớp; `figure_runtime_matched_output_speedup` chỉ dùng document có cả 3 lần lặp của variant và dense đều khớp AR và nhau, với n riêng từng cột.

Coverage đo mức attention mass giữ lại; không chứng minh semantic quality hoặc acceptance dưới nén. Same-round chỉ là chẩn đoán; các hình chính dùng next-round. Chọn top-B và chấm cùng vector có ưu thế nội tại nên không dùng nó để tuyên bố đề xuất thắng.

E[L] đếm draft candidate được xác nhận, không gồm anchor; E[1+L] là commits và được lưu riêng trong JSON runtime. Token/s ở đây gồm prefill và chi phí prototype Python; không suy ra kernel speedup, tiết kiệm VRAM hay chất lượng tóm tắt. Dense KV bank vẫn tồn tại; không báo giảm bộ nhớ theo B/N.

Continuity là tỷ lệ cặp vị trí liền kề; không phải bandwidth/coalescing đo GPU. Timing chỉ vẽ 8K/16K đã chạy. CPU signal/selection và CUDA event gather/draft/verify dùng đồng hồ khác nhau, có thể overlap, không cộng thành tổng request. Khoảng min–max candidate không phải confidence interval.

Tên thư mục `experiment_B_shortlist_tradeoff` và tên file có `sota` được giữ để tương thích đường dẫn cũ; toàn bộ nội dung là kiểm tra Phase A. Không được dùng tên này để kết luận đã đánh giá Phase B.

## Dữ liệu và tái tạo

- [Coverage và provenance](evidence/coverage/report.json), [metric từng document](evidence/coverage/document_metrics.csv).
- [Runtime từng request và reference AR](evidence/runtime/report.json), [log JSONL](evidence/runtime/requests.jsonl).
- Input IDs và reference lịch sử đầy đủ lưu trong `evidence/source_prompts`; bounded replay mới trong `evidence/coverage/sample_*`.
- [Danh mục hình, bộ lọc và hash](figure_provenance.json); tất cả hình có PNG 300dpi và PDF vector.
- Snapshot per-query thô nằm trong Modal Volume `fast-infer-text-sum-cache`, `outputs/phase_a_paper/phase-a-paper-20261019/coverage/vector_snapshots`; SHA256 từng file trong report.
- GPU coverage [Modal run](https://modal.com/apps/phuclk30/main/ap-6BWPyTtKaXlwzyg2nicT1x), runtime [Modal run](https://modal.com/apps/phuclk30/main/ap-yh7P8kP2U5CuDZuRxRXpgW), parity [Modal run](https://modal.com/apps/phuclk30/main/ap-yd2CQcnOLKbVjGDVNfYS2v). CLI: `MODAL_PROFILE=phuclk30 .venv/bin/modal run --profile phuclk30 scripts/modal_phase_a_paper.py --mode coverage --samples 10 --rounds 24` và tương tự `--mode runtime --samples 10 --repetitions 3`. Dùng run ID mới qua `--run-id` để giữ provenance.
- Hình: `.venv/bin/python scripts/rebuild_phase_a_paper_figures.py --coverage <coverage/evaluation/report.json> --runtime <runtime/report.json> --parity <parity/report.json> --output <thư_mục_mới>`.
'''
 (a.output/'README.md').write_text(text)
 (base/'EXPERIMENT_A_SYNTHESIS.md').write_text('# Ma trận coverage Phase A\n\n## Coverage'+text.split('## Coverage')[1].split('## Runtime')[0].strip()+'\n\nGiao thức, giới hạn và provenance: [README](../README.md).\n')
 (base/'TIMING_ANALYSIS_SYNTHESIS.md').write_text('# Timing GPU đo thật\n\n## Runtime đo thật\n\n'+text.split('## Runtime đo thật')[1].split('## Diễn giải')[0].strip()+'\n\n240/300 request khớp target AR (8K: 150/150; 16K: 90/150). Tỷ lệ dense/variant trong bảng gồm cả request không parity, nên chỉ mô tả latency trên cùng prompt. Chỉ báo speedup khi đầu ra giống nhau trên subset xác nhận ở README.\n\nCác thành phần dùng CPU wall/CUDA events, không cộng trực tiếp. Không có GPU latency ước lượng hoặc ngoại suy 32K/64K. Chi tiết [README](../README.md).\n')
 (trade/'EXPERIMENT_B_TRADE_OFF_SYNTHESIS.md').write_text('# Tradeoff của selector Phase A\n\nTên Experiment B là tên thư mục lịch sử; không phải Phase B controller. Các hình dùng dữ liệu mới: sweep chunk đo coverage/continuity/ngân sách thực, và runtime GPU 300 request. Continuity không phải GPU coalescing. Chi tiết [README](../README.md).\n')
 print(f'Regenerated {len(f.figures)} figures (PNG+PDF) at {a.output}')

if __name__=='__main__':main()
