#!/usr/bin/env python3
"""Archive the old paper directory and install reviewed replacement atomically."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil
from pathlib import Path


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);p.add_argument('--target',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);a=p.parse_args()
 stage=a.stage.resolve();target=a.target.resolve();archive=a.archive.resolve()
 if not target.is_dir() or not stage.is_dir() or archive.exists():raise FileExistsError('Require old target + new stage, and an unused archive path')
 for rel in ('README.md','figure_provenance.json','evidence/coverage/report.json','evidence/runtime/report.json','evidence/runtime/parity_diagnostic.json'):
  if not (stage/rel).is_file():raise FileNotFoundError(stage/rel)
 old=[]
 for path in sorted(x for x in target.rglob('*') if x.is_file()):
  rel=path.relative_to(target);relstr=rel.as_posix()
  if rel.parts[0]=='Old_layer_18':
   old.append({'old_path':relstr,'status':'archived_as_historical','replacement':None,
    'reason':'Bản layer-18 cũ trùng/sai logic; lưu trong archive để truy vết, không trộn với rerun.'})
   continue
  candidates=[stage/rel]
  if len(rel.parts)==1:candidates.extend(stage.rglob(path.name))
  replacement=next((x for x in candidates if x.is_file()),None)
  if replacement:
   old.append({'old_path':relstr,'status':'replaced','replacement':replacement.relative_to(stage).as_posix(),'sha256':sha(replacement)})
  else:
   old.append({'old_path':relstr,'status':'archived_no_equivalent','replacement':None,
    'reason':'Artifact legacy phụ thuộc số đo/claim tổng hợp cũ, không tái tạo trong đánh giá riêng Phase A.'})
 mapping={'schema_version':'phase-a.paper-replacement.v1','old_directory':str(target),
  'archive_directory':str(archive),'new_directory':str(target),'old_file_count':len(old),
  'replaced_count':sum(x['status']=='replaced' for x in old),'archive_only_count':sum(x['status']!='replaced' for x in old),
  'files':old}
 (stage/'figure_replacement_manifest.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n')
 with (stage/'README.md').open('a') as f:f.write(f"\n## Thay thế artifact cũ\n\nBản trước được lưu nguyên trạng tại [`{archive.name}`](../{archive.name}/). Danh mục ánh xạ từng file cũ sang bản thay thế nằm trong [figure_replacement_manifest.json](figure_replacement_manifest.json). Các entry `Old_layer_18` nằm trong archive và không được giữ trong thư mục đang dùng.\n")
 archive.parent.mkdir(parents=True,exist_ok=True)
 os.rename(target,archive)
 try:os.rename(stage,target)
 except Exception:
  os.rename(archive,target);raise
 print(json.dumps({k:v for k,v in mapping.items() if k!='files'},ensure_ascii=False))
 print(f'Installed: {target}')

if __name__=='__main__':main()
