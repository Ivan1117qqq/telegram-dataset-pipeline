"""Local audit harness. Fresh outputs only; never writes raw or legacy data/ outputs."""
import argparse, csv, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/validation"
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--phase",choices=["before","after"],required=True);args=ap.parse_args()
 dst=BASE/args.phase;dst.mkdir(parents=True,exist_ok=True)
 if args.phase=="before":
  snap=BASE/"original_scripts";snap.mkdir(exist_ok=True)
  for p in (ROOT/"scripts").glob("*.py"):
   if p.name!="verify_existing_pipeline.py":shutil.copy2(p,snap/p.name)
  protected=list((ROOT/"data").rglob("*"))+list(ROOT.glob("messages*.html"))
  for name in ("photos","files","video_files","stickers","css","js","images"):
   protected+=list((ROOT/name).rglob("*"))
  manifest={str(p.relative_to(ROOT)):digest(p) for p in protected if p.is_file()}
  (BASE/"protected_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
 fresh=dst/"processed/messages.jsonl"
 jobs=[
 ("preprocess_chat.py",["--input-dir",str(ROOT),"--output",str(fresh)]),
 ("profile_dataset.py",["--input",str(fresh),"--output-dir",str(dst/"analysis")]),
 ("segment_conversations.py",["--input",str(fresh),"--output-dir",str(dst/"segmentation")]),
 ("analyze_single_conversations.py",["--messages",str(fresh),"--segmentation",str(dst/"segmentation/segmentation_60m.jsonl"),"--output-dir",str(dst/"segmentation")]),
 ("analyze_sender_roles.py",["--input",str(fresh),"--output-dir",str(dst/"roles")]),
 ("reconstruct_agent_conversations.py",["--messages",str(fresh),"--baseline",str(dst/"segmentation/segmentation_60m.jsonl"),"--mapping",str(ROOT/"data/roles/sender_role_mapping.csv"),"--output-dir",str(dst/"reconstruction")]),
 ("audit_agent_interactions.py",["--messages",str(fresh),"--mapping",str(ROOT/"data/roles/sender_role_mapping.csv"),"--output-dir",str(dst/"agent_audit")])]
 (dst/"roles").mkdir(exist_ok=True)
 shutil.copy2(ROOT/"data/roles/sender_role_mapping.csv",dst/"roles/sender_role_mapping.csv")
 runs=[]
 for name,argv in jobs:
  cmd=[sys.executable,str(ROOT/"scripts"/name),*argv]
  r=subprocess.run(cmd,cwd=ROOT,capture_output=True,encoding="utf-8",errors="replace",env={**os.environ,"PYTHONUTF8":"1","PYTHONDONTWRITEBYTECODE":"1"})
  (dst/(name+".log")).write_text(r.stdout+"\n"+r.stderr,encoding="utf-8")
  runs.append({"script":name,"command":cmd,"exit_code":r.returncode})
  print(name,r.returncode)
  if r.returncode:break
 (dst/"run_manifest.json").write_text(json.dumps(runs,ensure_ascii=False,indent=2),encoding="utf-8")
 comparisons=[]
 for p in dst.rglob("*"):
  if not p.is_file() or p.suffix not in (".csv",".json",".jsonl",".txt",".md"):continue
  rel=p.relative_to(dst);old=ROOT/"data"/rel
  if not old.exists():continue
  eq=p.read_bytes()==old.read_bytes()
  semantic=None
  if p.suffix==".jsonl":
   semantic=[json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]==[json.loads(l) for l in old.read_text(encoding="utf-8").splitlines()]
  elif p.suffix==".csv":
   semantic=list(csv.DictReader(p.open(encoding="utf-8-sig")))==list(csv.DictReader(old.open(encoding="utf-8-sig")))
  comparisons.append({"file":str(rel),"byte_equal":eq,"structured_equal":semantic})
 (dst/"old_output_comparison.json").write_text(json.dumps(comparisons,indent=2),encoding="utf-8")
 manifest=json.loads((BASE/"protected_manifest.json").read_text(encoding="utf-8"))
 changed=[p for p,h in manifest.items() if not (ROOT/p).is_file() or digest(ROOT/p)!=h]
 assert not changed,changed
 print("Protected files unchanged:",len(manifest))
 if any(r["exit_code"] for r in runs):raise SystemExit(1)
if __name__=="__main__":main()

