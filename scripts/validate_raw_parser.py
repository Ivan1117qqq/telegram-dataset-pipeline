"""Independent raw-HTML verification: ElementTree-backed HTML reader, no production parser import."""
import argparse,csv,hashlib,json,random,re,sys
from collections import Counter
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote,urlsplit
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
from analyze_sender_roles import safe
class Reader(HTMLParser):
 def __init__(self):
  super().__init__(convert_charrefs=True);self.root=ET.Element("root");self.stack=[self.root]
 def handle_starttag(self,tag,attrs):
  n=ET.SubElement(self.stack[-1],tag,dict(attrs))
  if tag not in {"area","base","br","col","embed","hr","img","input","link","meta","param","source","track","wbr"}:self.stack.append(n)
 def handle_endtag(self,tag):
  for i in range(len(self.stack)-1,0,-1):
   if self.stack[i].tag==tag:del self.stack[i:];return
 def handle_data(self,text):
  n=self.stack[-1]
  if len(n):n[-1].tail=(n[-1].tail or "")+text
  else:n.text=(n.text or "")+text
def has(n,c):return c in n.get("class","").split()
def direct(n,c):return [x for x in n if has(x,c)]
def text(n,skipdate=False):
 if n is None:return ""
 if skipdate and has(n,"date"):return ""
 if n.tag=="br":return "\n"
 return (n.text or "")+"".join(text(c,skipdate)+(c.tail or "") for c in n)
def first(n,c):return next((x for x in n if has(x,c)),None)
def reply(n,source):
 x=first(n,"reply_to")
 if x is None:return None
 a=next(x.iter("a"));u=urlsplit(a.get("href"));mid=re.search(r"message(-?\d+)$",u.fragment)[1]
 return {"message_id":mid,"source_file":unquote(u.path) or source,"href":a.get("href")}
def raw_rows():
 rows=[];counts={};fragments={}
 for p in sorted(ROOT.glob("messages*.html"),key=lambda p:(int(re.search(r"\d+",p.stem)[0]) if re.search(r"\d+",p.stem) else 1,p.name)):
  reader=Reader();reader.feed(p.read_text(encoding="utf-8"));nodes=[n for n in reader.root.iter("div") if has(n,"message")]
  stat=Counter(blocks=len(nodes));prev=None
  for n in nodes:
   mid=n.get("id")[7:];b=first(n,"body");service=has(n,"service")
   if service:
    value=text(b).strip();stat["service_blocks"]+=1;prev=None
    if re.fullmatch(r"\d{1,2} [A-Za-z]+ \d{4}",value):stat["date_separators"]+=1;continue
    rows.append(dict(message_id=mid,source_file=p.name,sender=None,datetime=None,text=value,reply=None,forwarded=[],attachments=[],types=["service"]));continue
   stat["default"]+=1
   if has(n,"joined"):stat["joined"]+=1
   sender=first(b,"from_name")
   sender=text(sender).strip() if sender is not None else prev if has(n,"joined") else None
   prev=sender
   raw_date=first(b,"date").get("title");datepart,offset=raw_date.split(" UTC")
   dt=datetime.fromisoformat(datetime.strptime(datepart,"%d.%m.%Y %H:%M:%S").isoformat()+offset).isoformat()
   scopes=[b]+[x for x in b.iter("div") if x is not b and has(x,"forwarded") and has(x,"body")]
   parts=[text(x).strip() for scope in scopes for x in direct(scope,"text")]
   forwards=[]
   for scope in scopes[1:]:
    fn=first(scope,"from_name");fd=next((x for x in fn.iter() if has(x,"date")),None) if fn is not None else None
    forwards.append({"sender":text(fn,True).strip() if fn is not None else None,
       "datetime":datetime.strptime(fd.get("title"),"%d.%m.%Y %H:%M:%S UTC%z").isoformat() if fd is not None else None,
       "reply":reply(scope,p.name),"text":"\n".join(text(x).strip() for x in direct(scope,"text"))})
   media=[]
   for wrap in b.iter("div"):
    if not has(wrap,"media_wrap"):continue
    for item in wrap:
     title=next((text(x).strip() for x in item.iter() if has(x,"title")),None)
     kind=("file" if has(item,"media_file") else "sticker" if has(item,"sticker_wrap") or title=="Sticker"
           else "animation" if has(item,"animated_wrap") or title=="Animation" else "photo" if has(item,"photo_wrap") or title=="Photo" else "video")
     path=item.get("href");local=bool(path and (p.parent/unquote(path)).is_file())
     media.append({"type":kind,"path":path,"title":title,"html_class":item.get("class"),"reference_exists":bool(path),"local_exists":local})
   value="\n".join(v for v in parts if v)
   types=["normal"]+([ "joined"] if has(n,"joined") else [])+(["reply"] if reply(b,p.name) else [])+(["forwarded"] if forwards else [])+(["media"] if media else [])+(["text_attachment"] if value and media else [])+(["mention"] if any("ShowMention" in a.get("onclick","") for a in b.iter("a")) else [])+(["emoji"] if any(ord(c)>0xffff for c in value) else [])+(["multiline"] if "\n" in value else [])
   rows.append(dict(message_id=mid,source_file=p.name,sender=sender,datetime=dt,text=value,reply=reply(b,p.name),forwarded=forwards,attachments=media,types=types))
  counts[p.name]=dict(stat)
 return rows,counts
def dumpcsv(path,rs):
 with path.open("w",encoding="utf-8-sig",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader()
  for r in rs:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--phase",required=True);args=ap.parse_args()
 out=ROOT/"outputs/validation";phase=out/args.phase
 raw,counts=raw_rows();parsed=[json.loads(x) for x in (phase/"processed/messages.jsonl").read_text(encoding="utf-8").splitlines()]
 by={(r["source_file"],r["message_id"]):r for r in parsed};rawkeys={(r["source_file"],r["message_id"]) for r in raw}
 checks=[]
 for i,r in enumerate(raw):
  p=by.get((r["source_file"],r["message_id"]),{})
  f=p.get("forwarded",[]);pf=[{k:x.get(k) for k in ("sender","datetime","reply","text")} for x in f]
  ra=[{k:x[k] for k in ("type","path","title","html_class")} for x in r["attachments"]];pa=[{k:x.get(k) for k in ("type","path","title","html_class")} for x in p.get("attachments",[])]
  rr=r["reply"]["message_id"] if r["reply"] else None
  c={"sample_id":i+1,"message_id":r["message_id"],"source_file":r["source_file"],"sample_types":r["types"]}
  for key,a,b in [("sender",r["sender"],p.get("sender")),("datetime",r["datetime"],p.get("datetime")),("text",r["text"],p.get("text")),("reply_to",rr,p.get("reply_to")),("forwarded",r["forwarded"],pf),("attachment",ra,pa)]:
   label={"reply_to":"reply"}.get(key,key);c["raw_"+key]=safe(a) if key=="text" else a;c["parsed_"+key]=safe(b) if key=="text" and b is not None else b;c[label+"_match"]=a==b
  c["overall_pass"]=all(c[k] for k in ("sender_match","datetime_match","text_match","reply_match","forwarded_match","attachment_match"))
  c["notes"]="Comparison before masking; forwarded.text missing is schema loss, not outer sender mismatch." if not c["forwarded_match"] else "Exact decoded field comparison; UI excluded."
  # Do not expose unmasked forwarded text in the human-readable CSV.
  c["raw_forwarded"]=[{**x,"text":safe(x["text"])} for x in r["forwarded"]]
  c["parsed_forwarded"]=[{**x,"text":safe(x["text"]) if x["text"] is not None else None} for x in pf]
  checks.append(c)
 rng=random.Random(20260929);chosen={}
 for kind in ("normal","joined","reply","forwarded","media","service","text_attachment","mention","emoji","multiline"):
  pool=[c for c in checks if kind in c["sample_types"]]
  for c in rng.sample(pool,min(4,len(pool))):chosen[c["sample_id"]]=c
 for c in rng.sample([c for c in checks if c["sample_id"] not in chosen],50-len(chosen)):chosen[c["sample_id"]]=c
 selected=sorted(chosen.values(),key=lambda c:c["sample_id"]);dumpcsv(phase/"parser_sample_validation.csv",selected)
 if args.phase=="after":dumpcsv(out/"parser_sample_validation.csv",selected)
 normal=[r for r in parsed if r["message_type"]=="message"];raw_normal=[r for r in raw if "service" not in r["types"]]
 replies=[r for r in raw if r["reply"]];missing=[r for r in replies if (r["reply"]["source_file"],r["reply"]["message_id"]) not in rawkeys]
 attachment=[a for r in raw for a in r["attachments"]]
 summary={"raw_files":counts,"raw_blocks":sum(c["blocks"] for c in counts.values()),"parsed_messages":len(parsed),"date_separators":sum(c["date_separators"] for c in counts.values()),
  "key_sets_equal":set(by)==rawkeys,"duplicate_parsed_ids":len(parsed)-len({r["message_id"] for r in parsed}),
  "normal_missing_senders":sum(not r["sender"] or r["sender"] in ("None","UNKNOWN","NaN") for r in normal),
  "field_mismatches":{k:sum(not c[k] for c in checks) for k in ("sender_match","datetime_match","text_match","reply_match","forwarded_match","attachment_match")},
  "sample_count":len(selected),"sample_pass":sum(c["overall_pass"] for c in selected),
  "sample_type_coverage":dict(Counter(t for c in selected for t in c["sample_types"])),
  "reply":{"total_reply_messages":len(replies),"valid_reply_target":len(replies)-len(missing),"broken_reply_target":len(missing),"cross_file_reply":sum(r["reply"]["source_file"]!=r["source_file"] for r in replies),
   "missing_target":len(missing),"parser_related_missing":0 if all(by[(r["source_file"],r["message_id"])]["reply_to"]==r["reply"]["message_id"] for r in missing) else None,
   "missing_examples":[{"source":r["source_file"],"id":r["message_id"],"reference":r["reply"],"classification":"absent_from_supplied_export; deletion not provable"} for r in missing]},
  "attachment_count":len(attachment),"local_attachment_exists":sum(a["local_exists"] for a in attachment),"local_attachment_missing":sum(not a["local_exists"] for a in attachment),
  "attachment_extensions":dict(Counter(Path(a["path"]).suffix.lower() for a in attachment)),
  "negative_time_gaps":sum(b["datetime"]<a["datetime"] for a,b in zip(normal,normal[1:]))}
 (phase/"raw_verification.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
 dumpcsv(phase/"sender_distribution.csv",[{"sender":s,"message_count":n} for s,n in Counter(r["sender"] for r in normal).most_common()])
 # Attachment references are inspected only with is_file; no content read or OCR.
 dumpcsv(phase/"attachment_validation.csv",[{"message_id":r["message_id"],**a} for r in raw for a in r["attachments"]])
 thresholds=[]
 for m in (10,30,60,120):
  cs=[json.loads(l) for l in (phase/f"segmentation/segmentation_{m}m.jsonl").read_text(encoding="utf-8").splitlines()]
  chat=[c for c in cs if c["candidate_type"]=="chat"];sizes=sorted(len(c["messages"]) for c in chat)
  owner={r["message_id"]:c["conversation_id"] for c in cs for r in c["messages"]}
  edges=[r for r in normal if r["reply_to"] and by.get((r["reply_to_source_file"],r["reply_to"]),{}).get("message_type")=="message"]
  p=(len(sizes)-1)*.9;idx=int(p);q=sizes[idx]+(sizes[min(idx+1,len(sizes)-1)]-sizes[idx])*(p-idx)
  thresholds.append({"threshold":m,"conversation_count":len(chat),"single_message_count":sizes.count(1),"single_message_ratio":sizes.count(1)/len(chat),"median_messages":__import__("statistics").median(sizes),"p90_messages":q,"max_messages":max(sizes),"multi_sender_ratio":sum(len({r["sender"] for r in c["messages"]})>1 for c in chat)/len(chat),"direct_reply_edges":len(edges),"broken_direct_reply":sum(owner[r["message_id"]]!=owner[r["reply_to"]] for r in edges),"long_reply_edges":sum((datetime.fromisoformat(r["datetime"])-datetime.fromisoformat(by[(r["reply_to_source_file"],r["reply_to"])]["datetime"])).total_seconds()>86400 for r in edges),"all_messages_preserved":Counter(r["message_id"] for c in cs for r in c["messages"])==Counter(r["message_id"] for r in parsed)})
 dumpcsv(phase/"threshold_validation.csv",thresholds)
 print(json.dumps(summary,ensure_ascii=True,indent=2))
if __name__=="__main__":main()

