"""Analyze singleton candidates without changing segmentation. Standard library only."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import re
import statistics

def redact(text):
    text=re.sub(r"(?i)(?:https?://|www\.)[^\s<>\u3000]+","[URL]",text)
    text=re.sub(r"(?i)[\w.+-]+@[\w.-]+\.[a-z]{2,}","[EMAIL]",text)
    text=re.sub(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])","[IP]",text)
    text=re.sub(r"(?i)\b[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\b","[ID]",text)
    text=re.sub(r"(?i)(\b(?:token|hash|password|passwd|secret|api[_ -]?key)\b\s*[:=\uff1a]\s*)[^\s&<>]+",r"\1[REDACTED]",text)
    text=re.sub(r"(?i)\b[0-9a-f]{24,}\b","[ID]",text)
    def phone(m):
        v=m[0]
        return "[PHONE_CANDIDATE]" if not re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}",v) and 8<=len(re.sub(r"\D","",v))<=15 else v
    return re.sub(r"(?<![\w.])\+?\d[\d ()-]{6,}\d(?![\w.])",phone,text)

def write_csv(path, rows, fields=None):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})

def main():
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--segmentation",type=Path,default=root/"data/segmentation/segmentation_60m.jsonl")
    ap.add_argument("--messages",type=Path,default=root/"data/processed/messages.jsonl")
    ap.add_argument("--output-dir",type=Path,default=root/"data/segmentation")
    ap.add_argument("--seed",type=int,default=20260914)
    args=ap.parse_args()
    protected=list((root/"data/segmentation").glob("segmentation_*"))+[args.messages]+[
        root/"scripts/preprocess_chat.py",root/"scripts/profile_dataset.py",root/"scripts/segment_conversations.py"]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    candidates=[json.loads(x) for x in args.segmentation.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows=[json.loads(x) for x in args.messages.read_text(encoding="utf-8").splitlines() if x.strip()]
    by={r["message_id"]:r for r in rows}
    order={r["message_id"]:i for i,r in enumerate(rows)}
    chat=[c for c in candidates if c["candidate_type"]=="chat"]
    singles=[c for c in chat if c["message_count"]==1]
    owner={m["message_id"]:c for c in chat for m in c["messages"]}
    normal=[r for r in rows if r["message_type"]=="message"]
    assert set(owner)=={r["message_id"] for r in normal}
    times={r["message_id"]:datetime.fromisoformat(r["datetime"]) for r in normal}
    def target(r):
        t=by.get(r.get("reply_to"))
        if t and t["source_file"]==(r.get("reply_to_source_file") or r["source_file"]):
            return t
        return None
    topics=defaultdict(list)
    for r in normal:
        topic=owner[r["message_id"]]["topic_root"]
        if topic is not None:
            topics[topic].append(r)
    neighbors={}
    for topic, group in topics.items():
        group.sort(key=lambda r:(times[r["message_id"]],order[r["message_id"]]))
        for i,r in enumerate(group):
            neighbors[r["message_id"]]=(group[i-1] if i else None,group[i+1] if i+1<len(group) else None)
    inbound=defaultdict(list)
    long=[]
    missing=[]
    system_refs=0
    normal_edges=0
    for r in normal:
        if not r.get("reply_to"):continue
        t=target(r)
        if t is None:
            missing.append(r["message_id"]);continue
        if t["message_type"]!="message":
            system_refs+=1;continue
        normal_edges+=1
        inbound[t["message_id"]].append(r)
        gap=(times[r["message_id"]]-times[t["message_id"]]).total_seconds()
        if gap>86400:
            long.append({"message_id":r["message_id"],"datetime":r["datetime"],"sender":r["sender"],
                "topic_root":owner[r["message_id"]]["topic_root"],
                "reply_to":t["message_id"],"target_datetime":t["datetime"],"target_sender":t["sender"],
                "gap_seconds":gap,"gap_hours":gap/3600,"gap_days":gap/86400,
                "conversation_id":owner[r["message_id"]]["conversation_id"],
                "target_conversation_id":owner[t["message_id"]]["conversation_id"],
                "same_candidate":owner[r["message_id"]]["conversation_id"]==owner[t["message_id"]]["conversation_id"]})
    long.sort(key=lambda r:-r["gap_seconds"])
    def mini(r):
        if r is None:return None
        return {"message_id":r["message_id"],"datetime":r["datetime"],"sender":r["sender"],
                "text":redact(r["text"]),"reply_to":r["reply_to"],
                "conversation_id":owner[r["message_id"]]["conversation_id"]}
    def relation(a,b):
        if b is None:return None
        ta,tb=target(a),target(b)
        return {"single_replies_to_neighbor":bool(ta and ta["message_id"]==b["message_id"]),
                "neighbor_replies_to_single":bool(tb and tb["message_id"]==a["message_id"]),
                "same_resolved_reply_target":bool(ta and tb and ta["message_id"]==tb["message_id"]),
                "shared_target_is_system":bool(ta and tb and ta["message_id"]==tb["message_id"] and ta["message_type"]=="service")}
    analyses=[]
    for c in sorted(singles,key=lambda c:order[c["messages"][0]["message_id"]]):
        r=c["messages"][0];mid=r["message_id"]
        prev,nxt=neighbors.get(mid,(None,None))
        gp=(times[mid]-times[prev["message_id"]]).total_seconds() if prev else None
        gn=(times[nxt["message_id"]]-times[mid]).total_seconds() if nxt else None
        rp,rn=relation(r,prev),relation(r,nxt)
        def direct(rel):
            return bool(rel and (rel["single_replies_to_neighbor"] or rel["neighbor_replies_to_single"]))
        t=target(r)
        replygap=(times[mid]-times[t["message_id"]]).total_seconds() if t and t["message_type"]=="message" else None
        pclose=(gp is not None and gp<=3600) or direct(rp)
        nclose=(gn is not None and gn<=3600) or direct(rn)
        if replygap is not None and replygap>86400:
            label="LONG_DISTANCE_REPLY"
        elif c["topic_root"] is None or (r["reply_to"] and t is None):
            label="UNKNOWN"
        elif pclose and nclose:label="POSSIBLE_BOTH"
        elif pclose:label="POSSIBLE_PREVIOUS"
        elif nclose:label="POSSIBLE_NEXT"
        else:label="STRUCTURALLY_ISOLATED"
        analyses.append({"conversation_id":c["conversation_id"],"message_id":mid,
            "datetime":r["datetime"],"sender":r["sender"],"text":redact(r["text"]),"text_redacted":True,
            "topic_root":c["topic_root"],"reply_to":r["reply_to"],
            "reply_target_type":"none" if not r["reply_to"] else "missing" if t is None else t["message_type"],
            "reply_target_gap_seconds":replygap,"incoming_normal_reply_count":len(inbound[mid]),
            "previous_message":mini(prev),"next_message":mini(nxt),
            "gap_previous_seconds":gp,"gap_next_seconds":gn,
            "same_sender_previous":r["sender"]==prev["sender"] if prev else None,
            "same_sender_next":r["sender"]==nxt["sender"] if nxt else None,
            "previous_sender_equals_next":prev["sender"]==nxt["sender"] if prev and nxt else None,
            "previous_reply_relationship":rp,"next_reply_relationship":rn,
            "structural_classification":label})
    labels=["STRUCTURALLY_ISOLATED","POSSIBLE_PREVIOUS","POSSIBLE_NEXT","POSSIBLE_BOTH","LONG_DISTANCE_REPLY","UNKNOWN"]
    classifications={label:sum(a["structural_classification"]==label for a in analyses) for label in labels}
    def gap_bin(g):
        if g is None:return "no_neighbor"
        return ("<1 min" if g<60 else "1-5 min" if g<300 else "5-10 min" if g<600 else
                "10-30 min" if g<1800 else "30-60 min" if g<3600 else "1-2 hr" if g<=7200 else ">2 hr")
    bin_labels=["<1 min","1-5 min","5-10 min","10-30 min","30-60 min","1-2 hr",">2 hr","no_neighbor"]
    bins={side:{label:sum(gap_bin(a[f"gap_{side}_seconds"])==label for a in analyses) for label in bin_labels} for side in ("previous","next")}
    proximity={}
    for minutes in (5,10,30,60):
        p=[a["gap_previous_seconds"] is not None and a["gap_previous_seconds"]<=minutes*60 for a in analyses]
        n=[a["gap_next_seconds"] is not None and a["gap_next_seconds"]<=minutes*60 for a in analyses]
        proximity[str(minutes)]={"previous_within":sum(p),"next_within":sum(n),
                                 "both_sides_within":sum(a and b for a,b in zip(p,n)),
                                 "either_side_within":sum(a or b for a,b in zip(p,n))}
    continuity={}
    for field in ("same_sender_previous","same_sender_next","previous_sender_equals_next"):
        eligible=sum(a[field] is not None for a in analyses)
        count=sum(a[field] is True for a in analyses)
        continuity[field]={"equal_count":count,"eligible_count":eligible,
                           "percent_among_eligible":100*count/eligible if eligible else None,
                           "percent_of_all_singles":100*count/len(analyses)}
    stats={"baseline":str(args.segmentation),"final_threshold_selected":False,
        "single_chat_candidates":len(analyses),"excluded_single_system_events":sum(c["candidate_type"]=="system_event" and c["message_count"]==1 for c in candidates),
        "known_topic_single_count":sum(a["topic_root"] is not None for a in analyses),
        "unknown_topic_single_count":sum(a["topic_root"] is None for a in analyses),
        "classification_rules":{"proximity_seconds":3600,"long_reply_seconds_strictly_greater_than":86400,
            "priority":"LONG_DISTANCE_REPLY > UNKNOWN > POSSIBLE_BOTH > POSSIBLE_PREVIOUS/NEXT > STRUCTURALLY_ISOLATED",
            "link":"Known same-topic neighbor within <=60m OR direct reply between singleton and neighbor. Merely sharing a topic/system reply target is not a direct relationship.",
            "unknown":"No confirmed topic or unresolved outgoing reply. Unknown topics are never pooled.",
            "isolated":"Known topic, neither side meets proximity/direct relationship. Missing neighbor at dataset boundary counts as no observed evidence, not proof of isolation."},
        "classifications":classifications,
        "classification_percent_all_singles":{k:100*v/len(analyses) for k,v in classifications.items()},
        "gap_bins":bins,"gap_bin_boundaries_seconds":"[0,60),[60,300),[300,600),[600,1800),[1800,3600),[3600,7200],(7200,+inf); no_neighbor includes unknown topic",
        "no_neighbor_breakdown":{side:{"unknown_topic":sum(a["topic_root"] is None for a in analyses),
              "known_topic_export_boundary":sum(a["topic_root"] is not None and a[f"{side}_message"] is None for a in analyses)} for side in ("previous","next")},
        "proximity_minutes_inclusive":proximity,"sender_continuity":continuity,
        "reply_target_type_counts":dict(Counter(a["reply_target_type"] for a in analyses)),
        "single_neighbor_direct_relationship_count":sum(any(rel and (rel["single_replies_to_neighbor"] or rel["neighbor_replies_to_single"]) for rel in (a["previous_reply_relationship"],a["next_reply_relationship"])) for a in analyses),
        "long_distance_replies_all_normal_messages":{"count":len(long),
            "median_gap_seconds":statistics.median(r["gap_seconds"] for r in long) if long else None,
            "max_gap_seconds":max((r["gap_seconds"] for r in long),default=None),
            "resolved_normal_reply_edges_examined":normal_edges,
            "system_target_edges_without_exact_time_excluded":system_refs,
            "missing_targets_excluded":len(missing),
            "single_source_count":sum(r["message_id"] in {a["message_id"] for a in analyses} for r in long)},
        "redaction":"CSV text and review text mask URL/email/IP/phone candidates/labelled credentials. Sender display names retained. Heuristic, not comprehensive anonymization. Original text remains in unchanged input.",
        "input_hashes":hashes}
    # Coverage-first stratified random sample. Categories overlap.
    def both(a,seconds):
        return all(a[f"gap_{side}_seconds"] is not None and a[f"gap_{side}_seconds"]<seconds for side in ("previous","next"))
    pools={
        "both_sides_under_5m":[a for a in analyses if both(a,300)],
        "both_sides_5_to_10m":[a for a in analyses if both(a,600) and not both(a,300)],
        "both_sides_10_to_30m":[a for a in analyses if both(a,1800) and not both(a,600)],
        "same_sender":[a for a in analyses if a["same_sender_previous"] is True or a["same_sender_next"] is True],
        "different_sender":[a for a in analyses if a["same_sender_previous"] is False or a["same_sender_next"] is False],
        "normal_direct_reply_related":[a for a in analyses if a["reply_target_type"]=="message" or a["incoming_normal_reply_count"]>0],
        "system_reply_reference_only":[a for a in analyses if a["reply_target_type"]=="service"],
        "no_reply":[a for a in analyses if a["reply_to"] is None],
        "large_gap_over_24h":[a for a in analyses if any(a[f"gap_{side}_seconds"] is not None and a[f"gap_{side}_seconds"]>86400 for side in ("previous","next"))],
        "unknown_topic":[a for a in analyses if a["topic_root"] is None]}
    rng=random.Random(args.seed)
    selected={}
    for topic in sorted({a["topic_root"] for a in analyses if a["topic_root"] is not None}):
        pool=[a for a in analyses if a["topic_root"]==topic]
        a=rng.choice(pool);selected[a["message_id"]]=a
    coverage={}
    for label,pool in pools.items():
        available=[a for a in pool if a["message_id"] not in selected]
        chosen=rng.sample(available,min(12,len(available),150-len(selected)))
        selected.update((a["message_id"],a) for a in chosen)
        coverage[label]={"eligible_count":len(pool),"newly_selected":len(chosen)}
    remaining=[a for a in analyses if a["message_id"] not in selected]
    selected.update((a["message_id"],a) for a in rng.sample(remaining,min(150-len(selected),len(remaining))))
    sample=sorted(selected.values(),key=lambda a:order[a["message_id"]])
    for label,pool in pools.items():
        coverage[label]["final_selected_count"]=sum(a["message_id"] in selected for a in pool)
    stats["review_sampling"]={"seed":args.seed,"count":len(sample),"coverage":coverage,
        "selected_ids":[a["message_id"] for a in sample],
        "known_topics_in_population":len(topics),"known_topics_in_single_population":len({a["topic_root"] for a in analyses if a["topic_root"] is not None}),
        "known_topics_in_sample":len({a["topic_root"] for a in sample if a["topic_root"] is not None}),
        "method":"One random case per singleton topic, then up to 12 new cases per stratum, then random fill to 150. Not representative of class prevalence."}
    def show(r):
        result=[f"[{r['datetime']}] {r['sender']} (message{r['message_id']})",
                redact(r["text"]) if r["text"].strip() else "[no text]"]
        if r["reply_to"]:result.append(f"  reply_to: message{r['reply_to']}")
        if r.get("attachment_type"):result.append(f"  [attachment: {r['attachment_type']}; path omitted]")
        return result+[""]
    def show_context(neighbor):
        if neighbor is None:return ["[No confirmed same-topic neighbor; see topic/gap metadata]",""]
        c=owner[neighbor["message_id"]]
        group=sorted(c["messages"],key=lambda r:(times[r["message_id"]],order[r["message_id"]]))
        idx=next(i for i,r in enumerate(group) if r["message_id"]==neighbor["message_id"])
        start=max(0,min(idx-2,len(group)-5))
        excerpt=group[start:start+5]
        result=[f"Candidate: {c['conversation_id']} | total messages={len(group)} | showing {len(excerpt)}",
                f"Chronological neighbor: message{neighbor['message_id']} (context excerpt centered on this member)",
                "Candidate members can span across target time because direct reply chains may interleave."]
        for r in excerpt:result+=show(r)
        return result
    review=["150 singleton cases; 60m is analysis baseline only.",
            "Known topic chronological neighbors, stable source order for equal timestamps. Unknown topic not pooled.",
            "Previous/next conversation excerpts contain up to 5 existing candidate members, including the actual neighbor.",
            "If both neighbors belong to one candidate it appears on both sides; no new segmentation is created.",
            "Text redaction is heuristic. Sender names retained. No role or semantic labels.",""]
    for i,a in enumerate(sample,1):
        review+=["="*64,f"Single Case #{i:03d}",f"Target: message{a['message_id']}",f"Topic: {a['topic_root']}",
                 f"Candidate: {a['conversation_id']}","="*64,"","Previous conversation:",""]
        review+=show_context(a["previous_message"])
        review+=["---","TARGET SINGLE MESSAGE:",""]+show(by[a["message_id"]])
        review+=["---","Next conversation:",""]+show_context(a["next_message"])
        review += [f"Gap Previous: {a['gap_previous_seconds']} seconds",
                   f"Gap Next: {a['gap_next_seconds']} seconds",
                   f"Same Sender Previous: {a['same_sender_previous']}",
                   f"Same Sender Next: {a['same_sender_next']}",
                   f"Previous Sender == Next Sender: {a['previous_sender_equals_next']}",
                   "Reply Relationship: "+json.dumps({"previous":a["previous_reply_relationship"],"next":a["next_reply_relationship"],"target_type":a["reply_target_type"]}),
                   f"Current Structural Classification: {a['structural_classification']}",""]
    output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    write_csv(output/"single_message_analysis.csv",analyses)
    fields=["message_id","datetime","sender","topic_root","reply_to","target_datetime","target_sender","gap_seconds","gap_hours","gap_days","conversation_id","target_conversation_id","same_candidate"]
    write_csv(output/"long_distance_replies.csv",long,fields)
    (output/"single_message_statistics.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (output/"single_message_review.txt").write_text("\n".join(review),encoding="utf-8")
    report=["# Step 4B Single-message Conversation Analysis","",
        f"分析基準：segmentation_60m.jsonl；共有 {len(analyses)} 個單則 chat candidates。50 個 system_event 單點另行排除，60m 並非 final threshold。",
        "本次只做結構候選分類，未進行角色、Intent、Slot 或 LLM labeling，未更動既有 segmentation。","",
        "## 方法","",
        "- 在同一個明確 topic 的全部一般訊息中找時間前後鄰居；同秒時以原 JSONL 順序決定。不使用全域鄰居。",
        "- 前後 conversation 是鄰居原本所屬 candidate，review 各最多 5 則，且包含鄰居本身。不是重切的新 conversation。",
        "- 未知 topic 不共用群組；缺少鄰居可能是未知 topic 或匯出邊界，兩者分開統計。",
        "- 分類鄰近門檻 <=60 分鐘；超過 24 小時的已解析一般訊息 direct reply 優先標為 LONG_DISTANCE_REPLY。",
        "- UNKNOWN：topic 不明或 outgoing reply 目標缺失。其他已知 topic 案例依雙側／單側鄰近分成 POSSIBLE_BOTH／PREVIOUS／NEXT，皆不符合則 STRUCTURALLY_ISOLATED。",
        "- 相同 system/topic reply target 不等於兩訊息存在 direct reply；CSV 同時保留這兩種不同關係。",
        "- 結構標籤不是 ground truth；隔離只是資料中未觀察到門檻內關係，不代表語意完全獨立。",
        "- CSV 的 text 與 review 已遮蔽常見識別模式，sender 顯示名稱保留；原文仍在未修改的來源中。","",
        "## 結構分類","","| 分類 | 數量 | 占全部單則 % |","|---|---:|---:|"]
    report += [f"| {k} | {v} | {100*v/len(analyses):.2f} |" for k,v in classifications.items()]
    report += ["","## 時間鄰近","","within 採 <= 邊界；both 是兩側都符合，either 是至少一側符合。","",
               "| 分鐘 | 前側 | 後側 | 雙側都有 | 任一側 |","|---|---:|---:|---:|---:|"]
    report += [f"| {m} | {v['previous_within']} | {v['next_within']} | {v['both_sides_within']} | {v['either_side_within']} |" for m,v in proximity.items()]
    report += ["","| 間隔 | Previous | Next |","|---|---:|---:|"]
    report += [f"| {b} | {bins['previous'][b]} | {bins['next'][b]} |" for b in bin_labels]
    report += ["","時間分箱前五組左閉右開，1–2 hr 包含恰好兩小時；no_neighbor 包含 topic 未知，明細見 JSON。","",
               "## Sender continuity","","| 比較 | 相同 | 可比較 | 可比較者中比例 % | 全部單則比例 % |","|---|---:|---:|---:|---:|"]
    report += [f"| {k} | {v['equal_count']} | {v['eligible_count']} | {v['percent_among_eligible']:.2f} | {v['percent_of_all_singles']:.2f} |" for k,v in continuity.items()]
    report += ["","## Long-distance direct reply","",
        f"掃描全部 {normal_edges} 條可解析的一般訊息 direct reply，>24h 共 {len(long)} 條；並非只掃 singleton。",
        f"Median gap：{stats['long_distance_replies_all_normal_messages']['median_gap_seconds']} 秒；Max gap：{stats['long_distance_replies_all_normal_messages']['max_gap_seconds']} 秒。",
        f"其中 singleton 為來源的案例：{stats['long_distance_replies_all_normal_messages']['single_source_count']}。強回覆已同組保留，故單則中缺少此類案例是預期結果。",
        f"{system_refs} 條 system/topic 回覆沒有精確目標時間、{len(missing)} 條目標缺失，均無法計算，未猜測時間。",
        "完整訊息 ID、時間差與原 candidate 對照見 long_distance_replies.csv，沒有拆開任何 long-distance reply。","",
        "## Over-segmentation 的結構證據","",
        f"{proximity['5']['both_sides_within']} 則 singleton 的前後同 topic 訊息都在 5 分鐘內；{proximity['60']['either_side_within']} 則在 60 分鐘內至少一側有鄰居。",
        "這是目前單點切分值得懷疑、需要人工檢查的結構證據；但同 topic 且時間接近仍可能是不同工作，不能由此證明應合併的真實數量。",
        "既有演算法限制弱合併需同 sender 且同 topic 相鄰，保留獨立 reply chain。不同 sender 的近距離延續討論可能因此被拆散。",
        "不能把 POSSIBLE_BOTH 直接變成合併兩側的指令，也不能將所有 STRUCTURALLY_ISOLATED 當作真實獨立對話。","",
        "## 人工 Review","",
        f"固定 seed={args.seed}，抽取 {len(sample)} 個不同 singleton；先覆蓋各 topic，再分層補入近距離、同／不同 sender、reply/no-reply、長間隔與未知 topic，最後隨機補足。",
        "normal direct-reply-related 若為 0，明確記錄不可抽樣；不以共同 topic 引用冒充 direct reply。",
        "抽樣偏重邊界案例，不用於推估分類比例。全部 942 筆的統計才是母體分布。",
        "```json",json.dumps(stats["review_sampling"],ensure_ascii=False,indent=2),"```",""]
    (output/"single_message_report.md").write_text("\n".join(report),encoding="utf-8")
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()),"Protected input modified"
    print(json.dumps({k:v for k,v in stats.items() if k not in ("input_hashes","review_sampling","redaction","classification_rules")},ensure_ascii=True,indent=2))
    print(json.dumps({"sample_coverage":coverage,"inputs_unchanged":True},ensure_ascii=True))

if __name__=="__main__":
    main()
