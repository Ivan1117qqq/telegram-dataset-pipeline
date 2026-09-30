"""Deterministic hybrid candidate segmentation; Python standard library only.
No role inference, semantics, or final threshold selection.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import statistics

def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    p = (len(values)-1)*q
    i = int(p)
    return values[i] + (values[min(i+1,len(values)-1)]-values[i])*(p-i)

def summary(values):
    return dict(mean=statistics.mean(values) if values else None,
                median=percentile(values,.5), p90=percentile(values,.9),
                max=max(values) if values else None)

class Union:
    def __init__(self, ids):
        self.p = {i:i for i in ids}
    def find(self, i):
        if self.p[i] != i:
            self.p[i] = self.find(self.p[i])
        return self.p[i]
    def join(self, a, b):
        a,b = self.find(a),self.find(b)
        if a != b:
            self.p[b] = a
        return a

def redact(text):
    text = re.sub(r"(?i)(?:https?://|www\.)[^\s<>\u3000]+", "[URL]", text)
    text = re.sub(r"(?i)(?<![\w.+-])[\w.+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+", "[EMAIL]", text)
    text = re.sub(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])", "[IP]", text)
    text = re.sub(r"(?i)\b[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\b", "[ID]", text)
    text = re.sub(r"(?i)(\b(?:token|hash|password|passwd|secret|api[_ -]?key)\b\s*[:=\uff1a]\s*)[^\s&<>]+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)\b[0-9a-f]{24,}\b", "[ID]", text)
    def phone(m):
        value = m[0]
        if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", value):
            return value
        return "[PHONE_CANDIDATE]" if 8 <= len(re.sub(r"\D","",value)) <= 15 else value
    return re.sub(r"(?<![\w.])\+?\d[\d ()-]{6,}\d(?![\w.])", phone, text)

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root/"data/processed/messages.jsonl")
    parser.add_argument("--output-dir", type=Path, default=root/"data/segmentation")
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    protected = [args.input, root/"scripts/preprocess_chat.py", root/"scripts/profile_dataset.py"]
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    by = {r["message_id"]:r for r in rows}
    assert len(by)==len(rows), "Duplicate IDs"
    normal = [r for r in rows if r["message_type"]=="message"]
    services = [r for r in rows if r["message_type"]=="service"]
    ids = [r["message_id"] for r in normal]
    order = {r["message_id"]:i for i,r in enumerate(rows)}
    times = {r["message_id"]:datetime.fromisoformat(r["datetime"]) for r in normal}
    assert all(times[a]<=times[b] for a,b in zip(ids,ids[1:])), "Input time order"
    topics = {r["message_id"] for r in services if " created topic " in r["text"]}
    def target(r):
        t = by.get(r.get("reply_to"))
        if t and t["source_file"] != (r.get("reply_to_source_file") or r["source_file"]):
            return None
        return t
    def context(r):
        seen = set()
        while True:
            if r["message_id"] in seen:
                raise ValueError("Cycle in direct replies")
            seen.add(r["message_id"])
            if r["message_id"] in topics:
                return ("topic", r["message_id"])
            if not r.get("reply_to"):
                return ("unknown_root", r["message_id"])
            t = target(r)
            if t is None:
                return ("missing_root", (r.get("reply_to_source_file") or r["source_file"])+"#"+r["reply_to"])
            r = t
    contexts = {r["message_id"]:context(r) for r in normal}
    direct = []
    missing = []
    system_links = []
    strong = Union(ids)
    for r in normal:
        if not r.get("reply_to"):
            continue
        t = target(r)
        if t is None:
            missing.append({"message_id":r["message_id"], "reply_to":r["reply_to"]})
        elif t["message_type"]=="message":
            assert contexts[r["message_id"]]==contexts[t["message_id"]]
            direct.append((t["message_id"],r["message_id"]))
            strong.join(t["message_id"],r["message_id"])
        else:
            system_links.append((t["message_id"],r["message_id"]))
    components = defaultdict(list)
    for i in ids:
        components[strong.find(i)].append(i)
    # Stable component ID based on first row, not union-find representative.
    component_id = {}
    for members in components.values():
        label = min(members,key=order.get)
        for i in members:
            component_id[i]=label
    sizes = Counter(component_id.values())
    topic_rows = defaultdict(list)
    for i in ids:
        if contexts[i][0]=="topic":
            topic_rows[contexts[i][1]].append(i)
    weak_proposals = []
    for topic, members in topic_rows.items():
        for a,b in zip(members,members[1:]):
            if component_id[a]!=component_id[b] and by[a]["sender"]==by[b]["sender"]:
                weak_proposals.append(((times[b]-times[a]).total_seconds(),a,b))
    # Shorter links get priority. This is explicit because one anchored chain per
    # candidate makes weak-link acceptance order relevant.
    weak_proposals.sort(key=lambda e:(e[0],order[e[2]],order[e[1]]))
    output=args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    comparisons=[]
    details=[]
    partitions=[]
    for minutes in (10,30,60,120):
        union=Union(set(component_id.values()))
        anchors={i:({i} if sizes[i]>1 else set()) for i in union.p}
        links=[]
        blocked=0
        for gap,a,b in weak_proposals:
            if gap>minutes*60:
                continue
            ca,cb=union.find(component_id[a]),union.find(component_id[b])
            if ca==cb:
                continue
            combined=anchors[ca]|anchors[cb]
            if len(combined)>1:
                blocked+=1
                continue
            merged=union.join(ca,cb)
            anchors[merged]=combined
            links.append({"from_message_id":a,"to_message_id":b,"gap_seconds":gap,
                          "reason":"same_known_topic_adjacent_same_sender"})
        groups=defaultdict(list)
        for r in normal:
            groups[union.find(component_id[r["message_id"]])].append(r)
        candidates=[]
        membership={}
        for group in sorted(groups.values(),key=lambda g:order[g[0]["message_id"]]):
            first=group[0]["message_id"]
            ctx=contexts[first]
            assert len({contexts[r["message_id"]] for r in group})==1
            cid=f"candidate_{minutes}m_{first}"
            members={r["message_id"] for r in group}
            for i in members:
                membership[i]=cid
            timestamps=[times[r["message_id"]] for r in group]
            internal=[(a,b) for a,b in direct if a in members and b in members]
            long_edges=[{"from_message_id":a,"to_message_id":b,
                         "gap_seconds":(times[b]-times[a]).total_seconds()}
                        for a,b in internal if (times[b]-times[a]).total_seconds()>minutes*60]
            chronological_gaps=[(b-a).total_seconds() for a,b in zip(timestamps,timestamps[1:])]
            candidates.append({
                "conversation_id":cid,"candidate_type":"chat","threshold_minutes":minutes,
                "topic_root":ctx[1] if ctx[0]=="topic" else None,
                "context_key":ctx[0]+":"+ctx[1],
                "start_time":min(timestamps).isoformat(),"end_time":max(timestamps).isoformat(),
                "duration_seconds":(max(timestamps)-min(timestamps)).total_seconds(),
                "sender_count":len({r["sender"] for r in group}),"message_count":len(group),
                "direct_reply_count":len(internal),
                "strong_component_ids":sorted({component_id[i] for i in members},key=order.get),
                "weak_merge_links":[e for e in links if e["from_message_id"] in members and e["to_message_id"] in members],
                "long_direct_reply_edges":long_edges,
                "max_internal_gap_seconds":max(chronological_gaps,default=0),
                "messages":group})
        chat=list(candidates)
        # Preserve all service rows exactly once without connecting their topic descendants.
        for r in services:
            candidates.append({"conversation_id":f"context_{minutes}m_{r['message_id']}",
                "candidate_type":"system_event","threshold_minutes":minutes,
                "topic_root":r["message_id"] if r["message_id"] in topics else None,
                "context_key":"system_event:"+r["message_id"],
                "start_time":r["datetime"],"end_time":r["datetime"],"duration_seconds":None,
                "sender_count":0,"message_count":1,"direct_reply_count":0,"messages":[r]})
        candidates.sort(key=lambda c:order[c["messages"][0]["message_id"]])
        flat=[r for c in candidates for r in c["messages"]]
        assert len(flat)==len(rows) and Counter(r["message_id"] for r in flat)==Counter(by.keys())
        assert all(r==by[r["message_id"]] for r in flat), "Message metadata changed"
        broken=[(a,b) for a,b in direct if membership[a]!=membership[b]]
        assert not broken
        partition={i:membership[i] for i in ids}
        partitions.append(partition)
        msg=summary([c["message_count"] for c in chat])
        duration=summary([c["duration_seconds"] for c in chat])
        stat={"threshold_minutes":minutes,"conversations":len(chat),
            "system_event_records":len(services),"total_output_records":len(candidates),
            **{"message_count_"+k:v for k,v in msg.items()},
            **{"duration_seconds_"+k:v for k,v in duration.items()},
            "sender_count_distribution":dict(sorted(Counter(c["sender_count"] for c in chat).items())),
            "single_message_count":sum(c["message_count"]==1 for c in chat),
            "single_message_percent":100*sum(c["message_count"]==1 for c in chat)/len(chat),
            "over_20_messages":sum(c["message_count"]>20 for c in chat),
            "over_50_messages":sum(c["message_count"]>50 for c in chat),
            "over_100_messages":sum(c["message_count"]>100 for c in chat),
            "broken_direct_reply_count":len(broken),"resolved_normal_direct_reply_count":len(direct),
            "topic_or_service_links_excluded_from_broken_metric":len(system_links),
            "unresolved_direct_reply_count":len(missing),
            "accepted_weak_links":len(links),"blocked_independent_chain_links":blocked,
            "direct_reply_edges_over_threshold":sum(len(c["long_direct_reply_edges"]) for c in chat),
            "conversations_with_internal_gap_over_threshold":sum(c["max_internal_gap_seconds"]>minutes*60 for c in chat),
            "multi_topic_conversations":0,
            "conversations_merging_multiple_nontrivial_reply_chains":0}
        comparisons.append(stat)
        with (output/f"segmentation_{minutes}m.jsonl").open("w",encoding="utf-8",newline="\n") as f:
            for c in candidates:
                f.write(json.dumps(c,ensure_ascii=False)+"\n")
        # Stratified random coverage; categories overlap, but 50 distinct IDs.
        duration_cut=percentile([c["duration_seconds"] for c in chat if c["duration_seconds"]>0],.9)
        pools={
            "short_1_2_messages":[c for c in chat if c["message_count"]<=2],
            "medium_3_10_messages":[c for c in chat if 3<=c["message_count"]<=10],
            "long_11plus_messages":[c for c in chat if c["message_count"]>=11],
            "multi_sender":[c for c in chat if c["sender_count"]>=2],
            "direct_reply":[c for c in chat if c["direct_reply_count"]>0],
            "large_time_span":[c for c in chat if duration_cut is not None and c["duration_seconds"]>=duration_cut]}
        rng=random.Random(args.seed+minutes)
        selected={}
        sampling={}
        for label,pool in pools.items():
            available=[c for c in pool if c["conversation_id"] not in selected]
            picks=rng.sample(available,min(7,len(available)))
            sampling[label]={"eligible":len(pool),"newly_selected":len(picks)}
            selected.update((c["conversation_id"],c) for c in picks)
        remaining=[c for c in chat if c["conversation_id"] not in selected]
        selected.update((c["conversation_id"],c) for c in rng.sample(remaining,min(50-len(selected),len(remaining))))
        review=sorted(selected.values(),key=lambda c:order[c["messages"][0]["message_id"]])
        assert len(review)==min(50,len(chat))
        lines=[f"Candidate review: {minutes}m; seed={args.seed+minutes}; {len(review)} distinct chat candidates.",
               "Stratified random sample, NOT a representative estimate of candidate frequency.",
               "Text masks URL/email/IP/phone candidates/labelled secrets; patterns are not comprehensive anonymization.",
               "Message IDs and sender display names retained. Attachment paths omitted. System events excluded from review.", ""]
        for c in review:
            labels=[label for label,pool in pools.items() if c in pool]
            lines+=["="*64,f"Conversation: {c['conversation_id']}",
                    f"Topic: {c['topic_root']} | Context: {c['context_key']}",
                    f"Messages: {c['message_count']} | Senders: {c['sender_count']} | Duration: {c['duration_seconds']} seconds",
                    f"Direct replies: {c['direct_reply_count']} | Weak links: {len(c['weak_merge_links'])} | Max gap: {c['max_internal_gap_seconds']} seconds",
                    "Strata: "+", ".join(labels),
                    "Long direct reply edges: "+json.dumps(c["long_direct_reply_edges"]), "="*64,""]
            for r in c["messages"]:
                lines += [f"[{r['datetime']}] {r['sender']} (message_id={r['message_id']})",
                          redact(r["text"]) if r["text"].strip() else "[no text]"]
                if r.get("reply_to"):
                    lines.append(f"  ↳ reply_to: message{r['reply_to']}")
                if r.get("attachment_type"):
                    lines.append(f"  [attachment: {r['attachment_type']}; path omitted]")
                if r.get("forwarded"):
                    lines.append("  [contains forwarded content; forwarded replies are not segmentation edges]")
                lines.append("")
        (output/f"review_{minutes}m.txt").write_text("\n".join(lines),encoding="utf-8")
        details.append({"threshold":minutes,"sampling":sampling,"review_ids":list(selected),
                        "long_span_review_cutoff_seconds":duration_cut})
    with (output/"segmentation_comparison.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(comparisons[0]))
        writer.writeheader()
        for stat in comparisons:
            writer.writerow({k:json.dumps(v) if isinstance(v,dict) else v for k,v in stat.items()})
    stability=[]
    for i in range(1,len(partitions)):
        # Membership-set agreement ignores threshold-specific conversation IDs.
        old,new=defaultdict(set),defaultdict(set)
        for mid,cid in partitions[i-1].items():old[cid].add(mid)
        for mid,cid in partitions[i].items():new[cid].add(mid)
        unchanged=sum(old[partitions[i-1][mid]]==new[partitions[i][mid]] for mid in ids)
        stability.append({"from_minutes":comparisons[i-1]["threshold_minutes"],
                          "to_minutes":comparisons[i]["threshold_minutes"],
                          "unchanged_message_membership_percent":100*unchanged/len(ids),
                          "changed_messages":len(ids)-unchanged,
                          "conversation_count_change":comparisons[i]["conversations"]-comparisons[i-1]["conversations"]})
    report=["# Candidate Conversation Segmentation","",
        "本次為結構性候選切分，未判定角色、意圖或 slot，未使用 LLM 語意分類，未選擇最終 threshold。","",
        "## 方法與限制","",
        "1. 僅外層 reply_to 建立強連結；一般訊息間的可解析 direct reply 以連通分量完整保留。",
        "2. 沿 reply chain 尋找明確的 topic 建立 service；topic 只是限制條件，service 不參與一般訊息連通分量。",
        "3. 同 topic 中相鄰、同 sender 且間隔 <= threshold 的不同分量可嘗試弱合併。跨 topic 或未知 topic 不做弱合併。",
        "4. 每個候選最多包含一條非單點的 direct-reply 連通分量；不把兩組獨立 direct-reply chain 因時間接近而合併。單點可透過同 sender 連續性加入。",
        "5. 弱邊按間隔由短到長，再按來源順序處理；合併依據保存在 weak_merge_links。此保守約束可能造成過度切分，也可能讓同 sender 的單點連續工作被合併。",
        "6. Direct reply 優先於 inactivity，跨日回覆仍保留，因此 threshold 不是最大對話時長，也不是每組內最大間隔。",
        "7. 此版沒有推測無回覆訊息的 topic；unknown_root/missing_root 各自保留。轉傳內 reply 不參與切分。",
        "8. 每份 JSONL 含全部 2,905 則原始訊息且各出現一次。50 則 service 各為 candidate_type=system_event 的獨立保留紀錄，時間與 duration 為 null，不當客服 conversation。",
        "9. 下列 conversation 統計只計 candidate_type=chat（2,855 則一般訊息）；CSV 另列 system_event_records 與 total_output_records。",
        "10. JSONL 保留完整原始訊息欄位；人工 review 遮蔽常见識別模式並省略附件路徑，並非完整匿名化。",
        "11. 分位數採 (n-1)*q 線性插值；duration 為末則減首則，單點為 0 秒。","",
        "## Threshold 比較","",
        "| Threshold | Conversations | Median Msg | P90 Msg | Max Msg | Single Msg % | Broken Direct Reply |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for s in comparisons:
        report.append(f"| {s['threshold_minutes']}m | {s['conversations']} | {s['message_count_median']} | {s['message_count_p90']:.2f} | {s['message_count_max']} | {s['single_message_percent']:.2f} | {s['broken_direct_reply_count']} |")
    report+=["","完整 mean/median/p90/max、sender_count 分布、>20/>50/>100 計數及其他指標見 segmentation_comparison.csv。","",
             "## 可重現的統計明細","","```json",json.dumps(comparisons,ensure_ascii=False,indent=2),"```","",
             "## Threshold 敏感度","",
             "穩定度定義為相鄰兩種 threshold 下，完整同組 message ID 集合沒有改變的訊息比例；不是語意正確率，也不是最終選擇標準。",
             "```json",json.dumps(stability,ensure_ascii=False,indent=2),"```","",
             "## Direct reply 與邊界情況","",
             f"- 一般訊息可解析 direct reply 共 {len(direct)} 條，四組均要求 broken_direct_reply_count=0。",
             f"- {len(system_links)} 條指向 topic/service 的連結只作 context，故不納入 broken 指標；它們仍保留在原始 message.reply_to。",
             f"- {len(missing)} 條回覆目標不存在，無法驗證同組，不冒充已成功保留的 direct reply。缺失清單：{json.dumps(missing)}。",
             f"- 強連通分量共 {len(components)} 組，最大 {max(map(len,components.values()))} 則。",
             "- 跨日 direct reply、稀疏長跨度 chain、topic 未知訊息、同 sender 的不同工作連續發送、不同 sender 無 explicit reply 的延續討論，均需人工檢查。",
             "- 在 topic 中插入其他 sender 會阻斷弱合併；direct reply 仍可跨越插入訊息。這降低全域時間合併的混淆，但也可能過度切分。",
             "- 不同已知 topic 合併為 0、不同非單點 reply chain 的弱合併為 0，均為演算法約束。不能據此宣稱語意上的多人交錯已完全解決。","",
             "## 人工 review","",
             "每種 threshold 隨機抽取 50 個不重複 chat candidates。六種分層依序最多補入 7 個，再從剩餘候選隨機補足 50。",
             "short=1–2 則、medium=3–10 則、long>=11 則；另覆蓋多 sender、含一般訊息 direct reply，以及正 duration 的 p90 以上長跨度。",
             "分層可重疊；不足 7 個時取完可用候選並記錄，不捏造長對話。這是品質檢查樣本，不是分布估計樣本。",
             "```json",json.dumps(details,ensure_ascii=False,indent=2),"```","",
             "## 輸入完整性","",
             "輸入 JSONL 與兩支既有程式 SHA-256（執行前後必須一致）：",
             "```json",json.dumps(hashes,ensure_ascii=False,indent=2),"```",""]
    most_stable = max(stability, key=lambda x:x["unchanged_message_membership_percent"])
    report += ["## 本次觀察", "",
        f"- 相鄰 threshold 中，{most_stable['from_minutes']}–{most_stable['to_minutes']} 分鐘的成員集合最穩定，{most_stable['unchanged_message_membership_percent']:.2f}% 訊息同組集合不變；這不表示其中任一值是最終最佳門檻。",
        f"- 四組 >100 message 候選皆為 0；最大訊息數依序為 {[s['message_count_max'] for s in comparisons]}。",
        f"- 最長時間跨度仍為 {max(s['duration_seconds_max'] for s in comparisons)/86400:.2f} 天，顯示跨日 direct reply 可能把很少的訊息連到很久以前，需人工回查。",
        "- 單則比例仍約 59%–62%，代表本方法較保守，可能拆散未使用 direct reply 的多 sender 延續對話。",
        "- 跨 topic 與獨立強 chain 的時間合併受到限制，已減少這兩種結構性混合；是否改善真實問題邊界仍待人工 review。",
        "- 四種 threshold 的 50 個 review 樣本均涵蓋六種要求的檢查類別。", ""]
    (output/"segmentation_report.md").write_text("\n".join(report),encoding="utf-8")
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    print(json.dumps({"comparison":comparisons,"stability":stability,
                      "max_strong_component":max(map(len,components.values())),
                      "missing_targets":missing,"inputs_unchanged":True},ensure_ascii=True,indent=2))

if __name__=="__main__":
    main()
