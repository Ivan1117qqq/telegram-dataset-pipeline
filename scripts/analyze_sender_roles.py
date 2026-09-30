"""Behavior-only sender profiles and blank human role mapping. Standard library."""
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

def csv_write(path, rows, fields=None):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})

def desc(values):
    return {"count":len(values),"mean":statistics.mean(values) if values else None,
            "median":statistics.median(values) if values else None,
            "min":min(values) if values else None,"max":max(values) if values else None,
            "coefficient_of_variation":statistics.pstdev(values)/statistics.mean(values) if values and statistics.mean(values)>0 else None}

def safe(text):
    # More conservative than counting patterns: redact the entire URL, including query.
    text=re.sub(r"(?i)(?:https?://|www\.)[^\s<>\u3000]+","[URL]",text)
    text=re.sub(r"(?i)[\w.+-]+@[\w.-]+\.[a-z]{2,}","[EMAIL]",text)
    text=re.sub(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])","[IP]",text)
    text=re.sub(r"(?i)\b[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\b","[ID]",text)
    text=re.sub(r"(?i)(\b(?:token|hash|password|passwd|secret|api[_ -]?key|account|username|user[_ -]?id|login)\b\s*[:=\uff1a]\s*)[^\s&<>]+",r"\1[REDACTED]",text)
    text=re.sub(r"((?:帳號|账号|用戶名|用户名|會員編號|会员编号|密碼|密码)\s*[:=\uff1a]?\s*)[A-Za-z0-9_.@+-]+",r"\1[ACCOUNT]",text)
    text=re.sub(r"`[^`\n]+`","[QUOTED_VALUE]",text)
    text=re.sub(r"(?i)\b[0-9a-f]{16,}\b","[ID]",text)
    text=re.sub(r"(?<!\w)@[A-Za-z0-9_]+","[HANDLE]",text)
    def phone(m):
        v=m[0]
        return "[PHONE_OR_ID]" if not re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}",v) and 7<=len(re.sub(r"\D","",v))<=15 else v
    text=re.sub(r"(?<![\w.])\+?\d[\d ()-]{5,}\d(?![\w.])",phone,text)
    # Mixed letter/digit identifiers may be accounts; false positives are acceptable in review.
    text=re.sub(r"\b(?=[A-Za-z0-9_.-]{4,}\b)(?=[A-Za-z0-9_.-]*[A-Za-z])(?=[A-Za-z0-9_.-]*\d)[A-Za-z0-9_.-]+","[IDENTIFIER]",text)
    return text

def graph_analysis(matrix, names):
    n=len(names)
    weights=[[matrix[i][j]+matrix[j][i] if i!=j else 0 for j in range(n)] for i in range(n)]
    total=sum(map(sum,weights))/2
    degrees=[sum(row) for row in weights]
    groups=[{i} for i in range(n) if degrees[i]]
    isolated=[names[i] for i in range(n) if not degrees[i]]
    # Greedy positive modularity merging, weighted undirected graph, no role labels.
    while True:
        best=None
        for i in range(len(groups)):
            for j in range(i+1,len(groups)):
                cross=sum(weights[a][b] for a in groups[i] for b in groups[j])
                gain=cross/total-sum(degrees[a] for a in groups[i])*sum(degrees[b] for b in groups[j])/(2*total*total) if total else 0
                if gain>1e-12 and (best is None or gain>best[0]+1e-12):
                    best=(gain,i,j)
        if best is None:break
        _,i,j=best
        groups[i]|=groups[j];del groups[j]
    groups.sort(key=lambda g:(-sum(degrees[i] for i in g),min(g)))
    modularity=sum(sum(weights[a][b] for a in g for b in g)/(2*total)-
                   (sum(degrees[a] for a in g)/(2*total))**2 for g in groups) if total else 0
    internal=sum(weights[a][b] for g in groups for a in g for b in g)/2
    triangles=sum(weights[i][j]>0 and weights[i][k]>0 and weights[j][k]>0
                  for i in range(n) for j in range(i+1,n) for k in range(j+1,n))
    # Exhaustive weighted maximum cut via Gray-code traversal, vertex 0 fixed.
    sides=[0]*n;current=0;maximum=-1;best_sides=None;previous=0
    for number in range(1<<(n-1) if 0 < n <= 20 else 0):
        gray=number^(number>>1)
        if number:
            flip=(gray^previous).bit_length() # bit 0 maps to vertex 1
            old=sides[flip]
            current+=sum(w if sides[j]==old else -w for j,w in enumerate(weights[flip]) if j!=flip)
            sides[flip]=1-old
        if current>maximum:
            maximum=current;best_sides=sides[:]
        previous=gray
    return {"method":"Greedy weighted modularity communities, separate from exhaustive two-way maximum-cut diagnostic. Both ignore self edges and sender names.",
            "cross_sender_weight":total,"undirected_edges":sum(weights[i][j]>0 for i in range(n) for j in range(i+1,n)),
            "communities":[[names[i] for i in sorted(g)] for g in groups],"isolated_senders":isolated,
            "modularity":modularity,"within_community_weight_fraction":internal/total if total else None,
            "triangle_count":triangles,
            "maximum_two_way_cross_weight_fraction":maximum/total if total and best_sides is not None else None,
            "maximum_cut_status":"exact" if best_sides is not None else "not_computed_node_limit",
            "maximum_cut_groups":[[names[i] for i in range(n) if best_sides is not None and best_sides[i]==side] for side in (0,1)],
            "interpretation":"Communities measure dense internal interaction, maximum cut measures across-group interaction; neither identifies USER/AGENT. Optimized cut is descriptive, not statistical significance."}

def main():
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",type=Path,default=root/"data/processed/messages.jsonl")
    ap.add_argument("--output-dir",type=Path,default=root/"data/roles")
    ap.add_argument("--evidence-sender", default="Watchdog", help="Named behavior case study only; never a role label")
    ap.add_argument("--seed",type=int,default=20260914)
    args=ap.parse_args()
    protected=[args.input]+list((root/"data/segmentation").glob("segmentation_*"))+[
        root/"scripts/segment_conversations.py",root/"scripts/preprocess_chat.py",root/"scripts/profile_dataset.py"]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    rows=[json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    normal=[r for r in rows if r["message_type"]=="message"]
    by={r["message_id"]:r for r in rows};order={r["message_id"]:i for i,r in enumerate(rows)}
    senders=sorted({r["sender"] for r in normal},key=lambda s:(-sum(r["sender"]==s for r in normal),s))
    if not senders: raise ValueError("No normal messages")
    sender_count=len(senders)
    index={s:i for i,s in enumerate(senders)}
    times={r["message_id"]:datetime.fromisoformat(r["datetime"]) for r in normal}
    def target(r):
        t=by.get(r.get("reply_to"))
        return t if t and t["source_file"]==(r.get("reply_to_source_file") or r["source_file"]) else None
    def topic(r):
        seen=set()
        while True:
            if r["message_id"] in seen:raise ValueError("Reply cycle")
            seen.add(r["message_id"])
            if r["message_type"]=="service":
                return r["message_id"] if " created topic " in r["text"] else None
            t=target(r)
            if not t:return None
            r=t
    topics={r["message_id"]:topic(r) for r in normal}
    reply=[[0]*sender_count for _ in senders];adjacency=[[0]*sender_count for _ in senders]
    incoming=defaultdict(list);resolved=[]
    for r in normal:
        t=target(r)
        if t and t["message_type"]=="message":
            reply[index[r["sender"]]][index[t["sender"]]]+=1
            incoming[t["message_id"]].append(r)
            resolved.append((r,t))
    groups=defaultdict(list)
    for r in normal:
        if topics[r["message_id"]] is not None:groups[topics[r["message_id"]]].append(r)
    neighbors={}
    for group in groups.values():
        group.sort(key=lambda r:(times[r["message_id"]],order[r["message_id"]]))
        for i,r in enumerate(group):
            neighbors[r["message_id"]]=(group[i-1] if i else None,group[i+1] if i+1<len(group) else None)
        for a,b in zip(group,group[1:]):
            if (times[b["message_id"]]-times[a["message_id"]]).total_seconds()<=600:
                adjacency[index[a["sender"]]][index[b["sender"]]]+=1
    global_switch=Counter((a["sender"],b["sender"]) for a,b in zip(normal,normal[1:]) if a["sender"]!=b["sender"])
    def ranked(matrix,i,incoming_direction=False):
        values=[(senders[j],matrix[j][i] if incoming_direction else matrix[i][j]) for j in range(sender_count) if j!=i]
        return [{"sender":s,"count":v} for s,v in sorted(values,key=lambda x:(-x[1],x[0])) if v]
    def partners(matrix,i):
        return [{"sender":senders[j],"count":matrix[i][j]+matrix[j][i]}
                for j in sorted((j for j in range(sender_count) if j!=i),key=lambda j:(-(matrix[i][j]+matrix[j][i]),senders[j]))
                if matrix[i][j]+matrix[j][i]>0]
    profiles=[]
    sampling={}
    samples={}
    for sender in senders:
        i=index[sender];group=[r for r in normal if r["sender"]==sender]
        texts=[r for r in group if r["text"].strip()]
        lengths=[len(r["text"]) for r in group]
        attached=lambda r:bool(r.get("attachments") or r.get("attachment_type"))
        types=Counter("none" if not r["reply_to"] else "missing" if target(r) is None else target(r)["message_type"] for r in group)
        profile={"sender":sender,"message_count":len(group),"text_message_count":len(texts),
            "attachment_message_count":sum(attached(r) for r in group),
            "attachment_item_count":sum(len(r.get("attachments",[])) for r in group),
            "first_datetime":group[0]["datetime"],"last_datetime":group[-1]["datetime"],
            "active_days":len({times[r["message_id"]].date() for r in group}),
            "reply_sent_count":sum(bool(r["reply_to"]) for r in group),
            "reply_sent_to_normal_count":types["message"],"reply_sent_to_system_count":types["service"],
            "reply_sent_unresolved_count":types["missing"],"reply_received_count":sum(reply[j][i] for j in range(sender_count)),
            "self_reply_count":reply[i][i],
            "mean_text_length_including_empty":statistics.mean(lengths),"median_text_length_including_empty":statistics.median(lengths),
            "mean_nonempty_text_length":statistics.mean(len(r["text"]) for r in texts) if texts else None,
            "median_nonempty_text_length":statistics.median(len(r["text"]) for r in texts) if texts else None,
            "url_message_count":sum(bool(re.search(r"(?i)(?:https?://|www\.)",r["text"])) for r in group),
            "url_message_ratio":sum(bool(re.search(r"(?i)(?:https?://|www\.)",r["text"])) for r in group)/len(group),
            "attachment_message_ratio":sum(attached(r) for r in group)/len(group),
            "question_mark_message_count":sum("?" in re.sub(r"(?i)(?:https?://|www\.)\S+","",r["text"]) or "？" in r["text"] for r in group),
            "known_topic_count":len({topics[r["message_id"]] for r in group if topics[r["message_id"]] is not None}),
            "most_replied_to":ranked(reply,i),"most_replied_by":ranked(reply,i,True),
            "reply_interactions_other_senders":partners(reply,i),
            "adjacency_interactions_other_senders":partners(adjacency,i),
            "same_topic_switches_to":ranked(adjacency,i),"same_topic_switches_from":ranked(adjacency,i,True),
            "global_chronological_switches_to":[{"sender":s,"count":global_switch[(sender,s)]} for s in senders if s!=sender and global_switch[(sender,s)]],
            "global_chronological_switches_from":[{"sender":s,"count":global_switch[(s,sender)]} for s in senders if s!=sender and global_switch[(s,sender)]],
            "same_sender_topic_adjacency_count":adjacency[i][i]}
        profiles.append(profile)
        # Marker-based sampling only: no semantic question/role classifier.
        pools={"question_mark":[r for r in texts if "?" in re.sub(r"(?i)(?:https?://|www\.)\S+","",r["text"]) or "？" in r["text"]],
               "reply_to_other_sender":[r for r in texts if target(r) and target(r)["message_type"]=="message" and target(r)["sender"]!=sender],
               "replied_by_other_sender":[r for r in texts if any(t["sender"]!=sender for t in incoming[r["message_id"]])],
               "attachment_context":[r for r in texts if attached(r) or any(n and attached(n) for n in neighbors.get(r["message_id"],(None,None))) or (target(r) and attached(target(r)))],
               "long_100plus_chars":[r for r in texts if len(r["text"])>=100],
               "short_20or_fewer_chars":[r for r in texts if len(r["text"])<=20]}
        rng=random.Random(args.seed+i)
        selected={}
        dates=sorted({times[r["message_id"]].date() for r in texts})
        for date in rng.sample(dates,min(8,len(dates))):
            r=rng.choice([r for r in texts if times[r["message_id"]].date()==date])
            selected[r["message_id"]]=r
        for label,pool in pools.items():
            available=[r for r in pool if r["message_id"] not in selected]
            chosen=rng.sample(available,min(3,len(available),30-len(selected)))
            selected.update((r["message_id"],r) for r in chosen)
        remaining=[r for r in texts if r["message_id"] not in selected]
        selected.update((r["message_id"],r) for r in rng.sample(remaining,min(30-len(selected),len(remaining))))
        samples[sender]=sorted(selected.values(),key=lambda r:order[r["message_id"]])
        sampling[sender]={"selected_ids":list(selected),"count":len(selected),
            "distinct_sample_days":len({times[r["message_id"]].date() for r in selected.values()}),
            "strata":{label:{"eligible":len(pool),"sampled":sum(r["message_id"] in selected for r in pool)} for label,pool in pools.items()}}
    w=[r for r in normal if r["sender"]==args.evidence_sender]
    counts=Counter(r["text"] for r in w)
    def template(text):
        text=safe(text)
        text=re.sub(r"/[A-Za-z][A-Za-z0-9_]*","[COMMAND]",text)
        text=re.sub(r"\d+","[NUMBER]",text)
        return re.sub(r"\s+"," ",text).strip()
    templates=Counter(template(r["text"]) for r in w)
    wgaps=[(times[b["message_id"]]-times[a["message_id"]]).total_seconds() for a,b in zip(w,w[1:])]
    watch={"message_count":len(w),"distinct_exact_texts":len(counts),
        "exact_duplicate_groups":sum(v>1 for v in counts.values()),
        "messages_in_exact_duplicate_groups":sum(v for v in counts.values() if v>1),
        "duplicate_occurrences_beyond_first":sum(v-1 for v in counts.values()),
        "top_exact_redacted":[{"text":safe(t),"count":v} for t,v in counts.most_common(8)],
        "normalized_template_count":len(templates),
        "top_normalized_templates":[{"template":t,"count":v} for t,v in templates.most_common(8)],
        "normalization":"Mask identifiers/quoted values, commands, numeric runs and whitespace; collisions are possible, not verified semantic templates.",
        "checkmark_prefix_count":sum(r["text"].lstrip().startswith("\u2705") for r in w),
        "warning_or_failure_prefix_count":sum(r["text"].lstrip().startswith(("\u274c","\u26a0","\u2757")) for r in w),
        "multiline_count":sum("\n" in r["text"] for r in w),
        "colon_field_candidate_count":sum(bool(re.search(r"(?m)^.{1,24}[:\uff1a]",r["text"])) for r in w),
        "interarrival_seconds":desc(wgaps),
        "interarrival_within_60_seconds":sum(g<=60 for g in wgaps),
        "hour_counts":dict(sorted(Counter(times[r["message_id"]].hour for r in w).items())),
        "top_exact_send_times":[{"time":t,"count":v} for t,v in Counter(times[r["message_id"]].strftime("%H:%M:%S") for r in w).most_common(5)],
        "inference":"Repeated confirmation/status formats are evidence consistent with automation, not proof of a BOT role. Interarrival regularity is measured independently."}
    reply_graph=graph_analysis(reply,senders)
    adjacency_graph=graph_analysis(adjacency,senders)
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    csv_write(out/"sender_behavior.csv",profiles)
    for filename,matrix in (("sender_reply_matrix.csv",reply),("sender_adjacency_matrix.csv",adjacency)):
        csv_write(out/filename,[{"sender":sender,**{name:matrix[i][j] for j,name in enumerate(senders)}} for i,sender in enumerate(senders)])
    mapping=out/"sender_role_mapping.csv"
    if mapping.exists():
        with mapping.open(encoding="utf-8-sig",newline="") as f:existing=list(csv.DictReader(f))
        if len(existing)!=len(senders) or {r["sender"] for r in existing}!=set(senders):
            raise ValueError("Existing mapping sender set differs; refusing to overwrite")
        if any(r["role"] not in ("","USER","AGENT","INTERNAL","BOT","UNKNOWN") for r in existing):
            raise ValueError("Invalid existing role")
        # Preserve any manual labels on reruns.
    else:
        csv_write(mapping,[{"sender":sender,"role":"","confidence":"","notes":""} for sender in senders])
    def display(r):
        if r is None:return ["[No confirmed same-topic neighbor / unavailable target]"]
        if r["message_type"]=="service":
            return [f"[System event message{r['message_id']}; exact time and sender unavailable; event text omitted]"]
        result=[f"[{r['datetime']}] message{r['message_id']}",f"Sender: {r['sender']}","Text: "+safe(r["text"])]
        if r.get("attachment_type"):result.append("[Attachment: "+r["attachment_type"]+"; path omitted]")
        return result
    review=["# Sender Role Review","",
        "角色與信心由人工填寫；名稱與 prefix 僅為 metadata，不是角色證據。",
        "所有比例以一般訊息為分母；文字長度包含空文字。Reply sent 包含 system/topic 參照，矩陣只計一般訊息 direct reply。",
        "Previous/Next 為同一已確認 topic 的時間鄰居，不設時間上限；間隔由時間戳可見，未知 topic 不使用全域鄰居。",
        "抽樣為分層隨機：問號線索、回覆他人、被他人回覆、附件上下文、長短訊息及不同日期。問號線索不是語意詢問句分類。",
        "文字以規則遮蔽完整 URL、Email、電話候選、帳號標記、token、引號值與混合字母數字 ID；可能誤遮蔽或漏掉自由文字個資，原文未修改。",
        "允許 role：USER / AGENT / INTERNAL / BOT / UNKNOWN；目前所有選項均未勾選。",""]
    for profile in profiles:
        sender=profile["sender"]
        review += ["## Sender: "+sender,"","### Statistics","",
            "```json",json.dumps({k:v for k,v in profile.items() if not isinstance(v,list)},ensure_ascii=False,indent=2),"```","",
            "### Most interacted senders","",
            "- Direct reply（雙向加總、排除 self）："+json.dumps(profile["reply_interactions_other_senders"][:5],ensure_ascii=False),
            "- 同 topic <=10m adjacency（雙向加總、排除 self）："+json.dumps(profile["adjacency_interactions_other_senders"][:5],ensure_ascii=False),
            "- 最常 reply 對方："+json.dumps(profile["most_replied_to"][:5],ensure_ascii=False),
            "- 最常被對方 reply："+json.dumps(profile["most_replied_by"][:5],ensure_ascii=False),
            "","### Representative messages","",
            "抽樣覆蓋："+json.dumps(sampling[sender],ensure_ascii=False),""]
        for i,r in enumerate(samples[sender],1):
            prev,nxt=neighbors.get(r["message_id"],(None,None))
            review += [f"#### Sample {i:02d} — message{r['message_id']}","",
                "```text"]+display(r)+["","Previous:"]+display(prev)+["","Next:"]+display(nxt)+["","Reply target:"]+display(target(r))+["```",""]
        review += ["### Human review","", "Proposed Role:","",
                   "- [ ] USER","- [ ] AGENT","- [ ] INTERNAL","- [ ] BOT","- [ ] UNKNOWN","",
                   "Confidence:","","- [ ] HIGH","- [ ] MEDIUM","- [ ] LOW","","Notes:","","---",""]
    (out/"sender_role_review.md").write_text("\n".join(review),encoding="utf-8")
    report=["# Step 5 Sender Role Analysis","",
        "此階段未自動填 role/confidence，未修改 segmentation、來源或既有程式，未進行 Intent / Slot / LLM labeling。","",
        "## 統計定義","",
        "- 一般訊息 2,855 則、18 個顯示名稱；system event 不納入 mapping。",
        "- reply matrix：row=回覆發送者、column=被回覆訊息 sender；只計可解析的一般訊息 direct reply，包含對角線 self reply。各條回覆只計一次，不設時間門檻。",
        "- adjacency matrix：row=較早訊息 sender、column=較晚訊息 sender；在同已確認 topic 的全部一般訊息依時間相鄰，間隔 <=600 秒，包含對角線。未知 topic 不混合。",
        "- 相同秒以原始訊息順序決定。兩種 matrix 的方向語意不同，絕不合成單一權重。",
        "- most interacted 排除 self，各矩陣分別加總雙向；行為 CSV 保留全部其他 sender 的計數。",
        "- Global switching 與同 topic <=10m switching 分開存放於行為 CSV，前者可能跨 topic 或隔日，不當回覆證據。",
        "- attachment item count 與含附件訊息數均提供；text length 為 Unicode code points。比例為 0–1。",
        "- 問號、長短、附件等只是可重現抽樣線索，不是角色或意圖標籤。",
        "- mapping 是 CSV，無法內建儲存格驗證；程式定義並在重跑時檢查 role 合法值，保留既有人工填寫值，不覆寫。","",
        "## Sender 與最常互動對象","",
        "| Sender | Messages | Direct reply partner（雙向） | Adjacency partner（雙向） |",
        "|---|---:|---|---|"]
    for p in profiles:
        def top(items):
            if not items:return "無"
            maximum=items[0]["count"]
            return "、".join(x["sender"] for x in items if x["count"]==maximum)+f" ({maximum})"
        report.append(f"| {p['sender']} | {p['message_count']} | {top(p['reply_interactions_other_senders'])} | {top(p['adjacency_interactions_other_senders'])} |")
    report += ["","## Interaction graph","",
        "社群法尋找群內密集互動；最大二分 cut 尋找跨組互動，兩種群組不是同一概念，也不能命名 USER/AGENT。最大 cut 為 18 節點枚舉精確值；社群為 greedy modularity，非唯一最佳分群。","",
        "### Direct reply graph","```json",json.dumps(reply_graph,ensure_ascii=False,indent=2),"```","",
        "### Adjacency graph","```json",json.dumps(adjacency_graph,ensure_ascii=False,indent=2),"```","",
        "若存在三角形，圖並非嚴格二分；即使最大 cut 高，也不能確認 USER ↔ AGENT，因為同事協作、通知回覆等也會產生相似結構。",
        "不以名稱、prefix 或分群結果填入角色。人工 review 可檢查同一 sender 在不同 topic/date 中是否表現多種互動形式；雙向回覆本身無法證明多角色。","",
        "## Watchdog evidence","```json",json.dumps(watch,ensure_ascii=False,indent=2),"```","",
        "重複成功／狀態格式支持自動化行為的可能性，但不能直接標 BOT。固定模板與固定週期是不同特徵；需分別看模板頻率與發送間隔變異，不能由其中之一推論另一項。",
        "模板 normalization 遮蔽數字、識別碼、引號值後可能碰撞，因此提供原文精確重複數與正規化模板數兩種結果。","",
        "## 人工角色確認","",
        "sender_role_review.md 每人最多 30 則文字樣本，顯示同 topic 前後訊息及原 direct reply target。system target 只顯示 ID 與類型。",
        "sender_role_mapping.csv 初次建立時 role/confidence/notes 全空白。合法 role：USER、AGENT、INTERNAL、BOT、UNKNOWN。",
        "此版輸出不選擇 final threshold，也不改切分規則。","",
        "## Validation metadata","```json",json.dumps({"reply_matrix_total":sum(map(sum,reply)),
            "adjacency_matrix_total":sum(map(sum,adjacency)),"reply_self_edges":sum(reply[i][i] for i in range(sender_count)),
            "adjacency_self_edges":sum(adjacency[i][i] for i in range(sender_count)),
            "unknown_topic_messages_excluded_from_adjacency":sum(t is None for t in topics.values()),
            "sample_counts":{s:len(v) for s,v in samples.items()},"protected_sha256":hashes},ensure_ascii=False,indent=2),"```",""]
    (out/"sender_role_report.md").write_text("\n".join(report),encoding="utf-8")
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    print(json.dumps({"partners":[{"sender":p["sender"],"count":p["message_count"],
        "reply":p["reply_interactions_other_senders"][:2],"adjacency":p["adjacency_interactions_other_senders"][:2]} for p in profiles],
        "reply_graph":reply_graph,"adjacency_graph":adjacency_graph,"watchdog":watch,
        "samples":{s:len(v) for s,v in samples.items()},"inputs_unchanged":True},ensure_ascii=True,indent=2))

if __name__=="__main__":
    main()
