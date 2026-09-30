"""Step 6B: role-grounded agent context audit, no semantic labels.
Standard library only. Uses the existing review redactor without running its main.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
from analyze_sender_roles import safe

QUESTION_MARKERS = ["?", "？", "請問", "是否", "哪個", "哪一個", "什麼",
                    "多少", "可以提供", "麻煩提供", "能否提供", "方便提供", "有沒有", "是否可以"]
MANUAL = ["customer_service_interaction", "agent_behavior", "information_missing_before",
          "agent_asked_missing_information", "information_provided_after",
          "potential_missing_information", "notes"]

def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                        for k, v in row.items()})

def qtile(values, q):
    if not values:
        return None
    a = sorted(values)
    p = (len(a)-1)*q
    i = int(p)
    return a[i] + (a[min(i+1, len(a)-1)]-a[i])*(p-i)

class Union:
    def __init__(self, ids):
        self.p = {i: i for i in ids}
    def find(self, i):
        if self.p[i] != i:
            self.p[i] = self.find(self.p[i])
        return self.p[i]
    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a

def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--messages", type=Path, default=root/"data/processed/messages.jsonl")
    ap.add_argument("--mapping", type=Path, default=root/"data/roles/sender_role_mapping.csv")
    ap.add_argument("--output-dir", type=Path, default=root/"data/agent_audit")
    args = ap.parse_args()
    protected = [args.messages, args.mapping] + list((root/"scripts").glob("*.py"))
    protected += list((root/"data/segmentation").glob("segmentation_*"))
    protected += list((root/"data/reconstruction").glob("agent_aware_*.jsonl"))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    rows = [json.loads(line) for line in args.messages.read_text(encoding="utf-8").splitlines() if line.strip()]
    by = {r["message_id"]: r for r in rows}
    if len(by) != len(rows):
        raise ValueError("Duplicate message IDs")
    order = {r["message_id"]: i for i, r in enumerate(rows)}
    with args.mapping.open(encoding="utf-8-sig", newline="") as f:
        mapping = list(csv.DictReader(f))
    roles = {r["sender"]: r["role"] for r in mapping}
    if len(roles) != len(mapping):
        raise ValueError("Duplicate mapping sender")
    normal = [r for r in rows if r["message_type"] == "message"]
    if any(r["sender"] not in roles or not roles[r["sender"]] for r in normal):
        raise ValueError("Missing/blank manual role; no role will be inferred")
    times = {r["message_id"]: datetime.fromisoformat(r["datetime"]) for r in normal}
    def role(r):
        return roles[r["sender"]] if r["message_type"] == "message" else "SYSTEM_EVENT"
    agents = [r for r in normal if role(r) == "AGENT"]
    if not agents:
        raise ValueError("No manually mapped AGENT messages")
    def target(r):
        t = by.get(r.get("reply_to"))
        return t if t and t["source_file"] == (r.get("reply_to_source_file") or r["source_file"]) else None
    def topic(r):
        seen = set()
        while True:
            if r["message_id"] in seen:
                raise ValueError("Reply cycle")
            seen.add(r["message_id"])
            if r["message_type"] == "service":
                return r["message_id"] if " created topic " in r["text"] else None
            t = target(r)
            if t is None:
                return None
            r = t
    topics = {r["message_id"]: topic(r) for r in normal}
    chain = Union([r["message_id"] for r in normal])
    incoming = defaultdict(list)
    excluded_cross_topic = []
    for r in normal:
        t = target(r)
        if t and t["message_type"] == "message":
            if topics[r["message_id"]] != topics[t["message_id"]]:
                excluded_cross_topic.append({"message_id": r["message_id"], "reply_to": t["message_id"]})
                continue
            incoming[t["message_id"]].append(r)
            chain.join(r["message_id"], t["message_id"])
    component = defaultdict(list)
    for r in normal:
        component[chain.find(r["message_id"])].append(r)
    topic_rows = defaultdict(list)
    for r in normal:
        if topics[r["message_id"]] is not None:
            topic_rows[topics[r["message_id"]]].append(r)
    def sorted_rows(group):
        return sorted(group, key=lambda r: (times[r["message_id"]], order[r["message_id"]]))
    for k in topic_rows:
        topic_rows[k] = sorted_rows(topic_rows[k])
    contexts = {}
    features = []
    for a in agents:
        mid = a["message_id"]
        related = component[chain.find(mid)]
        # Unknown topic: only the proven direct-reply component supplies candidates.
        # We never pool unrelated null-topic messages by global proximity.
        pool = topic_rows[topics[mid]] if topics[mid] is not None else sorted_rows(related)
        idx = next(i for i, r in enumerate(pool) if r["message_id"] == mid)
        before = [r for r in pool[:idx] if 0 <= (times[mid]-times[r["message_id"]]).total_seconds() <= 3600][-10:]
        after = [r for r in pool[idx+1:] if 0 <= (times[r["message_id"]]-times[mid]).total_seconds() <= 3600][:10]
        window = {r["message_id"] for r in before + [a] + after}
        context_ids = window | {r["message_id"] for r in related}
        t = target(a)
        system_targets = {t["message_id"]} if t and t["message_type"] == "service" else set()
        # Include terminal system references of the agent's chain as metadata only;
        # they cannot join episodes or bring all topic descendants into a chain.
        for r in related:
            rt = target(r)
            if rt and rt["message_type"] == "service":
                system_targets.add(rt["message_id"])
        contexts[mid] = {"window_ids": window, "normal_ids": context_ids,
                         "chain_ids": {r["message_id"] for r in related},
                         "system_reference_ids": system_targets}
        human_before = [r for r in pool[:idx] if role(r) != "BOT"]
        human_after = [r for r in pool[idx+1:] if role(r) != "BOT"]
        prev = human_before[-1] if human_before else None
        nxt = human_after[0] if human_after else None
        others_after = [r for r in human_after if r["sender"] != a["sender"]]
        nonagent_after = [r for r in human_after if role(r) != "AGENT"]
        clean = re.sub(r"(?i)(?:https?://|www\.)\S+", "", a.get("own_text", a["text"]))
        hits = [m for m in QUESTION_MARKERS if m in clean]
        f = {"message_id": mid, "datetime": a["datetime"], "sender": a["sender"],
             "role": role(a), "topic_root": topics[mid], "topic_unknown": topics[mid] is None,
             "text": safe(a["text"]), "text_redacted": True, "has_text": bool(a["text"].strip()),
             "has_attachment": bool(a.get("attachments") or a.get("attachment_type")),
             "reply_to": a["reply_to"], "has_reply_to": bool(a["reply_to"]),
             "reply_target_resolved": t is not None,
             "reply_target_type": t["message_type"] if t else None,
             "reply_target_sender": t["sender"] if t else None,
             "reply_target_is_agent": role(t) == "AGENT" if t else None,
             "reply_target_gap_seconds": (times[mid]-times[t["message_id"]]).total_seconds()
                 if t and t["message_type"] == "message" else None,
             "previous_human_message_id": prev["message_id"] if prev else None,
             "next_human_message_id": nxt["message_id"] if nxt else None,
             "previous_human_sender": prev["sender"] if prev else None,
             "next_human_sender": nxt["sender"] if nxt else None,
             "gap_previous_seconds": (times[mid]-times[prev["message_id"]]).total_seconds() if prev else None,
             "gap_next_seconds": (times[nxt["message_id"]]-times[mid]).total_seconds() if nxt else None,
             "question_like": bool(hits), "question_text_scope": "own_text" if "own_text" in a else "legacy_aggregate", "question_marker_hits": hits,
             "window_before_count": len(before), "window_after_count": len(after),
             "normal_context_count": len(context_ids),
             "chain_extension_count": len(context_ids-window),
             "no_normal_context": len(context_ids) == 1,
             "incoming_direct_reply_ids": [r["message_id"] for r in incoming[mid]]}
        def within(group, minutes):
            return [r["message_id"] for r in group if 0 <= (times[r["message_id"]]-times[mid]).total_seconds() <= minutes*60]
        for minutes in (5,10,30,60):
            possible = within(nonagent_after, minutes)
            f[f"response_within_{minutes}m"] = bool(possible)
            f[f"response_message_ids_{minutes}m"] = possible
            f[f"other_human_followup_within_{minutes}m"] = bool(within(others_after, minutes))
            explicit = [r for r in incoming[mid] if role(r) not in ("AGENT","BOT")
                        and order[r["message_id"]] > order[mid]]
            f[f"direct_nonagent_reply_within_{minutes}m"] = bool(within(explicit, minutes))
        features.append(f)
    # Union contexts when actual normal-message sets overlap; shared service refs
    # never count. This also deduplicates chain-expanded overlap.
    eu = Union([r["message_id"] for r in agents])
    merge_evidence = []
    for i,a in enumerate(agents):
        for b in agents[i+1:]:
            x,y = a["message_id"], b["message_id"]
            if topics[x] != topics[y]:
                continue
            same_chain = chain.find(x) == chain.find(y)
            overlap = contexts[x]["normal_ids"] & contexts[y]["normal_ids"]
            window_overlap = contexts[x]["window_ids"] & contexts[y]["window_ids"]
            if topics[x] is None and not same_chain:
                continue
            if overlap or same_chain:
                eu.join(x,y)
                merge_evidence.append({"agent_message_a":x,"agent_message_b":y,
                    "same_direct_reply_component":same_chain,
                    "window_overlap_ids":sorted(window_overlap,key=order.get),
                    "expanded_context_overlap_ids":sorted(overlap,key=order.get)})
    agroups = defaultdict(list)
    for a in agents:
        agroups[eu.find(a["message_id"])].append(a["message_id"])
    episodes = []
    feature_by = {f["message_id"]:f for f in features}
    for number,anchors in enumerate(sorted(agroups.values(),key=lambda ids:min(order[i] for i in ids)),1):
        eid = f"agent_episode_{number:03d}"
        mids = set().union(*(contexts[i]["normal_ids"] for i in anchors))
        refs = set().union(*(contexts[i]["system_reference_ids"] for i in anchors))
        messages = []
        for mid in sorted(mids,key=lambda i:(times[i],order[i])):
            r = by[mid]
            provenance = []
            for anchor in anchors:
                if mid not in contexts[anchor]["normal_ids"]:
                    continue
                reasons = []
                if mid == anchor: reasons.append("agent_anchor")
                if mid in contexts[anchor]["window_ids"] and mid != anchor: reasons.append("same_topic_time_window")
                if mid in contexts[anchor]["chain_ids"] and mid != anchor: reasons.append("resolved_direct_reply_component")
                provenance.append({"agent_message_id":anchor,
                    "gap_seconds_from_agent":(times[mid]-times[anchor]).total_seconds(),
                    "outside_60m":abs((times[mid]-times[anchor]).total_seconds())>3600,
                    "inclusion_reasons":reasons})
            enriched = {**r,"role":role(r),"topic_root":topics[mid],"context_provenance":provenance}
            if role(r) == "AGENT":
                enriched["question_like"] = feature_by[mid]["question_like"]
                enriched["question_marker_hits"] = feature_by[mid]["question_marker_hits"]
            messages.append(enriched)
        ep = {"episode_id":eid,"topic_root":topics[anchors[0]],
            "topic_unknown":topics[anchors[0]] is None,
            "start_time":messages[0]["datetime"],"end_time":messages[-1]["datetime"],
            "agent_message_count":sum(r["role"]=="AGENT" for r in messages),
            "message_count":len(messages),
            "agent_senders":sorted({r["sender"] for r in messages if r["role"]=="AGENT"}),
            "other_senders":sorted({r["sender"] for r in messages if r["role"] not in ("AGENT","BOT")}),
            "bot_senders":sorted({r["sender"] for r in messages if r["role"]=="BOT"}),
            "agent_anchor_ids":anchors,
            "question_like_agent_count":sum(feature_by[i]["question_like"] for i in anchors),
            "episode_merge_evidence":[e for e in merge_evidence if e["agent_message_a"] in anchors and e["agent_message_b"] in anchors],
            "messages":messages,
            "system_references":[{**by[i],"role":"SYSTEM_EVENT","exact_time_unavailable":True}
                                 for i in sorted(refs,key=order.get)]}
        assert {r["message_id"] for r in messages if r["role"]=="AGENT"} == set(anchors)
        episodes.append(ep)
        for mid in anchors:
            feature_by[mid]["episode_id"] = eid
    # No overlapping normal-message contexts remain between same-topic episodes.
    seen = set()
    for ep in episodes:
        ids = {r["message_id"] for r in ep["messages"]}
        assert not seen & ids
        seen |= ids
        assert len({r["topic_root"] for r in ep["messages"]}) == 1
    output = args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    annotation = []
    for ep in episodes:
        row = {k:ep[k] for k in ("episode_id","topic_root","start_time","end_time","message_count",
               "agent_message_count","agent_senders","other_senders","question_like_agent_count")}
        row.update({k:"" for k in MANUAL})
        annotation.append(row)
    annotation_path = output/"agent_episode_annotation.csv"
    if annotation_path.exists():
        with annotation_path.open(encoding="utf-8-sig",newline="") as f:
            old = list(csv.DictReader(f))
        if any(r.get(k) for r in old for k in MANUAL):
            raise ValueError("Existing manual annotation is nonempty; refusing to overwrite audit outputs")
    review_path = output/"agent_episode_review.md"
    if review_path.exists() and re.search(r"\[[xX]\]",review_path.read_text(encoding="utf-8")):
        raise ValueError("Existing review contains checked annotations; refusing to overwrite")
    write_csv(output/"agent_messages.csv",features)
    write_csv(annotation_path,annotation)
    with (output/"agent_episodes.jsonl").open("w",encoding="utf-8",newline="\n") as f:
        for ep in episodes:
            f.write(json.dumps(ep,ensure_ascii=False)+"\n")
    review = ["# Agent Interaction Episode Review","",
        "人工 role mapping 直接使用。所有 annotation 選項留白；question_like 只是文字規則，不是 clarification 標籤。",
        "一般 window 為同 topic 前後各最多10則、各60分鐘；一般訊息 direct-reply component 完整保留。",
        "系統 target 另列 system_references，無精確時間，不計入 episode message_count，也不因共同 topic system target 合併 episode。",
        "UNKNOWN 視為非BOT participant，並不推測其真實職能。Review 文字採規則遮蔽，完整原文保留在 JSONL。",
        "後續發言不等於直接回覆或問題已解決。超窗 direct-chain context 會逐筆標明相對 anchor 的時間差。",""]
    for ep in episodes:
        review += ["="*64,f"## Episode: {ep['episode_id']}",
            f"Topic: {ep['topic_root']} | topic_unknown: {ep['topic_unknown']}",
            f"Start: {ep['start_time']}",f"End: {ep['end_time']}",
            "Agents: "+", ".join(ep["agent_senders"]),
            "Other Senders: "+", ".join(ep["other_senders"]),
            f"Agent Messages: {ep['agent_message_count']} | All normal messages: {ep['message_count']}",
            "="*64,""]
        for r in ep["messages"]:
            text = safe(r["text"]).replace("`","ˋ")
            review += [f"[{r['datetime']}] [{r['role']}] {r['sender']} — message{r['message_id']}",
                       "```text",text or "[no text]","```"]
            if r.get("attachment_type"):
                review.append("[attachment: "+r["attachment_type"]+"; path omitted]")
            if r["role"]=="AGENT":
                review += ["```yaml","question_like: "+str(r["question_like"]).lower(),
                    "reply_to: "+json.dumps(r["reply_to"]),
                    "episode_anchor: true","```"]
            elif r.get("reply_to"):
                review.append("reply_to: message"+r["reply_to"])
            extensions = [v for v in r["context_provenance"] if "resolved_direct_reply_component" in v["inclusion_reasons"]]
            if extensions:
                review.append("Direct-chain context: "+json.dumps(extensions,ensure_ascii=False))
            review.append("")
        if ep["system_references"]:
            review += ["System reply targets（僅 metadata，不是 episode bridge）:"]
            for r in ep["system_references"]:
                review.append(f"- message{r['message_id']}: [SYSTEM_EVENT; sender/time unavailable] "+safe(r["text"]))
            review.append("")
        review += ["---","### 人工 Annotation",""]
        for title,options in [
            ("Is customer-service interaction",["YES","NO","UNCERTAIN"]),
            ("Agent behavior",["CLARIFICATION","INFORMATION_REQUEST","ACKNOWLEDGEMENT","STATUS_UPDATE","RESOLUTION","HANDOFF","OTHER","UNCERTAIN"]),
            ("Is information missing before Agent message",["YES","NO","UNCERTAIN"]),
            ("Did Agent ask for missing information",["YES","NO","UNCERTAIN"]),
            ("Did someone provide information afterward",["YES","NO","UNCERTAIN"])]:
            review += [title+":",""]+["- [ ] "+opt for opt in options]+[""]
        review += ["Potential missing information:","","---","","Notes:","","---",""]
    review_path.write_text("\n".join(review),encoding="utf-8")
    question = [f for f in features if f["question_like"]]
    stats = {
        "agent_message_count":len(agents),
        "agent_sender_counts":dict(Counter(r["sender"] for r in agents)),
        "agent_date_counts":dict(sorted(Counter(times[r["message_id"]].date().isoformat() for r in agents).items())),
        "agent_topic_counts":dict(Counter(str(topics[r["message_id"]]) for r in agents)),
        "known_agent_topic_count":len({topics[r["message_id"]] for r in agents if topics[r["message_id"]] is not None}),
        "unknown_topic_agent_count":sum(topics[r["message_id"]] is None for r in agents),
        "episode_count":len(episodes),
        "episode_message_count":{"median":qtile([e["message_count"] for e in episodes],.5),
            "p90":qtile([e["message_count"] for e in episodes],.9),
            "max":max(e["message_count"] for e in episodes)},
        "question_like_agent_count":len(question),
        "agent_reply_target_counts":dict(Counter(f["reply_target_type"] if f["reply_target_resolved"] else "missing" if f["has_reply_to"] else "none" for f in features)),
        "agent_has_reply_to_count":sum(f["has_reply_to"] for f in features),
        "single_agent_message_episodes":sum(e["agent_message_count"]==1 for e in episodes),
        "agent_messages_without_normal_context":[f["message_id"] for f in features if f["no_normal_context"]],
        "agent_messages_without_time_window_neighbors":[f["message_id"] for f in features if f["window_before_count"]+f["window_after_count"]==0],
        "question_like_followup_counts":{str(m):{
            "nonagent_later_message":sum(f[f"response_within_{m}m"] for f in question),
            "explicit_nonagent_direct_reply":sum(f[f"direct_nonagent_reply_within_{m}m"] for f in question),
            "other_human_later_message_including_other_agents":sum(f[f"other_human_followup_within_{m}m"] for f in question)}
            for m in (5,10,30,60)},
        "all_agent_followup_counts":{str(m):{
            "nonagent_later_message":sum(f[f"response_within_{m}m"] for f in features),
            "explicit_nonagent_direct_reply":sum(f[f"direct_nonagent_reply_within_{m}m"] for f in features),
            "other_human_later_message_including_other_agents":sum(f[f"other_human_followup_within_{m}m"] for f in features)}
            for m in (5,10,30,60)},
        "unique_normal_messages_in_episodes":len(seen),
        "chain_extension_message_occurrences":sum(f["chain_extension_count"] for f in features),
        "excluded_cross_topic_direct_edges":excluded_cross_topic,
        "manual_annotation_columns":MANUAL}
    report = ["# Step 6B Agent Interaction Audit","",
        "只提供人工確認 AGENT 的結構統計與 review；未判斷 clarification ground truth、資訊是否補足、問題是否解決或資料是否足夠。","",
        "## 定義","",
        "- role 僅來自人工 CSV；system event 不屬於 sender mapping。",
        "- 同 topic normal message 前後各最多10則且各 <=60分鐘。這個數量上限只約束一般 context window。",
        "- 對 anchor 的一般訊息 direct reply 取完整無向連通分量，保留祖先、後代與同 chain 分支；不經 system/topic 節點連接其他訊息。",
        "- AGENT direct reply target 若為 system event，完整 metadata 放在 system_references 並於 review 顯示；無時間不補猜。message_count 與時間範圍只計一般訊息。",
        "- topic unknown 時，只在已證實 direct reply component 中找時間鄰近；不以全域時間把未知 topic 與已知 topic 混合。",
        "- 同 topic 的 context 實際一般訊息集合有交集（含 direct-chain 延伸）或共享一般訊息 reply component，就合併 episode，採傳遞閉包。共用 system reference 不觸發合併。",
        "- Previous/Next human 是同 topic 非BOT 的前後一般訊息，可超過60分鐘，也可能與 anchor 同 sender；另提供 gap。",
        "- response_within_* 指同 topic 在指定時間內有 non-AGENT、非BOT 後續訊息，不等於對 anchor 的明確回覆；direct_nonagent_reply_within_* 另計 explicit reply_to。其他 human 欄位可包括其他 AGENT，但排除 anchor 自己。",
        "- follow-up 統計掃描時間範圍內全部後續訊息，不受 context 的10則上限限制；同秒以來源順序判定後續。",
        "- question_like 在去除 URL 後以問號與固定中文字串判定；不是語意分類，也不證明是追問缺漏資訊。",
        "- CSV/review 文字採規則遮蔽，可能誤遮或漏掉自由文字個資；JSONL 保留原文字及附件 metadata，沒有讀取附件。",
        "- 人工 annotation 全空白；若重跑遇到非空 CSV annotation 或已勾選 review，程式會拒絕覆寫。","",
        "## 統計","```json",json.dumps(stats,ensure_ascii=False,indent=2),"```","",
        "## 每日 AGENT 訊息","","| 日期 | 數量 |","|---|---:|"]
    report += [f"| {k} | {v} |" for k,v in stats["agent_date_counts"].items()]
    report += ["","## Topic 分布","","| Topic root | AGENT 訊息 |","|---|---:|"]
    report += [f"| {k} | {v} |" for k,v in stats["agent_topic_counts"].items()]
    report += ["","## Episode 大小","","| Episode | Topic | 一般訊息 | AGENT 訊息 | Question-like |","|---|---|---:|---:|---:|"]
    report += [f"| {e['episode_id']} | {e['topic_root']} | {e['message_count']} | {e['agent_message_count']} | {e['question_like_agent_count']} |" for e in episodes]
    report += ["","## Question-like 後續發言","","| 分鐘 | non-AGENT 後續發言 | 明確 reply anchor |","|---|---:|---:|"]
    report += [f"| {m} | {v['nonagent_later_message']} | {v['explicit_nonagent_direct_reply']} |" for m,v in stats["question_like_followup_counts"].items()]
    report += ["","以上以 question-like AGENT 訊息為計數單位，不是回覆訊息總數；同一後續訊息可能落在多個 anchor 的時間窗。",
        "無上下文定義為沒有任何其他一般訊息（含 direct-chain 延伸）；即使有 topic system reference，也不把它當作互動內容。",
        "Episode 不代表最終 conversation 或已確認客服事件，所有人工欄位待填。","",
        "## 輸入完整性","```json",json.dumps(hashes,ensure_ascii=False,indent=2),"```",""]
    (output/"agent_audit_report.md").write_text("\n".join(report),encoding="utf-8")
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    print(json.dumps(stats,ensure_ascii=True,indent=2))
    print("Protected inputs unchanged; annotations left blank.")

if __name__ == "__main__":
    main()

