"""Read-only profiling of processed Telegram messages (Python standard library).
Run: python scripts/profile_dataset.py
No session splitting, role inference, or intent classification is performed.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import random
import re
import statistics

def quantile(values, q):
    if not values:
        return None
    a = sorted(values)
    index = (len(a) - 1) * q
    low = int(index)
    return a[low] + (a[min(low + 1, len(a) - 1)] - a[low]) * (index - low)

def describe(values):
    return {"count": len(values), "mean": statistics.mean(values) if values else None,
            "median": quantile(values, .5), "p90": quantile(values, .9),
            "p95": quantile(values, .95), "min": min(values) if values else None,
            "max": max(values) if values else None}

URL = re.compile(r"(?i)(?:https?://|www\.)[^\s<>\u3000]+")
EMAIL = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+")
IP = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
UUID = re.compile(r"(?i)\b[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\b")
PHONE = re.compile(r"(?<![\w.])(?:\+?\d[\d ()-]{6,}\d)(?![\w.])")
SECRET = re.compile(r"(?i)(\b(?:token|hash|password|passwd|secret|api[_ -]?key)\b\s*[:=\uff1a]\s*)[^\s&<>]+")
LONG_ID = re.compile(r"(?i)\b[0-9a-f]{24,}\b")

def phone_candidates(text):
    # Remove URLs/IPs/UUIDs and common dates first to limit false positives.
    clean = URL.sub(" ", text)
    clean = EMAIL.sub(" ", clean)
    clean = IP.sub(" ", clean)
    clean = UUID.sub(" ", clean)
    clean = re.sub(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", " ", clean)
    return [m.group() for m in PHONE.finditer(clean)
            if 8 <= len(re.sub(r"\D", "", m.group())) <= 15]

def attachments(row):
    return bool(row.get("attachments") or row.get("attachment_type") or row.get("attachment_path"))

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=root / "data/processed/messages.jsonl")
    ap.add_argument("--output-dir", type=Path, default=root / "data/analysis")
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()
    original_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    normal = [r for r in rows if r["message_type"] == "message"]
    timed = [r for r in normal if r.get("datetime")]
    dates = [datetime.fromisoformat(r["datetime"]) for r in timed]
    assert len({(r["source_file"], r["message_id"]) for r in rows}) == len(rows)
    assert all(b >= a for a, b in zip(dates, dates[1:])), "Time order is not monotonic"
    assert all(t.utcoffset() == dates[0].utcoffset() for t in dates), "Mixed timezone offsets"
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    by = {(r["source_file"], r["message_id"]): r for r in rows}
    def key(r):
        return (r["source_file"], r["message_id"])
    def parent(r):
        return (r.get("reply_to_source_file") or r["source_file"], r["reply_to"]) if r.get("reply_to") else None

    senders = sorted({r["sender"] for r in normal if r["sender"]})
    aliases = {name: f"Sender {i:02d}" for i, name in enumerate(senders, 1)}
    # Redact known display names inside text as well as sample speaker labels.
    known_names = dict(aliases)
    for row in normal:
        for forward in row.get("forwarded", []):
            name = forward.get("sender")
            if name and name not in known_names:
                known_names[name] = "[FORWARDED_NAME]"
    def redact(text):
        text = URL.sub("[URL]", text)
        text = EMAIL.sub("[EMAIL]", text)
        text = IP.sub("[IP]", text)
        text = UUID.sub("[ID]", text)
        text = SECRET.sub(lambda m: m[1] + "[REDACTED]", text)
        text = LONG_ID.sub("[ID]", text)
        for value in sorted(phone_candidates(text), key=len, reverse=True):
            text = text.replace(value, "[PHONE_CANDIDATE]")
        for name in sorted(known_names, key=len, reverse=True):
            text = text.replace(name, known_names[name])
        return text

    sender_stats = []
    for sender in senders:
        group = [r for r in normal if r["sender"] == sender]
        ts = [r["datetime"] for r in group if r.get("datetime")]
        sender_stats.append({"sender": sender, "message_count": len(group),
            "text_message_count": sum(bool(r["text"].strip()) for r in group),
            "attachment_message_count": sum(attachments(r) for r in group),
            "first_datetime": min(ts) if ts else None, "last_datetime": max(ts) if ts else None})
    sender_stats.sort(key=lambda x: (-x["message_count"], x["sender"]))
    with (output / "sender_statistics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sender_stats[0]))
        writer.writeheader()
        writer.writerows(sender_stats)

    daily = Counter(t.date().isoformat() for t in dates)
    day = dates[0].date()
    while day <= dates[-1].date():
        daily.setdefault(day.isoformat(), 0)
        day += timedelta(days=1)
    service_daily = Counter(r.get("date_context") for r in rows if r["message_type"] == "service")
    gaps = [(b - a).total_seconds() for a, b in zip(dates, dates[1:])]
    bins = {"<1 minute": 0, "1-5 minutes": 0, "5-10 minutes": 0,
            "10-30 minutes": 0, "30-60 minutes": 0, "1-6 hours": 0, ">6 hours": 0}
    for gap in gaps:
        bucket = ("<1 minute" if gap < 60 else "1-5 minutes" if gap < 300 else
                  "5-10 minutes" if gap < 600 else "10-30 minutes" if gap < 1800 else
                  "30-60 minutes" if gap < 3600 else "1-6 hours" if gap <= 21600 else ">6 hours")
        bins[bucket] += 1
    time_stats = {"scope": "Normal messages with exact datetime, original JSONL order; local UTC+08:00",
        "total_records": len(rows), "timed_normal_messages": len(timed),
        "excluded_untimed_records": len(rows) - len(timed),
        "first_datetime": dates[0].isoformat(), "last_datetime": dates[-1].isoformat(),
        "calendar_days_inclusive": len(daily), "active_days": sum(v > 0 for v in daily.values()),
        "daily_message_counts": dict(sorted(daily.items())),
        "service_counts_by_date_context_not_exact_time": dict(sorted((k, v) for k, v in service_daily.items() if k)),
        "hourly_message_counts": {str(h).zfill(2): sum(t.hour == h for t in dates) for h in range(24)},
        "adjacent_gap_seconds": describe(gaps), "adjacent_gap_bins": bins,
        "bin_boundaries": "[0,60), [60,300), [300,600), [600,1800), [1800,3600), [3600,21600], (21600,+inf), seconds",
        "zero_second_gaps": gaps.count(0), "negative_gaps": sum(g < 0 for g in gaps),
        "quantiles": "Linear interpolation at index (n-1)*q"}

    chains = {}
    cycles = []
    for row in normal:
        current = row
        visited = set()
        depth = 0
        while True:
            k = key(current)
            if k in visited:
                termination, root_key = "cycle", k
                cycles.append({"source_file": row["source_file"], "message_id": row["message_id"]})
                break
            visited.add(k)
            target = parent(current)
            if target is None:
                termination, root_key = ("service" if current["message_type"] == "service" else "normal_no_reply"), k
                break
            if target not in by:
                termination, root_key = "missing_target", target
                break
            depth += 1
            current = by[target]
        chains[key(row)] = (depth, termination, root_key)
    reply_rows = [r for r in normal if parent(r)]
    target_types = Counter()
    targets = Counter(parent(r) for r in reply_rows)
    for row in reply_rows:
        target = by.get(parent(row))
        kind = ("missing" if target is None else "topic_creation_service"
                if target["message_type"] == "service" and " created topic " in target["text"]
                else target["message_type"])
        target_types[kind] += 1
    root_members = defaultdict(list)
    for row in normal:
        root_members[chains[key(row)][2]].append(row)
    root_stats = []
    for root_key, members in root_members.items():
        root_row = by.get(root_key)
        root_stats.append({"root_source_file": root_key[0], "root_message_id": root_key[1],
            "root_type": "missing" if root_row is None else root_row["message_type"],
            "is_topic_creation": bool(root_row and " created topic " in root_row["text"]),
            "normal_message_count": len(members), "distinct_senders": len({r["sender"] for r in members}),
            "span_seconds": (datetime.fromisoformat(members[-1]["datetime"]) -
                             datetime.fromisoformat(members[0]["datetime"])).total_seconds(),
            "max_observed_depth_edges": max(chains[key(r)][0] for r in members)})
    root_stats.sort(key=lambda r: -r["normal_message_count"])
    # These fixed windows measure activity overlap; they do not define sessions.
    windows = defaultdict(list)
    for row, dt in zip(timed, dates):
        window = dt.replace(minute=dt.minute // 5 * 5, second=0, microsecond=0)
        windows[window].append(row)
    busy = [group for group in windows.values() if len(group) >= 3]
    multi = [group for group in busy if len({r["sender"] for r in group}) >= 3]
    multi_roots = [group for group in multi if len({chains[key(r)][2] for r in group}) >= 2]
    aba = sum(a["sender"] == c["sender"] != b["sender"] and
              (dates[i + 2] - dates[i]).total_seconds() <= 600
              for i, (a, b, c) in enumerate(zip(timed, timed[1:], timed[2:])))
    switches = sum(a["sender"] != b["sender"] for a, b in zip(timed, timed[1:]))
    near_pairs = [(a, b) for a, b, gap in zip(timed, timed[1:], gaps) if gap < 300]
    near_root_switches = sum(chains[key(a)][2] != chains[key(b)][2] for a, b in near_pairs)
    runs = []
    for row in normal:
        if runs and runs[-1]["sender"] == row["sender"]:
            runs[-1]["length"] += 1
        else:
            runs.append({"sender": row["sender"], "length": 1})
    depths = [chains[key(r)][0] for r in normal]
    conversation = {"scope": "Normal messages; outer reply_to only. Forwarded reply metadata excluded.",
        "normal_message_count": len(normal), "messages_with_reply_to": len(reply_rows),
        "reply_to_rate_normal_messages": len(reply_rows) / len(normal),
        "reply_to_rate_all_records": len(reply_rows) / len(rows),
        "direct_reply_target_types": dict(target_types),
        "top_direct_reply_targets": [{"source_file": k[0], "message_id": k[1], "count": v}
                                     for k, v in targets.most_common(20)],
        "reply_chain_definition": "Observed resolved parent edges from each normal message to terminal ancestor; includes service roots. Missing edge is not counted. This is depth, not component size or duration.",
        "reply_chain_depth_edges_distribution": dict(sorted(Counter(depths).items())),
        "reply_chain_depth_edges_statistics": describe(depths),
        "reply_chain_depth_for_messages_with_reply": dict(sorted(Counter(chains[key(r)][0] for r in reply_rows).items())),
        "chain_termination_counts": dict(Counter(chains[key(r)][1] for r in normal)),
        "cycles": cycles,
        "reply_root_groups_not_sessions": root_stats,
        "interleaving": {
            "definition": "Observable sender/root alternation, not proof of simultaneous speech or independent semantic conversations.",
            "adjacent_sender_switches": switches, "adjacent_pair_count": len(gaps),
            "sender_switch_rate": switches / len(gaps),
            "ABA_triples_within_10_minutes": aba,
            "fixed_5minute_active_windows": len(windows), "windows_with_at_least_3_messages": len(busy),
            "windows_with_at_least_3_senders": len(multi),
            "three_sender_window_rate_among_windows_with_3_messages": len(multi) / len(busy) if busy else None,
            "three_sender_windows_with_multiple_reply_roots": len(multi_roots),
            "pairs_under_5_minutes": len(near_pairs), "reply_root_switches_under_5_minutes": near_root_switches,
            "reply_root_switch_rate_under_5_minutes": near_root_switches / len(near_pairs) if near_pairs else None},
        "same_sender_runs": {"definition": "Maximal adjacent normal-message runs; untimed service records ignored; no time threshold.",
            "run_length_distribution": dict(sorted(Counter(r["length"] for r in runs).items())),
            "run_count": len(runs), "multi_message_run_count": sum(r["length"] >= 2 for r in runs),
            "messages_in_multi_message_runs": sum(r["length"] for r in runs if r["length"] >= 2),
            "maximum_run_length": max(r["length"] for r in runs),
            "per_sender": [{"sender": s, "runs": sum(r["sender"] == s for r in runs),
                            "max_run_length": max(r["length"] for r in runs if r["sender"] == s)} for s in senders]}}

    texts = [r["text"] for r in rows]
    normal_texts = [r["text"] for r in normal]
    counts = Counter(t for t in normal_texts if t.strip())
    duplicate = {t: n for t, n in counts.items() if n > 1}
    pii_flags = [{"url": bool(URL.search(t)), "email": bool(EMAIL.search(t)),
                  "phone_candidate": bool(phone_candidates(t)), "ip_candidate": bool(IP.search(t)),
                  "credential_candidate": bool(SECRET.search(t))} for t in texts]
    text_stats = {"scope": "Top 50 and exact duplicates use nonempty normal-message text, no normalization. Length is Python Unicode code points, including whitespace.",
        "empty_text_all_records": sum(not t.strip() for t in texts),
        "empty_text_normal_messages": sum(not t.strip() for t in normal_texts),
        "length_all_records": describe([len(t) for t in texts]),
        "length_normal_messages_including_empty": describe([len(t) for t in normal_texts]),
        "length_nonempty_normal_messages": describe([len(t) for t in normal_texts if t.strip()]),
        "most_common_50_texts": [{"rank": i, "text_redacted": redact(t), "count": n}
                                for i, (t, n) in enumerate(counts.most_common(50), 1)],
        "exact_duplicate_nonempty_text": {"distinct_texts": len(counts), "repeated_text_groups": len(duplicate),
            "messages_in_repeated_groups": sum(duplicate.values()),
            "redundant_occurrences_beyond_first": sum(n - 1 for n in duplicate.values())},
        "all_record_text_duplicates_including_empty": {
            "repeated_text_groups": sum(n > 1 for n in Counter(texts).values()),
            "messages_in_repeated_groups": sum(n for n in Counter(texts).values() if n > 1),
            "redundant_occurrences_beyond_first": sum(n - 1 for n in Counter(texts).values())},
        "potential_pii_message_counts_all_records": {k: sum(f[k] for f in pii_flags)
            for k in ["url", "email", "phone_candidate", "ip_candidate", "credential_candidate"]},
        "messages_with_url_email_or_phone_candidate": sum(f["url"] or f["email"] or f["phone_candidate"] for f in pii_flags),
        "pii_method": {"url": "http(s) or www links", "email": "Email-like address regex",
            "phone": "8-15 digits with optional +/spaces/parentheses/hyphens; exclude URLs, emails, IPv4, UUID, YYYY-MM-DD dates",
            "limitations": "Heuristic candidates, not verified personal data; telephone matches may be numeric identifiers, URLs are not necessarily personal. Categories overlap. Names/addresses/free-form identifiers cannot be comprehensively detected by regex.",
            "redaction": "Sample and top texts mask URLs, email, phone candidates, IPs, UUIDs, labelled secrets, long hexadecimal identifiers, and known sender/forwarder names. Attachment paths and system-event prose are omitted from sample."}}

    selected = sorted(random.Random(args.seed).sample(range(len(rows)), min(100, len(rows))))
    included = set()
    lines = ["Random 100 anchors from ALL JSONL records; seed=" + str(args.seed),
             "Each block retains previous/next 3 records in original order. Context may repeat across blocks.",
             "Speaker aliases carry no role meaning. Pattern redaction is heuristic; this file is for local review.",
             "System-event prose and attachment paths are omitted. No sessions have been inferred.", ""]
    for number, index in enumerate(selected, 1):
        lines.append(f"=== Sample {number:03d}; anchor row {index + 1}; message_id={rows[index]['message_id']} ===")
        for j in range(max(0, index - 3), min(len(rows), index + 4)):
            included.add(j)
            row = rows[j]
            anchor = " [ANCHOR]" if j == index else ""
            if row["datetime"]:
                stamp = datetime.fromisoformat(row["datetime"]).strftime("%Y-%m-%d %H:%M:%S %z")
            else:
                stamp = (row.get("date_context") or "unknown date") + " [exact time unavailable]"
            lines.append(f"[{stamp}]{anchor} (row={j + 1}, id={row['message_id']}, reply_to={row.get('reply_to')})")
            if row["message_type"] == "service":
                lines.append("SYSTEM: [system-event text omitted]")
            else:
                text = redact(row["text"]) if row["text"].strip() else "[no text]"
                lines.append(f"{aliases[row['sender']]}: {text}")
                if attachments(row):
                    lines.append(f"[attachment type: {row.get('attachment_type')}; path omitted]")
                if row.get("forwarded"):
                    lines.append("[contains forwarded content]")
            lines.append("")
    (output / "sample_conversations.txt").write_text("\n".join(lines), encoding="utf-8")
    text_stats["sample"] = {"seed": args.seed, "anchor_count": len(selected),
        "anchor_rows_1based": [i + 1 for i in selected], "unique_records_including_context": len(included),
        "population": "all records, including service", "context_before_after": 3}
    dump(output / "time_statistics.json", time_stats)
    dump(output / "conversation_statistics.json", conversation)
    dump(output / "text_statistics.json", text_stats)

    gapstats = time_stats["adjacent_gap_seconds"]
    inter = conversation["interleaving"]
    report = [
        "# Dataset Profiling", "",
        "本階段僅分析；未推測客服／玩家身分，未進行 conversation segmentation 或 Intent Classification。",
        f"輸入：{len(rows):,} 筆，一般聊天 {len(normal):,} 筆、系統事件 {len(rows)-len(normal)} 筆。",
        f"輸入 SHA-256：{original_hash}", "",
        "## 統計口徑", "",
        "- Sender、回覆與連續發言：一般聊天；sender 是顯示名稱，不代表已去重的真實人物。",
        "- 時間：僅精確 datetime，保留 JSONL 原始順序及 UTC+08:00；系統事件只另列 date_context 日期數量。",
        "- 附件訊息含有文字加附件，因此 sender 的文字數與附件數可重疊。",
        "- 分位數使用 (n-1)*q 線性插值；時間間隔單位為秒。",
        "- 高頻文字與重複：非空的一般訊息原文精確比較；不做大小寫或空白正規化。",
        "- 抽樣與高頻文字已做規則遮蔽；疑似電話等只是候選，不是已驗證個資。未輸出個資命中原文。", "",
        "## Sender", "",
        "| Sender | 訊息 | 有文字 | 有附件 | 第一則 | 最後一則 |",
        "|---|---:|---:|---:|---|---|"]
    for s in sender_stats:
        report.append(f"| {s['sender'].replace('|', '/')} | {s['message_count']} | {s['text_message_count']} | {s['attachment_message_count']} | {s['first_datetime']} | {s['last_datetime']} |")
    report += ["", "## 時間", "",
        f"範圍：{time_stats['first_datetime']} 至 {time_stats['last_datetime']}。",
        f"共 {len(daily)} 個曆日，{time_stats['active_days']} 天有一般訊息。",
        f"相鄰時間間隔共 {len(gaps)} 組；median={gapstats['median']:.2f} 秒、mean={gapstats['mean']:.2f} 秒、p90={gapstats['p90']:.2f} 秒、p95={gapstats['p95']:.2f} 秒。",
        "包含隔夜與跨日間隔；不可直接解讀為客服回應時間。", "",
        "| 間隔 | 數量 | 比例 |", "|---|---:|---:|"]
    report += [f"| {k} | {v} | {v / len(gaps):.2%} |" for k, v in bins.items()]
    report += ["", "邊界：前五區間左閉右開；1–6 小時包含恰好 6 小時；最後一區間為 >6 小時。", "",
               "### 每日一般訊息數（含零訊息日）", "", "| 日期 | 數量 |", "|---|---:|"]
    report += [f"| {k} | {v} |" for k, v in sorted(daily.items())]
    report += ["", "### 每小時分布（UTC+08:00）", "", "| 小時 | 數量 |", "|---|---:|"]
    report += [f"| {k} | {v} |" for k, v in time_stats["hourly_message_counts"].items()]
    report += ["", "## 回覆與交錯結構", "",
        f"具有 reply_to：{len(reply_rows):,} 則；一般訊息使用率 {len(reply_rows)/len(normal):.2%}，全部資料使用率 {len(reply_rows)/len(rows):.2%}。",
        "直接目標分類：" + json.dumps(dict(target_types), ensure_ascii=False) + "。",
        "Reply chain 長度定義為已解析的父連結邊數，包含 service 根；缺失目標的那條邊不計入。此數字不是對話回合數。",
        "深度分布：" + json.dumps(conversation["reply_chain_depth_edges_distribution"]) + "。",
        "終止類型：" + json.dumps(conversation["chain_termination_counts"]) + "。",
        f"偵測到 {len(cycles)} 個循環起點。",
        f"相鄰 sender 切換 {switches}/{len(gaps)}（{switches/len(gaps):.2%}）；10 分鐘內 A-B-A 交替 {aba} 組。",
        f"固定 5 分鐘視窗中，{len(busy)} 個含至少 3 則訊息，{len(multi)} 個含至少 3 位 sender，其中 {len(multi_roots)} 個跨越不同 reply root。",
        f"小於 5 分鐘的相鄰訊息有 {len(near_pairs)} 組，其中 {near_root_switches} 組切換 reply root（{inter['reply_root_switch_rate_under_5_minutes']:.2%}）。",
        "這些是多人及回覆脈絡交錯的結構證據，不代表同時發話，也不保證每個 root 對應獨立語意對話。",
        f"同 sender 連續至少 2 則的 run 有 {conversation['same_sender_runs']['multi_message_run_count']} 組；最長 {conversation['same_sender_runs']['maximum_run_length']} 則。",
        "連續 run 不設時間門檻，忽略無精確時間的系統事件；完整分布見 conversation_statistics.json。", "",
        "### 最大的 reply root 群組（不是 session）", "",
        "| 根 ID | 根類型 | 是否建立 topic | 一般訊息數 | sender 數 | 跨度天數 | 最大深度 |",
        "|---|---|---|---:|---:|---:|---:|"]
    for r in root_stats[:10]:
        report.append(f"| {r['root_message_id']} | {r['root_type']} | {r['is_topic_creation']} | {r['normal_message_count']} | {r['distinct_senders']} | {r['span_seconds']/86400:.2f} | {r['max_observed_depth_edges']} |")
    report += ["", "## 文字", "",
        f"空文字：{text_stats['empty_text_all_records']} 則。",
        "一般訊息字數（含空文字）：" + json.dumps(text_stats["length_normal_messages_including_empty"], ensure_ascii=False) + "。",
        "一般訊息字數（僅非空）：" + json.dumps(text_stats["length_nonempty_normal_messages"], ensure_ascii=False) + "。",
        f"完全重複非空文字：{len(duplicate)} 組，涉及 {sum(duplicate.values())} 則；扣除每組第一則後有 {sum(n-1 for n in duplicate.values())} 則重複出現。",
        "疑似個資訊息數：" + json.dumps(text_stats["potential_pii_message_counts_all_records"], ensure_ascii=False) + "。",
        "各個資類別可重疊；URL 不一定含個資，電話候選可能是其他數字識別碼。正規表示式無法完整識別自由文字姓名與地址。",
        "最常見 50 個文字的遮蔽版本與次數見 text_statistics.json；排名依遮蔽前的精確原文計算，遮蔽後顯示相同不表示原文相同。", "",
        "## 抽樣", "",
        f"固定 seed={args.seed}，由全部 {len(rows)} 筆均勻抽取 {len(selected)} 個不重複 anchor，保留前後各 3 筆，邊界以實際資料為限。",
        f"共涵蓋 {len(included)} 個不重複資料列，區塊間可重複 context。sample_conversations.txt 中 [ANCHOR] 標示抽中訊息。",
        "時間、訊息 ID、reply_to 與 speaker alias 可供追蹤；附件路徑及系統事件原文不展示。此檔案是人工檢查樣本，不是切分後的對話。", "",
        "## 後續 segmentation 的主要困難", "",
        "reply_to 大量指向建立 topic 的 service 節點，代表長期 thread 歸屬的可能性高；不能將 reply root 直接等同一次客服事件。",
        "同一 root 可跨越多天，短時間內又可切換不同 root 與 sender；只用全域時間間隔容易合併不同脈絡，或拆散被其他訊息插入的延續討論。",
        "低 reply-chain 深度不表示問題已結束。缺失回覆目標、轉傳內回覆與外層回覆的不同作用範圍，以及純附件缺少文字，均限制僅靠目前 metadata 判斷問題起訖。",
        "本次只提供結構證據，尚未選定時間門檻、session 規則或任何角色標籤。", ""]
    (output / "profiling_report.md").write_text("\n".join(report), encoding="utf-8")
    assert hashlib.sha256(args.input.read_bytes()).hexdigest() == original_hash, "Input changed"
    print(json.dumps({"senders": sender_stats, "time_range": [time_stats["first_datetime"], time_stats["last_datetime"]],
        "gaps": gapstats, "gap_bins": bins, "reply_rate": conversation["reply_to_rate_normal_messages"],
        "reply_targets": dict(target_types), "depths": conversation["reply_chain_depth_edges_distribution"],
        "interleaving": inter, "top_roots": root_stats[:3],
        "text": {k: text_stats[k] for k in ["empty_text_all_records", "length_normal_messages_including_empty",
            "exact_duplicate_nonempty_text", "potential_pii_message_counts_all_records"]},
        "input_unchanged": True}, ensure_ascii=True, indent=2))

if __name__ == "__main__":
    main()
