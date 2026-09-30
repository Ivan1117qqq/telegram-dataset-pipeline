"""Merge-only reconstruction from manual role mapping; no semantic/role inference."""
import argparse
from collections import Counter,defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import re
import statistics

def csv_out(path, rows, fields=None):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def quant(a,q):
    if not a:return None
    a=sorted(a);x=(len(a)-1)*q;i=int(x)
    return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
class Union:
    def __init__(self,ids):self.p={i:i for i in ids};self.members={i:{i} for i in ids}
    def find(self,i):
        if self.p[i]!=i:self.p[i]=self.find(self.p[i])
        return self.p[i]
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[b]=a;self.members[a]|=self.members[b]
        return a
def safe(t):
    t=re.sub(r"(?i)(?:https?://|www\.)[^\s<>]+","[URL]",t)
    t=re.sub(r"(?i)[\w.+-]+@[\w.-]+\.[a-z]{2,}","[EMAIL]",t)
    t=re.sub(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])","[IP]",t)
    t=re.sub(r"(?i)\b[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\b","[ID]",t)
    t=re.sub(r"(?i)(\b(?:token|hash|password|secret|api[_ -]?key|account|username|user[_ -]?id)\b\s*[:=\uff1a]\s*)[^\s&<>]+",r"\1[REDACTED]",t)
    t=re.sub(r"((?:帳號|账号|密碼|密码)\s*[:=\uff1a]?\s*)[A-Za-z0-9_.@+-]+",r"\1[ACCOUNT]",t)
    t=re.sub(r"(?i)\b[0-9a-f]{16,}\b","[ID]",t)
    t=re.sub(r"(?<!\w)\+?\d[\d ()-]{6,}\d(?!\w)","[PHONE_OR_ID]",t)
    return t

def main():
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline",type=Path,default=root/"data/segmentation/segmentation_60m.jsonl")
    ap.add_argument("--messages",type=Path,default=root/"data/processed/messages.jsonl")
    ap.add_argument("--mapping",type=Path,default=root/"data/roles/sender_role_mapping.csv")
    ap.add_argument("--output-dir",type=Path,default=root/"data/reconstruction")
    ap.add_argument("--seed",type=int,default=20260914)
    args=ap.parse_args()
    protected=[args.baseline,args.messages,args.mapping]+[root/"scripts"/n for n in
        ("preprocess_chat.py","profile_dataset.py","segment_conversations.py","analyze_single_conversations.py","analyze_sender_roles.py")]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    with args.mapping.open(encoding="utf-8-sig",newline="") as f:mapping=list(csv.DictReader(f))
    roles={r["sender"]:r["role"] for r in mapping}
    if len(roles)!=len(mapping):raise ValueError("Duplicate mapping sender")
    raw=[json.loads(x) for x in args.messages.read_text(encoding="utf-8").splitlines() if x.strip()]
    baseline=[json.loads(x) for x in args.baseline.read_text(encoding="utf-8").splitlines() if x.strip()]
    by={r["message_id"]:r for r in raw};order={r["message_id"]:i for i,r in enumerate(raw)}
    normal=[r for r in raw if r["message_type"]=="message"]
    absent={r["sender"] for r in normal}-set(roles)
    if absent:raise ValueError("Missing sender mapping: "+repr(absent))
    if any(not roles[r["sender"]] for r in normal):raise ValueError("Blank manual role; cannot infer")
    chats=[c for c in baseline if c["candidate_type"]=="chat"]
    systems=[c for c in baseline if c["candidate_type"]!="chat"]
    bc={c["conversation_id"]:c for c in baseline}
    owner={r["message_id"]:c["conversation_id"] for c in baseline for r in c["messages"]}
    assert len(owner)==len(raw) and all(r==by[r["message_id"]] for c in baseline for r in c["messages"])
    times={r["message_id"]:datetime.fromisoformat(r["datetime"]) for r in normal}
    ids=[c["conversation_id"] for c in chats]
    def role(r):return roles[r["sender"]] if r["message_type"]=="message" else "SYSTEM_EVENT"
    def is_unknown(s):return roles[s]=="UNKNOWN"
    def humans(members):
        return {r["sender"] for cid in members for r in bc[cid]["messages"] if r["message_type"]=="message" and role(r)!="BOT"}
    def agents(members):return {s for s in humans(members) if roles[s]=="AGENT"}
    def others(members):return {s for s in humans(members) if roles[s]!="AGENT"}
    def target(r):
        t=by.get(r.get("reply_to"))
        return t if t and t["source_file"]==(r.get("reply_to_source_file") or r["source_file"]) else None
    direct=[];missing=[];system_refs=0
    msg_union=Union([r["message_id"] for r in normal])
    for r in normal:
        if not r.get("reply_to"):continue
        t=target(r)
        if t is None:missing.append(r["message_id"]);continue
        if t["message_type"]!="message":system_refs+=1;continue
        direct.append((t["message_id"],r["message_id"]))
        msg_union.join(t["message_id"],r["message_id"])
    comp_sizes=Counter(msg_union.find(r["message_id"]) for r in normal)
    strong_components={cid:{msg_union.find(r["message_id"]) for r in bc[cid]["messages"]
        if comp_sizes[msg_union.find(r["message_id"])]>1} for cid in ids}
    long_edges=[(a,b) for a,b in direct if (times[b]-times[a]).total_seconds()>86400]
    assert all(owner[a]==owner[b] for a,b in direct), "Baseline broken direct reply requires review"
    topic_groups=defaultdict(list);neighbors={}
    for r in normal:
        t=bc[owner[r["message_id"]]]["topic_root"]
        if t is not None:topic_groups[t].append(r)
    # Actual same-topic chronological message boundaries, not candidate min/max
    # intervals: old replies can make those intervals span months.
    opportunities=[];runs_by_topic={}
    for topic,group in topic_groups.items():
        group.sort(key=lambda r:(times[r["message_id"]],order[r["message_id"]]))
        runs=[]
        for i,r in enumerate(group):
            neighbors[r["message_id"]]=(group[i-1] if i else None,group[i+1] if i+1<len(group) else None)
            cid=owner[r["message_id"]]
            if runs and runs[-1]["cid"]==cid:runs[-1]["rows"].append(r)
            else:runs.append({"cid":cid,"rows":[r]})
        runs_by_topic[topic]=runs
        for ra,rb in zip(runs,runs[1:]):
            a,b=ra["rows"][-1],rb["rows"][0]
            opportunities.append((ra["cid"],rb["cid"],a,b,"same_topic_boundary"))
    # Cross-topic/unknown global boundaries are audited as explicit rejections.
    for a,b in zip(normal,normal[1:]):
        ca,cb=owner[a["message_id"]],owner[b["message_id"]]
        if ca!=cb and (bc[ca]["topic_root"]!=bc[cb]["topic_root"] or bc[ca]["topic_root"] is None):
            opportunities.append((ca,cb,a,b,"global_guardrail_only"))
    def reconstruct(groups,minutes,accepted):
        result=[]
        for members in groups:
            originals=sorted((r for cid in members for r in bc[cid]["messages"]),key=lambda r:order[r["message_id"]])
            cids=sorted(members,key=lambda cid:min(order[r["message_id"]] for r in bc[cid]["messages"]))
            event=all(bc[cid]["candidate_type"]!="chat" for cid in members)
            mids={r["message_id"] for r in originals}
            ag=agents(members);non=others(members)
            links=[e for e in accepted if e["candidate_a"] in members and e["candidate_b"] in members]
            internal=[(a,b) for a,b in direct if a in mids and b in mids]
            cc=set().union(*(strong_components.get(cid,set()) for cid in members))
            # Risk B: same confirmed agent near >=2 distinct non-agent participants.
            agent_contacts=defaultdict(list)
            for a,b in zip(originals,originals[1:]):
                if a["message_type"]!="message" or b["message_type"]!="message":continue
                if (times[b["message_id"]]-times[a["message_id"]]).total_seconds()>600:continue
                if role(a)=="AGENT" and role(b) not in ("AGENT","BOT"):agent_contacts[a["sender"]].append((times[b["message_id"]],b["sender"]))
                if role(b)=="AGENT" and role(a) not in ("AGENT","BOT"):agent_contacts[b["sender"]].append((times[b["message_id"]],a["sender"]))
            concurrent_contacts=False
            for events in agent_contacts.values():
                for i,(time,person) in enumerate(events):
                    people={p for t,p in events[i:] if (t-time).total_seconds()<=600}
                    if len(people)>=2:
                        concurrent_contacts=True
                        break
            flags=[]
            if len(non)>=3:flags.append("A_three_or_more_non_agents")
            if concurrent_contacts:flags.append("B_same_agent_multiple_non_agents_within_10m")
            if len(members)>=4:flags.append("C_four_or_more_source_candidates")
            if len(originals)>20:flags.append("D_over_20_messages")
            if len(cc)>=2:flags.append("E_multiple_independent_reply_components")
            risk="HIGH" if len(flags)>=2 or "D_over_20_messages" in flags else "MEDIUM" if flags else "LOW"
            ts=[times[r["message_id"]] for r in originals if r["message_type"]=="message"]
            messages=[{**r,"role":role(r),"topic_root":bc[owner[r["message_id"]]]["topic_root"],
                       "baseline_candidate_id":owner[r["message_id"]]} for r in originals]
            reason=sorted({v for e in links for v in e["merge_reason"]})
            result.append({"conversation_id":f"agent_{minutes}m_{originals[0]['message_id']}",
                "candidate_type":"system_event" if event else "chat",
                "source_candidate_ids":cids,"topic_root":bc[cids[0]]["topic_root"],
                "start_time":min(ts).isoformat() if ts else None,"end_time":max(ts).isoformat() if ts else None,
                "duration_seconds":(max(ts)-min(ts)).total_seconds() if ts else None,
                "message_count":len(originals),"sender_count":len({r["sender"] for r in originals if r["sender"]}),
                "agent_senders":sorted(ag),"non_agent_senders":sorted(non),
                "contains_agent":bool(ag),"contains_bot":any(role(r)=="BOT" for r in originals),
                "merge_reason":reason,"merge_score":max((e["merge_score"] for e in links),default=0),
                "merge_edges":links,"has_long_distance_reply":any(a in mids and b in mids for a,b in long_edges),
                "independent_reply_component_count":len(cc),"overmerge_risk":risk,"overmerge_risk_flags":flags,
                "risk_predates_reconstruction":len(members)==1 and bool(flags),"messages":messages})
        return sorted(result,key=lambda c:order[c["messages"][0]["message_id"]])
    def metrics(label,cs):
        g=[c for c in cs if c["candidate_type"]=="chat"];n=len(g)
        counts=[c["message_count"] for c in g];dur=[c["duration_seconds"] for c in g]
        owners={r["message_id"]:c["conversation_id"] for c in cs for r in c["messages"]}
        return {"threshold":label,"conversations":n,"singleton_count":counts.count(1),
            "singleton_percent":100*counts.count(1)/n,"message_count_mean":statistics.mean(counts),
            "message_count_median":quant(counts,.5),"message_count_p90":quant(counts,.9),"message_count_max":max(counts),
            "duration_seconds_median":quant(dur,.5),"duration_seconds_p90":quant(dur,.9),"duration_seconds_max":max(dur),
            "multi_sender_percent":100*sum(c["sender_count"]>1 for c in g)/n,
            "contains_agent_percent":100*sum(c["contains_agent"] for c in g)/n,
            "zero_agent_conversations":sum(len(c["agent_senders"])==0 for c in g),
            "one_agent_conversations":sum(len(c["agent_senders"])==1 for c in g),
            "multiple_agent_conversations":sum(len(c["agent_senders"])>1 for c in g),
            "over_20_messages":sum(v>20 for v in counts),"over_50_messages":sum(v>50 for v in counts),
            "over_100_messages":sum(v>100 for v in counts),
            "direct_reply_broken_count":sum(owners[a]!=owners[b] for a,b in direct),
            "risk_medium_count":sum(c["overmerge_risk"]=="MEDIUM" for c in g),
            "risk_high_count":sum(c["overmerge_risk"]=="HIGH" for c in g),
            "risk_medium_newly_merged_count":sum(c["overmerge_risk"]=="MEDIUM" and len(c["source_candidate_ids"])>1 for c in g),
            "risk_high_newly_merged_count":sum(c["overmerge_risk"]=="HIGH" and len(c["source_candidate_ids"])>1 for c in g),
            "system_event_records":len(cs)-n,"preserved_message_count":sum(c["message_count"] for c in cs)}
    baseline_out=reconstruct([{c["conversation_id"]} for c in baseline],0,[])
    comparisons=[metrics("Baseline",baseline_out)]
    singleton_rows=[];summary=[];out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    for minutes in (10,30,60):
        # Triple support is computed from unchanged baseline runs, not recursively inferred roles.
        bridge=defaultdict(set)
        bridge_witness=defaultdict(list)
        for topic,runs in runs_by_topic.items():
            for a,b,c in zip(runs,runs[1:],runs[2:]):
                x,y,z,w=a["rows"][-1],b["rows"][0],b["rows"][-1],c["rows"][0]
                gaps=((times[y["message_id"]]-times[x["message_id"]]).total_seconds(),
                      (times[w["message_id"]]-times[z["message_id"]]).total_seconds())
                if max(gaps)>minutes*60 or any(role(r)=="BOT" for r in (x,y,z,w)):continue
                reasons=set()
                if role(x) not in ("AGENT","BOT") and role(y)==role(z)=="AGENT" and x["sender"]==w["sender"]:
                    reasons={"agent_bridge","shared_non_agent_participant","other_agent_same_other"}
                elif role(x)==role(w)=="AGENT" and role(y) not in ("AGENT","BOT") and y["sender"]==z["sender"]:
                    reasons={"agent_other_agent"}
                if reasons:
                    for ca,cb in ((a["cid"],b["cid"]),(b["cid"],c["cid"])):
                        bridge[frozenset((ca,cb))]|=reasons
                        bridge_witness[frozenset((ca,cb))].append({
                            "candidate_sequence":[a["cid"],b["cid"],c["cid"]],
                            "boundary_message_ids":[r["message_id"] for r in (x,y,z,w)],
                            "gap_seconds":list(gaps),"reasons":sorted(reasons),
                            "shared_non_agent_participants":[x["sender"]] if "other_agent_same_other" in reasons else []})
        proposals=[]
        for ca,cb,a,b,scope in opportunities:
            shared=humans({ca})&humans({cb});shared_non=others({ca})&others({cb})
            gap=(times[b["message_id"]]-times[a["message_id"]]).total_seconds()
            reasons=[];score=0;reject=None
            ta,tb=bc[ca]["topic_root"],bc[cb]["topic_root"]
            if ta!=tb:reject="different_topic"
            elif ta is None:reject="unknown_topic_no_direct_reply"
            elif role(a)=="BOT" or role(b)=="BOT":reject="bot_boundary_no_evidence"
            elif gap>minutes*60:reject="outside_merge_window"
            else:
                support=bridge.get(frozenset((ca,cb)),set())
                if support:reasons=sorted(support);score=90
                elif (role(a)=="AGENT" and role(b) not in ("AGENT","BOT")) or (role(b)=="AGENT" and role(a) not in ("AGENT","BOT")):
                    reasons=["agent_response_pair","agent_to_other" if role(a)=="AGENT" else "other_to_agent"];score=80
                elif any(not is_unknown(s) for s in shared):
                    reasons=["participant_continuity"]
                    if shared_non:reasons.append("shared_non_agent_participant")
                    score=50
                else:reject="unknown_only_or_no_role_participant_evidence"
            proposals.append({"candidate_a":ca,"candidate_b":cb,"topic_root":ta,
                "gap_seconds":gap,"roles_at_boundary":[role(a),role(b)],
                "boundary_message_ids":[a["message_id"],b["message_id"]],
                "shared_participants":sorted(shared),"shared_non_agent_participants":sorted(shared_non),
                "direct_reply_between":False,"merge_reason":reasons,"merge_score":score,
                "decision":"NO_MERGE","rejection_reason":reject,"evidence_scope":scope,
                "bridge_witnesses":bridge_witness.get(frozenset((ca,cb)),[]) if score==90 else []})
        proposals.sort(key=lambda e:(-e["merge_score"],e["gap_seconds"],order[e["boundary_message_ids"][1]]))
        union=Union(ids);accepted=[]
        for e in proposals:
            if e["rejection_reason"]:continue
            ca,cb=e["candidate_a"],e["candidate_b"];ra,rb=union.find(ca),union.find(cb)
            if ra==rb:e["rejection_reason"]="already_connected";continue
            ma,mb=union.members[ra],union.members[rb]
            oa,ob=others(ma),others(mb)
            # Recheck at COMPONENT level so a shared agent cannot transitively join
            # disjoint non-agent cases after individually plausible merges.
            if oa and ob and not oa&ob:
                e["rejection_reason"]="different_non_agent_participants_component_guard";continue
            sa=set().union(*(strong_components[i] for i in ma))
            sb=set().union(*(strong_components[i] for i in mb))
            if sa and sb and not sa&sb and not (oa&ob and (agents(ma)|agents(mb)) and e["merge_score"]>=80):
                e["rejection_reason"]="independent_reply_structures_without_strong_shared_context";continue
            e["decision"]="MERGE";e["rejection_reason"]=None
            e["component_shared_non_agent_participants"]=sorted(oa&ob)
            union.join(ra,rb);accepted.append(dict(e))
        groups=[union.members[i] for i in ids if union.find(i)==i]+[{c["conversation_id"]} for c in systems]
        result=reconstruct(groups,minutes,accepted)
        own={r["message_id"]:c for c in result for r in c["messages"]}
        assert len(own)==len(raw) and sum(c["message_count"] for c in result)==len(raw)
        for c in baseline:assert len({own[r["message_id"]]["conversation_id"] for r in c["messages"]})==1
        for c in result:
            assert len({bc[cid]["topic_root"] for cid in c["source_candidate_ids"]})==1
            for r in c["messages"]:assert all(r[k]==v for k,v in by[r["message_id"]].items())
        assert all(own[a]["conversation_id"]==own[b]["conversation_id"] for a,b in direct)
        with (out/f"agent_aware_{minutes}m.jsonl").open("w",encoding="utf-8",newline="\n") as f:
            for c in result:f.write(json.dumps(c,ensure_ascii=False)+"\n")
        # A uniform schema includes dynamic component evidence for rejected edges too.
        for e in proposals:e.setdefault("component_shared_non_agent_participants",[])
        csv_out(out/f"merge_audit_{minutes}m.csv",proposals)
        stat=metrics(f"{minutes}m",result);comparisons.append(stat)
        local=[]
        for c in chats:
            if c["message_count"]!=1:continue
            r=c["messages"][0];new=own[r["message_id"]];prev,nxt=neighbors.get(r["message_id"],(None,None))
            incident=[e for e in new["merge_edges"] if c["conversation_id"] in (e["candidate_a"],e["candidate_b"])]
            row={"threshold_minutes":minutes,"message_id":r["message_id"],"baseline_conversation_id":c["conversation_id"],
                "new_conversation_id":new["conversation_id"],"was_merged":len(new["source_candidate_ids"])>1,
                "new_message_count":new["message_count"],"merge_reason":sorted({x for e in incident for x in e["merge_reason"]}),
                "conversation_merge_reason":new["merge_reason"],"agent_involved":new["contains_agent"],
                "shared_non_agent_participant":sorted({s for e in incident for s in e["component_shared_non_agent_participants"]}|{s for e in incident for s in e["shared_non_agent_participants"]}|{s for e in incident for w in e["bridge_witnesses"] for s in w["shared_non_agent_participants"]}),
                "gap_previous":(times[r["message_id"]]-times[prev["message_id"]]).total_seconds() if prev else None,
                "gap_next":(times[nxt["message_id"]]-times[r["message_id"]]).total_seconds() if nxt else None}
            local.append(row);singleton_rows.append(row)
        singles_merged={r["new_conversation_id"] for r in local if r["was_merged"]}
        chat=[c for c in result if c["candidate_type"]=="chat"]
        pools={"merged_baseline_singleton":[c for c in chat if c["conversation_id"] in singles_merged],
            "other_to_agent":[c for c in chat if "other_to_agent" in c["merge_reason"] or "other_agent_same_other" in c["merge_reason"]],
            "agent_to_other":[c for c in chat if "agent_to_other" in c["merge_reason"] or "other_agent_same_other" in c["merge_reason"]],
            "other_agent_same_other":[c for c in chat if "other_agent_same_other" in c["merge_reason"]],
            "multiple_agents":[c for c in chat if len(c["agent_senders"])>1],
            "multiple_non_agents":[c for c in chat if len(c["non_agent_senders"])>1],
            "high_risk":[c for c in chat if c["overmerge_risk"]=="HIGH"],
            "unmerged_singleton":[c for c in chat if c["message_count"]==1],
            "long_distance_reply":[c for c in chat if c["has_long_distance_reply"]]}
        rng=random.Random(args.seed+minutes);selected={};coverage={}
        # Scarce categories first, so available examples cannot be crowded out.
        for label,pool in sorted(pools.items(),key=lambda kv:len(kv[1])):
            available=[c for c in pool if c["conversation_id"] not in selected]
            picks=rng.sample(available,min(5,len(available),60-len(selected)))
            selected.update((c["conversation_id"],c) for c in picks)
        remaining=[c for c in chat if c["conversation_id"] not in selected]
        selected.update((c["conversation_id"],c) for c in rng.sample(remaining,min(60-len(selected),len(remaining))))
        for label,pool in pools.items():coverage[label]={"available":len(pool),"selected":sum(c["conversation_id"] in selected for c in pool)}
        review=[f"Agent-aware {minutes}m; 60 stratified review candidates; roles copied from manual CSV.",
            "Unavailable strata are explicitly reported, not fabricated. Risk is heuristic, not truth.",
            "Baseline candidate intervals may interleave. Boundary blocks list actual accepted graph edges; chronological switches without an edge are labelled accordingly.",
            "Text masks common identifiers; source JSONL remains complete. Pattern redaction is not comprehensive anonymization.",
            "Coverage: "+json.dumps(coverage),""] 
        for c in sorted(selected.values(),key=lambda c:order[c["messages"][0]["message_id"]]):
            review+=["="*64,"Conversation: "+c["conversation_id"],"Source candidates: "+json.dumps(c["source_candidate_ids"]),
                "Merge reason: "+json.dumps(c["merge_reason"]),"Overmerge risk: "+c["overmerge_risk"]+" "+json.dumps(c["overmerge_risk_flags"]),
                "Topic: "+str(c["topic_root"]),"Messages: "+str(c["message_count"]),"Agents: "+", ".join(c["agent_senders"]),
                "Other participants: "+", ".join(c["non_agent_senders"]),"Long-distance reply: "+str(c["has_long_distance_reply"]),"="*64]
            for e in c["merge_edges"]:
                review+=["---------- MERGED BOUNDARY ----------",e["candidate_a"]+" -> "+e["candidate_b"],
                    "reason: "+json.dumps(e["merge_reason"]),"gap: "+str(e["gap_seconds"])+" seconds",
                    "boundary message IDs: "+json.dumps(e["boundary_message_ids"])]
            previous_cid=None
            for r in c["messages"]:
                cid=r["baseline_candidate_id"]
                if cid!=previous_cid:review.append("--- BASELINE CANDIDATE: "+cid+" ---")
                review += [f"[{r['datetime']}] [{r['role']}] {r['sender']} (message{r['message_id']})",safe(r["text"]) or "[no text]"]
                if r["reply_to"]:review.append("reply_to: message"+r["reply_to"])
                if r.get("attachment_type"):review.append("[attachment: "+r["attachment_type"]+"; path omitted]")
                review.append("");previous_cid=cid
        (out/f"review_agent_{minutes}m.txt").write_text("\n".join(review),encoding="utf-8")
        summary.append({"threshold":minutes,"accepted_merge_edges":len(accepted),
            "reason_edge_counts":dict(Counter(v for e in accepted for v in e["merge_reason"])),
            "rejection_counts":dict(Counter(e["rejection_reason"] for e in proposals if e["decision"]=="NO_MERGE")),
            "baseline_singletons_merged":sum(r["was_merged"] for r in local),
            "baseline_singletons_still_single":sum(not r["was_merged"] for r in local),
            "merged_singleton_destination_size_distribution":dict(sorted(Counter(r["new_message_count"] for r in local if r["was_merged"]).items())),
            "review_coverage":coverage,"review_ids":list(selected)})
    csv_out(out/"singleton_reconstruction.csv",singleton_rows)
    csv_out(out/"reconstruction_comparison.csv",comparisons)
    report=["# Step 6 Agent-aware Conversation Reconstruction","",
        "人工 mapping 原值直接使用，不讀名稱或 notes 推測角色。此版只有 merge，未拆 baseline，也未選 final dataset。","",
        "## Mapping 與證據覆蓋","",
        "Role sender counts："+json.dumps(dict(Counter(roles[r["sender"]] for r in mapping)),ensure_ascii=False),
        "Role message counts："+json.dumps(dict(Counter(role(r) for r in normal)),ensure_ascii=False),
        "非標準 role 保留原值，作 non-BOT non-agent participant；UNKNOWN 可被已確認 AGENT 的回應證據帶入，但 UNKNOWN-only continuity 不足以合併。",
        "BOT 只依 CSV 的 BOT 識別，不能依名字或先前行為分析自行加標籤。system event 全部獨立保留。","",
        "## 可重現規則","",
        "- Node 是 baseline candidate。Edge 機會來自同 topic 訊息時間序列中相鄰的不同 candidate run；同秒按來源順序。",
        "- 使用實際訊息邊界 gap，不使用跨月 candidate 的 min/max 區間距離。BOT 邊界直接拒絕，不跳過 BOT 尋找替代 bridge。",
        "- OTHER→AGENT→SAME OTHER 與 AGENT→SAME OTHER→AGENT 的三段 baseline run，兩段 gap 都需符合 window，score=90；附 bridge reason。",
        "- AGENT↔non-BOT OTHER 邊界，score=80；UNKNOWN 在此有 confirmed AGENT 支持，不單獨作強證據。",
        "- 共享非 BOT 且非 UNKNOWN participant 的同 topic 邊界，score=50。共用 AGENT 仍需通過不同 non-agent guard。",
        "- score 是規則優先序而非概率。先高分、再短 gap、再來源順序；conversation score 為已接受 edge 最高分，所有 edge 明細完整保留。",
        "- 每次 union 都對已合併 component 重查 non-agent 集合：雙方非空且無交集則拒絕，避免共用 AGENT 造成傳遞式混案。",
        "- 雙方已有獨立一般訊息 reply component 時，除非共享 non-agent 且有 AGENT 強往返證據，否則拒絕。",
        "- 不跨 topic；unknown/null topic 不做弱合併。一般訊息 direct reply 先驗證 baseline 完整，若 baseline 已破壞會中止；本資料沒有跨 baseline direct edge。",
        "- audit 記錄同 topic 邊界機會，另記錄全域相鄰的跨 topic/unknown 拒絕；不是所有 node pair 的笛卡兒積。already_connected 代表該 pair 已被其他 edge 接通，不再重複 union。",
        "- baseline 全部訊息原欄位保留，另加 role/topic_root/baseline_candidate_id。50 個 system_event 保留且不計入 conversation 比較分母。",
        "- 84 條 >24h direct reply 不拆，只 flag；不存在的回覆目標不可計算。","",
        "## Overmerge risk 定義","",
        "A: >=3 non-agent；B: 同一 AGENT 在任一10分鐘視窗內，透過本組 <=10m 相鄰互動接觸至少2不同 non-agent；C: >=4來源 candidate；D: >20訊息；E: >=2獨立非單點 direct-reply component。",
        "D 或至少2旗標為 HIGH；1旗標為 MEDIUM；0為 LOW。這是 heuristic。baseline 原有風險會保留，CSV 另列 newly_merged 風險數。",
        "UNKNOWN 列入 non-agent participant 計數，BOT 排除；B 為結構性 proxy，並不宣稱是不同案件。","",
        "## 比較","","| Threshold | Conversations | Singleton | Singleton % | Median Msg | P90 Msg | Max Msg | Multi-sender % | >20 | Broken reply |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in comparisons:
        report.append(f"| {s['threshold']} | {s['conversations']} | {s['singleton_count']} | {s['singleton_percent']:.2f} | {s['message_count_median']} | {s['message_count_p90']} | {s['message_count_max']} | {s['multi_sender_percent']:.2f} | {s['over_20_messages']} | {s['direct_reply_broken_count']} |")
    report+=["","完整指標見 reconstruction_comparison.csv。Singleton trace 每個 baseline singleton 每個 window 一列，共 942×3；was_merged 指其 baseline node 參與實際 union。","",
        "## Merge 與 Review 覆蓋","```json",json.dumps(summary,ensure_ascii=False,indent=2),"```","",
        "Review 每組60個不同 chat candidates，稀少類別先抽，每類最多補入5個，再隨機補足。不存在的 bridge、多AGENT 或 HIGH 例子會列 available=0，不捏造。",
        "Review 先顯示所有接受的 graph merge boundary（原因、gap、訊息ID），再按時間顯示所有訊息及原 baseline candidate 切換，避免把 interleaving 的視覺切換誤認為合併邊。",
        "本 mapping 的已確認 AGENT 覆蓋率有限。比較改善幅度時需同時考慮 UNKNOWN-only 的大量訊息不具足夠 role evidence；不能只為降低 singleton 放寬成時間合併。","",
        "## 完整性","```json",json.dumps(hashes,ensure_ascii=False,indent=2),"```",""]
    report += ["## 本次結果判讀", "",
        f"人工確認 AGENT 共{sum(role(r)=='AGENT' for r in normal)}則訊息，占{100*sum(role(r)=='AGENT' for r in normal)/len(normal):.2f}%；UNKNOWN-only互動不足以支持本次自動role-aware合併。",
        "10/30/60分鐘合併的原baseline singleton數："+str([s["baseline_singletons_merged"] for s in summary])+"。",
        "60分鐘分母（conversation數）變小而singleton數不變，因此singleton比例可高於30分鐘，並非重新拆開訊息。",
        "就本次可觀察的singleton改善與HIGH風險數比較，30分鐘較值得優先人工review；這不是最終門檻選擇或語意品質保證。",
        "baseline既有風險與本次新合併的風險分開列出；風險旗標非ground truth，不能用零broken reply證明無overmerge。",
        "各組 >20/>50/>100 及 broken direct reply 詳見比較表與 CSV；所有輸出均完成 baseline 不拆分與訊息保留驗證。", ""]
    (out/"reconstruction_report.md").write_text("\n".join(report),encoding="utf-8")
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    print(json.dumps({"mapping":dict(Counter(roles.values())),"role_message_counts":dict(Counter(role(r) for r in normal)),
        "comparison":comparisons,"summary":[{k:v for k,v in s.items() if k!="review_ids"} for s in summary],
        "input_unchanged":True},ensure_ascii=True,indent=2))

if __name__=="__main__":main()
