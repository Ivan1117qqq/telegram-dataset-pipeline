"""Preprocess Telegram HTML using only Python's standard library.

Run: python scripts/preprocess_chat.py
Inputs are read as UTF-8, in natural filename order. No attachment is opened.
Date separators are excluded; system events are retained with message_type=service.
Unknown structures abort before replacing output. See preprocessing_report.json.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

VOID = set("area base br col embed hr img input link meta param source track wbr".split())

class Node:
    def __init__(self, tag, attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []
    @property
    def classes(self):
        return set(self.attrs.get("class", "").split())
    def walk(self):
        yield self
        for child in self.elements():
            yield from child.walk()
    def elements(self):
        return [c for c in self.children if isinstance(c, Node)]
    def direct(self, cls):
        return [c for c in self.elements() if cls in c.classes]

class Tree(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("document")
        self.stack = [self.root]
    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in VOID:
            self.stack.append(node)
    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)
    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if len(self.stack) == 1 or self.stack[-1].tag != tag:
            raise ValueError(f"Unexpected closing tag: {tag}, line {self.getpos()[0]}")
        self.stack.pop()
    def handle_data(self, data):
        self.stack[-1].children.append(data)

def content(node, skip_date=False):
    if node.tag in {"script", "style"} or (skip_date and "date" in node.classes):
        return ""
    if node.tag == "br":
        return "\n"
    return "".join(c if isinstance(c, str) else content(c, skip_date)
                   for c in node.children)

def one(nodes, label, optional=False):
    if len(nodes) == 1:
        return nodes[0]
    if not nodes and optional:
        return None
    raise ValueError(f"{label}: expected one element, got {len(nodes)}")

def timestamp(node):
    if node is None:
        return None
    return datetime.strptime(node.attrs["title"], "%d.%m.%Y %H:%M:%S UTC%z").isoformat()

MONTHS = {m: i for i, m in enumerate(
    "January February March April May June July August September October November December".split(), 1)}
def separator_date(value):
    match = re.fullmatch(r"(\d{1,2}) (" + "|".join(MONTHS) + r") (\d{4})", value)
    if not match:
        return None
    day, month, year = match.groups()
    return datetime(int(year), MONTHS[month], int(day)).date().isoformat()

def reply_info(body, source):
    block = one(body.direct("reply_to"), "reply_to", optional=True)
    if block is None:
        return None
    link = one([n for n in block.walk() if n.tag == "a"], "reply anchor")
    href = link.attrs.get("href", "")
    url = urlsplit(href)
    page = unquote(url.path)
    match = re.fullmatch(r"(?:go_to_)?message(-?\d+)", url.fragment)
    if (not match or url.scheme or url.netloc or url.query or
            (page and not re.fullmatch(r"messages(?:\d+|\(\d+\))?\.html", page))):
        raise ValueError(f"Unknown reply link: {href}")
    return {"message_id": match[1], "source_file": page or source, "href": href}

def attachments(body, export_dir=None):
    result = []
    for wrap in (n for n in body.walk() if "media_wrap" in n.classes):
        items = wrap.elements()
        if not items:
            raise ValueError("Empty media_wrap")
        for item in items:
            classes = item.classes
            titles = [content(n).strip() for n in item.walk() if "title" in n.classes]
            title = titles[0] if titles else None
            if "media_file" in classes:
                kind = "file"
            elif "media_voice_message" in classes or "voice_message_wrap" in classes:
                kind = "voice"
            elif "media_audio_file" in classes or "audio_file_wrap" in classes:
                kind = "audio"
            elif "sticker_wrap" in classes or ("media_photo" in classes and title == "Sticker"):
                kind = "sticker"
            elif "animated_wrap" in classes or ("media_video" in classes and title == "Animation"):
                kind = "animation"
            elif "photo_wrap" in classes or ("media_photo" in classes and title == "Photo"):
                kind = "photo"
            elif "video_file_wrap" in classes or ("media_video" in classes and title in {"Video file", "Video"}):
                kind = "video"
            else:
                raise ValueError(f"Unknown media structure: {sorted(classes)}, title={title!r}")
            path = item.attrs.get("href")
            not_exported = "Not included, change data exporting settings to download." in content(item)
            if not path and not not_exported:
                raise ValueError("Media has no original href")
            exists = False if not path else None
            if path and export_dir is not None:
                url = urlsplit(path)
                if not url.scheme and not url.netloc:
                    base = Path(export_dir).resolve()
                    local = (base / unquote(url.path)).resolve()
                    if not local.is_relative_to(base):
                        raise ValueError("Attachment path escapes export directory")
                    exists = local.is_file()
            result.append({"type": kind, "path": path, "title": title,
                           "html_class": item.attrs.get("class"),
                           "reference_exists": bool(path), "local_exists": exists,
                           "availability": "not_exported" if not_exported else
                               "present" if exists else "missing" if exists is False else "unverified"})
    return result

def validate_body(body):
    allowed = {"date", "from_name", "reply_to", "text", "reactions", "media_wrap", "forwarded"}
    for child in body.elements():
        if not child.classes & allowed:
            raise ValueError(f"Unknown body child: {child.tag} {child.attrs}")
        if "forwarded" in child.classes and "body" in child.classes:
            validate_body(child)

def parse_file(path, report, state=None):
    # State is explicit and belongs to ONE export; service/date rows reset it.
    state = {} if state is None else state
    tree = Tree()
    tree.feed(path.read_text(encoding="utf-8-sig"))
    tree.close()
    if len(tree.stack) != 1:
        shell = [(n.tag, n.classes) for n in tree.stack[1:]]
        expected = [("html", set()), ("body", set()), ("div", {"page_wrap"}),
                    ("div", {"page_body", "chat_page"}), ("div", {"history"})]
        if shell != expected:
            raise ValueError("Unclosed HTML tags inside message or unknown page structure")
        report["warnings"].append({"source_file": path.name,
            "reason": "Missing closing page-shell tags; all message elements are closed"})
    history = one([n for n in tree.root.walk() if "history" in n.classes], "history")
    all_messages = [n for n in tree.root.walk() if "message" in n.classes]
    messages = history.direct("message")
    if len(all_messages) != len(messages):
        raise ValueError("Message outside direct history children")
    rows, previous_sender, date_context = [], state.get("sender"), state.get("date_context")
    stats = Counter(html_message_blocks=len(messages))
    for message in messages:
        html_id = message.attrs.get("id", "")
        try:
            match = re.fullmatch(r"message(-?\d+)", html_id)
            if not match:
                raise ValueError("Missing/invalid message ID")
            body = one(message.direct("body"), "message body")
            row = dict(message_id=match[1], datetime=None, sender=None, text="",
                       reply_to=None, attachment_type=None, attachment_path=None,
                       source_file=path.name)
            if message.classes == {"message", "service"}:
                value = content(body).strip()
                date = separator_date(value)
                if date:
                    date_context = date
                    stats["date_separators"] += 1
                    previous_sender = None
                    continue
                # Service prose does not provide reliable structured actor/time fields.
                if any(n.tag not in {"div", "a", "span", "br", "b", "strong", "i", "em"}
                       for n in body.walk()):
                    raise ValueError("Unknown service content")
                row.update(message_type="service", text=value, date_context=date_context)
                previous_sender = None
            elif ({"message", "default", "clearfix"} <= message.classes
                  and message.classes <= {"message", "default", "clearfix", "joined"}):
                validate_body(body)
                sender = one(body.direct("from_name"), "sender", optional=True)
                inherited = sender is None and "joined" in message.classes
                row["sender"] = content(sender).strip() if sender else previous_sender if inherited else None
                if not row["sender"]:
                    raise ValueError("Missing sender without an available joined predecessor")
                previous_sender = row["sender"]
                row["datetime"] = timestamp(one(body.direct("date"), "message date"))
                row["message_type"] = "message"
                row["sender_inherited"] = inherited
                scopes = [body] + [n for n in body.walk()
                                   if n is not body and {"forwarded", "body"} <= n.classes]
                texts = [content(n).strip() for scope in scopes for n in scope.direct("text")]
                row["text"] = "\n".join(t for t in texts if t)
                row["own_text"] = "\n".join(content(n).strip() for n in body.direct("text"))
                reply = reply_info(body, path.name)
                row["reply_to"] = reply["message_id"] if reply else None
                row["reply_to_source_file"] = reply["source_file"] if reply else None
                forwarded = []
                for scope in scopes[1:]:
                    name = one(scope.direct("from_name"), "forwarded sender", optional=True)
                    dates = [n for n in name.walk() if "date" in n.classes] if name else []
                    forwarded.append({
                        "sender": content(name, skip_date=True).strip() if name else None,
                        "datetime": timestamp(one(dates, "forwarded date", optional=True)),
                        "reply": reply_info(scope, path.name),
                        "text": "\n".join(content(n).strip() for n in scope.direct("text"))})
                row["forwarded"] = forwarded
                media = attachments(body, path.parent)
                row["attachments"] = media
                if media:
                    row["attachment_type"] = media[0]["type"] if len(media) == 1 else "multiple"
                    row["attachment_path"] = media[0]["path"] if len(media) == 1 else None
                if not row["text"] and not media:
                    report["warnings"].append({"source_file": path.name, "message_id": match[1],
                                               "reason": "No text or attachment"})
                stats["joined_sender_inherited"] += inherited
                stats["forwarded_messages"] += bool(forwarded)
            else:
                raise ValueError(f"Unknown message classes: {sorted(message.classes)}")
            rows.append(row)
        except (ValueError, KeyError) as exc:
            previous_sender = None
            report["errors"].append({"source_file": path.name, "html_id": html_id, "reason": str(exc)})
    stats["output_records"] = len(rows)
    stats["normal_messages"] = sum(r["message_type"] == "message" for r in rows)
    stats["service_messages"] = len(rows) - stats["normal_messages"]
    report["files"][path.name] = dict(stats)
    state.update(sender=previous_sender, date_context=date_context)
    return rows

def page_order(path):
    """Order one export's pages by first exact timestamp, then natural page name.

    Overlapping/duplicate exports are never silently deduplicated.
    """
    raw = path.read_text(encoding="utf-8-sig")
    match = re.search(r'title="(\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2} UTC[+-]\d{2}:\d{2})"', raw)
    stamp = datetime.strptime(match[1], "%d.%m.%Y %H:%M:%S UTC%z").timestamp() if match else float("-inf")
    number = re.search(r"\d+", path.stem)
    return stamp, 0 if path.name == "messages.html" else 1, int(number[0]) if number else 0, path.name

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=root)
    parser.add_argument("--output", type=Path, default=root / "data/processed/messages.jsonl")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to((root / "data/raw").resolve()):
        parser.error("data/raw is read-only; choose an output outside it")
    paths = sorted(args.input_dir.glob("messages*.html"), key=page_order)
    if not paths:
        parser.error("No messages*.html found. Pass ONE export folder; independent datasets must be processed separately.")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = {"files": {}, "errors": [], "warnings": [], "source_sha256": hashes}
    rows = []
    state = {}
    for path in paths:
        try:
            rows.extend(parse_file(path, report, state))
        except (ValueError, UnicodeError) as exc:
            report["errors"].append({"source_file": path.name, "reason": str(exc)})
    seen = set()
    for row in rows:
        if row["message_id"] in seen:
            report["errors"].append({"message_id": row["message_id"], "reason": "Duplicate message ID"})
        seen.add(row["message_id"])
    keys = {(r["source_file"], r["message_id"]) for r in rows}
    for row in rows:
        references = []
        if row["reply_to"]:
            references.append((row["reply_to_source_file"], row["reply_to"], "reply"))
        for forwarded in row.get("forwarded", []):
            if forwarded["reply"]:
                ref = forwarded["reply"]
                references.append((ref["source_file"], ref["message_id"], "forwarded_reply"))
        for source, target, scope in references:
            if (source, target) not in keys:
                report["warnings"].append({"source_file": row["source_file"],
                    "message_id": row["message_id"], "reason": "Reply target absent from export",
                    "target_source_file": source, "target_id": target, "scope": scope})
    normal = [r for r in rows if r["message_type"] == "message"]
    if any(datetime.fromisoformat(b["datetime"]) < datetime.fromisoformat(a["datetime"])
           for a, b in zip(normal, normal[1:])):
        report["errors"].append({"reason": "Non-chronological export pages; do not combine independent or overlapping exports"})
    report["summary"] = {
        "total_records": len(rows), "normal_messages": len(normal),
        "service_messages": len(rows) - len(normal),
        "text_records_including_service": sum(bool(r["text"]) for r in rows),
        "normal_text_messages": sum(bool(r["text"]) for r in normal),
        "attachment_only_messages": sum(not r["text"] and bool(r["attachments"]) for r in normal),
        "distinct_sender_names": len({r["sender"] for r in normal}),
        "attachment_types": dict(Counter(a["type"] for r in normal for a in r["attachments"])),
        "parse_errors": len(report["errors"]), "warnings": len(report["warnings"])}
    report["sources_unchanged"] = all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[p.name] for p in paths)
    if not report["sources_unchanged"]:
        raise RuntimeError("Source changed during preprocessing")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.output.with_name("preprocessing_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=True, indent=2))
    if report["errors"]:
        print("Unknown/invalid structures: see preprocessing_report.json; JSONL was not written.")
        return 1
    temporary = args.output.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
