"""Synthetic-only Telegram fixtures. No company messages in tests."""
import contextlib,io,json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import preprocess_chat as parser
def msg(mid,body,sender="Alice",joined=False):
 name=f'<div class="from_name">{sender}</div>' if sender is not None else ""
 return f'<div class="message default clearfix{" joined" if joined else ""}" id="message{mid}"><div class="body"><div class="date details" title="19.06.2025 19:02:{int(mid)%60:02d} UTC+08:00">19:02</div>{name}{body}</div></div>'
def document(body):return '<html><body><div class="history">'+body+'</div></body></html>'
def report():return {"files":{},"errors":[],"warnings":[]}
class ParserTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def parse(self,body,name="messages.html",state=None):
  p=self.root/name;p.write_text(document(body),encoding="utf-8");r=report()
  rows=parser.parse_file(p,r) if state is None else parser.parse_file(p,r,state)
  return rows,r
 def test_joined_sender(self):
  rows,r=self.parse(msg(1,'<div class="text">one</div>')+msg(2,'<div class="text">two</div>',None,True))
  self.assertFalse(r["errors"]);self.assertEqual(rows[1]["sender"],"Alice")
 def test_missing_sender_without_joined_is_error(self):
  rows,r=self.parse(msg(1,'<div class="text">x</div>',None));self.assertTrue(r["errors"])
 def test_service_resets_sender(self):
  rows,r=self.parse(msg(1,"")+'<div class="message service" id="message-2"><div class="body details">Alice invited Bob</div></div>'+msg(2,"",None,True))
  self.assertTrue(r["errors"]);self.assertIsNone(rows[1]["sender"])
 def test_date_separator_resets_sender(self):
  rows,r=self.parse(msg(1,"")+'<div class="message service" id="message-1"><div class="body details">20 June 2025</div></div>'+msg(2,"",None,True))
  self.assertTrue(r["errors"]);self.assertEqual(r["files"]["messages.html"]["date_separators"],1)
 def test_utf8_multiline_emoji_inline_mention(self):
  rows,r=self.parse(msg(1,'<div class="text">中文😀<br><a onclick="ShowMentionName()">小明</a> &amp; <b>您好</b></div>'))
  self.assertEqual(rows[0]["text"],"中文😀\n小明 & 您好")
  self.assertEqual(rows[0]["datetime"],"2025-06-19T19:02:01+08:00")
 def test_cross_file_reply(self):
  rows,r=self.parse(msg(1,'<div class="reply_to"><a href="messages2.html#go_to_message22">reply</a></div>'))
  self.assertEqual((rows[0]["reply_to"],rows[0]["reply_to_source_file"]),("22","messages2.html"))
 def test_parenthesized_reply_filename(self):
  rows,r=self.parse(msg(1,'<div class="reply_to"><a href="messages(1).html#go_to_message22">reply</a></div>'))
  self.assertFalse(r["errors"]);self.assertEqual(rows[0]["reply_to_source_file"],"messages(1).html")
 def test_forwarded_sender_time_and_separate_text(self):
  body='<div class="text">Outer note</div><div class="forwarded body"><div class="from_name">Merchant<span class="date" title="18.06.2025 01:02:03 UTC+08:00">old date</span></div><div class="text">Forwarded detail</div></div>'
  rows,r=self.parse(msg(1,body))
  self.assertEqual(rows[0]["sender"],"Alice")
  f=rows[0]["forwarded"][0];self.assertEqual(f["sender"],"Merchant");self.assertEqual(f["datetime"],"2025-06-18T01:02:03+08:00")
  self.assertEqual(f.get("text"),"Forwarded detail")
 def test_file_exists_metadata(self):
  (self.root/"files").mkdir();(self.root/"files/demo.pdf").write_bytes(b"synthetic")
  rows,r=self.parse(msg(1,'<div class="media_wrap"><a class="media_file" href="files/demo.pdf"><div class="title">demo.pdf</div></a></div>'))
  self.assertTrue(rows[0]["attachments"][0].get("local_exists"))
 def test_missing_file_metadata(self):
  rows,r=self.parse(msg(1,'<div class="media_wrap"><a class="media_file" href="files/missing.pdf"><div class="title">missing.pdf</div></a></div>'))
  a=rows[0]["attachments"][0];self.assertTrue(a.get("reference_exists"));self.assertIs(a.get("local_exists"),False)
 def test_not_exported_photo(self):
  rows,r=self.parse(msg(1,'<div class="media_wrap"><div class="media_photo"><div class="title">Photo</div><div class="description">Not included, change data exporting settings to download.</div></div></div>'))
  self.assertFalse(r["errors"]);self.assertEqual(rows[0]["attachments"][0]["availability"],"not_exported")
 def test_voice(self):
  rows,r=self.parse(msg(1,'<div class="media_wrap"><a class="media_voice_message" href="voice_messages/test.ogg"><div class="title">Voice message</div></a></div>'))
  self.assertFalse(r["errors"]);self.assertEqual(rows[0]["attachment_type"],"voice")
 def test_cross_page_joined(self):
  state={};self.parse(msg(1,'<div class="text">first</div>'),state=state)
  rows,r=self.parse(msg(2,'<div class="text">continued</div>',None,True),"messages2.html",state)
  self.assertFalse(r["errors"]);self.assertEqual(rows[0]["sender"],"Alice")
 def test_unknown_structure_is_not_guessed(self):
  rows,r=self.parse(msg(1,'<div class="unrecognized">??</div>'));self.assertTrue(r["errors"])
 def test_multi_html_cli_chronology(self):
  (self.root/"messages.html").write_text(document(msg(1,"first")),encoding="utf-8")
  (self.root/"messages(1).html").write_text(document(msg(2,"second")),encoding="utf-8")
  out=self.root/"output.jsonl"
  result=subprocess.run([sys.executable,str(ROOT/"scripts/preprocess_chat.py"),"--input-dir",str(self.root),"--output",str(out)],capture_output=True)
  self.assertEqual(result.returncode,0,result.stderr)
  self.assertEqual([json.loads(x)["message_id"] for x in out.read_text(encoding="utf-8").splitlines()],["1","2"])
 def test_duplicate_message_id_fails(self):
  for name in ("messages.html","messages2.html"):(self.root/name).write_text(document(msg(1,"")),encoding="utf-8")
  out=self.root/"out.jsonl";r=subprocess.run([sys.executable,str(ROOT/"scripts/preprocess_chat.py"),"--input-dir",str(self.root),"--output",str(out)],capture_output=True)
  self.assertNotEqual(r.returncode,0);self.assertFalse(out.exists())
 def test_profile_url_after_chinese(self):
  import profile_dataset
  self.assertIsNotNone(profile_dataset.URL.search('連結https://example.test/?id=synthetic'))
 def test_sender_analysis_accepts_two_senders(self):
  records=[dict(message_id=str(i),source_file='messages.html',message_type='message',
    sender=sender,datetime=f'2025-06-19T19:02:0{i}+08:00',text='synthetic',
    reply_to=None,attachments=[],attachment_type=None,forwarded=[]) for i,sender in [(1,'Alice'),(2,'Bob')]]
  p=self.root/'input.jsonl';p.write_text('\n'.join(json.dumps(r) for r in records),encoding='utf-8')
  result=subprocess.run([sys.executable,str(ROOT/'scripts/analyze_sender_roles.py'),'--input',str(p),'--output-dir',str(self.root/'roles')],capture_output=True)
  self.assertEqual(result.returncode,0,result.stderr)
  import csv
  with (self.root/'roles/sender_reply_matrix.csv').open(encoding='utf-8-sig') as f:
   self.assertEqual(len(list(csv.DictReader(f))),2)
 def test_segmentation_keeps_five_hour_reply(self):
  records=[dict(message_id=str(i),source_file='messages.html',message_type='message',
    sender=sender,datetime=f'2025-06-19T{hour}:00:00+08:00',text='synthetic',
    reply_to='1' if i==2 else None,reply_to_source_file='messages.html' if i==2 else None,
    attachments=[],attachment_type=None,forwarded=[]) for i,sender,hour in [(1,'Alice','10'),(2,'Bob','15')]]
  p=self.root/'input.jsonl';p.write_text('\n'.join(json.dumps(r) for r in records),encoding='utf-8')
  result=subprocess.run([sys.executable,str(ROOT/'scripts/segment_conversations.py'),'--input',str(p),'--output-dir',str(self.root/'segments')],capture_output=True)
  self.assertEqual(result.returncode,0,result.stderr)
  for minutes in (10,30,60,120):
   cs=[json.loads(l) for l in (self.root/f'segments/segmentation_{minutes}m.jsonl').read_text(encoding='utf-8').splitlines()]
   self.assertEqual(len(cs),1);self.assertEqual(cs[0]['message_count'],2)
if __name__=="__main__":unittest.main()
