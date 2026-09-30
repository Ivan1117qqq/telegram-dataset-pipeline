# Telegram Dataset Audit Pipeline

本機處理 Telegram Desktop HTML 匯出資料，用於研究前的解析、統計與人工檢查。
程式不呼叫外部 LLM、embedding 或上傳 API；純 Python 處理不消耗模型 token。

## 安裝與測試

已在 Python 3.13 驗證。處理程式使用 Python 標準函式庫，pytest 僅為測試依賴。

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

測試使用合成 Telegram HTML，不包含真實聊天。

## 使用不同資料集

每個 Telegram export 放在獨立目錄，例如 `data/raw/dataset_001/`，內含
`messages*.html` 及原有附件目錄。原始資料僅供讀取，不需提交 Git。
既有匯出若已在專案根目錄，可使用 `--input-dir .`，不必搬動原始資料。

```powershell
python scripts/preprocess_chat.py --input-dir data/raw/dataset_001 --output outputs/dataset_001/processed/messages.jsonl
python scripts/profile_dataset.py --input outputs/dataset_001/processed/messages.jsonl --output-dir outputs/dataset_001/analysis
python scripts/segment_conversations.py --input outputs/dataset_001/processed/messages.jsonl --output-dir outputs/dataset_001/segmentation
python scripts/analyze_sender_roles.py --input outputs/dataset_001/processed/messages.jsonl --output-dir outputs/dataset_001/roles --evidence-sender ""
```

每個 export 分開執行，避免不同群組的 message ID 碰撞。
先檢查 preprocessing report 的 errors / warnings，再進行下游分析。
角色 mapping 必須針對每份資料由人工填寫，不可沿用另一份資料的身份。
Agent audit 範例（填完 mapping 後執行）：

```powershell
python scripts/audit_agent_interactions.py --messages outputs/dataset_001/processed/messages.jsonl --mapping outputs/dataset_001/roles/sender_role_mapping.csv --output-dir outputs/dataset_001/agent_audit
```

其他 script 的參數可用 `python scripts/<script>.py --help` 查看。

## Scripts

| Script | 功能 |
| --- | --- |
| preprocess_chat.py | Canonical HTML parser；保留訊息、回覆、轉寄及附件 metadata |
| profile_dataset.py | Sender、時間、reply 與文字統計 |
| segment_conversations.py | 多 threshold 的 candidate segmentation |
| analyze_single_conversations.py | Singleton 結構與鄰近關係分析 |
| analyze_sender_roles.py | 行為統計、互動矩陣、空白人工 role mapping |
| reconstruct_agent_conversations.py | 依人工角色進行候選合併 |
| audit_agent_interactions.py | Agent context、episode 與人工 annotation template |
| verify_existing_pipeline.py | 原始 audit 的重跑 harness；依賴本機 snapshot / mapping |
| validate_raw_parser.py | 原始 audit 的獨立 HTML 比對工具；依賴本機 audit 輸出 |

最後兩支是特定 audit 工作目錄的工具，並非新 clone 後即可執行的通用入口。
不要在其他資料集直接重跑 `--phase before` 覆蓋既有 audit 證據。

## 已知限制與研究邊界

- 已驗證一份真實 Telegram export 及合成測試；其他匯出版本仍需抽樣比對。
- 部分下游 Markdown 報告仍有原資料集的固定敘述，換資料時需檢查；不得直接把文字敘述當成重新計算的統計。
- joined sender、跨頁、轉寄、附件及 reply 有測試；不明結構不應靠猜測補值。
- `text` 保留原先合併文字；`own_text` 與 `forwarded[].text` 區分實際發送者文字與轉寄內容。
- 附件 reference 與本機檔案存在狀態分開記錄；不做 OCR。
- Conversation segment / episode / question_like 都是候選，不是 case、clarification 或 action ground truth。
- 沒有 DB/API 執行紀錄時，不能由聊天文字宣稱實際執行過 TOOL。
- 尚未提供通用的「論文資料適用性」自動判定；需人工檢查資訊缺漏與追問案例。

## Git 上傳範圍

`.gitignore` 採白名單：只允許此 README、設定、指定 Python scripts 與合成測試。
原始 HTML、附件、data/、outputs/、reports/、角色 mapping 及分析抽樣均不提交。
新增程式時需要明確更新白名單。不要使用 `git add -f` 將排除的資料強制加入。

推送前用 `git diff --cached --name-only` 檢查上傳清單。
先在 GitHub 建立空白 repository，再設定自己的 URL：

```powershell
git remote add origin https://github.com/YOUR_ACCOUNT/telegram-dataset-pipeline.git
git push -u origin main
```

若本機尚未初始化或提交：

```powershell
git init -b main
git add .gitignore README.md requirements-dev.txt pytest.ini scripts tests
git diff --cached --name-only
git commit -m "Add local Telegram dataset audit pipeline"
```

驗證報告與實際資料只保存在原本本機工作目錄，不隨程式發佈。
