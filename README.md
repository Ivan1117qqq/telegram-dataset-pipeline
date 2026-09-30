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

### 1. 放置資料

Repository 已包含以下空目錄，clone 後不需自行建立。`.gitkeep` 只是讓 Git 保留目錄，可以留著。

```text
data/raw/dataset_001/     第一份 Telegram export 放這裡
data/raw/dataset_002/     第二份 Telegram export 放這裡
outputs/dataset_001/      第一份資料的處理結果
outputs/dataset_002/      第二份資料的處理結果
```

將 Telegram 匯出資料夾的**內容**複製到對應目錄，完成後應為：

```text
data/raw/dataset_001/
├── .gitkeep
├── messages.html
├── messages2.html         有幾頁就保留幾頁
├── photos/
├── files/
└── ...                   其餘 Telegram 匯出檔案也保留
```

不要多包一層成 `dataset_001/ChatExport_xxx/messages.html`；若已如此放置，
請將 `--input-dir` 改為實際包含 HTML 的目錄。Parser 不會遞迴搜尋子目錄。
不要合併不同 export 的 HTML，也不要為了符合範例自行改名。
第三份以上可自行新增 `data/raw/dataset_003/`，輸出用 `outputs/dataset_003/`；仍會被 Git 排除。

### 2. 在專案根目錄執行

VS Code 選「終端機 → 新增終端機」。下列指令以 PowerShell 為例，
目前目錄必須包含 `scripts/` 與本 README。先確認：

```powershell
Get-Location
Test-Path scripts/preprocess_chat.py
python --version
```

`Test-Path` 應顯示 `True`。若 Python 指令不存在，先安裝 Python 3.13 並重開終端機；
若電腦只有 `py` 指令，可把下列 `python` 換成 `py -3.13`。
只做資料處理不需要安裝 pytest，也不需要 API key。

每個 Telegram export 放在獨立目錄，例如 `data/raw/dataset_001/`，內含
`messages*.html` 及原有附件目錄。原始資料僅供讀取，不需提交 Git。
既有匯出若已在專案根目錄，可使用 `--input-dir .`，不必搬動原始資料。

```powershell
python scripts/preprocess_chat.py --input-dir data/raw/dataset_001 --output outputs/dataset_001/processed/messages.jsonl
```

先檢查 `outputs/dataset_001/processed/preprocessing_report.json` 的 errors / warnings；
有錯誤時先停下來確認原始結構。解析成功後再執行統計：

```powershell
python scripts/profile_dataset.py --input outputs/dataset_001/processed/messages.jsonl --output-dir outputs/dataset_001/analysis
```

優先閱讀 `outputs/dataset_001/analysis/profiling_report.md` 與
`sample_conversations.txt`，核對統計並人工查看上下文。
如需候選切分，再執行（不代表最終案件標註）：

```powershell
python scripts/segment_conversations.py --input outputs/dataset_001/processed/messages.jsonl --output-dir outputs/dataset_001/segmentation
```

切分結果與 review 文件在 `outputs/dataset_001/segmentation/`。
上述程式會自動建立輸出子目錄。相同輸出路徑重跑會更新產物，
若要保留不同實驗版本，請改用新的輸出目錄。

### 3. 第二份資料的指令

```powershell
python scripts/preprocess_chat.py --input-dir data/raw/dataset_002 --output outputs/dataset_002/processed/messages.jsonl
```

確認解析報告後執行：

```powershell
python scripts/profile_dataset.py --input outputs/dataset_002/processed/messages.jsonl --output-dir outputs/dataset_002/analysis
python scripts/segment_conversations.py --input outputs/dataset_002/processed/messages.jsonl --output-dir outputs/dataset_002/segmentation
```

### 4. 角色與 Agent 分析（選用）

先建立行為統計與人工填寫的角色表：

```powershell
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

`.gitignore` 採白名單：只允許此 README、設定、指定 Python scripts、合成測試及四個目錄佔位 `.gitkeep`。
原始 HTML、附件、data/ 與 outputs/ 內的實際資料、reports/、角色 mapping 及分析抽樣均不提交。
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
