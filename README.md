# GitHub Issue Summarize

GitHub Issue/PR の更新情報を取得し、Issue/PR ごとに JSON ファイルとして保存したうえで、OpenAI互換APIとして公開されたOllama上のLLMで週次報告CSVを作成するためのスクリプトです。

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

`.env` の `GITHUB_TOKEN` に GitHub Personal Access Token を設定してください。

Ollamaを使う環境では、必要に応じて以下も設定します。

```env
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen3.5:9b
API_MODE=ollama
```

## A. GitHubデータ取得

過去 7 日間の更新を取得する例:

```powershell
python github_issue_fetcher.py owner/repo --output-dir output/issues
```

期間を明示する例:

```powershell
python github_issue_fetcher.py owner/repo --start-at 2026-05-01T00:00:00+00:00 --end-at 2026-05-08T00:00:00+00:00 --output-dir output/issues
```

各 JSON には Issue/PR の基本情報、期間内コメント、期間内タイムラインイベントが含まれます。

## B. LLM要約とCSV出力

Aで作成したJSONフォルダを読み込み、CSVを出力します。

```powershell
python summarize_issues.py --input-dir output/issues --output-csv output/weekly_report.csv
```

モデルやOllamaの接続先を指定する例:

```powershell
python summarize_issues.py --input-dir output/issues --output-csv output/weekly_report.csv --model qwen3.5:9b
```

1件あたりの入力が長い場合は、直近の内容を優先してデフォルト2,000文字にトリミングします。変更する場合は `--max-chars` を使います。

```powershell
python summarize_issues.py --input-dir output/issues --output-csv output/weekly_report.csv --max-chars 6000
```

GTX1080 Ti 11GBなどVRAMに余裕が少ない環境では、まず以下の軽量設定を推奨します。

```powershell
python summarize_issues.py --input-dir output/issues --output-csv output/weekly_report.csv --api-mode ollama --max-chars 1200 --max-tokens 160 --num-ctx 2048
```

CSVは1件ごとに保存され、同じ出力先で再実行すると要約済みの行はスキップします。最初から作り直したい場合は `--no-resume` を付けます。

`LLMによる進捗要約` が空のCSVができた場合は、同じ出力先で再実行してください。空の行だけ再処理されます。

`qwen3.5` などのThinkingモデルでは、OllamaのOpenAI互換APIだと最終回答の `content` が空になることがあります。このため、Ollama利用時は既定の `--api-mode ollama` のまま実行してください。これはOllamaネイティブAPI `/api/chat` に `think:false` を渡します。

接続確認だけを軽く行う場合は、先頭1件だけ処理できます。

```powershell
python summarize_issues.py --input-dir output/issues --output-csv output/test_report.csv --api-mode ollama --limit 1 --max-chars 1200 --max-tokens 160 --num-ctx 2048
```

Ollama側でGPUが使われているかは、サーバー側で次を確認します。

```bash
nvidia-smi
docker logs <ollama-container-name>
```

`nvidia-smi` でVRAM使用量が増えない場合は、DockerのGPU割り当てを確認してください。例:

```bash
docker run --gpus all -p 11434:11434 -v ollama:/root/.ollama ollama/ollama
```

## 別サーバーでのOllama動作確認手順

1. Python 3.10以上を用意します。

2. このプロジェクトを配置し、依存関係をインストールします。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Ollamaを起動し、モデルを取得します。

```bash
ollama serve
ollama pull qwen3.5:9b
```

4. 別ターミナルでOpenAI互換APIの疎通を確認します。

```bash
curl http://localhost:11434/v1/models
```

5. `.env` を作成します。

```env
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen3.5:9b
API_MODE=ollama
```

Ollamaが別ホストで動いている場合は、`OPENAI_BASE_URL=http://<server-host>:11434/v1` のように変更します。

6. Aの出力済みJSONを `output/issues` に置き、Bを実行します。

```bash
python summarize_issues.py --input-dir output/issues --output-csv output/test_report.csv --api-mode ollama --limit 1 --max-chars 1200 --max-tokens 160 --num-ctx 2048
```

7. 1件の要約が成功したら、全件を処理します。

```bash
python summarize_issues.py --input-dir output/issues --output-csv output/weekly_report.csv
```

8. `output/weekly_report.csv` に以下の列で出力されていることを確認します。

`種別`, `番号`, `タイトル`, `ステータス`, `担当者`, `ラベル`, `LLMによる進捗要約`, `URL`
