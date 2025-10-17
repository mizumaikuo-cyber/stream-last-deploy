# 会社内検索アプリ (Company Inner Search App)

社内文書の検索と社内問い合わせに対応するStreamlitアプリです。RAG + 会話履歴で、関連するドキュメントの提示や問い合わせ対応を行います。

## 必要条件
- Python 3.10 以降
- Windows: `requirements_windows.txt`
- macOS: `requirements_mac.txt`

## セットアップ（Windows）
1. 仮想環境を作成・有効化
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. 依存ライブラリのインストール
   ```cmd
   pip install -r requirements_windows.txt
   ```
3. 環境変数の設定（`.env`）
   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```

## 起動
```cmd
streamlit run main.py
```

## 構成
- `data/` RAGで参照する社内文書群
- `initialize.py` 初期化（ログ、Retriever構築、データ読み込み）
- `utils.py` LLM呼び出し、Chain構築など
- `components.py` 画面コンポーネント（UI）
- `constants.py` 各種定数
- `.streamlit/` Streamlit設定

## ログ
- `logs/` に日次ローテーションでログを出力します（`.gitignore`対象）

## デプロイのメモ
- 環境変数とベクトルストアの永続化（Chroma）戦略を検討してください。
- WindowsでのUnicode/CP932問題に対応する正規化処理を含みます。
