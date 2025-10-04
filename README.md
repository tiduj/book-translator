# 📚 Book Translator

AI-powered translation system that actually understands context. Built for books, documents, and anything that needs more than word-by-word translation.

## 🎯 How It Works

```
Original Text
     ↓
┌─────────────────────────────────────┐
│  Stage 1: Primary Translation       │
│  • Full context awareness           │
│  • Genre-specific adaptation        │
│  • Idiom localization              │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│  Stage 2: Self-Reflection          │
│  • AI critiques its own work       │
│  • Checks accuracy & naturalness   │
│  • Polishes final output           │
└─────────────────────────────────────┘
     ↓
Final Translation
```

**No Google Translate. No DeepL. Pure LLM pipeline.**

## ⚡ Quick Start

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull a model
ollama pull llama3

# 3. Clone & run
git clone <repo-url>
cd BTrans
pip install -r requirements.txt
python translator.py
```

Open **http://localhost:5001** → paste text → translate

## 🎨 Features

| Feature | Description |
|---------|-------------|
| **Smart Chunking** | Splits by paragraphs, not arbitrary limits |
| **Context Memory** | Each chunk knows what came before |
| **Genre Modes** | Fiction, technical, academic, business, poetry |
| **Two-Stage Quality** | Draft → Reflection → Final |
| **Multi-Format Export** | TXT, PDF, EPUB, clipboard |
| **Translation History** | Auto-saved, searchable, re-exportable |
| **Progress Tracking** | Real-time updates, 3-panel view |

## 🔧 Supported Languages

English • Russian • Spanish • French • German • Italian • Portuguese • Chinese • Japanese • Korean

## 💡 Example

**Input (English):**
```
He kicked the bucket last night.
```

**Bad translation (literal):**
```
Он пнул ведро прошлой ночью.
```

**Our translation (Stage 1 + 2):**
```
Он умер прошлой ночью.
```

Stage 1 understands idiom. Stage 2 ensures naturalness.

## 🎛️ Configuration

Edit `translator.py` or use defaults:

```python
PORT = 5001
CHUNK_SIZE = 1000  # characters
OLLAMA_URL = "http://localhost:11434"

# Temperature by genre
TEMPERATURES = {
    'fiction': 0.7,    # more creative
    'technical': 0.3,  # more precise
    'poetry': 0.8      # most creative
}
```

## 📁 Project Structure

```
BTrans/
├── translator.py          # Backend (Flask + Ollama)
├── static/index.html      # Frontend (vanilla JS)
├── requirements.txt       # 4 dependencies
├── logs/                  # Rotation logs
├── translations/          # Exported files
└── cache.db              # SQLite cache
```

## 🐛 Troubleshooting

**"Connection refused"**
```bash
ollama serve  # start Ollama
```

**"Model not found"**
```bash
ollama list           # check installed
ollama pull llama3    # install model
```

**Slow performance**
- Use smaller models: `llama3:8b` instead of `llama3:70b`
- Reduce chunk size: 500 chars instead of 1000
- Check CPU/RAM usage in UI

**Clear cache**
```bash
rm cache.db translations.db
```

## 🚀 Advanced Usage

**Custom model:**
```bash
ollama pull mistral
# Select "mistral" in UI dropdown
```

**Batch processing:**
```python
# API endpoint
POST /api/translate
{
  "text": "...",
  "source_lang": "en",
  "target_lang": "ru",
  "genre": "fiction",
  "model": "llama3"
}
```

**Export formats:**
- TXT: plain text
- PDF: formatted with metadata
- EPUB: e-reader compatible
- Clipboard: instant copy

## 📊 Performance

| Model | Speed | Quality | RAM |
|-------|-------|---------|-----|
| llama3:8b | Fast | Good | 8GB |
| llama3:70b | Slow | Excellent | 64GB |
| mistral:7b | Fast | Good | 8GB |
| gemma:7b | Fast | Good | 8GB |

## 🤝 Contributing

Found a bug? Have an idea? PRs welcome.

## 📄 License

MIT - do whatever you want

---

**Why this exists:** Machine translation is fast but dumb. This is slower but actually understands what it's translating.
