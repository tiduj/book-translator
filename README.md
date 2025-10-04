<div align="center">
  <img src="https://github.com/user-attachments/assets/f62278a1-ec33-4096-aa13-a855dd7bda4f" alt="Logo">
  <br> 
</div>
<div align="center">
<p><strong>Book Translator</strong></p>
  <p>A platform for translating books and large text documents.</p>
  <p><strong>Two-step process. For better quality.</strong></p>
</div>
  <p>The tool processes text files using Ollama LLM models with a two-stage approach: primary translation followed by AI self-reflection and refinement for better results. Suitable for translators, publishers, authors, researchers and content creators who need to translate large text documents.</p>
Support for multiple languages including English, Russian, Spanish, French, German, Italian, Portuguese, Chinese, and Japanese.  Genre-specific modes (fiction, technical, academic, business, poetry), real-time translation progress tracking for both stages, translation history and status monitoring, automatic error recovery and retry mechanisms, and multi-format export (TXT, PDF, EPUB).  <br>   <br> 

<img width="1191" height="889" alt="Screenshot 2025-10-04 at 12 54 17" src="https://github.com/user-attachments/assets/35838c1c-26be-4594-a759-dbaa76a9494f" />

## Requirements

- Python 3.7+
- [Ollama](https://ollama.ai/) installed and running
- Recommended: 8GB+ RAM for basic models, 64GB for large models

## Installation

1. **Install Ollama**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

2. **Clone the repository**
```bash
git clone https://github.com/KazKozDev/book-translator.git
cd book-translator
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Pull an Ollama model (choose any you prefer)**
```bash
# Example with gpt-oss:20b
ollama pull gpt-oss:20b

# Or use other models:
# ollama pull llama3.2
# ollama pull qwen2.5
# ollama pull gemma3:12b
# ollama pull phi3
```

5. **Start the application**

**Option 1: Quick Launch (macOS)**
```bash
./Launch\ Book-Translator.command
```
This will automatically:
- Kill any process on port 5001
- Start the Flask server
- Open http://localhost:5001 in your browser
- Clear translation cache

**Option 2: Manual start**
```bash
python translator.py
# Then open http://localhost:5001 in your browser
```

## Project Structure

```
book-translator/
├── Launch Book-Translator.command  # Quick launch script (macOS)
├── translator.py          # Flask backend
├── static/index.html      # Frontend interface
├── requirements.txt       # Python dependencies
├── logs/                  # Application logs
├── translations/          # Exported files
├── cache.db              # Translation cache
└── translations.db        # Translation history
```

## Configuration

Default settings in `translator.py`:
- Port: 5001
- Chunk size: 1000 characters
- Temperature varies by genre (0.3-0.8)

## License

MIT License - see [LICENSE](LICENSE)

---

If you like this project, please give it a star ⭐
For questions, feedback, or support, open an issue or submit a PR.

---

**Note:** The previous version of this project is available in the `archive-old-version` branch.
