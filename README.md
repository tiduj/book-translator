![Book Translator](https://raw.githubusercontent.com/KazKozDev/book-translator/main/banner.jpg)

# Book Translator

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Flask](https://img.shields.io/badge/flask-%23000.svg?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![React](https://img.shields.io/badge/react-%2320232a.svg?style=flat&logo=react&logoColor=%2361DAFB)](https://reactjs.org/)
[![Tailwind CSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=flat&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

A web-based application for translating books and text documents between multiple languages using Ollama AI models. Built with Python (Flask) backend and React frontend.

## Features

- Translate text documents between multiple languages:
  - English
  - Russian
  - German
  - French
  - Spanish
  - Italian
  - Chinese
  - Japanese
- Automatic language detection
- Support for multiple Ollama AI models
- Real-time translation progress tracking
- Translation history with status tracking
- Resume interrupted translations
- Download translated files
- Modern, responsive UI built with React and Tailwind CSS

![Book Translator](https://raw.githubusercontent.com/KazKozDev/book-translator/main/demo.jpg)

## Prerequisites

- Python 3.7+
- [Ollama](https://ollama.ai/) installed and running
- Node.js (optional, for development)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/book-translator.git
cd book-translator
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure Ollama is running and you have at least one model pulled:
```bash
ollama pull aya-expanse:32b
```

## Running the Application

1. Start the Flask backend:
```bash
python translator.py
```

2. Access the application:
Open `http://localhost:5001` in your web browser

## Project Structure

```
book-translator/
├── translator.py        # Flask backend
├── static/             # Static files
│   └── index.html      # React frontend
├── uploads/            # Temporary upload directory
├── translations/       # Completed translations directory
└── translations.db     # SQLite database
```

## API Endpoints

- `GET /models` - Get available Ollama models
- `GET /translations` - Get translation history
- `GET /translations/<id>` - Get specific translation details
- `POST /translate` - Start new translation
- `GET /download/<id>` - Download completed translation
- `GET /health` - Check service health

## Database Schema

### Translations Table
- id: Primary key
- filename: Original file name
- source_lang: Source language code
- target_lang: Target language code
- model: AI model used
- status: Translation status
- progress: Translation progress (0-100)
- current_chunk: Current chunk being processed
- total_chunks: Total number of chunks
- original_text: Source text
- translated_text: Translated text
- detected_language: Detected language (for auto detection)
- created_at: Creation timestamp
- updated_at: Last update timestamp

### Chunks Table
- id: Primary key
- translation_id: Foreign key to translations
- chunk_number: Chunk sequence number
- original_text: Original chunk text
- translated_text: Translated chunk text
- status: Chunk status

## Features in Detail

### Translation Process
1. File upload and validation
2. Text chunking for efficient processing
3. Language detection (if auto-detect is selected)
4. Chunk-by-chunk translation with progress tracking
5. Real-time progress updates via Server-Sent Events
6. Automatic translation resumption on interruption

### UI Features
- Drag-and-drop file upload
- Real-time progress bar
- Side-by-side original and translated text preview
- Translation history with filtering and sorting
- Dark/light mode support
- Responsive design for mobile and desktop

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details

## Acknowledgments

- [Ollama](https://ollama.ai/) for providing the AI models
- [Flask](https://flask.palletsprojects.com/) for the backend framework
- [React](https://reactjs.org/) for the frontend framework
- [Tailwind CSS](https://tailwindcss.com/) for styling
