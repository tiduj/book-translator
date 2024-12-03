<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 320">
  <defs>
    <linearGradient id="modernGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#000000;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#18181b;stop-opacity:1" />
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  <rect width="1280" height="320" fill="url(#modernGrad)"/>
  <path d="M0 40 L1280 40 M0 80 L1280 80 M0 120 L1280 120 M0 160 L1280 160 M0 200 L1280 200 M0 240 L1280 240 M0 280 L1280 280" stroke="#27272a" stroke-width="0.5"/>
  <g transform="translate(100, 100)">
    <g filter="url(#glow)">
      <path d="M20,20 L60,20 A10,10 0 0 1 70,30 L70,90 A10,10 0 0 1 60,100 L20,100 A10,10 0 0 1 10,90 L10,30 A10,10 0 0 1 20,20 Z" fill="none" stroke="#06b6d4" stroke-width="2"/>
      <path d="M25,30 L55,30 M25,45 L55,45 M25,60 L55,60" stroke="#06b6d4" stroke-width="2"/>
    </g>
    <text x="100" y="45" font-family="Arial" font-size="48" font-weight="bold" fill="white" filter="url(#glow)">Book Translator</text>
    <text x="100" y="80" font-family="Arial" font-size="20" fill="#94a3b8">Multilingual Translation Platform</text>
    <g transform="translate(100, 120)">
      <rect x="0" y="0" rx="20" ry="20" width="90" height="40" fill="#06b6d4" fill-opacity="0.1" stroke="#06b6d4" stroke-width="1"/>
      <text x="45" y="25" font-family="Arial" font-size="14" fill="#06b6d4" text-anchor="middle">Python</text>
      <rect x="100" y="0" rx="20" ry="20" width="90" height="40" fill="#06b6d4" fill-opacity="0.1" stroke="#06b6d4" stroke-width="1"/>
      <text x="145" y="25" font-family="Arial" font-size="14" fill="#06b6d4" text-anchor="middle">React</text>
      <rect x="200" y="0" rx="20" ry="20" width="90" height="40" fill="#06b6d4" fill-opacity="0.1" stroke="#06b6d4" stroke-width="1"/>
      <text x="245" y="25" font-family="Arial" font-size="14" fill="#06b6d4" text-anchor="middle">Flask</text>
    </g>
  </g>
  <g transform="translate(800, 0)" fill="#06b6d4" fill-opacity="0.03">
    <circle cx="200" cy="100" r="80"/>
    <circle cx="300" cy="200" r="120"/>
    <circle cx="150" cy="250" r="60"/>
  </g>
</svg>

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