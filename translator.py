#!/usr/bin/env python3
import json
import requests
import time
from typing import List, Dict
from tqdm import tqdm
import os
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, Response, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Folders setup
UPLOAD_FOLDER = 'uploads'
TRANSLATIONS_FOLDER = 'translations'
STATIC_FOLDER = 'static'
DB_PATH = 'translations.db'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSLATIONS_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Initialize database
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS translations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                source_lang TEXT NOT NULL,
                target_lang TEXT NOT NULL,
                model TEXT NOT NULL,
                status TEXT NOT NULL,
                progress REAL DEFAULT 0,
                current_chunk INTEGER DEFAULT 0,
                total_chunks INTEGER DEFAULT 0,
                original_text TEXT,
                translated_text TEXT,
                detected_language TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                translation_id INTEGER,
                chunk_number INTEGER,
                original_text TEXT,
                translated_text TEXT,
                status TEXT,
                FOREIGN KEY (translation_id) REFERENCES translations (id)
            )
        ''')

init_db()

class BookTranslator:
    def __init__(self, model_name: str = "aya-expanse:32b", chunk_size: int = 2000):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        self.chunk_size = chunk_size
        
        # Expanded language support
        self.language_prompts = {
            'auto': {
                'en': "Determine the language of this text and translate it to English. Format: [Detected: Language] Translation:\n\n",
                'ru': "Определи язык этого текста и переведи его на русский. Формат: [Обнаружен: Язык] Перевод:\n\n",
                'de': "Bestimme die Sprache dieses Textes und übersetze ihn ins Deutsche. Format: [Erkannt: Sprache] Übersetzung:\n\n",
                'fr': "Déterminez la langue de ce texte et traduisez-le en français. Format: [Détecté: Langue] Traduction:\n\n",
                'es': "Determina el idioma de este texto y tradúcelo al español. Formato: [Detectado: Idioma] Traducción:\n\n",
                'it': "Determina la lingua di questo testo e traducilo in italiano. Formato: [Rilevato: Lingua] Traduzione:\n\n",
                'zh': "判断这段文字的语言并将其翻译成中文。格式：[检测到：语言] 翻译：\n\n",
                'ja': "このテキストの言語を判定し、日本語に翻訳してください。形式：[検出：言語] 翻訳：\n\n"
            },
            'ru': {
                'en': "Translate this Russian text to English. Skip any confirmations or additional notes:\n\n",
                'de': "Übersetze diesen russischen Text ins Deutsche. Überspringe Bestätigungen:\n\n",
                'fr': "Traduis ce texte russe en français. Ignore les confirmations:\n\n",
                'es': "Traduce este texto ruso al español. Omite confirmaciones:\n\n",
                'it': "Traduci questo testo russo in italiano. Salta le conferme:\n\n",
                'zh': "将这段俄语文本翻译成中文。跳过确认：\n\n",
                'ja': "このロシア語のテキストを日本語に翻訳してください。確認は省略：\n\n"
            },
            'en': {
                'ru': "Переведи этот английский текст на русский язык. Пропусти подтверждения:\n\n",
                'de': "Übersetze diesen englischen Text ins Deutsche. Überspringe Bestätigungen:\n\n",
                'fr': "Traduis ce texte anglais en français. Ignore les confirmations:\n\n",
                'es': "Traduce este texto inglés al español. Omite confirmaciones:\n\n",
                'it': "Traduci questo testo inglese in italiano. Salta le conferme:\n\n",
                'zh': "将这段英语文本翻译成中文。跳过确认：\n\n",
                'ja': "この英語のテキストを日本語に翻訳してください。確認は省略：\n\n"
            },
            'de': {
                'ru': "Переведи этот немецкий текст на русский язык. Пропусти подтверждения:\n\n",
                'en': "Translate this German text to English. Skip confirmations:\n\n",
                'fr': "Traduis ce texte allemand en français. Ignore les confirmations:\n\n",
                'es': "Traduce este texto alemán al español. Omite confirmaciones:\n\n",
                'it': "Traduci questo testo tedesco in italiano. Salta le conferme:\n\n",
                'zh': "将这段德语文本翻译成中文。跳过确认：\n\n",
                'ja': "このドイツ語のテキストを日本語に翻訳してください。確認は省略：\n\n"
            },
            'fr': {
                'ru': "Переведи этот французский текст на русский язык. Пропусти подтверждения:\n\n",
                'en': "Translate this French text to English. Skip confirmations:\n\n",
                'de': "Übersetze diesen französischen Text ins Deutsche. Überspringe Bestätigungen:\n\n",
                'es': "Traduce este texto francés al español. Omite confirmaciones:\n\n",
                'it': "Traduci questo testo francese in italiano. Salta le conferme:\n\n",
                'zh': "将这段法语文本翻译成中文。跳过确认：\n\n",
                'ja': "このフランス語のテキストを日本語に翻訳してください。確認は省略：\n\n"
            },
            'es': {
                'ru': "Переведи этот испанский текст на русский язык. Пропусти подтверждения:\n\n",
                'en': "Translate this Spanish text to English. Skip confirmations:\n\n",
                'de': "Übersetze diesen spanischen Text ins Deutsche. Überspringe Bestätigungen:\n\n",
                'fr': "Traduis ce texte espagnol en français. Ignore les confirmations:\n\n",
                'it': "Traduci questo testo spagnolo in italiano. Salta le conferme:\n\n",
                'zh': "将这段西班牙语文本翻译成中文。跳过确认：\n\n",
                'ja': "このスペイン語のテキストを日本語に翻訳してください。確認は省略：\n\n"
            }
        }

    def detect_language(self, text: str) -> str:
        try:
            prompt = "Determine the language of this text. Respond with only the language code (en, ru, de, fr, es, it, zh, ja):\n\n" + text[:500]
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }

            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            detected_lang = result['response'].strip().lower()
            
            # Map full language names to codes if necessary
            lang_map = {
                'english': 'en',
                'russian': 'ru',
                'german': 'de',
                'french': 'fr',
                'spanish': 'es',
                'italian': 'it',
                'chinese': 'zh',
                'japanese': 'ja'
            }
            
            return lang_map.get(detected_lang, detected_lang)
            
        except Exception as e:
            print(f"Language detection error: {e}")
            return 'unknown'

    def get_available_models(self) -> List[str]:
        try:
            response = requests.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            models = response.json()
            return [model['name'] for model in models['models']]
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []

    def read_file(self, input_path: str) -> str:
        try:
            with open(input_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except UnicodeDecodeError:
            with open(input_path, 'r', encoding='cp1251') as file:
                return file.read()

    def split_into_chunks(self, text: str) -> List[str]:
        paragraphs = text.replace('\r\n', '\n').split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            if len(paragraph) > self.chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                sentences = paragraph.split('. ')
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    if not sentence.endswith('.'):
                        sentence += '.'
                    
                    sentence_length = len(sentence)
                    
                    if current_length + sentence_length > self.chunk_size:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                        current_chunk = [sentence]
                        current_length = sentence_length
                    else:
                        current_chunk.append(sentence)
                        current_length += sentence_length
            else:
                if current_length + len(paragraph) > self.chunk_size:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [paragraph]
                    current_length = len(paragraph)
                else:
                    current_chunk.append(paragraph)
                    current_length += len(paragraph)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def translate_chunk(self, chunk: str, source_lang: str, target_lang: str, detected_lang: str = None) -> str:
        if not chunk.strip():
            return ""
            
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # If source language is 'auto', use the detected language for translation
                actual_source = detected_lang if source_lang == 'auto' else source_lang
                
                if actual_source == 'unknown':
                    # If language detection failed, use auto translation prompt
                    prompt_template = self.language_prompts['auto'].get(target_lang)
                else:
                    # Use specific language pair prompt
                    prompt_template = self.language_prompts.get(actual_source, {}).get(target_lang)
                
                if not prompt_template:
                    raise ValueError(f"Unsupported language pair: {actual_source} -> {target_lang}")
                
                prompt = prompt_template + chunk
                
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }

                response = requests.post(self.api_url, json=payload, timeout=180)
                response.raise_for_status()
                result = response.json()
                return result['response'].strip()

            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 5)
                else:
                    return ""

    def translate_text(self, text: str, source_lang: str, target_lang: str, translation_id: int):
        detected_lang = None
        if source_lang == 'auto':
            detected_lang = self.detect_language(text[:1000])  # Use first 1000 chars for detection
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute('''
                    UPDATE translations 
                    SET detected_language = ? 
                    WHERE id = ?
                ''', (detected_lang, translation_id))
        
        chunks = self.split_into_chunks(text)
        total_chunks = len(chunks)
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''
                UPDATE translations 
                SET total_chunks = ? 
                WHERE id = ?
            ''', (total_chunks, translation_id))
            
            for i, chunk in enumerate(chunks, 1):
                conn.execute('''
                    INSERT INTO chunks (translation_id, chunk_number, original_text, status)
                    VALUES (?, ?, ?, ?)
                ''', (translation_id, i, chunk, 'pending'))
        
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute('''
                SELECT current_chunk 
                FROM translations 
                WHERE id = ?
            ''', (translation_id,))
            start_chunk = cur.fetchone()[0] or 0
        
        translated_chunks = []
        
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute('''
                SELECT chunk_number, translated_text 
                FROM chunks 
                WHERE translation_id = ? AND status = 'completed'
                ORDER BY chunk_number
            ''', (translation_id,))
            existing_translations = {row[0]: row[1] for row in cur.fetchall()}
            translated_chunks.extend([existing_translations[i] for i in range(1, start_chunk + 1) if i in existing_translations])
        
        for i in range(start_chunk + 1, total_chunks + 1):
            chunk = chunks[i - 1]
            translated_text = self.translate_chunk(chunk, source_lang, target_lang, detected_lang)
            
            if translated_text:
                translated_chunks.append(translated_text)
                
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute('''
                        UPDATE chunks
                        SET translated_text = ?, status = 'completed'
                        WHERE translation_id = ? AND chunk_number = ?
                    ''', (translated_text, translation_id, i))
# Update translation progress
                    progress = (i / total_chunks) * 100
                    conn.execute('''
                        UPDATE translations 
                        SET progress = ?, current_chunk = ?, translated_text = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    ''', (progress, i, '\n\n'.join(translated_chunks), translation_id))
                    
                    yield {
                        'progress': progress,
                        'translated_text': '\n\n'.join(translated_chunks),
                        'current_chunk': i,
                        'total_chunks': total_chunks,
                        'detected_language': detected_lang if source_lang == 'auto' else None
                    }
                    
            if i < total_chunks:
                time.sleep(2)
                
        # Mark translation as completed
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''
                UPDATE translations 
                SET status = 'completed', progress = 100,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (translation_id,))
            
        yield {
            'progress': 100,
            'translated_text': '\n\n'.join(translated_chunks),
            'status': 'completed',
            'detected_language': detected_lang if source_lang == 'auto' else None
        }
        
# Flask routes
@app.route('/')
def serve_frontend():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/models', methods=['GET'])
def get_models():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        available_models = response.json()
        
        models = []
        for model in available_models['models']:
            models.append({
                'name': model['name'],
                'size': model.get('size', 'Unknown'),
                'modified': model.get('modified', 'Unknown')
            })
            
        return jsonify({'models': models})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/translations', methods=['GET'])
def get_translations():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute('''
                SELECT id, filename, source_lang, target_lang, model,
                       status, progress, detected_language, created_at, updated_at
                FROM translations
                ORDER BY created_at DESC
            ''')
            translations = [dict(row) for row in cur.fetchall()]
        return jsonify({'translations': translations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/translations/<int:translation_id>', methods=['GET'])
def get_translation(translation_id):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute('SELECT * FROM translations WHERE id = ?', (translation_id,))
            translation = cur.fetchone()
            if translation:
                return jsonify(dict(translation))
            return jsonify({'error': 'Translation not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/translate', methods=['POST'])
def translate():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    source_lang = request.form.get('sourceLanguage')
    target_lang = request.form.get('targetLanguage')
    model_name = request.form.get('model')
    
    if not all([file, source_lang, target_lang, model_name]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Read file content
        text = None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='cp1251') as f:
                text = f.read()
                
        # Create translation record
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute('''
                INSERT INTO translations (
                    filename, source_lang, target_lang, model,
                    status, original_text
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (filename, source_lang, target_lang, model_name, 'in_progress', text))
            translation_id = cur.lastrowid
            
        # Initialize translator
        translator = BookTranslator(model_name=model_name)
        
        def generate():
            try:
                for update in translator.translate_text(text, source_lang, target_lang, translation_id):
                    yield f"data: {json.dumps(update)}\n\n"
            except Exception as e:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute('''
                        UPDATE translations 
                        SET status = 'error'
                        WHERE id = ?
                    ''', (translation_id,))
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                
        return Response(generate(), mimetype='text/event-stream')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Cleanup uploaded file
        try:
            os.remove(filepath)
        except:
            pass
            
@app.route('/download/<int:translation_id>', methods=['GET'])
def download_translation(translation_id):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute('''
                SELECT filename, translated_text
                FROM translations
                WHERE id = ? AND status = 'completed'
            ''', (translation_id,))
            result = cur.fetchone()
            
            if not result:
                return jsonify({'error': 'Translation not found or not completed'}), 404
            
            filename, translated_text = result
            
            # Create download file
            download_path = os.path.join(TRANSLATIONS_FOLDER, f'translated_{filename}')
            with open(download_path, 'w', encoding='utf-8') as f:
                f.write(translated_text)
                
            return send_file(
                download_path,
                as_attachment=True,
                download_name=f'translated_{filename}'
            )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Check if Ollama API is accessible
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        return jsonify({'status': 'healthy', 'ollama': 'connected'})
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 503
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)