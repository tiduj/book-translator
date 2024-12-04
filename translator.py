import json
import requests
import time
from typing import List, Dict, Optional, Tuple, Callable
from tqdm import tqdm
import os
import sqlite3
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler
import hashlib
import traceback
import psutil
import threading
from collections import deque
from dataclasses import dataclass, field
from functools import wraps
from flask import Flask, request, jsonify, Response, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Folders setup
UPLOAD_FOLDER = 'uploads'
TRANSLATIONS_FOLDER = 'translations'
STATIC_FOLDER = 'static'
LOG_FOLDER = 'logs'
DB_PATH = 'translations.db'
CACHE_DB_PATH = 'cache.db'

# Create necessary directories
for folder in [UPLOAD_FOLDER, TRANSLATIONS_FOLDER, STATIC_FOLDER, LOG_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Logger setup
class AppLogger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.app_logger = self._setup_logger(
            'app_logger',
            os.path.join(log_dir, 'app.log')
        )
        
        self.translation_logger = self._setup_logger(
            'translation_logger',
            os.path.join(log_dir, 'translations.log')
        )
        
        self.api_logger = self._setup_logger(
            'api_logger',
            os.path.join(log_dir, 'api.log')
        )

    def _setup_logger(self, name, log_file):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        return logger

# Initialize logger
logger = AppLogger()

# Monitoring setup
@dataclass
class TranslationMetrics:
    total_requests: int = 0
    successful_translations: int = 0
    failed_translations: int = 0
    average_translation_time: float = 0
    translation_times: deque = field(default_factory=lambda: deque(maxlen=100))

class AppMonitor:
    def __init__(self):
        self.metrics = TranslationMetrics()
        self._lock = threading.Lock()
        self.start_time = time.time()
        
    def record_translation_attempt(self, success: bool, translation_time: float):
        with self._lock:
            self.metrics.total_requests += 1
            if success:
                self.metrics.successful_translations += 1
                self.metrics.translation_times.append(translation_time)
                self.metrics.average_translation_time = (
                    sum(self.metrics.translation_times) / len(self.metrics.translation_times)
                )
            else:
                self.metrics.failed_translations += 1
    
    def get_system_metrics(self) -> Dict:
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'uptime': time.time() - self.start_time
        }
    
    def get_metrics(self) -> Dict:
        with self._lock:
            return {
                'translation_metrics': {
                    'total_requests': self.metrics.total_requests,
                    'successful_translations': self.metrics.successful_translations,
                    'failed_translations': self.metrics.failed_translations,
                    'average_translation_time': self.metrics.average_translation_time,
                    'success_rate': (
                        self.metrics.successful_translations / self.metrics.total_requests * 100
                    ) if self.metrics.total_requests > 0 else 0
                },
                'system_metrics': self.get_system_metrics()
            }

# Initialize monitor
monitor = AppMonitor()

# Translation cache setup
class TranslationCache:
    def __init__(self, db_path: str = CACHE_DB_PATH):
        self.db_path = db_path
        self._init_cache_db()
    
    def _init_cache_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS translation_cache (
                    hash_key TEXT PRIMARY KEY,
                    source_lang TEXT,
                    target_lang TEXT,
                    original_text TEXT,
                    translated_text TEXT,
                    created_at TIMESTAMP,
                    last_used TIMESTAMP
                )
            ''')

    def _generate_hash(self, text: str, source_lang: str, target_lang: str) -> str:
        key = f"{text}:{source_lang}:{target_lang}"
        return hashlib.sha256(key.encode()).hexdigest()
    
    def get_cached_translation(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        hash_key = self._generate_hash(text, source_lang, target_lang)
        
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute('''
                SELECT translated_text, created_at
                FROM translation_cache
                WHERE hash_key = ?
            ''', (hash_key,))
            
            result = cur.fetchone()
            if result:
                translated_text, created_at = result
                conn.execute('''
                    UPDATE translation_cache
                    SET last_used = CURRENT_TIMESTAMP
                    WHERE hash_key = ?
                ''', (hash_key,))
                return translated_text
        
        return None
    
    def cache_translation(self, text: str, translated_text: str, source_lang: str, target_lang: str):
        hash_key = self._generate_hash(text, source_lang, target_lang)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO translation_cache
                (hash_key, source_lang, target_lang, original_text, translated_text, created_at, last_used)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (hash_key, source_lang, target_lang, text, translated_text))
    
    def cleanup_old_entries(self, days: int = 30):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                DELETE FROM translation_cache
                WHERE last_used < datetime('now', '-? days')
            ''', (days,))

# Initialize cache
cache = TranslationCache()

# Error handling setup
class TranslationError(Exception):
    pass

def with_error_handling(f: Callable):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except requests.Timeout as e:
            logger.app_logger.error(f"Timeout error: {str(e)}")
            raise TranslationError("Translation service timeout")
        except requests.RequestException as e:
            logger.app_logger.error(f"Request error: {str(e)}")
            raise TranslationError("Translation service unavailable")
        except sqlite3.Error as e:
            logger.app_logger.error(f"Database error: {str(e)}")
            raise TranslationError("Database error occurred")
        except Exception as e:
            logger.app_logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
            raise TranslationError("An unexpected error occurred")
    return wrapper

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
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                error_message TEXT
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
                error_message TEXT,
                attempts INTEGER DEFAULT 0,
                FOREIGN KEY (translation_id) REFERENCES translations (id)
            )
        ''')

init_db()

class BookTranslator:
    def __init__(self, model_name: str = "aya-expanse:32b", chunk_size: int = 1000):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        self.chunk_size = chunk_size
        self.session = requests.Session()
        self.session.mount('http://', requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=10
        ))
        
        # Language prompts setup
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
                'de': "Translate this English text to German. Skip confirmations:\n\n",
                'fr': "Translate this English text to French. Skip confirmations:\n\n",
                'es': "Translate this English text to Spanish. Skip confirmations:\n\n",
                'it': "Translate this English text to Italian. Skip confirmations:\n\n",
                'zh': "Translate this English text to Chinese. Skip confirmations:\n\n",
                'ja': "Translate this English text to Japanese. Skip confirmations:\n\n"
            },
            'de': {
                'ru': "Переведи этот немецкий текст на русский язык. Пропусти подтверждения:\n\n",
                'en': "Translate this German text to English. Skip confirmations:\n\n",
                'fr': "Übersetze diesen deutschen Text ins Französische. Überspringe Bestätigungen:\n\n",
                'es': "Übersetze diesen deutschen Text ins Spanische. Überspringe Bestätigungen:\n\n",
                'it': "Übersetze diesen deutschen Text ins Italienische. Überspringe Bestätigungen:\n\n",
                'zh': "Übersetze diesen deutschen Text ins Chinesische. Überspringe Bestätigungen:\n\n",
                'ja': "Übersetze diesen deutschen Text ins Japanische. Überspringe Bestätigungen:\n\n"
            },
            'fr': {
                'ru': "Переведи этот французский текст на русский язык. Пропусти подтверждения:\n\n",
                'en': "Translate this French text to English. Skip confirmations:\n\n",
                'de': "Translate this French text to German. Skip confirmations:\n\n",
                'es': "Translate this French text to Spanish. Skip confirmations:\n\n",
                'it': "Translate this French text to Italian. Skip confirmations:\n\n",
                'zh': "Translate this French text to Chinese. Skip confirmations:\n\n",
                'ja': "Translate this French text to Japanese. Skip confirmations:\n\n"
            },
            'es': {
                'ru': "Переведи этот испанский текст на русский язык. Пропусти подтверждения:\n\n",
                'en': "Translate this Spanish text to English. Skip confirmations:\n\n",
                'de': "Translate this Spanish text to German. Skip confirmations:\n\n",
                'fr': "Translate this Spanish text to French. Skip confirmations:\n\n",
                'it': "Translate this Spanish text to Italian. Skip confirmations:\n\n",
                'zh': "Translate this Spanish text to Chinese. Skip confirmations:\n\n",
                'ja': "Translate this Spanish text to Japanese. Skip confirmations:\n\n"
            }
        }

    @with_error_handling
    def detect_language(self, text: str) -> str:
        try:
            prompt = "Determine the language of this text. Respond with only the language code (en, ru, de, fr, es, it, zh, ja):\n\n" + text[:500]
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            response = self.session.post(
                self.api_url,
                json=payload,
                timeout=(30, 30)
            )
            response.raise_for_status()
            result = response.json()
            detected_lang = result['response'].strip().lower()
            
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
            logger.translation_logger.error(f"Language detection error: {str(e)}")
            return 'unknown'
        
    @with_error_handling
    def get_available_models(self) -> List[str]:
        response = self.session.get(
            "http://localhost:11434/api/tags",
            timeout=(5, 5)
        )
        response.raise_for_status()
        models = response.json()
        return [model['name'] for model in models['models']]

    def split_into_chunks(self, text: str) -> List[str]:
        MAX_CHUNK_SIZE = 4096
        paragraphs = text.replace('\r\n', '\n').split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            if len(paragraph) > MAX_CHUNK_SIZE:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                    
                sentences = paragraph.split('. ')
                current_sentence = []
                current_sentence_length = 0
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    if not sentence.endswith('.'):
                        sentence += '.'
                        
                    sentence_length = len(sentence)
                    
                    if current_sentence_length + sentence_length > MAX_CHUNK_SIZE:
                        if current_sentence:
                            chunks.append('. '.join(current_sentence) + '.')
                            current_sentence = [sentence]
                            current_sentence_length = sentence_length
                        else:
                            words = sentence.split()
                            current_words = []
                            current_word_length = 0
                            
                            for word in words:
                                word_length = len(word + ' ')
                                if current_word_length + word_length > MAX_CHUNK_SIZE:
                                    if current_words:
                                        chunks.append(' '.join(current_words))
                                    current_words = [word]
                                    current_word_length = word_length
                                else:
                                    current_words.append(word)
                                    current_word_length += word_length
                                    
                            if current_words:
                                chunks.append(' '.join(current_words))
                    else:
                        current_sentence.append(sentence)
                        current_sentence_length += sentence_length
                        
                if current_sentence:
                    chunks.append('. '.join(current_sentence) + '.')
            else:
                if current_length + len(paragraph) > MAX_CHUNK_SIZE:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = [paragraph]
                    current_length = len(paragraph)
                else:
                    current_chunk.append(paragraph)
                    current_length += len(paragraph)
                    
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
        return chunks

    @with_error_handling
    def translate_chunk(self, chunk: str, source_lang: str, target_lang: str, detected_lang: str = None) -> str:
        if not chunk.strip():
            return ""
        
        cached_translation = cache.get_cached_translation(chunk, source_lang, target_lang)
        if cached_translation:
            logger.translation_logger.info("Cache hit for chunk")
            return cached_translation
        
        max_retries = 5
        backoff_factor = 2
        
        for attempt in range(max_retries):
            try:
                actual_source = detected_lang if source_lang == 'auto' else source_lang
                
                if actual_source == 'unknown':
                    prompt_template = self.language_prompts['auto'].get(target_lang)
                else:
                    prompt_template = self.language_prompts.get(actual_source, {}).get(target_lang)
                    
                if not prompt_template:
                    raise ValueError(f"Unsupported language pair: {actual_source} -> {target_lang}")
                    
                prompt = prompt_template + chunk
                
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
                
                response = self.session.post(
                    self.api_url,
                    json=payload,
                    timeout=(300, 300)
                )
                response.raise_for_status()
                result = response.json()
                translated_text = result['response'].strip()
                
                cache.cache_translation(chunk, translated_text, source_lang, target_lang)
                
                return translated_text
            
            except requests.Timeout as e:
                wait_time = (backoff_factor ** attempt) * 5
                logger.translation_logger.warning(
                    f"Timeout on attempt {attempt + 1}, waiting {wait_time} seconds..."
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    logger.translation_logger.error(f"Final timeout error: {str(e)}")
                    raise
                
            except Exception as e:
                wait_time = (backoff_factor ** attempt) * 5
                logger.translation_logger.warning(
                    f"Error on attempt {attempt + 1}: {str(e)}, waiting {wait_time} seconds..."
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    logger.translation_logger.error(f"Final error: {str(e)}")
                    raise
                
    def translate_text(self, text: str, source_lang: str, target_lang: str, translation_id: int):
        start_time = time.time()
        success = False
        
        try:
            detected_lang = None
            if source_lang == 'auto':
                detected_lang = self.detect_language(text[:1000])
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
                try:
                    chunk = chunks[i - 1]
                    translated_text = self.translate_chunk(chunk, source_lang, target_lang, detected_lang)
                    
                    if translated_text:
                        translated_chunks.append(translated_text)
                        
                        with sqlite3.connect(DB_PATH) as conn:
                            conn.execute('''
                                UPDATE chunks
                                SET translated_text = ?, status = 'completed',
                                    attempts = attempts + 1
                                WHERE translation_id = ? AND chunk_number = ?
                            ''', (translated_text, translation_id, i))
                            
                            progress = (i / total_chunks) * 100
                            current_translation = '\n\n'.join(translated_chunks)
                            
                            conn.execute('''
                                UPDATE translations 
                                SET progress = ?, current_chunk = ?, translated_text = ?,
                                    updated_at = CURRENT_TIMESTAMP
                                WHERE id = ?
                            ''', (progress, i, current_translation, translation_id))
                            
                            yield {
                                'progress': progress,
                                'translated_text': current_translation,
                                'current_chunk': i,
                                'total_chunks': total_chunks,
                                'detected_language': detected_lang if source_lang == 'auto' else None
                            }
                            
                except Exception as e:
                    logger.translation_logger.error(f"Error translating chunk {i}: {str(e)}")
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute('''
                            UPDATE chunks
                            SET status = 'error', error_message = ?,
                                attempts = attempts + 1
                            WHERE translation_id = ? AND chunk_number = ?
                        ''', (str(e), translation_id, i))
                    raise
                
                if i < total_chunks:
                    time.sleep(2)  # Rate limiting
                    
            # Mark translation as completed
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute('''
                    UPDATE translations 
                    SET status = 'completed', progress = 100,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (translation_id,))
                
            success = True
            yield {
                'progress': 100,
                'translated_text': '\n\n'.join(translated_chunks),
                'status': 'completed',
                'detected_language': detected_lang if source_lang == 'auto' else None
            }
            
        except Exception as e:
            logger.translation_logger.error(f"Translation failed: {str(e)}")
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute('''
                    UPDATE translations 
                    SET status = 'error', error_message = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (str(e), translation_id))
            raise
        finally:
            translation_time = time.time() - start_time
            monitor.record_translation_attempt(success, translation_time)

# Translation Recovery
class TranslationRecovery:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        
    def get_failed_translations(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute('''
                SELECT * FROM translations 
                WHERE status = 'error'
                ORDER BY created_at DESC
            ''')
            return [dict(row) for row in cur.fetchall()]
        
    def retry_translation(self, translation_id: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE translations
                SET status = 'pending', progress = 0, error_message = NULL,
                    current_chunk = 0, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (translation_id,))
            
            conn.execute('''
                UPDATE chunks
                SET status = 'pending', error_message = NULL
                WHERE translation_id = ? AND status = 'error'
            ''', (translation_id,))
            
    def cleanup_failed_translations(self, days: int = 7):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                DELETE FROM translations
                WHERE status = 'error'
                AND created_at < datetime('now', '-? days')
            ''', (days,))

recovery = TranslationRecovery()

# Health checking middleware
@app.before_request
def check_ollama():
    if request.endpoint != 'health_check':
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.app_logger.error(f"Ollama health check failed: {str(e)}")
            return jsonify({
                'error': 'Translation service is not available'
            }), 503

# Flask routes
@app.route('/')
def serve_frontend():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/models', methods=['GET'])
@with_error_handling
def get_models():
    translator = BookTranslator()
    available_models = translator.get_available_models()
    models = []
    for model_name in available_models:
        models.append({
            'name': model_name,
            'size': 'Unknown',
            'modified': 'Unknown'
        })
    return jsonify({'models': models})

@app.route('/translations', methods=['GET'])
@with_error_handling
def get_translations():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute('''
            SELECT id, filename, source_lang, target_lang, model,
                   status, progress, detected_language, created_at, 
                   updated_at, error_message
            FROM translations
            ORDER BY created_at DESC
        ''')
        translations = [dict(row) for row in cur.fetchall()]
    return jsonify({'translations': translations})

@app.route('/translations/<int:translation_id>', methods=['GET'])
@with_error_handling
def get_translation(translation_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute('SELECT * FROM translations WHERE id = ?', (translation_id,))
        translation = cur.fetchone()
        if translation:
            return jsonify(dict(translation))
        return jsonify({'error': 'Translation not found'}), 404

@app.route('/translate', methods=['POST'])
@with_error_handling
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
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        text = None
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='cp1251') as f:
                text = f.read()
                
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute('''
                INSERT INTO translations (
                    filename, source_lang, target_lang, model,
                    status, original_text
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (filename, source_lang, target_lang, model_name, 'in_progress', text))
            translation_id = cur.lastrowid
            
        translator = BookTranslator(model_name=model_name)
        
        def generate():
            try:
                for update in translator.translate_text(text, source_lang, target_lang, translation_id):
                    # Send raw text without additional escaping
                    yield f"data: {json.dumps(update, ensure_ascii=False)}\n\n"
            except Exception as e:
                error_message = str(e)
                logger.translation_logger.error(f"Translation error: {error_message}")
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute('''
                        UPDATE translations 
                        SET status = 'error', error_message = ?
                        WHERE id = ?
                    ''', (error_message, translation_id))
                yield f"data: {json.dumps({'error': error_message})}\n\n"
                
        return Response(generate(), mimetype='text/event-stream')
    
    except Exception as e:
        logger.app_logger.error(f"Translation request error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            os.remove(filepath)
        except Exception as e:
            logger.app_logger.error(f"Failed to cleanup uploaded file: {str(e)}")

@app.route('/download/<int:translation_id>', methods=['GET'])
@with_error_handling
def download_translation(translation_id):
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
        
        # Create download file with raw text
        download_path = os.path.join(TRANSLATIONS_FOLDER, f'translated_{filename}')
        with open(download_path, 'w', encoding='utf-8') as f:
            f.write(translated_text)
            
        return send_file(
            download_path,
            as_attachment=True,
            download_name=f'translated_{filename}'
        )

@app.route('/failed-translations', methods=['GET'])
@with_error_handling
def get_failed_translations():
    return jsonify(recovery.get_failed_translations())

@app.route('/retry-translation/<int:translation_id>', methods=['POST'])
@with_error_handling
def retry_failed_translation(translation_id):
    recovery.retry_translation(translation_id)
    return jsonify({'status': 'success'})

@app.route('/metrics', methods=['GET'])
def get_metrics():
    return jsonify(monitor.get_metrics())

@app.route('/health', methods=['GET'])
def health_check():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('SELECT 1')
            
        disk_usage = psutil.disk_usage('/')
        if disk_usage.percent > 90:
            logger.app_logger.warning("Low disk space")
            
        return jsonify({
            'status': 'healthy',
            'ollama': 'connected',
            'database': 'connected',
            'disk_usage': f"{disk_usage.percent}%"
        })
    except Exception as e:
        logger.app_logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

def cleanup_old_data():
    while True:
        try:
            logger.app_logger.info("Running cleanup task")
            cache.cleanup_old_entries(days=30)
            recovery.cleanup_failed_translations(days=7)
            time.sleep(24 * 60 * 60)  # Run daily
        except Exception as e:
            logger.app_logger.error(f"Cleanup task error: {str(e)}")
            time.sleep(60 * 60)  # Retry in an hour

cleanup_thread = threading.Thread(target=cleanup_old_data, daemon=True)
cleanup_thread.start()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
