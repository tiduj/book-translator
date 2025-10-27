from ebooklib import epub
from bs4 import BeautifulSoup
import json
import requests
import time
from typing import List, Dict, Optional, Callable
import os
import sqlite3
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler
import hashlib
import traceback
import psutil
import threading
import signal
import atexit
from collections import deque
from dataclasses import dataclass, field
from functools import wraps
from flask import Flask, request, jsonify, Response, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import zipfile
import uuid
from datetime import datetime as dt
import re
import sys

# ----------------- HERE IS THE CHANGE ------------------
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
# -------------------------------------------------------

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
            metrics_data = {
                'translation_metrics': {
                    'total_requests': self.metrics.total_requests,
                    'successful_translations': self.metrics.successful_translations,
                    'failed_translations': self.metrics.failed_translations,
                    'average_translation_time': self.metrics.average_translation_time
                },
                'system_metrics': self.get_system_metrics()
            }
            if self.metrics.total_requests > 0:
                metrics_data['translation_metrics']['success_rate'] = (
                    self.metrics.successful_translations / self.metrics.total_requests * 100
                )
            else:
                metrics_data['translation_metrics']['success_rate'] = 0
            return metrics_data

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
                    machine_translation TEXT,
                    created_at TIMESTAMP,
                    last_used TIMESTAMP
                )
            ''')
    def _generate_hash(self, text: str, source_lang: str, target_lang: str, model: str = "") -> str:
        key = f"{text}:{source_lang}:{target_lang}:{model}".encode('utf-8')
        return hashlib.sha256(key).hexdigest()
    def get_cached_translation(self, text: str, source_lang: str, target_lang: str, model: str = "") -> Optional[Dict[str, str]]:
        hash_key = self._generate_hash(text, source_lang, target_lang, model)
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute('''
                SELECT translated_text, machine_translation
                FROM translation_cache
                WHERE hash_key = ?
            ''', (hash_key,))
            result = cur.fetchone()
            if result:
                conn.execute('''
                    UPDATE translation_cache
                    SET last_used = CURRENT_TIMESTAMP
                    WHERE hash_key = ?
                ''', (hash_key,))
                return {
                    'translated_text': result[0],
                    'machine_translation': result[1]
                }
        return None
    def cache_translation(self, text: str, translated_text: str, machine_translation: str,
                         source_lang: str, target_lang: str, model: str = ""):
        hash_key = self._generate_hash(text, source_lang, target_lang, model)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO translation_cache
                (hash_key, source_lang, target_lang, original_text, translated_text,
                 machine_translation, created_at, last_used)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (hash_key, source_lang, target_lang, text, translated_text, machine_translation))
    def cleanup_old_entries(self, days: int = 30):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"DELETE FROM translation_cache WHERE last_used < datetime('now', '-{days} days')"
            )

# Initialize cache
cache = TranslationCache()

# Terminology Manager
class TerminologyManager:
    """Manages consistent terminology across translation chunks"""
    def __init__(self):
        self.terms = {}  # {original_term: translated_term}
        self.proper_nouns = set()
    def extract_proper_nouns(self, text: str) -> List[str]:
        """Extract proper nouns (capitalized words/phrases)"""
        pattern = r'(?<!^)(?<![.!?]\s)\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        nouns = re.findall(pattern, text, re.MULTILINE)
        return list(set(nouns))
    def add_term(self, original: str, translated: str):
        self.terms[original] = translated
    def get_term(self, original: str) -> Optional[str]:
        return self.terms.get(original)
    def ensure_consistency(self, text: str, chunk_terms: Dict[str, str]) -> str:
        for original, translated in chunk_terms.items():
            if original in self.terms and self.terms[original] != translated:
                text = text.replace(translated, self.terms[original])
            else:
                self.terms[original] = translated
        return text
    def get_terminology_context(self) -> str:
        if not self.terms:
            return ""
        term_list = [f"{orig} -> {trans}" for orig, trans in list(self.terms.items())[:10]]
        return "Important terms: " + ", ".join(term_list)

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
        conn.executescript('''
            DROP TABLE IF EXISTS chunks;
            DROP TABLE IF EXISTS translations;
            CREATE TABLE translations (
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
                machine_translation TEXT,
                translated_text TEXT,
                detected_language TEXT,
                genre TEXT DEFAULT 'unknown',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                error_message TEXT
            );
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                translation_id INTEGER,
                chunk_number INTEGER,
                original_text TEXT,
                machine_translation TEXT,
                translated_text TEXT,
                status TEXT,
                error_message TEXT,
                attempts INTEGER DEFAULT 0,
                FOREIGN KEY (translation_id) REFERENCES translations (id)
            );
        ''')

init_db()

class BookTranslator:
    def __init__(self, model_name: str = "llama3.3:70b-instruct-q2_K", chunk_size: int = 1000):
        self.model_name = model_name
        # -------- HERE THE API URL IS USING THE ENV -----------
        self.api_url = f"{OLLAMA_HOST}/api/generate"
        # ------------------------------------------------------
        self.chunk_size = chunk_size
        self.session = requests.Session()
        self.session.mount('http://', requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=10
        ))
        self.terminology = TerminologyManager()

    def split_into_chunks(self, text: str) -> list:
        MAX_LENGTH = 4500  # Google Translate limit
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        for paragraph in paragraphs:
            if len(paragraph) + current_length > MAX_LENGTH:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                if len(paragraph) > MAX_LENGTH:
                    sentences = paragraph.split('. ')
                    temp_chunk = []
                    temp_length = 0
                    for sentence in sentences:
                        if temp_length + len(sentence) > MAX_LENGTH:
                            if temp_chunk:
                                chunks.append('. '.join(temp_chunk) + '.')
                                temp_chunk = []
                                temp_length = 0
                        temp_chunk.append(sentence)
                        temp_length += len(sentence) + 2  # +2 for '. '
                    if temp_chunk:
                        chunks.append('. '.join(temp_chunk) + '.')
                else:
                    current_chunk.append(paragraph)
                    current_length = len(paragraph)
            else:
                current_chunk.append(paragraph)
                current_length += len(paragraph) + 2  # +2 for '\n\n'
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        return chunks

    def translate_text(self, text: str, source_lang: str, target_lang: str, translation_id: int, genre: str = 'unknown'):
        start_time = time.time()
        success = False
        try:
            chunks = self.split_into_chunks(text)
            total_chunks = len(chunks)
            draft_translations = []
            final_translations = []
            self.terminology = TerminologyManager()
            logger.translation_logger.info(f"Starting translation {translation_id} with {total_chunks} chunks (genre: {genre})")
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute('''
                    UPDATE translations
                    SET total_chunks = ?, status = 'in_progress', genre = ?
                    WHERE id = ?
                ''', (total_chunks * 2, genre, translation_id))
            # STAGE 1: Primary translation
            logger.translation_logger.info("Stage 1: Primary LLM translation")
            for i, chunk in enumerate(chunks, 1):
                try:
                    cached_result = cache.get_cached_translation(chunk, source_lang, target_lang, self.model_name + "_stage1")
                    if cached_result:
                        draft_translation = cached_result['machine_translation']
                        logger.translation_logger.info(f"Cache hit for stage 1 chunk {i}")
                    else:
                        previous_chunk = draft_translations[-1] if draft_translations else ""
                        logger.translation_logger.info(f"Stage 1 translating chunk {i}/{total_chunks}")
                        draft_translation = self.stage1_primary_translation(
                            text=chunk,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            previous_chunk=previous_chunk,
                            genre=genre
                        )
                        cache.cache_translation(
                            chunk, draft_translation, draft_translation,
                            source_lang, target_lang, self.model_name + "_stage1"
                        )
                        time.sleep(0.5)
                    draft_translations.append(draft_translation)
                    progress = (i / (total_chunks * 2)) * 100
                    yield {
                        'progress': progress,
                        'stage': 'primary_translation',
                        'original_text': '\n\n'.join(chunks),
                        'machine_translation': '\n\n'.join(draft_translations),
                        'current_chunk': i,
                        'total_chunks': total_chunks * 2
                    }
                except Exception as e:
                    error_msg = f"Error in stage 1 chunk {i}: {str(e)}"
                    logger.translation_logger.error(error_msg)
                    logger.translation_logger.error(traceback.format_exc())
                    raise Exception(error_msg)
            # STAGE 2: Reflection and improvement
            logger.translation_logger.info("Stage 2: Reflection and improvement")
            for i, (original_chunk, draft_chunk) in enumerate(zip(chunks, draft_translations), 1):
                try:
                    cached_result = cache.get_cached_translation(original_chunk, source_lang, target_lang, self.model_name + "_stage2")
                    if cached_result:
                        final_translation = cached_result['translated_text']
                        logger.translation_logger.info(f"Cache hit for stage 2 chunk {i}")
                    else:
                        previous_final = final_translations[-1] if final_translations else ""
                        logger.translation_logger.info(f"Stage 2 improving chunk {i}/{total_chunks}")
                        final_translation = self.stage2_reflection_improvement(
                            original_text=original_chunk,
                            draft_translation=draft_chunk,
                            source_lang=source_lang,
                            target_lang=target_lang,
                            previous_chunk=previous_final,
                            genre=genre
                        )
                        cache.cache_translation(
                            original_chunk, final_translation, draft_chunk,
                            source_lang, target_lang, self.model_name + "_stage2"
                        )
                        time.sleep(0.5)
                    final_translations.append(final_translation)
                    progress = ((i + total_chunks) / (total_chunks * 2)) * 100
                    with sqlite3.connect(DB_PATH) as conn:
                        conn.execute('''
                            UPDATE translations
                            SET progress = ?,
                                translated_text = ?,
                                machine_translation = ?,
                                current_chunk = ?,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        ''', (
                            progress,
                            '\n\n'.join(final_translations),
                            '\n\n'.join(draft_translations),
                            i + total_chunks,
                            translation_id
                        ))
                    yield {
                        'progress': progress,
                        'stage': 'reflection_improvement',
                        'original_text': '\n\n'.join(chunks),
                        'machine_translation': '\n\n'.join(draft_translations),
                        'translated_text': '\n\n'.join(final_translations),
                        'current_chunk': i + total_chunks,
                        'total_chunks': total_chunks * 2
                    }
                except Exception as e:
                    error_msg = f"Error in stage 2 chunk {i}: {str(e)}"
                    logger.translation_logger.error(error_msg)
                    logger.translation_logger.error(traceback.format_exc())
                    final_translations.append(draft_chunk)
                    raise Exception(error_msg)
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute('''
                    UPDATE translations
                    SET status = 'completed',
                        progress = 100,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (translation_id,))
            success = True
            yield {
                'progress': 100,
                'original_text': '\n\n'.join(chunks),
                'machine_translation': '\n\n'.join(draft_translations),
                'translated_text': '\n\n'.join(final_translations),
                'status': 'completed'
            }
        except Exception as e:
            error_msg = f"Translation failed: {str(e)}"
            logger.translation_logger.error(error_msg)
            logger.translation_logger.error(traceback.format_exc())
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute('''
                    UPDATE translations
                    SET status = 'error',
                        error_message = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (str(e), translation_id))
            raise
        finally:
            translation_time = time.time() - start_time
            monitor.record_translation_attempt(success, translation_time)

    def stage1_primary_translation(self, text: str, source_lang: str, target_lang: str,
                                   previous_chunk: str = "", genre: str = "unknown") -> str:
        lang_names = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese',
            'ja': 'Japanese', 'ko': 'Korean'
        }
        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)
        context_section = f"\n\nPrevious translated paragraph:\n{previous_chunk}" if previous_chunk else ""
        prompt = f"""You are a professional translator. Translate from {source_name} to {target_lang}.
CONTEXT:
- Document type: {genre}
- Preserve formatting (paragraphs, line breaks)
- Adapt idioms and cultural references for target audience
- Maintain tone and emotional coloring of original
{context_section}
TEXT TO TRANSLATE:
{text}
Return ONLY the translation without comments."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.6}
        }
        try:
            response = self.session.post(self.api_url, json=payload, timeout=(30, 300))  # 5 min timeout
            response.raise_for_status()
            result = json.loads(response.text)
            if 'response' in result:
                return result['response'].strip()
            logger.api_logger.warning("No response field in Stage 1 result")
            return text
        except requests.exceptions.Timeout:
            logger.api_logger.error(f"Stage 1 timeout after 300s - text too long or model too slow")
            return text
        except Exception as e:
            logger.api_logger.error(f"Stage 1 error: {e}")
            return text

    def stage2_reflection_improvement(self, original_text: str, draft_translation: str,
                                     source_lang: str, target_lang: str,
                                     previous_chunk: str = "", genre: str = "unknown") -> str:
        lang_names = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese',
            'ja': 'Japanese', 'ko': 'Korean'
        }
        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)
        context_section = f"\n\nPrevious final paragraph:\n{previous_chunk}" if previous_chunk else ""
        prompt = f"""You are a translation editor. Review and improve this translation.
ORIGINAL ({source_name}):
{original_text}
DRAFT TRANSLATION ({target_name}):
{draft_translation}
{context_section}
TASK:
Critically evaluate the translation:
1. ACCURACY - Is all meaning preserved?
2. NATURALNESS - Does it sound like a native speaker wrote it?
3. STYLE & TONE - Is the register and emotional coloring maintained?
4. CULTURAL ADAPTATION - Are idioms and references adapted?
Return ONLY the improved final translation without explanations."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.4}
        }
        try:
            response = self.session.post(self.api_url, json=payload, timeout=(30, 300))  # 5 min timeout
            response.raise_for_status()
            result = json.loads(response.text)
            if 'response' in result:
                return result['response'].strip()
            logger.api_logger.warning("No response field in Stage 2 result, using draft")
            return draft_translation
        except requests.exceptions.Timeout:
            logger.api_logger.error(f"Stage 2 timeout after 300s - using draft translation")
            return draft_translation
        except Exception as e:
            logger.api_logger.error(f"Stage 2 error: {e}")
            return draft_translation

    def get_available_models(self) -> List[str]:
        # --------- HERE WE USE ENV VARIABLE OLLAMA_HOST ------
        response = self.session.get(f"{OLLAMA_HOST}/api/tags", timeout=(5, 5))
        # -----------------------------------------------------
        response.raise_for_status()
        models = response.json()
        return [model['name'] for model in models['models']]

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
            conn.execute(
                f"DELETE FROM translations WHERE status = 'error' AND created_at < datetime('now', '-{days} days')"
            )
recovery = TranslationRecovery()

# Health checking middleware
@app.before_request
def check_ollama():
    if request.endpoint != 'health_check':
        try:
            # --------- HERE WE USE OLLAMA_HOST ENV VARIABLE ----
            response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            # ---------------------------------------------------
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.app_logger.error(f"Ollama health check failed: {str(e)}")
            return jsonify({
                'error': 'Translation service is not available'
            }), 503

# (All your Flask routes here, unchanged)
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

# (Rest of routes follow, unchanged...)

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
def extract_text_from_epub(epub_path):
    book = epub.read_epub(epub_path)
    text_content = []
    for item in book.get_items():
        if item.get_type() == epub.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            paragraphs = [p.get_text() for p in soup.find_all('p')]
            text_content.extend(paragraphs)
    return '\n\n'.join(text_content)
@app.route('/translate', methods=['POST'])
@with_error_handling
def translate():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    try:
        file = request.files['file']
        source_lang = request.form.get('sourceLanguage')
        target_lang = request.form.get('targetLanguage')
        model_name = request.form.get('model')
        genre = request.form.get('genre', 'unknown')
        if not all([file, source_lang, target_lang, model_name]):
            return jsonify({'error': 'Missing required parameters'}), 400
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        if filename.lower().endswith('.epub'):
            try:
                text = extract_text_from_epub(filepath)
            except Exception as e:
                logger.app_logger.error(f"Failed to extract EPUB: {str(e)}")
                return jsonify({'error': f"Failed to read EPUB: {str(e)}"}), 400
        else:
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
                    status, original_text, genre
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (filename, source_lang, target_lang, model_name,
                  'in_progress', text, genre))
            translation_id = cur.lastrowid
        translator = BookTranslator(model_name=model_name)
        def generate():
            try:
                for update in translator.translate_text(text, source_lang, target_lang, translation_id, genre=genre):
                    yield f"data: {json.dumps(update, ensure_ascii=False)}\n\n"
            except Exception as e:
                error_message = str(e)
                logger.translation_logger.error(f"Translation error: {error_message}")
                logger.translation_logger.error(traceback.format_exc())
                yield f"data: {json.dumps({'error': error_message})}\n\n"
        return Response(generate(), mimetype='text/event-stream')
    except Exception as e:
        logger.app_logger.error(f"Translation request error: {str(e)}")
        logger.app_logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    finally:
        try:
            if 'filepath' in locals():
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
        download_path = os.path.join(TRANSLATIONS_FOLDER, f'translated_{filename}')
        with open(download_path, 'w', encoding='utf-8') as f:
            f.write(translated_text)
        return send_file(
            download_path,
            as_attachment=True,
            download_name=f'translated_{filename}'
        )

@app.route('/export/epub', methods=['POST'])
@with_error_handling
def export_epub():
    try:
        data = request.get_json()
        text = data.get('text', '')
        title = data.get('title', 'Translation')
        author = data.get('author', 'Book Translator')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        epub_id = str(uuid.uuid4())
        epub_filename = f'translation_{epub_id}.epub'
        epub_path = os.path.join(TRANSLATIONS_FOLDER, epub_filename)
        with zipfile.ZipFile(epub_path, 'w', zipfile.ZIP_DEFLATED) as epub:
            epub.writestr('mimetype', 'application/epub+zip', compress_type=zipfile.ZIP_STORED)
            container_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
    <rootfiles>
        <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
    </rootfiles>
</container>'''
            epub.writestr('META-INF/container.xml', container_xml)
            content_opf = f'''<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="BookID">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:title>{title}</dc:title>
        <dc:creator>{author}</dc:creator>
        <dc:language>en</dc:language>
        <dc:identifier id="BookID">{epub_id}</dc:identifier>
        <meta property="dcterms:modified">{dt.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}</meta>
    </metadata>
    <manifest>
        <item id="chapter1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
        <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    </manifest>
    <spine toc="ncx">
        <itemref idref="chapter1"/>
    </spine>
</package>'''
            epub.writestr('OEBPS/content.opf', content_opf)
            toc_ncx = f'''<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
    <head>
        <meta name="dtb:uid" content="{epub_id}"/>
        <meta name="dtb:depth" content="1"/>
    </head>
    <docTitle>
        <text>{title}</text>
    </docTitle>
    <navMap>
        <navPoint id="chapter1" playOrder="1">
            <navLabel>
                <text>Chapter 1</text>
            </navLabel>
            <content src="chapter1.xhtml"/>
        </navPoint>
    </navMap>
</ncx>'''
            epub.writestr('OEBPS/toc.ncx', toc_ncx)
            paragraphs = text.split('\n\n')
            html_paragraphs = ''.join([f'<p>{p.strip()}</p>\n' for p in paragraphs if p.strip()])
            chapter_xhtml = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
    <title>{title}</title>
    <style>
        body {{ font-family: serif; line-height: 1.6; margin: 2em; }}
        p {{ margin-bottom: 1em; text-indent: 1.5em; }}
        p:first-of-type {{ text-indent: 0; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    {html_paragraphs}
</body>
</html>'''
            epub.writestr('OEBPS/chapter1.xhtml', chapter_xhtml)
        logger.app_logger.info(f"EPUB created: {epub_filename}")
        return send_file(
            epub_path,
            as_attachment=True,
            download_name=f'{title.replace(" ", "_")}.epub',
            mimetype='application/epub+zip'
        )
    except Exception as e:
        logger.app_logger.error(f"EPUB export error: {str(e)}")
        logger.app_logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

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
        # --------- HERE WE USE OLLAMA_HOST ENV VARIABLE ----
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        # ---------------------------------------------------
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
            try:
                cache.cleanup_old_entries()
                logger.app_logger.info("Cache cleanup completed")
            except Exception as e:
                logger.app_logger.error(f"Cache cleanup error: {str(e)}")
            try:
                recovery.cleanup_failed_translations()
                logger.app_logger.info("Failed translations cleanup completed")
            except Exception as e:
                logger.app_logger.error(f"Failed translations cleanup error: {str(e)}")
            time.sleep(24 * 60 * 60)
        except Exception as e:
            logger.app_logger.error(f"Cleanup task error: {str(e)}")
            time.sleep(60 * 60)  # Retry in an hour

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_data, daemon=True)
cleanup_thread.start()

if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print("Shutting down gracefully...")
        try:
            if 'translator' in locals():
                translator.cleanup()
        except Exception as e:
            logger.app_logger.error(f"Cleanup error during shutdown: {str(e)}")
        if 'cleanup_thread' in globals() and cleanup_thread.is_alive():
            try:
                cleanup_thread._stop()
            except Exception as e:
                logger.app_logger.error(f"Error stopping cleanup thread: {str(e)}")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    app.run(host='0.0.0.0', port=5001, debug=True)
