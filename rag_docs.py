import os
import re
import json
import pickle
import time
import logging
import shutil
import asyncio
import hashlib
import threading
from typing import List, Tuple, Dict, Optional, Union, Any, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from abc import ABC, abstractmethod
import aiohttp
import requests

import numpy as np
import faiss
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# Конфигурация по умолчанию
DEFAULT_CONFIG = {
    # Пути к файлам и директориям
    "DOCS_FOLDER": "docs",
    "CACHE_FOLDER": "cache",
    "LOGS_FOLDER": "logs",

    # Настройки индексации
    "MAX_TOKENS": 256,  # длина чанка в словах
    "MIN_TOKENS": 128,   # минимальная длина чанка
    "OVERLAP": 128,      # перекрытие между чанками
    "CONTEXT_WINDOW": 1024,  # размер контекстного окна для чанков

    # Модели эмбеддингов
    "EMB_MODEL_ID": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
    "CROSS_ENCODER_ID": "cross-encoder/ms-marco-MiniLM-L-6-v2",

    # Настройки LLM
    "DEFAULT_LLM_PROVIDER": "ollama",
    "OLLAMA_API_URL": "http://localhost:11434/api/generate",
    "OLLAMA_MODEL": "deepseek-r1:14b",
    "OPENAI_API_KEY": "",
    "OPENAI_MODEL": "gpt-4o-mini",

    # Общие настройки
    "CACHE_ENABLED": True,
    "CACHE_TTL_HOURS": 24,
    "CACHE_MAX_ITEMS": 1000,
    "ADAPTIVE_CHUNKING": True,
    "EXTRACT_METADATA": True,
    "PARALLEL_PROCESSING": False,
    "MAX_WORKERS": 4,
    "API_MAX_RETRIES": 3,
    "LEMMATIZATION_ENABLED": True,
}

# Настройка логирования
def setup_logging(config):
    log_dir = config.get("LOGS_FOLDER", "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"rag_logs_{datetime.now().strftime('%Y%m%d')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("rag_docs")

#######################################################################
# Структуры данных для документов и чанков
#######################################################################

@dataclass
class ChunkMetadata:
    """Метаданные фрагмента документа"""
    doc_id: int
    chunk_id: int
    filename: str
    text: str
    heading: str = ""       # заголовок блока
    section: str = ""       # раздел
    article: str = ""       # статья
    chapter: str = ""       # глава
    paragraph: str = ""     # параграф/пункт
    part: str = ""          # часть
    raw_text: str = ""      # исходный текст (без форматирования)
    start_idx: int = 0      # индекс начала в исходном тексте
    end_idx: int = 0        # индекс конца в исходном тексте
    prev_chunk_id: int = -1 # ID предыдущего чанка (для контекста)
    next_chunk_id: int = -1 # ID следующего чанка (для контекста)
    context: str = ""       # контекстный текст (заголовки, начало документа и т.д.)

    def format_reference(self) -> str:
        """Форматирование ссылки на фрагмент для отображения"""
        parts = []

        if self.chapter:
            parts.append(f"Глава {self.chapter}")

        if self.article:
            article_ref = f"Статья {self.article}"
            if self.heading:
                article_ref += f": {self.heading}"
            parts.append(article_ref)
        elif self.heading:
            parts.append(self.heading)

        if self.part:
            parts.append(f"Часть {self.part}")

        if self.paragraph:
            parts.append(f"Пункт {self.paragraph}")

        if parts:
            return " | ".join(parts)
        else:
            return f"Документ: {self.filename}"

    def get_dictionary(self) -> Dict:
        """Получение словаря с метаданными"""
        return {k: v for k, v in asdict(self).items()
                if k not in ('raw_text', 'context')}


@dataclass
class DocumentMetadata:
    """Метаданные документа"""
    doc_id: int
    filename: str
    title: str = ""
    type: str = ""        # тип документа (закон, приказ и т.д.)
    date: str = ""        # дата документа
    number: str = ""      # номер документа
    author: str = ""      # автор/орган
    keywords: List[str] = field(default_factory=list)  # ключевые слова
    doc_structure: Dict = field(default_factory=dict)  # иерархическая структура

    @classmethod
    def extract_from_text(cls, doc_id: int, filename: str, text: str) -> 'DocumentMetadata':
        """Извлечение метаданных из текста документа"""
        meta = cls(doc_id=doc_id, filename=filename)

        # Извлекаем заголовок (ищем в первых 10 строках)
        lines = text.split('\n')
        for line in lines[:15]:
            line = line.strip()
            if not line:
                continue

            if line.startswith('# '):
                meta.title = line[2:].strip()
                break
            elif not meta.title and re.match(r'^[А-Я\s.,()«»-]+$', line[:40]):
                # Строка с заглавными буквами в начале - вероятно заголовок
                meta.title = line.strip()
                break

        # Если не нашли, берём первую непустую строку
        if not meta.title and lines[0].strip():
            meta.title = lines[0].strip()

        # Тип и номер документа
        doc_pattern = r"(Закон|Постановление|Указ|Приказ|Распоряжение|Кодекс|Федеральный закон)[\s\w]*?(?:от|№)\s*([0-9-А-Яа-я/]+)"
        doc_match = re.search(doc_pattern, text[:2000], re.IGNORECASE)
        if doc_match:
            meta.type = doc_match.group(1)
            meta.number = doc_match.group(2)

        # Дата документа
        date_patterns = [
            r"от\s*(\d{1,2}(?:\s*\S+|\.\d{2}\.)\s*\d{4})",  # от 01.01.2020
            r"от\s*(\d{1,2}\s+\w+\s+\d{4})",               # от 1 января 2020
            r"«(\d{1,2}»\s+\w+\s+\d{4})",                  # «1» января 2020
        ]

        for pattern in date_patterns:
            date_match = re.search(pattern, text[:2000])
            if date_match:
                meta.date = date_match.group(1)
                break

        # Автор/орган
        author_patterns = [
            r"(Президент[\s\w]+Федерации)[\s\n]+([А-Я]\.[А-Я]\.\s[А-Я][а-я]+)",
            r"(Председатель Правительства[\s\w]+Федерации)[\s\n]+([А-Я]\.[А-Я]\.\s[А-Я][а-я]+)",
            r"(Министр[\s\w]+)[\s\n]+([А-Я]\.[А-Я]\.\s[А-Я][а-я]+)",
        ]

        for pattern in author_patterns:
            author_match = re.search(pattern, text)
            if author_match:
                meta.author = f"{author_match.group(1)}, {author_match.group(2)}"
                break

        return meta

#######################################################################
# Поддержка лемматизации
#######################################################################

class Lemmatizer:
    """Класс для лемматизации текста"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.morph = None

        if enabled:
            try:
                import pymorphy2
                self.morph = pymorphy2.MorphAnalyzer()
                print("[rag_docs] Лемматизация включена с использованием pymorphy2")
            except ImportError:
                self.enabled = False
                print("[rag_docs] pymorphy2 не установлен. Лемматизация отключена.")

    def lemmatize(self, text: str) -> str:
        """Лемматизация текста"""
        if not self.enabled or not self.morph:
            return text.lower()

        words = re.findall(r'\w+', text.lower())
        lemmas = [self.morph.parse(word)[0].normal_form for word in words]
        return ' '.join(lemmas)

    def tokenize(self, text: str) -> List[str]:
        """Токенизация текста (для BM25)"""
        if not self.enabled or not self.morph:
            return [t.lower() for t in re.split(r'\W+', text) if t]

        words = re.findall(r'\w+', text.lower())
        return [self.morph.parse(w)[0].normal_form for w in words]

#######################################################################
# Кэширование запросов
#######################################################################

class QueryCache:
    """Кэш запросов с TTL и LRU-вытеснением"""

    def __init__(self, cache_path: str, max_items: int = 1000,
                 ttl_hours: int = 24, auto_save_interval: int = 20):
        self.cache_path = cache_path
        self.max_items = max_items
        self.ttl_hours = ttl_hours
        self.auto_save_interval = auto_save_interval
        self.cache = {}
        self.last_access = {}
        self.lock = threading.RLock()
        self.last_save_time = time.time()
        self.changes_since_save = 0
        self._load_cache()

    def _load_cache(self):
        """Загрузка кэша из файла"""
        if not os.path.exists(self.cache_path):
            return

        try:
            with open(self.cache_path, 'rb') as f:
                data = pickle.load(f)

            # Проверяем формат данных
            if isinstance(data, dict) and 'cache' in data and 'last_access' in data:
                self.cache = data['cache']
                self.last_access = data['last_access']

                # Удаляем просроченные записи
                self._cleanup_expired()
                print(f"[QueryCache] Загружен кэш запросов: {len(self.cache)} записей")
            else:
                # Старый формат или неверные данные
                print("[QueryCache] Обнаружен устаревший формат кэша, создаем новый")
                self.cache = {}
                self.last_access = {}
        except Exception as e:
            print(f"[QueryCache] Ошибка загрузки кэша: {e}")
            self.cache = {}
            self.last_access = {}

    def _cleanup_expired(self):
        """Удаление просроченных записей из кэша"""
        now = datetime.now()
        expiration = now - timedelta(hours=self.ttl_hours)

        expired_keys = [
            key for key, timestamp in self.last_access.items()
            if datetime.fromisoformat(timestamp) < expiration
        ]

        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
            if key in self.last_access:
                del self.last_access[key]

        return len(expired_keys)

    def _prune_cache(self):
        """Удаление наименее используемых записей если размер превышен"""
        if len(self.cache) <= self.max_items:
            return 0

        # Сортируем по времени последнего доступа
        items = [(k, datetime.fromisoformat(v))
                for k, v in self.last_access.items() if k in self.cache]
        items.sort(key=lambda x: x[1])  # Сортировка по дате доступа

        # Количество записей для удаления
        to_remove = len(self.cache) - self.max_items

        # Удаляем самые старые записи
        for i in range(min(to_remove, len(items))):
            key = items[i][0]
            if key in self.cache:
                del self.cache[key]
            if key in self.last_access:
                del self.last_access[key]

        return min(to_remove, len(items))

    def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша"""
        with self.lock:
            if key in self.cache:
                # Обновляем время последнего доступа
                self.last_access[key] = datetime.now().isoformat()
                return self.cache[key]
            return None

    def set(self, key: str, value: Any) -> None:
        """Добавление записи в кэш"""
        with self.lock:
            self.cache[key] = value
            self.last_access[key] = datetime.now().isoformat()
            self.changes_since_save += 1

            # Проверяем размер кэша и удаляем старые записи при необходимости
            self._prune_cache()

            # Автосохранение
            current_time = time.time()
            if (self.changes_since_save >= self.auto_save_interval or
                current_time - self.last_save_time > 60):  # Минимум раз в минуту при изменениях
                self.save()

    def save(self) -> bool:
        """Сохранение кэша в файл"""
        with self.lock:
            try:
                # Создаем родительскую директорию, если она не существует
                os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

                # Используем временный файл для атомарной записи
                temp_path = f"{self.cache_path}.tmp"
                with open(temp_path, 'wb') as f:
                    data = {
                        'cache': self.cache,
                        'last_access': self.last_access,
                        'saved_at': datetime.now().isoformat()
                    }
                    pickle.dump(data, f)

                # Атомарная замена
                if os.path.exists(temp_path):
                    os.replace(temp_path, self.cache_path)

                self.last_save_time = time.time()
                self.changes_since_save = 0
                return True
            except Exception as e:
                print(f"[QueryCache] Ошибка сохранения кэша: {e}")
                return False

    def clear(self) -> int:
        """Очистка всего кэша"""
        with self.lock:
            old_size = len(self.cache)
            self.cache = {}
            self.last_access = {}
            self.save()
            return old_size

    def stats(self) -> Dict:
        """Статистика кэша"""
        with self.lock:
            expired = self._cleanup_expired()
            return {
                'total_items': len(self.cache),
                'max_items': self.max_items,
                'ttl_hours': self.ttl_hours,
                'expired_removed': expired,
                'last_save': self.last_save_time,
                'changes_since_save': self.changes_since_save
            }

#######################################################################
# Модели эмбеддингов
#######################################################################

class EmbeddingModel:
    """Класс для работы с моделями эмбеддингов"""

    def __init__(self, model_id: str, cross_encoder_id: str = None):
        self.model_id = model_id
        self.cross_encoder_id = cross_encoder_id
        self._model = None
        self._cross_encoder = None
        self._lock = threading.RLock()

    @property
    def model(self):
        """Ленивая инициализация основной модели эмбеддингов"""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                        print(f"[EmbeddingModel] Загрузка модели {self.model_id}")
                        self._model = SentenceTransformer(self.model_id)
                    except ImportError:
                        raise ImportError("Не установлен пакет sentence_transformers. "
                                          "Установите через pip install sentence-transformers.")
                    except Exception as e:
                        raise RuntimeError(f"Ошибка загрузки модели {self.model_id}: {e}")
        return self._model

    @property
    def cross_encoder(self):
        """Ленивая инициализация cross-encoder модели"""
        if not self.cross_encoder_id:
            return None

        if self._cross_encoder is None:
            with self._lock:
                if self._cross_encoder is None:
                    try:
                        from sentence_transformers import CrossEncoder
                        print(f"[EmbeddingModel] Загрузка cross-encoder {self.cross_encoder_id}")
                        self._cross_encoder = CrossEncoder(self.cross_encoder_id)
                    except ImportError:
                        print("CrossEncoder не установлен, переранжирование отключено")
                        return None
                    except Exception as e:
                        print(f"Ошибка загрузки CrossEncoder: {e}, переранжирование отключено")
                        return None
        return self._cross_encoder

    def encode(self, texts: List[str], batch_size: int = 32,
               show_progress: bool = True) -> np.ndarray:
        """Кодирование текстов в эмбеддинги"""
        with self._lock:
            return self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress
            ).astype("float32")

    def rerank(self, query: str, texts: List[str],
               batch_size: int = 32) -> List[float]:
        """Переранжирование результатов с помощью cross-encoder"""
        if not self.cross_encoder:
            return [0.0] * len(texts)

        with self._lock:
            pairs = [(query, text) for text in texts]
            return self.cross_encoder.predict(pairs, batch_size=batch_size)

#######################################################################
# Преобразование и разбиение документов
#######################################################################

class DocumentStructure:
    """Класс для извлечения структуры документа"""

    # Регулярные выражения для извлечения структуры
    CHAPTER_PATTERNS = [
        re.compile(r'^##\s*Глава\s+(\d+(?:-\d+)?)(?:\.\s*(.*))?$', re.MULTILINE),
        re.compile(r'^Глава\s+(\d+(?:-\d+)?)(?:\.\s*(.*))?$', re.MULTILINE),
    ]

    ARTICLE_PATTERNS = [
        re.compile(r'^###\s*Статья\s+(\d+(?:-\d+)?)(?:\.\s*(.*))?$', re.MULTILINE),
        re.compile(r'^Статья\s+(\d+(?:-\d+)?)(?:\.\s*(.*))?$', re.MULTILINE),
    ]

    PART_PATTERN = re.compile(r'^####\s*Часть\s+(\d+(?:-\d+)?)(?:\.\s*(.*))?$', re.MULTILINE)
    PARAGRAPH_PATTERN = re.compile(r'^(\d+(?:\.\d+)*)\.\s+(.+)$', re.MULTILINE)
    SECTION_PATTERN = re.compile(r'^##\s*(?:Раздел|РАЗДЕЛ)\s+([IVX]+|[1-9][0-9]*)(?:\.\s*(.*))?$', re.MULTILINE)

    @classmethod
    def extract_structure(cls, text: str) -> Dict:
        """
        Извлекает иерархическую структуру документа:
        {
          "sections": [
            {
              "number": "I",
              "title": "...",
              "start": int,
              "end": int,
              "chapters": [
                {
                  "number": "1",
                  "title": "...",
                  "start": int,
                  "end": int,
                  "articles": [
                     {
                       "number": "1",
                       "title": "...",
                       "start": int,
                       "end": int,
                       "parts": [...]
                     }, ...
                  ]
                },
                ...
              ]
            },
            ...
          ]
        }
        """
        structure = {"sections": [], "chapters": [], "articles": []}

        # Извлекаем разделы
        sections = []
        for match in cls.SECTION_PATTERN.finditer(text):
            section_num = match.group(1)
            section_title = (match.group(2) or "").strip()
            sections.append({
                "number": section_num,
                "title": section_title,
                "start": match.start(),
                "end": None,
                "chapters": []
            })

        if sections:
            sections.sort(key=lambda s: s["start"])
            for i in range(len(sections) - 1):
                sections[i]["end"] = sections[i+1]["start"]
            sections[-1]["end"] = len(text)
            structure["sections"] = sections

        # Извлекаем главы
        chapters = []
        for pattern in cls.CHAPTER_PATTERNS:
            for match in pattern.finditer(text):
                chap_num = match.group(1)
                chap_title = (match.group(2) or "").strip()
                chapters.append({
                    "number": chap_num,
                    "title": chap_title,
                    "start": match.start(),
                    "end": None,
                    "articles": []
                })

        if chapters:
            chapters.sort(key=lambda c: c["start"])
            for i in range(len(chapters) - 1):
                chapters[i]["end"] = chapters[i+1]["start"]
            chapters[-1]["end"] = len(text)
            structure["chapters"] = chapters

            # Распределяем главы по разделам
            if sections:
                for chapter in chapters:
                    for section in sections:
                        if (chapter["start"] >= section["start"] and
                            (section["end"] is None or chapter["start"] < section["end"])):
                            section["chapters"].append(chapter)
                            break

        # Извлекаем статьи
        articles = []
        for pattern in cls.ARTICLE_PATTERNS:
            for match in pattern.finditer(text):
                art_num = match.group(1)
                art_title = (match.group(2) or "").strip()
                articles.append({
                    "number": art_num,
                    "title": art_title,
                    "start": match.start(),
                    "end": None,
                    "parts": []
                })

        if articles:
            articles.sort(key=lambda a: a["start"])
            for i in range(len(articles) - 1):
                articles[i]["end"] = articles[i+1]["start"]
            articles[-1]["end"] = len(text)
            structure["articles"] = articles

            # Распределяем статьи по главам
            for article in articles:
                assigned = False
                for chapter in chapters:
                    if (article["start"] >= chapter["start"] and
                        (chapter["end"] is None or article["start"] < chapter["end"])):
                        chapter["articles"].append(article)
                        assigned = True
                        break

                # Если нет глав или статья не попала ни в одну главу
                if not assigned and not chapters:
                    # Добавляем в корень
                    structure["articles"].append(article)

        # Извлекаем части статей
        parts = []
        for match in cls.PART_PATTERN.finditer(text):
            part_num = match.group(1)
            part_title = (match.group(2) or "").strip()
            parts.append({
                "number": part_num,
                "title": part_title,
                "start": match.start(),
                "end": None
            })

        if parts:
            parts.sort(key=lambda p: p["start"])
            for i in range(len(parts) - 1):
                parts[i]["end"] = parts[i+1]["start"]
            parts[-1]["end"] = len(text)

            # Распределяем части по статьям
            all_articles = structure["articles"].copy()
            for chapter in chapters:
                all_articles.extend(chapter["articles"])

            for part in parts:
                for article in all_articles:
                    if (part["start"] >= article["start"] and
                        (article["end"] is None or part["start"] < article["end"])):
                        article["parts"].append(part)
                        break

        return structure

    @classmethod
    def get_chunk_context(cls, text: str, chunk_start: int, chunk_end: int, structure: Dict) -> Dict:
        """
        Определяет, к каким структурным элементам относится данный чанк
        """
        context = {
            "section": "",
            "section_title": "",
            "chapter": "",
            "chapter_title": "",
            "article": "",
            "article_title": "",
            "part": "",
            "part_title": "",
            "paragraph": ""
        }

        # Поиск раздела
        for section in structure.get("sections", []):
            if (chunk_start >= section["start"] and
                (section["end"] is None or chunk_start < section["end"])):
                context["section"] = section["number"]
                context["section_title"] = section["title"]
                break

        # Поиск главы
        for chapter in structure.get("chapters", []):
            if (chunk_start >= chapter["start"] and
                (chapter["end"] is None or chunk_start < chapter["end"])):
                context["chapter"] = chapter["number"]
                context["chapter_title"] = chapter["title"]

                # Поиск статьи внутри главы
                for article in chapter.get("articles", []):
                    if (chunk_start >= article["start"] and
                        (article["end"] is None or chunk_start < article["end"])):
                        context["article"] = article["number"]
                        context["article_title"] = article["title"]

                        # Поиск части внутри статьи
                        for part in article.get("parts", []):
                            if (chunk_start >= part["start"] and
                                (part["end"] is None or chunk_start < part["end"])):
                                context["part"] = part["number"]
                                context["part_title"] = part["title"]
                                break
                        break
                break

        # Если не нашли в главах, ищем в корневых статьях
        if not context["article"]:
            for article in structure.get("articles", []):
                if (chunk_start >= article["start"] and
                    (article["end"] is None or chunk_start < article["end"])):
                    context["article"] = article["number"]
                    context["article_title"] = article["title"]

                    # Поиск части внутри статьи
                    for part in article.get("parts", []):
                        if (chunk_start >= part["start"] and
                            (part["end"] is None or chunk_start < part["end"])):
                            context["part"] = part["number"]
                            context["part_title"] = part["title"]
                            break
                    break

        # Поиск параграфа (пункта)
        chunk_text = text[chunk_start:chunk_end]
        paragraph_match = cls.PARAGRAPH_PATTERN.search(chunk_text)
        if paragraph_match:
            context["paragraph"] = paragraph_match.group(1)

        return context


class DocumentChunker:
    """Класс для разбиения документов на чанки с сохранением контекста"""

    def __init__(self, config: Dict):
        self.config = config
        self.max_tokens = config.get("MAX_TOKENS", 256)
        self.min_tokens = config.get("MIN_TOKENS", 50)
        self.overlap = config.get("OVERLAP", 64)
        self.context_window = config.get("CONTEXT_WINDOW", 512)
        self.adaptive_chunking = config.get("ADAPTIVE_CHUNKING", True)
        self.logger = logging.getLogger("rag_docs.chunker")

    @staticmethod
    def _safe_find(text: str, needle: str, start: int = 0) -> int:
        """
        Аналог str.find, но если подстрока не найдена,
        возвращает стартовую позицию, чтобы индекс всегда ≥ 0.
        """
        idx = text.find(needle, start)
        return idx if idx != -1 else start

    def _split_text_by_words(self, text: str, max_words: int,
                             overlap: int = 0) -> List[Tuple[str, int, int]]:
        """
        Разбивает текст на чанки по словам с перекрытием.
        Возвращает список кортежей (текст, abs_start, abs_end).
        Безопасен к коротким текстам: не уходит в бесконечный цикл
        и не использует отрицательные индексы.
        """
        words = text.split()
        if not words:
            return []

        chunks: List[Tuple[str, int, int]] = []
        start_idx = 0
        prev_text_pos = 0

        while start_idx < len(words):
            # если остаток короче MIN_TOKENS — добавляем к последнему чанку
            if len(words) - start_idx < self.min_tokens and chunks:
                last_text, last_abs_start, _ = chunks[-1]
                # расширяем уже сохранённый чанк «хвостом»
                extended = text[last_abs_start:]
                chunks[-1] = (extended, last_abs_start, len(text))
                break

            end_idx = min(start_idx + max_words, len(words))

            # --- безопасное вычисление абсолютных позиций ---
            if start_idx == 0:
                abs_start = 0
            else:
                abs_start = text.find(words[start_idx], prev_text_pos)
                if abs_start == -1:
                    abs_start = prev_text_pos

            abs_end_search = text.find(words[end_idx - 1], abs_start)
            if abs_end_search == -1:
                abs_end_search = len(text) - len(words[end_idx - 1])
            abs_end = abs_end_search + len(words[end_idx - 1])

            chunk_text = text[abs_start:abs_end]
            chunks.append((chunk_text, abs_start, abs_end))

            # если дошли до конца — прекращаем цикл
            if end_idx == len(words):
                break

            prev_text_pos = abs_start
            start_idx = end_idx - overlap
            if start_idx <= 0:  # защита от «негативного шага»
                start_idx = end_idx
        return chunks

    def _smart_split_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Умное разбиение текста с учетом структуры.
        Старается разбивать по абзацам, предложениям или смысловым границам.
        """
        # Сначала разбиваем по пустым строкам (параграфам)
        paragraphs = re.split(r'\n\s*\n', text.strip())

        chunks = []
        current_chunk = []
        current_words = 0
        start_text_idx = 0

        for para in paragraphs:
            para_words = len(para.split())

            # Если параграф сам по себе слишком большой, разбиваем его
            if para_words > self.max_tokens:
                # Если уже есть накопленный текст, сохраняем его как чанк
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    end_text_idx = text.find(current_chunk[-1], start_text_idx) + len(current_chunk[-1])
                    chunks.append((chunk_text, start_text_idx, end_text_idx))
                    current_chunk = []
                    current_words = 0

                # Разбиваем большой параграф по предложениям
                sentences = re.split(r'(?<=[.!?])\s+', para)

                temp_chunk = []
                temp_words = 0
                para_start_idx = text.find(para, start_text_idx)

                for sentence in sentences:
                    sent_words = len(sentence.split())

                    if temp_words + sent_words <= self.max_tokens:
                        temp_chunk.append(sentence)
                        temp_words += sent_words
                    else:
                        # Если предложение не помещается, сохраняем накопленный чанк
                        if temp_chunk:
                            chunk_text = ' '.join(temp_chunk)
                            sent_end_idx = text.find(temp_chunk[-1], para_start_idx) + len(temp_chunk[-1])
                            chunks.append((chunk_text, para_start_idx, sent_end_idx))
                            para_start_idx = sent_end_idx

                            # Если само предложение слишком длинное, разбиваем его на части
                            if sent_words > self.max_tokens:
                                sentence_chunks = self._split_text_by_words(sentence, self.max_tokens, self.overlap)
                                for s_chunk, s_start, s_end in sentence_chunks:
                                    # Корректируем индексы относительно оригинального текста
                                    abs_s_start = self._safe_find(text, s_chunk[:20], para_start_idx)
                                    abs_s_end = abs_s_start + len(s_chunk)
                                    chunks.append((s_chunk, abs_s_start, abs_s_end))
                                para_start_idx = abs_s_end
                            else:
                                # Начинаем новый чанк с этим предложением
                                temp_chunk = [sentence]
                                temp_words = sent_words
                        else:
                            # Разбиваем длинное предложение на части
                            sentence_chunks = self._split_text_by_words(sentence, self.max_tokens, self.overlap)
                            for s_chunk, s_start, s_end in sentence_chunks:
                                # Корректируем индексы относительно оригинального текста
                                abs_s_start = self._safe_find(text, s_chunk[:20], para_start_idx)
                                abs_s_end = abs_s_start + len(s_chunk)
                                chunks.append((s_chunk, abs_s_start, abs_s_end))
                            para_start_idx = abs_s_end

                # Добавляем оставшиеся предложения, если они есть
                if temp_chunk:
                    chunk_text = ' '.join(temp_chunk)
                    # Находим позицию этого текста в оригинале
                    last_sent_start = self._safe_find(text, temp_chunk[0], para_start_idx)
                    last_sent_end = last_sent_start + len(chunk_text)
                    chunks.append((chunk_text, last_sent_start, last_sent_end))

                start_text_idx = text.find(para, start_text_idx) + len(para)
            else:
                # Обычный параграф, добавляем его к текущему чанку, если помещается
                if current_words + para_words <= self.max_tokens:
                    current_chunk.append(para)
                    current_words += para_words
                else:
                    # Если не помещается, сохраняем накопленный чанк и начинаем новый
                    if current_chunk:
                        chunk_text = '\n\n'.join(current_chunk)
                        end_text_idx = text.find(current_chunk[-1], start_text_idx) + len(current_chunk[-1])
                        chunks.append((chunk_text, start_text_idx, end_text_idx))

                    current_chunk = [para]
                    current_words = para_words
                    start_text_idx = text.find(para, start_text_idx)

        # Добавляем последний чанк, если он есть
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            end_text_idx = min(text.find(current_chunk[-1], start_text_idx) + len(current_chunk[-1]), len(text))
            chunks.append((chunk_text, start_text_idx, end_text_idx))

        return chunks

    def _create_chunk_with_context(self,
                                  original_text: str,
                                  chunk_text: str,
                                  start_idx: int,
                                  end_idx: int,
                                  structure_context: Dict,
                                  doc_meta: DocumentMetadata,
                                  chunk_id: int) -> ChunkMetadata:
        """
        Создает метаданные чанка с контекстом.
        """
        # Создание основных метаданных
        meta = ChunkMetadata(
            doc_id=doc_meta.doc_id,
            chunk_id=chunk_id,
            filename=doc_meta.filename,
            text=chunk_text,
            start_idx=start_idx,
            end_idx=end_idx,
            raw_text=chunk_text
        )

        # Добавляем структурный контекст
        if structure_context:
            meta.section = structure_context.get("section", "")
            meta.chapter = structure_context.get("chapter", "")
            meta.article = structure_context.get("article", "")
            meta.part = structure_context.get("part", "")
            meta.paragraph = structure_context.get("paragraph", "")

            # Заголовки
            meta.heading = structure_context.get("article_title", "") or structure_context.get("chapter_title", "")

        # Добавляем окружающий контекст
        # Извлекаем до 100 слов перед началом чанка
        context_words = []

        if start_idx > 0:
            # Добавляем заголовок документа
            if doc_meta.title:
                context_words.append(f"Документ: {doc_meta.title}")

            # Добавляем иерархический контекст
            if meta.chapter:
                chapter_context = f"Глава {meta.chapter}"
                if structure_context.get("chapter_title"):
                    chapter_context += f": {structure_context['chapter_title']}"
                context_words.append(chapter_context)

            if meta.article:
                article_context = f"Статья {meta.article}"
                if structure_context.get("article_title"):
                    article_context += f": {structure_context['article_title']}"
                context_words.append(article_context)

            if meta.part:
                part_context = f"Часть {meta.part}"
                if structure_context.get("part_title"):
                    part_context += f": {structure_context['part_title']}"
                context_words.append(part_context)

        meta.context = "\n".join(context_words)

        return meta

    def _split_article_based(self,
                           text: str,
                           structure: Dict,
                           doc_meta: DocumentMetadata) -> List[Tuple[str, ChunkMetadata]]:
        """
        Разбивает документ на основе его структуры (главы, статьи, части).
        """
        results = []
        chunk_id_counter = 0

        # Сначала обрабатываем структуру статей внутри глав
        for ch in structure.get("chapters", []):
            chapter_label = f"Глава {ch['number']}"
            if ch["title"]:
                chapter_label += f". {ch['title']}"

            # Если в главе есть статьи
            if ch.get("articles"):
                for art in ch["articles"]:
                    article_label = f"Статья {art['number']}"
                    if art["title"]:
                        article_label += f". {art['title']}"

                    # Извлекаем текст статьи
                    art_text = text[art["start"]:art["end"]].strip()

                    # Если текст статьи слишком короткий, сохраняем как есть
                    if len(art_text.split()) <= self.max_tokens:
                        structure_context = {
                            "chapter": ch["number"],
                            "chapter_title": ch["title"],
                            "article": art["number"],
                            "article_title": art["title"]
                        }

                        meta = self._create_chunk_with_context(
                            original_text=text,
                            chunk_text=art_text,
                            start_idx=art["start"],
                            end_idx=art["end"],
                            structure_context=structure_context,
                            doc_meta=doc_meta,
                            chunk_id=chunk_id_counter
                        )

                        results.append((art_text, meta))
                        chunk_id_counter += 1
                    else:
                        # Разбиваем большую статью на части
                        if art.get("parts"):
                            # Если есть явно выделенные части, используем их
                            for part in art["parts"]:
                                part_text = text[part["start"]:part["end"]].strip()
                                structure_context = {
                                    "chapter": ch["number"],
                                    "chapter_title": ch["title"],
                                    "article": art["number"],
                                    "article_title": art["title"],
                                    "part": part["number"],
                                    "part_title": part["title"]
                                }

                                # Если часть слишком большая, разбиваем дальше
                                if len(part_text.split()) > self.max_tokens:
                                    chunks = self._smart_split_text(part_text)
                                    for chunk_text, rel_start, rel_end in chunks:
                                        abs_start = part["start"] + rel_start
                                        abs_end = part["start"] + rel_end

                                        meta = self._create_chunk_with_context(
                                            original_text=text,
                                            chunk_text=chunk_text,
                                            start_idx=abs_start,
                                            end_idx=abs_end,
                                            structure_context=structure_context,
                                            doc_meta=doc_meta,
                                            chunk_id=chunk_id_counter
                                        )

                                        results.append((chunk_text, meta))
                                        chunk_id_counter += 1
                                else:
                                    meta = self._create_chunk_with_context(
                                        original_text=text,
                                        chunk_text=part_text,
                                        start_idx=part["start"],
                                        end_idx=part["end"],
                                        structure_context=structure_context,
                                        doc_meta=doc_meta,
                                        chunk_id=chunk_id_counter
                                    )

                                    results.append((part_text, meta))
                                    chunk_id_counter += 1
                        else:
                            # Разбиваем статью по смысловым частям
                            chunks = self._smart_split_text(art_text)

                            structure_context = {
                                "chapter": ch["number"],
                                "chapter_title": ch["title"],
                                "article": art["number"],
                                "article_title": art["title"]
                            }

                            for chunk_text, rel_start, rel_end in chunks:
                                abs_start = art["start"] + rel_start
                                abs_end = art["start"] + rel_end

                                # Проверяем наличие параграфа
                                para_context = DocumentStructure.get_chunk_context(
                                    text, abs_start, abs_end, structure
                                )

                                structure_context.update({
                                    "paragraph": para_context.get("paragraph", "")
                                })

                                meta = self._create_chunk_with_context(
                                    original_text=text,
                                    chunk_text=chunk_text,
                                    start_idx=abs_start,
                                    end_idx=abs_end,
                                    structure_context=structure_context,
                                    doc_meta=doc_meta,
                                    chunk_id=chunk_id_counter
                                )

                                results.append((chunk_text, meta))
                                chunk_id_counter += 1
            else:
                # Глава без статей, обрабатываем как обычный текст
                chapter_text = text[ch["start"]:ch["end"]].strip()

                if len(chapter_text.split()) <= self.max_tokens:
                    structure_context = {
                        "chapter": ch["number"],
                        "chapter_title": ch["title"]
                    }

                    meta = self._create_chunk_with_context(
                        original_text=text,
                        chunk_text=chapter_text,
                        start_idx=ch["start"],
                        end_idx=ch["end"],
                        structure_context=structure_context,
                        doc_meta=doc_meta,
                        chunk_id=chunk_id_counter
                    )

                    results.append((chapter_text, meta))
                    chunk_id_counter += 1
                else:
                    # Разбиваем большую главу на части
                    chunks = self._smart_split_text(chapter_text)

                    structure_context = {
                        "chapter": ch["number"],
                        "chapter_title": ch["title"]
                    }

                    for chunk_text, rel_start, rel_end in chunks:
                        abs_start = ch["start"] + rel_start
                        abs_end = ch["start"] + rel_end

                        # Проверяем наличие параграфа
                        para_context = DocumentStructure.get_chunk_context(
                            text, abs_start, abs_end, structure
                        )

                        structure_context.update({
                            "paragraph": para_context.get("paragraph", "")
                        })

                        meta = self._create_chunk_with_context(
                            original_text=text,
                            chunk_text=chunk_text,
                            start_idx=abs_start,
                            end_idx=abs_end,
                            structure_context=structure_context,
                            doc_meta=doc_meta,
                            chunk_id=chunk_id_counter
                        )

                        results.append((chunk_text, meta))
                        chunk_id_counter += 1

        # Обрабатываем статьи, которые не входят в главы
        for art in structure.get("articles", []):
            # Проверяем, не обработана ли эта статья уже (в составе главы)
            is_processed = False
            for ch in structure.get("chapters", []):
                for ch_art in ch.get("articles", []):
                    if ch_art["start"] == art["start"]:
                        is_processed = True
                        break
                if is_processed:
                    break

            if is_processed:
                continue

            article_label = f"Статья {art['number']}"
            if art["title"]:
                article_label += f". {art['title']}"

            # Извлекаем текст статьи
            art_text = text[art["start"]:art["end"]].strip()

            # Если текст статьи слишком короткий, сохраняем как есть
            if len(art_text.split()) <= self.max_tokens:
                structure_context = {
                    "article": art["number"],
                    "article_title": art["title"]
                }

                meta = self._create_chunk_with_context(
                    original_text=text,
                    chunk_text=art_text,
                    start_idx=art["start"],
                    end_idx=art["end"],
                    structure_context=structure_context,
                    doc_meta=doc_meta,
                    chunk_id=chunk_id_counter
                )

                results.append((art_text, meta))
                chunk_id_counter += 1
            else:
                # Разбиваем большую статью аналогично предыдущему случаю
                if art.get("parts"):
                    # По частям
                    for part in art["parts"]:
                        part_text = text[part["start"]:part["end"]].strip()
                        structure_context = {
                            "article": art["number"],
                            "article_title": art["title"],
                            "part": part["number"],
                            "part_title": part["title"]
                        }

                        if len(part_text.split()) > self.max_tokens:
                            chunks = self._smart_split_text(part_text)
                            for chunk_text, rel_start, rel_end in chunks:
                                abs_start = part["start"] + rel_start
                                abs_end = part["start"] + rel_end

                                meta = self._create_chunk_with_context(
                                    original_text=text,
                                    chunk_text=chunk_text,
                                    start_idx=abs_start,
                                    end_idx=abs_end,
                                    structure_context=structure_context,
                                    doc_meta=doc_meta,
                                    chunk_id=chunk_id_counter
                                )

                                results.append((chunk_text, meta))
                                chunk_id_counter += 1
                        else:
                            meta = self._create_chunk_with_context(
                                original_text=text,
                                chunk_text=part_text,
                                start_idx=part["start"],
                                end_idx=part["end"],
                                structure_context=structure_context,
                                doc_meta=doc_meta,
                                chunk_id=chunk_id_counter
                            )

                            results.append((part_text, meta))
                            chunk_id_counter += 1
                else:
                    # По смысловым частям
                    chunks = self._smart_split_text(art_text)

                    structure_context = {
                        "article": art["number"],
                        "article_title": art["title"]
                    }

                    for chunk_text, rel_start, rel_end in chunks:
                        abs_start = art["start"] + rel_start
                        abs_end = art["start"] + rel_end

                        # Проверяем наличие параграфа
                        para_context = DocumentStructure.get_chunk_context(
                            text, abs_start, abs_end, structure
                        )

                        structure_context.update({
                            "paragraph": para_context.get("paragraph", "")
                        })

                        meta = self._create_chunk_with_context(
                            original_text=text,
                            chunk_text=chunk_text,
                            start_idx=abs_start,
                            end_idx=abs_end,
                            structure_context=structure_context,
                            doc_meta=doc_meta,
                            chunk_id=chunk_id_counter
                        )

                        results.append((chunk_text, meta))
                        chunk_id_counter += 1

        # Проверяем, покрыт ли весь документ
        covered_ranges = []
        for _, meta in results:
            covered_ranges.append((meta.start_idx, meta.end_idx))

        # Сортируем и объединяем перекрывающиеся диапазоны
        covered_ranges.sort()
        merged_ranges = []
        for start, end in covered_ranges:
            if not merged_ranges or start > merged_ranges[-1][1]:
                merged_ranges.append((start, end))
            else:
                merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))

        # Проверяем, есть ли непокрытые области
        current_pos = 0
        for start, end in merged_ranges:
            if start > current_pos:
                # Нашли непокрытую область
                uncovered_text = text[current_pos:start].strip()
                if uncovered_text and len(uncovered_text.split()) >= self.min_tokens:
                    self.logger.info(f"Найден непокрытый текст ({len(uncovered_text.split())} слов)")

                    # Разбиваем непокрытый текст
                    chunks = self._smart_split_text(uncovered_text)

                    for chunk_text, rel_start, rel_end in chunks:
                        abs_start = current_pos + rel_start
                        abs_end = current_pos + rel_end

                        # Определяем контекст
                        structure_context = DocumentStructure.get_chunk_context(
                            text, abs_start, abs_end, structure
                        )

                        meta = self._create_chunk_with_context(
                            original_text=text,
                            chunk_text=chunk_text,
                            start_idx=abs_start,
                            end_idx=abs_end,
                            structure_context=structure_context,
                            doc_meta=doc_meta,
                            chunk_id=chunk_id_counter
                        )

                        results.append((chunk_text, meta))
                        chunk_id_counter += 1

            current_pos = max(current_pos, end)

        # Проверяем хвост документа
        if current_pos < len(text):
            tail_text = text[current_pos:].strip()
            if tail_text and len(tail_text.split()) >= self.min_tokens:
                self.logger.info(f"Найден непокрытый хвост документа ({len(tail_text.split())} слов)")

                # Разбиваем хвост
                chunks = self._smart_split_text(tail_text)

                for chunk_text, rel_start, rel_end in chunks:
                    abs_start = current_pos + rel_start
                    abs_end = current_pos + rel_end

                    # Определяем контекст
                    structure_context = DocumentStructure.get_chunk_context(
                        text, abs_start, abs_end, structure
                    )

                    meta = self._create_chunk_with_context(
                        original_text=text,
                        chunk_text=chunk_text,
                        start_idx=abs_start,
                        end_idx=abs_end,
                        structure_context=structure_context,
                        doc_meta=doc_meta,
                        chunk_id=chunk_id_counter
                    )

                    results.append((chunk_text, meta))
                    chunk_id_counter += 1

        # Устанавливаем связи prev_chunk_id и next_chunk_id
        sorted_results = sorted(results, key=lambda x: x[1].start_idx)
        for i in range(len(sorted_results)):
            if i > 0:
                sorted_results[i][1].prev_chunk_id = sorted_results[i-1][1].chunk_id
            if i < len(sorted_results) - 1:
                sorted_results[i][1].next_chunk_id = sorted_results[i+1][1].chunk_id

        return sorted_results

    def _default_chunking(self,
                        text: str,
                        doc_meta: DocumentMetadata) -> List[Tuple[str, ChunkMetadata]]:
        """
        Стандартное разбиение текста на чанки (fallback).
        """
        chunks = self._smart_split_text(text)
        results = []

        for chunk_id, (chunk_text, start_idx, end_idx) in enumerate(chunks):
            meta = ChunkMetadata(
                doc_id=doc_meta.doc_id,
                chunk_id=chunk_id,
                filename=doc_meta.filename,
                text=chunk_text,
                start_idx=start_idx,
                end_idx=end_idx,
                raw_text=chunk_text
            )

            if doc_meta.title:
                meta.context = f"Документ: {doc_meta.title}"

            results.append((chunk_text, meta))

        # Устанавливаем связи prev_chunk_id и next_chunk_id
        for i in range(len(results)):
            if i > 0:
                results[i][1].prev_chunk_id = i - 1
            if i < len(results) - 1:
                results[i][1].next_chunk_id = i + 1

        return results

    def chunk_document(self,
                      text: str,
                      doc_meta: DocumentMetadata) -> List[Tuple[str, ChunkMetadata]]:
        """
        Основной метод разбиения документа на чанки.
        """
        try:
            if self.adaptive_chunking:
                # Извлекаем структуру документа
                structure = DocumentStructure.extract_structure(text)

                # Если нашли структуру, используем её для разбиения
                if structure.get("chapters") or structure.get("articles"):
                    self.logger.info(f"Найдена структура документа: {len(structure.get('chapters', []))} глав, "
                                     f"{len(structure.get('articles', []))} статей")
                    return self._split_article_based(text, structure, doc_meta)

            # Если нет структуры или adaptive_chunking отключен, используем стандартное разбиение
            self.logger.info("Используем стандартное разбиение текста")
            return self._default_chunking(text, doc_meta)
        except Exception:
            self.logger.exception("Ошибка при разбиении документа")  # напечатает весь стек
            return self._default_chunking(text, doc_meta)


#######################################################################
# LLM-провайдеры и генеративные модели
#######################################################################

class LLMProvider(ABC):
    """Абстрактный класс для провайдера LLM"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(f"rag_docs.llm.{self.__class__.__name__}")

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Генерация ответа на основе промпта"""
        pass


class OllamaProvider(LLMProvider):
    """Провайдер для Ollama API"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.api_url = config.get("OLLAMA_API_URL", "http://localhost:11434/api/generate")
        self.default_model = config.get("OLLAMA_MODEL", "deepseek-r1:14b")
        self.max_retries = config.get("API_MAX_RETRIES", 3)

    def generate(self, prompt: str,
                 model: str = None,
                 temperature: float = 0.5,
                 stream: bool = True,
                 **kwargs) -> str:
        """Генерация ответа с использованием Ollama API"""
        model = model or self.default_model

        payload = {
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "stream": stream,
                "num_ctx": kwargs.get("num_ctx", 8192)
            }
        }

        retries = 0
        while retries <= self.max_retries:
            try:
                resp = requests.post(self.api_url, json=payload, timeout=300)
                resp.raise_for_status()

                if stream:
                    answer = "".join(
                        json.loads(l).get("response", "")
                        for l in resp.iter_lines(decode_unicode=True)
                        if l.strip()
                    )
                else:
                    answer = resp.json().get("response", "")

                # Очищаем от служебных тегов, если они есть
                cleaned = re.sub(r"<think>[\s\S]*?</think>", "", answer, flags=re.I).strip()
                return cleaned or "(пустой ответ)"
            except Exception as e:
                retries += 1
                wait_time = 2 ** retries
                if retries <= self.max_retries:
                    self.logger.warning(f"Ошибка Ollama API: {e}, повтор через {wait_time} сек...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Не удалось получить ответ: {e}")
                    return f"Ошибка Ollama API: {e}"


class OpenAIProvider(LLMProvider):
    """Провайдер для OpenAI API"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.api_key = config.get("OPENAI_API_KEY", "")
        self.default_model = config.get("OPENAI_MODEL", "gpt-4o-mini")
        self.max_retries = config.get("API_MAX_RETRIES", 3)

        if not self.api_key:
            self.logger.warning("API ключ OpenAI не указан. Провайдер не будет работать.")

    def generate(self, prompt: str,
                 model: str = None,
                 temperature: float = 0.5,
                 **kwargs) -> str:
        """Генерация ответа с использованием OpenAI API"""
        if not self.api_key:
            return "Ошибка: API ключ OpenAI не указан. Укажите OPENAI_API_KEY в конфигурации."

        try:
            import openai

            # Настраиваем клиент
            openai.api_key = self.api_key

            # Создаем сообщения в формате чата
            messages = [
                {"role": "system", "content": "Ты – юридический ассистент. Используй только предоставленные фрагменты документов. Если данных нет – скажи об этом."},
                {"role": "user", "content": prompt}
            ]

            # Параметры запроса
            request_params = {
                "model": model or self.default_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": kwargs.get("max_tokens", 1000),
                "top_p": kwargs.get("top_p", 1.0),
                "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                "presence_penalty": kwargs.get("presence_penalty", 0.0)
            }

            # Выполняем запрос с повторными попытками при необходимости
            retries = 0
            while retries <= self.max_retries:
                try:
                    response = openai.ChatCompletion.create(**request_params)
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    retries += 1
                    wait_time = 2 ** retries
                    if retries <= self.max_retries:
                        self.logger.warning(f"Ошибка OpenAI API: {e}, повтор через {wait_time} сек...")
                        time.sleep(wait_time)
                    else:
                        self.logger.error(f"Не удалось получить ответ от OpenAI: {e}")
                        return f"Ошибка OpenAI API: {e}"
        except ImportError:
            return "Ошибка: пакет openai не установлен. Установите через pip install openai."
        except Exception as e:
            self.logger.error(f"Неожиданная ошибка: {e}")
            return f"Ошибка при работе с OpenAI API: {e}"


class LLMFactory:
    """Фабрика для создания экземпляров LLM-провайдеров"""

    @staticmethod
    def create_provider(provider_type: str, config: Dict) -> LLMProvider:
        """Создает экземпляр провайдера нужного типа"""
        provider_type = provider_type.lower()

        if provider_type == "ollama":
            return OllamaProvider(config)
        elif provider_type == "openai":
            return OpenAIProvider(config)
        else:
            raise ValueError(f"Неизвестный тип провайдера LLM: {provider_type}")

#######################################################################
# Основной класс RAG-системы
#######################################################################

class RAGSystem:
    """Основной класс RAG-системы для нормативных документов"""

    def __init__(self, config: Dict = None):
        """Инициализация системы"""
        self.config = config or DEFAULT_CONFIG

        # Настройка путей
        self.docs_folder = self.config.get("DOCS_FOLDER", "docs")
        self.cache_folder = self.config.get("CACHE_FOLDER", "cache")

        # Создаем папки, если их нет
        os.makedirs(self.docs_folder, exist_ok=True)
        os.makedirs(self.cache_folder, exist_ok=True)

        # Пути к файлам индексов и кэша
        self.index_faiss_path = os.path.join(self.cache_folder, "docs_index.faiss")
        self.index_meta_path = os.path.join(self.cache_folder, "docs_index_meta.pkl")
        self.index_bm25_path = os.path.join(self.cache_folder, "docs_bm25.pkl")
        self.query_cache_path = os.path.join(self.cache_folder, "query_cache.pkl")

        # Инициализация логирования
        self.logger = setup_logging(self.config)

        # Состояние индексов
        self.faiss_index = None
        self.chunks_meta = None
        self.doc_metadata = {}
        self._bm25 = None
        self._tokenized_chunks = None

        # Ленивая инициализация моделей
        self._embedding_model = None

        # Инициализация лемматизатора
        self.lemmatizer = Lemmatizer(self.config.get("LEMMATIZATION_ENABLED", True))

        # Инициализация кэша запросов
        if self.config.get("CACHE_ENABLED", True):
            self.query_cache = QueryCache(
                self.query_cache_path,
                max_items=self.config.get("CACHE_MAX_ITEMS", 1000),
                ttl_hours=self.config.get("CACHE_TTL_HOURS", 24)
            )
        else:
            self.query_cache = None

        # Блокировки для многопоточного доступа
        self._index_lock = threading.RLock()

    @property
    def embedding_model(self):
        """Ленивая инициализация модели эмбеддингов"""
        if self._embedding_model is None:
            emb_model_id = self.config.get("EMB_MODEL_ID", "sentence-transformers/multi-qa-mpnet-base-dot-v1")
            cross_encoder_id = self.config.get("CROSS_ENCODER_ID", None)
            self._embedding_model = EmbeddingModel(emb_model_id, cross_encoder_id)
        return self._embedding_model

    def _load_index(self) -> bool:
        """Загружает существующие индексы или создает новые"""
        with self._index_lock:
            if self.faiss_index is not None:
                return True

            if not (os.path.exists(self.index_faiss_path) and os.path.exists(self.index_meta_path)):
                self.logger.info("Нет готовых индексов, вызываем build_index...")
                return self.build_index()

            try:
                self.logger.info("Загрузка существующих индексов...")
                self.faiss_index = faiss.read_index(self.index_faiss_path)

                with open(self.index_meta_path, "rb") as f:
                    loaded_data = pickle.load(f)

                if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                    self.chunks_meta, self.doc_metadata = loaded_data
                else:
                    # Совместимость со старым форматом
                    self.chunks_meta = loaded_data
                    self.doc_metadata = {}

                if os.path.exists(self.index_bm25_path):
                    with open(self.index_bm25_path, "rb") as f:
                        self._tokenized_chunks = pickle.load(f)
                    self._bm25 = BM25Okapi(self._tokenized_chunks)
                else:
                    self.logger.warning("BM25 не найден, пересоздаём...")
                    self._regenerate_bm25_index()

                self.logger.info(f"Индексы загружены: {len(self.chunks_meta)} чанков, "
                                 f"{len(self.doc_metadata)} документов")
                return True
            except Exception as e:
                self.logger.error(f"Ошибка загрузки индексов: {e}, пересоздаём.")
                return self.build_index(force_rebuild=True)

    def _regenerate_bm25_index(self) -> bool:
        """Пересоздаёт BM25 индекс на основе chunks_meta"""
        if not self.chunks_meta:
            self.logger.error("Нет метаданных чанков, не можем создать BM25.")
            return False

        self.logger.info("Создание BM25 из существующих метаданных...")

        self._tokenized_chunks = [
            self.lemmatizer.tokenize(c.text) for c in self.chunks_meta
        ]
        self._bm25 = BM25Okapi(self._tokenized_chunks)

        with open(self.index_bm25_path, "wb") as f:
            pickle.dump(self._tokenized_chunks, f)

        self.logger.info(f"BM25 создан на {len(self._tokenized_chunks)} чанках.")
        return True

    def _process_document(self, doc_id: int, filename: str) -> Tuple[int, Optional[DocumentMetadata], List[Tuple[str, ChunkMetadata]]]:
        """Обработка одного документа"""
        chunks_with_meta = []
        doc_meta = None

        try:
            path = os.path.join(self.docs_folder, filename)
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()

            # Извлекаем метаданные документа
            if self.config.get("EXTRACT_METADATA", True):
                doc_meta = DocumentMetadata.extract_from_text(doc_id, filename, text)
            else:
                doc_meta = DocumentMetadata(doc_id=doc_id, filename=filename)

            # Разбиваем документ на чанки
            chunker = DocumentChunker(self.config)
            chunks_with_meta = chunker.chunk_document(text, doc_meta)

            self.logger.info(f"Документ {filename} обработан: {len(chunks_with_meta)} чанков")
            return doc_id, doc_meta, chunks_with_meta
        except Exception as e:
            self.logger.error(f"Ошибка при обработке {filename}: {e}")
            return doc_id, None, []

    def build_index(self, force_rebuild: bool = False) -> bool:
        """Создаёт индексы документов"""
        with self._index_lock:
            if not force_rebuild and os.path.exists(self.index_meta_path):
                try:
                    if self._load_index():
                        self.logger.info("Существующие индексы успешно загружены (no rebuild).")
                        return True
                except Exception as e:
                    self.logger.error(f"Ошибка при загрузке: {e}")
                    self.logger.info("Пересоздание индексов...")

            start_time = time.time()
            self.logger.info("Начало индексации документов...")

            doc_files = [
                (i, fn) for i, fn in enumerate(sorted(os.listdir(self.docs_folder)))
                if fn.endswith(".md") or fn.endswith(".txt")
            ]

            if not doc_files:
                err_msg = "Нет документов в папке docs/."
                self.logger.error(err_msg)
                raise RuntimeError(err_msg)

            all_chunks = []
            meta = []
            docs_meta = {}

            if self.config.get("PARALLEL_PROCESSING", True) and len(doc_files) > 1:
                max_workers = self.config.get("MAX_WORKERS", 4)
                self.logger.info(f"Параллельная обработка {len(doc_files)} документов (workers={max_workers})...")

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(self._process_document, doc_id, filename)
                        for doc_id, filename in doc_files
                    ]

                    for future in tqdm(as_completed(futures), total=len(futures), desc="Документы"):
                        doc_id, dmeta, chunk_list = future.result()
                        if dmeta:
                            docs_meta[doc_id] = dmeta
                        for chunk_text, chunk_meta in chunk_list:
                            all_chunks.append(chunk_text)
                            meta.append(chunk_meta)
            else:
                self.logger.info(f"Последовательная обработка {len(doc_files)} документов...")

                for doc_id, filename in tqdm(doc_files, desc="Документы"):
                    doc_id, dmeta, chunk_list = self._process_document(doc_id, filename)
                    if dmeta:
                        docs_meta[doc_id] = dmeta
                    for chunk_text, chunk_meta in chunk_list:
                        all_chunks.append(chunk_text)
                        meta.append(chunk_meta)

            if not all_chunks:
                err_msg = "Не получилось извлечь ни одного чанка."
                self.logger.error(err_msg)
                raise RuntimeError(err_msg)

            self.logger.info(f"Создание эмбеддингов для {len(all_chunks)} чанков...")
            embs = self.embedding_model.encode(all_chunks, batch_size=32, show_progress=True)

            self.faiss_index = faiss.IndexFlatL2(embs.shape[1])
            self.faiss_index.add(embs)

            self.logger.info("Создание BM25...")
            self._tokenized_chunks = [self.lemmatizer.tokenize(ch) for ch in all_chunks]
            self._bm25 = BM25Okapi(self._tokenized_chunks)

            self.logger.info("Сохранение индексов...")
            faiss.write_index(self.faiss_index, self.index_faiss_path)

            with open(self.index_meta_path, "wb") as f:
                pickle.dump((meta, docs_meta), f)

            with open(self.index_bm25_path, "wb") as f:
                pickle.dump(self._tokenized_chunks, f)

            self.chunks_meta = meta
            self.doc_metadata = docs_meta

            end_time = time.time()
            self.logger.info(f"Индексация завершена за {end_time - start_time:.2f} сек.")
            self.logger.info(f"Проиндексировано {len(all_chunks)} чанков из {len(doc_files)} файлов.")

            return True

    def search_docs(self, query: str, top_k: int = 12, bm_k: int = 8,
                   use_cache: bool = True, rerank: bool = True) -> List[Dict]:
        """Гибридный поиск (Faiss + BM25)"""
        # Кэширование запросов
        query_key = f"{query.strip().lower()}_{top_k}_{bm_k}_{rerank}"

        if use_cache and self.query_cache:
            cached_result = self.query_cache.get(query_key)
            if cached_result:
                self.logger.info(f"Найдено в кэше: '{query}'")
                return cached_result

        # Загружаем индексы, если они еще не загружены
        if not self._load_index():
            err_msg = "Не удалось загрузить индексы"
            self.logger.error(err_msg)
            raise RuntimeError(err_msg)

        self.logger.info(f"Поиск: '{query}' (top_k={top_k}, bm_k={bm_k}, rerank={rerank})")
        start_time = time.time()

        # Semantic search с использованием Faiss
        q_emb = self.embedding_model.encode([query]).astype("float32")
        D, I = self.faiss_index.search(q_emb, top_k)
        semantic_scores = {idx: 1.0 / (1.0 + float(dist)) for dist, idx in zip(D[0], I[0]) if idx != -1}

        # Lexical search с использованием BM25
        tokenized_query = self.lemmatizer.tokenize(query)
        bm_scores = self._bm25.get_scores(tokenized_query)
        top_bm_indices = np.argsort(bm_scores)[::-1][:bm_k]

        # Нормализация BM25 оценок
        max_bm_score = np.max(bm_scores) if bm_scores.size > 0 else 1.0
        lexical_scores = {
            idx: float(bm_scores[idx] / max_bm_score)
            for idx in top_bm_indices
        }

        # Объединение результатов и гибридное ранжирование
        candidate_indices = set(semantic_scores) | set(lexical_scores)

        hybrid_scores = []
        for idx in candidate_indices:
            sem_score = semantic_scores.get(idx, 0.0)
            lex_score = lexical_scores.get(idx, 0.0)

            # Гибридная оценка: комбинация семантической и лексической
            hybrid_score = 0.7 * sem_score + 0.3 * lex_score
            hybrid_scores.append((hybrid_score, idx))

        # Сортировка по гибридной оценке
        hybrid_scores.sort(key=lambda x: x[0], reverse=True)

        # Формирование результатов
        results = []
        indices_to_rerank = []
        chunks_to_rerank = []

        for score, idx in hybrid_scores[:top_k]:
            if idx >= len(self.chunks_meta):
                self.logger.warning(f"Индекс {idx} вне диапазона (len={len(self.chunks_meta)})")
                continue

            chunk_meta = self.chunks_meta[idx]
            doc_id = chunk_meta.doc_id

            # Информация о документе
            doc_info = {}
            if doc_id in self.doc_metadata:
                dm = self.doc_metadata[doc_id]
                doc_info = {
                    "doc_title": dm.title,
                    "doc_type": dm.type,
                    "doc_date": dm.date,
                    "doc_number": dm.number,
                    "doc_author": dm.author
                }

            # Формирование ссылки на фрагмент
            reference = chunk_meta.format_reference()

            # Создание записи результата
            result = {
                "hybrid_score": score,
                "semantic_score": semantic_scores.get(idx, 0.0),
                "lexical_score": lexical_scores.get(idx, 0.0),
                "filename": chunk_meta.filename,
                "chunk_text": chunk_meta.text,
                "reference": reference,
                "doc_id": doc_id,
                "chunk_id": chunk_meta.chunk_id,
                "article": chunk_meta.article,
                "chapter": chunk_meta.chapter,
                "section": chunk_meta.section,
                "heading": chunk_meta.heading,
                "prev_chunk_id": chunk_meta.prev_chunk_id,
                "next_chunk_id": chunk_meta.next_chunk_id,
                **doc_info
            }

            results.append(result)

            # Сохраняем информацию для переранжирования
            if rerank and self.embedding_model.cross_encoder:
                indices_to_rerank.append(len(results) - 1)
                chunks_to_rerank.append(chunk_meta.text)

        # Переранжирование с помощью Cross-encoder (если доступен)
        if rerank and self.embedding_model.cross_encoder and chunks_to_rerank:
            self.logger.info(f"Переранжирование {len(chunks_to_rerank)} результатов (Cross-Encoder)")

            ce_scores = self.embedding_model.rerank(query, chunks_to_rerank)

            for idx, ce_score in zip(indices_to_rerank, ce_scores):
                results[idx]["ce_score"] = float(ce_score)

            # Пересортировка результатов на основе cross-encoder оценок
            results.sort(key=lambda x: x.get("ce_score", 0.0), reverse=True)

        # Кэширование результатов
        if self.query_cache:
            self.query_cache.set(query_key, results[:top_k])

        end_time = time.time()
        self.logger.info(f"Поиск: {end_time - start_time:.3f} с, результатов: {len(results)}")

        return results[:top_k]

    def _build_context_from_results(self, results: List[Dict], include_context: bool = True) -> str:
        """Строит контекстное представление результатов поиска для промпта"""
        context_fragments = []

        for i, frag in enumerate(results):
            # Базовая ссылка на фрагмент
            ref = frag.get("reference") or f"ФРАГМЕНТ {i + 1}"

            # Информация о документе
            doc_info = ""
            if frag.get("doc_title"):
                doc_info = f" ({frag['doc_title']})"
                if frag.get("doc_number"):
                    doc_info += f" №{frag['doc_number']}"
                if frag.get("doc_date"):
                    doc_info += f" от {frag['doc_date']}"

            # Создаем форматированный заголовок фрагмента
            fragment_header = f"{ref}{doc_info}:"

            # Добавляем текст фрагмента
            chunk_text = frag['chunk_text']

            # Формируем полное представление фрагмента
            fragment_repr = f"{fragment_header}\n{chunk_text}"
            context_fragments.append(fragment_repr)

        # Объединяем все фрагменты в один контекст
        context = "\n\n".join(context_fragments)
        return context

    def _get_llm_provider(self, provider_type: str = None) -> LLMProvider:
        """Получает экземпляр провайдера LLM нужного типа"""
        provider_type = provider_type or self.config.get("DEFAULT_LLM_PROVIDER", "ollama")
        return LLMFactory.create_provider(provider_type, self.config)

    def answer_with_rag(self, question: str,
                       top_k: int = 10,
                       llm_provider: str = None,
                       temperature: float = 0.5,
                       rerank: bool = True,
                       **kwargs) -> Dict:
        """
        Генерация ответа на основе RAG

        Args:
            question: Вопрос пользователя
            top_k: Количество фрагментов для поиска
            llm_provider: Тип провайдера LLM ("ollama", "openai")
            temperature: Температура генерации
            rerank: Использовать переранжирование с помощью cross-encoder
            **kwargs: Дополнительные параметры для LLM

        Returns:
            Dict: Словарь с ответом и метаданными
        """
        self.logger.info(f"Запрос: '{question}'")

        # Ищем релевантные фрагменты
        fragments = self.search_docs(question, top_k=top_k, rerank=rerank)

        if not fragments:
            self.logger.warning("Нет релевантных фрагментов.")
            return {
                "answer": "Не найдены релевантные фрагменты документов.",
                "fragments": [],
                "time": 0
            }

        start_time = time.time()

        # Формируем системный промпт с инструкциями для модели
        system_prompt = (
            "Ты – юридический ассистент. Используй **только** предоставленные фрагменты закона. "
            "Не придумывай информацию. Если предоставленных данных недостаточно, честно скажи об этом. "
            "Указывай на какие статьи или фрагменты опираешься. Формат: 'Согласно статье X...' или "
            "'В соответствии с положениями...'. Если в предоставленных фрагментах нет ответа на "
            "вопрос - скажи об этом прямо."
        )

        # Строим контекст из найденных фрагментов
        context = self._build_context_from_results(fragments)

        # Полный промпт для LLM
        full_prompt = f"SYSTEM:\n{system_prompt}\n\n== КОНТЕКСТ ==\n{context}\n\nUSER:\n{question}\n\nASSISTANT:"

        # Получаем провайдера LLM
        provider = self._get_llm_provider(llm_provider)

        # Генерируем ответ
        answer = provider.generate(
            full_prompt,
            temperature=temperature,
            **kwargs
        )

        end_time = time.time()
        elapsed_time = end_time - start_time

        self.logger.info(f"Ответ сгенерирован за {elapsed_time:.3f} сек.")

        return {
            "answer": answer,
            "fragments": fragments,
            "time": elapsed_time
        }

    def get_document_by_id(self, doc_id: int) -> Optional[Dict]:
        """Получает информацию о документе по его ID"""
        self._load_index()

        if doc_id in self.doc_metadata:
            meta = self.doc_metadata[doc_id]
            return {
                "doc_id": meta.doc_id,
                "filename": meta.filename,
                "title": meta.title,
                "type": meta.type,
                "date": meta.date,
                "number": meta.number,
                "author": meta.author
            }
        return None

    def get_chunk_by_id(self, doc_id: int, chunk_id: int) -> Optional[Dict]:
        """Получает информацию о чанке по ID документа и ID чанка"""
        self._load_index()

        for chunk_meta in self.chunks_meta:
            if chunk_meta.doc_id == doc_id and chunk_meta.chunk_id == chunk_id:
                result = chunk_meta.get_dictionary()

                # Добавляем информацию о документе
                if doc_id in self.doc_metadata:
                    doc_meta = self.doc_metadata[doc_id]
                    result.update({
                        "doc_title": doc_meta.title,
                        "doc_type": doc_meta.type,
                        "doc_date": doc_meta.date,
                        "doc_number": doc_meta.number,
                        "doc_author": doc_meta.author
                    })

                # Добавляем ссылку на фрагмент
                result["reference"] = chunk_meta.format_reference()

                return result
        return None

    def get_adjacent_chunks(self, doc_id: int, chunk_id: int) -> Dict[str, Optional[Dict]]:
        """Получает соседние чанки для указанного чанка"""
        self._load_index()

        result = {
            "current": None,
            "previous": None,
            "next": None
        }

        current_chunk = None
        for chunk_meta in self.chunks_meta:
            if chunk_meta.doc_id == doc_id and chunk_meta.chunk_id == chunk_id:
                current_chunk = chunk_meta
                break

        if not current_chunk:
            return result

        # Текущий чанк
        result["current"] = self.get_chunk_by_id(doc_id, chunk_id)

        # Предыдущий чанк
        if current_chunk.prev_chunk_id >= 0:
            result["previous"] = self.get_chunk_by_id(doc_id, current_chunk.prev_chunk_id)

        # Следующий чанк
        if current_chunk.next_chunk_id >= 0:
            result["next"] = self.get_chunk_by_id(doc_id, current_chunk.next_chunk_id)

        return result

    def get_document_text(self, doc_id: int) -> Optional[str]:
        """Получает полный текст документа, собирая его из чанков"""
        self._load_index()

        if doc_id not in self.doc_metadata:
            return None

        # Получаем все чанки документа и сортируем по start_idx
        doc_chunks = [
            chunk for chunk in self.chunks_meta
            if chunk.doc_id == doc_id
        ]

        if not doc_chunks:
            return None

        # Сортируем чанки по позиции в документе
        doc_chunks.sort(key=lambda x: x.start_idx)

        # Путь к файлу
        filename = self.doc_metadata[doc_id].filename
        file_path = os.path.join(self.docs_folder, filename)

        try:
            # Пытаемся прочитать оригинальный файл
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except:
            # Если не получилось, собираем из чанков
            text_parts = [chunk.text for chunk in doc_chunks]
            return "\n\n".join(text_parts)

    def get_index_info(self) -> Dict:
        """Возвращает информацию о текущем индексе"""
        self._load_index()

        # Уникальные документы и файлы
        unique_docs = len(set(m.doc_id for m in self.chunks_meta)) if self.chunks_meta else 0
        unique_files = len(set(m.filename for m in self.chunks_meta)) if self.chunks_meta else 0

        info = {
            "total_chunks": len(self.chunks_meta) if self.chunks_meta else 0,
            "unique_documents": unique_docs,
            "unique_files": unique_files,
            "cross_encoder_enabled": self.embedding_model.cross_encoder is not None if self._embedding_model else False,
            "lemmatization_enabled": self.lemmatizer.enabled,
            "embedding_model": self.config.get("EMB_MODEL_ID"),
            "llm_provider": self.config.get("DEFAULT_LLM_PROVIDER"),
            "llm_model": self.config.get("OLLAMA_MODEL") if self.config.get("DEFAULT_LLM_PROVIDER") == "ollama"
                         else self.config.get("OPENAI_MODEL"),
            "cache_enabled": self.config.get("CACHE_ENABLED", True),
            "adaptive_chunking": self.config.get("ADAPTIVE_CHUNKING", True),
            "context_window": self.config.get("CONTEXT_WINDOW", 512),
        }

        # Добавляем статистику кэша запросов
        if self.query_cache:
            info["cache_stats"] = self.query_cache.stats()

        return info

    def clear_cache(self) -> Dict:
        """Очищает кэш запросов"""
        if not self.query_cache:
            return {"status": "error", "message": "Кэширование отключено"}

        old_size = self.query_cache.clear()

        return {
            "status": "success",
            "message": f"Кэш очищен (удалено {old_size} записей)."
        }

    def safely_rebuild_index(self) -> Dict:
        """Пересоздание индексов с резервным копированием"""
        try:
            # Создаем временные метки для бэкапов
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.join(self.cache_folder, f"backup_{timestamp}")
            os.makedirs(backup_dir, exist_ok=True)

            # Копируем существующие индексы в бэкап
            backup_files = {}
            for original_path in [self.index_faiss_path, self.index_meta_path, self.index_bm25_path]:
                if os.path.exists(original_path):
                    backup_name = os.path.basename(original_path)
                    backup_path = os.path.join(backup_dir, backup_name)
                    shutil.copy2(original_path, backup_path)
                    backup_files[original_path] = backup_path

            # Пересоздаем индексы
            self.build_index(force_rebuild=True)

            return {
                "status": "success",
                "message": f"Индексы успешно пересозданы ({len(self.chunks_meta)} чанков).",
                "backup_location": backup_dir
            }
        except Exception as e:
            self.logger.error(f"Ошибка при пересоздании: {e}")

            # Восстанавливаемся из бэкапа
            try:
                for original_path, backup_path in backup_files.items():
                    if os.path.exists(backup_path):
                        shutil.copy2(backup_path, original_path)

                self.logger.info("Восстановлены индексы из бэкапов.")

                return {
                    "status": "error",
                    "message": f"Ошибка при пересоздании: {e}. Восстановлено из бэкапа.",
                    "backup_location": backup_dir
                }
            except Exception as restore_error:
                self.logger.error(f"Ошибка при восстановлении: {restore_error}")

                return {
                    "status": "error",
                    "message": f"Критическая ошибка: {e}. Восстановление не удалось: {restore_error}"
                }

    async def async_search_docs(self, query: str, top_k: int = 20,
                               bm_k: int = 12, use_cache: bool = True,
                               rerank: bool = True) -> List[Dict]:
        """Асинхронная версия search_docs"""
        # Выполняем поиск в отдельном потоке, чтобы не блокировать event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search_docs(query, top_k, bm_k, use_cache, rerank)
        )

    async def async_answer_with_rag(self, question: str, top_k: int = 20,
                                   llm_provider: str = None,
                                   temperature: float = 0.5,
                                   rerank: bool = True, **kwargs) -> Dict:
        """Асинхронная версия answer_with_rag"""
        # Поиск фрагментов
        fragments = await self.async_search_docs(
            question, top_k=top_k, rerank=rerank
        )

        if not fragments:
            self.logger.warning("Нет релевантных фрагментов.")
            return {
                "answer": "Не найдены релевантные фрагменты документов.",
                "fragments": [],
                "time": 0
            }

        start_time = time.time()

        # Формируем системный промпт
        system_prompt = (
            "Ты – юридический ассистент. Используй **только** предоставленные фрагменты закона. "
            "Не придумывай информацию. Если предоставленных данных недостаточно, честно скажи об этом. "
            "Указывай на какие статьи или фрагменты опираешься. Формат: 'Согласно статье X...' или "
            "'В соответствии с положениями...'. Если в предоставленных фрагментах нет ответа на "
            "вопрос - скажи об этом прямо."
        )

        # Строим контекст
        context = self._build_context_from_results(fragments)

        # Полный промпт
        full_prompt = f"SYSTEM:\n{system_prompt}\n\n== КОНТЕКСТ ==\n{context}\n\nUSER:\n{question}\n\nASSISTANT:"

        # Получаем провайдера LLM
        provider = self._get_llm_provider(llm_provider)

        # Выполняем запрос в отдельном потоке
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            None,
            lambda: provider.generate(
                full_prompt, temperature=temperature, **kwargs
            )
        )

        end_time = time.time()
        elapsed_time = end_time - start_time

        self.logger.info(f"Ответ сгенерирован за {elapsed_time:.3f} сек.")

        return {
            "answer": answer,
            "fragments": fragments,
            "time": elapsed_time
        }


#######################################################################
# CLI и вспомогательные функции
#######################################################################

def get_default_rag_system(config: Dict = None) -> RAGSystem:
    """Возвращает экземпляр RAGSystem с настройками по умолчанию"""
    return RAGSystem(config or DEFAULT_CONFIG)


# Функция для создания парсера аргументов командной строки
def create_cli_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG-система для нормативных документов",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Основные команды
    parser.add_argument("--build", action="store_true",
                        help="Пересоздать индексы")
    parser.add_argument("--info", action="store_true",
                        help="Информация об индексе")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Очистить кэш запросов")
    parser.add_argument("--question", type=str,
                        help="Задать вопрос")

    # Дополнительные параметры
    parser.add_argument("--top-k", type=int, default=10,
                        help="Количество фрагментов для поиска")
    parser.add_argument("--temp", type=float, default=0.5,
                        help="Температура генерации")
    parser.add_argument("--provider", type=str, default="ollama",
                        choices=["ollama", "openai"],
                        help="Провайдер LLM")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Отключить переранжирование")
    parser.add_argument("--output", type=str,
                        help="Файл для сохранения результатов")

    return parser


# Функция для запуска CLI
def main():
    parser = create_cli_parser()
    args = parser.parse_args()

    # Создаем экземпляр RAG-системы
    rag = get_default_rag_system()

    if args.build:
        print("Пересоздание индексов...")
        rag.build_index(force_rebuild=True)
        print("Индексы созданы.")

    if args.info:
        info = rag.get_index_info()
        print("\nИнформация об индексе:")
        for k, v in info.items():
            if k != "cache_stats":
                print(f"  {k}: {v}")

        if "cache_stats" in info:
            print("\nСтатистика кэша:")
            for k, v in info["cache_stats"].items():
                print(f"  {k}: {v}")

    if args.clear_cache:
        result = rag.clear_cache()
        print(f"\n{result['message']}")

    if args.question:
        print(f"\nВопрос: {args.question}")
        print("\nПоиск и генерация ответа...")

        response = rag.answer_with_rag(
            args.question,
            top_k=args.top_k,
            llm_provider=args.provider,
            temperature=args.temp,
            rerank=not args.no_rerank
        )

        print("\nОтвет:")
        print(response["answer"])
        print(f"\nВремя генерации: {response['time']:.2f} сек.")

        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump({
                        "question": args.question,
                        "answer": response["answer"],
                        "fragments": response["fragments"],
                        "time": response["time"]
                    }, f, ensure_ascii=False, indent=2)
                print(f"\nРезультаты сохранены в {args.output}")
            except Exception as e:
                print(f"Ошибка при сохранении результатов: {e}")

    # Простейший тест, если без аргументов
    if not (args.build or args.info or args.clear_cache or args.question):
        test_q = "Какие виды оценки предусмотрены?"
        print(f"Тестовый вопрос: {test_q}")

        answer = rag.answer_with_rag(test_q, top_k=10)

        print("\nОтвет:")
        print(answer["answer"])
        print(f"\nВремя генерации: {answer['time']:.2f} сек.")

# Обеспечение обратной совместимости со старым кодом
def answer_with_rag(question: str, top_k: int = 20,
                   temperature: float = 0.5, **kwargs) -> str:
    """
    Обертка для совместимости со старым API.
    """
    rag_system = get_default_rag_system()
    response = rag_system.answer_with_rag(
        question=question,
        top_k=top_k,
        temperature=temperature,
        **kwargs
    )
    return response["answer"]

if __name__ == "__main__":
    main()