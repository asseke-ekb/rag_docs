# RAG-чат для работы с нормативными документами

Интерактивное веб-приложение на Flask для работы с документами через технологию Retrieval Augmented Generation (RAG). Позволяет пользователям задавать вопросы и получать ответы на основе базы нормативных документов.


## 🌟 Особенности

- **Интеллектуальный поиск документов** — гибридное использование семантического (Faiss) и лексического (BM25) поиска для нахождения наиболее релевантных фрагментов текста
- **Генеративные ответы** — использование современных языковых моделей (Ollama/OpenAI) для формирования человечных ответов на основе найденных документов
- **Удобный интерфейс** — современный и отзывчивый чат-интерфейс с визуальным оформлением сообщений
- **Адаптивная разбивка документов** — сохранение структуры нормативных документов при индексации (главы, статьи, пункты)
- **Переранжирование результатов** — использование cross-encoder для улучшения качества выдачи
- **Кэширование** — кэширование запросов для повышения производительности и экономии ресурсов

## 🛠️ Технологии

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **Векторное хранилище**: Faiss
- **Лексический поиск**: BM25
- **Эмбеддинги**: SentenceTransformers
- **LLM**: Поддержка Ollama (локально) и OpenAI (через API)
- **Контейнеризация**: Docker, Docker Compose

## 🚀 Установка и запуск

### Вариант 1: С использованием Docker (рекомендуется)

```bash
# Клонирование репозитория
git clone https://github.com/YOUR_USERNAME/rag-chat.git
cd rag-chat

# Запуск через Docker Compose
docker-compose up -d
```

Приложение будет доступно по адресу: http://localhost:5000

### Вариант 2: Локальная установка

```bash
# Клонирование репозитория
git clone https://github.com/YOUR_USERNAME/rag-chat.git
cd rag-chat

# Создание виртуального окружения
python -m venv venv

# Активация окружения
# На Windows:
venv\Scripts\activate
# На macOS/Linux:
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Запуск приложения
python app.py
```

Приложение будет доступно по адресу: http://127.0.0.1:5000

## 📋 Использование

### Добавление документов

1. Поместите ваши документы в формате Markdown (`.md`) или текстовом формате (`.txt`) в папку `docs/`
2. При первом запуске приложения документы будут автоматически проиндексированы
3. Для переиндексации после добавления новых документов, используйте API-запрос или соответствующую функцию в интерфейсе

### Взаимодействие с чатом

1. Откройте приложение в браузере
2. Введите ваш вопрос в поле в нижней части экрана
3. Получите ответ на основе релевантных фрагментов документов

## 📚 Структура проекта

```
project_root/
│
├── templates/             # HTML-шаблоны
│   └── index.html         # Основной шаблон чата
│
├── static/                # Статические файлы
│   ├── css/
│   │   └── styles.css     # CSS-стили
│   └── js/
│       └── script.js      # JavaScript для интерфейса
│
├── docs/                  # Папка для хранения документов
├── uploads/               # Папка для загруженных файлов
├── cache/                 # Кэш и индексы
├── logs/                  # Логи приложения
│
├── app.py                 # Основное Flask-приложение
├── rag_docs.py            # Модуль RAG-функциональности
├── requirements.txt       # Зависимости Python
├── Dockerfile             # Инструкции для сборки Docker-образа
└── docker-compose.yml     # Конфигурация Docker Compose
```

## ⚙️ Настройка

### Конфигурация LLM

По умолчанию используется локальная модель через Ollama. Для использования OpenAI API:

1. Откройте `rag_docs.py`
2. Найдите секцию `DEFAULT_CONFIG`
3. Измените параметр `DEFAULT_LLM_PROVIDER` на `"openai"`
4. Добавьте ваш API-ключ в `OPENAI_API_KEY`

### Настройка модели эмбеддингов

По умолчанию используется модель `sentence-transformers/multi-qa-mpnet-base-dot-v1`. Для изменения:

1. Откройте `rag_docs.py`
2. Найдите секцию `DEFAULT_CONFIG`
3. Измените параметр `EMB_MODEL_ID` на нужную модель из библиотеки SentenceTransformers

## 🛡️ Настройка безопасности

При развертывании в производственной среде рекомендуется:

1. Настроить HTTPS с использованием SSL-сертификата
2. Активировать защиту от CSRF-атак
3. Настроить аутентификацию пользователей при необходимости

## ✅ Возможности расширения

- Добавление поддержки PDF-документов
- Реализация функции поиска с подсветкой релевантных фрагментов
- Внедрение базы данных для хранения истории запросов
- Добавление мультиязычной поддержки
- Интеграция с другими LLM (Claude API, Mistral AI и т.д.)

## 📝 API Endpoints

- `GET /` - Главная страница с интерфейсом чата
- `POST /ask_docs` - Endpoint для отправки вопросов и получения ответов

## 📮 Контакты

- Создатель: [Asset Slambekov](asset.slambekov.96@gmail.com)
- GitHub: [asseke-ekb](https://github.com/asseke-ekb)

