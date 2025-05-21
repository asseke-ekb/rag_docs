############################################
# app.py
############################################
from flask import Flask, request, render_template, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import json
import ollama
# RAG-документация
from rag_docs import answer_with_rag

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'txt'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Получаем текущую дату для отображения в шаблоне
    from datetime import datetime
    current_date = datetime.now().strftime('%d %B %Y')
    return render_template('index.html', current_date=current_date)

@app.route('/index.html')
def docs_chat():
    """Страница RAG-чата по нормативной документации."""
    from datetime import datetime
    current_date = datetime.now().strftime('%d %B %Y')
    return render_template('docs_chat.html', current_date=current_date)

@app.route('/ask_docs', methods=['POST'])
def ask_docs():
    """
    Вопрос по документации (RAG).
    """
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Вопрос обязателен."}), 400

    try:
        answer = answer_with_rag(question, top_k=10)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Для параллельности можно:
    # app.run(debug=True, threaded=True)
    # или использовать gunicorn с 2+ воркерами
    app.run(debug=True)
