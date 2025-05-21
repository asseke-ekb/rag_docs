const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const chatContainer = document.getElementById('chatContainer');

// Функция для автоматического изменения высоты текстового поля
function autoResize() {
  questionInput.style.height = 'auto';
  questionInput.style.height = (questionInput.scrollHeight) + 'px';
}

// Текущее время в формате ЧЧ:ММ
function getCurrentTime() {
  const now = new Date();
  return `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
}

// Создаем сообщение пользователя
function createUserMessage(text) {
  const messageDiv = document.createElement('div');
  messageDiv.className = 'message message-user';
  messageDiv.innerHTML = `
    <div class="message-content">${text}</div>
    <span class="message-time">${getCurrentTime()}</span>
  `;
  return messageDiv;
}

// Создаем сообщение ассистента
function createAssistantMessage(html) {
  const messageDiv = document.createElement('div');
  messageDiv.className = 'message message-assistant';
  messageDiv.innerHTML = `
    <div class="message-content">${html}</div>
    <span class="message-time">${getCurrentTime()}</span>
  `;
  return messageDiv;
}

// Создаем индикатор печатания
function createTypingIndicator() {
  const typingDiv = document.createElement('div');
  typingDiv.className = 'typing-indicator';
  typingDiv.id = 'typingIndicator';
  typingDiv.innerHTML = `
    <div class="typing-animation">
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    </div>
  `;
  return typingDiv;
}

// Прокрутка чата вниз
function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Функция имитации печатания и добавления сообщения
function simulateTypingAndAppend(html) {
  // Создаем сообщение ассистента
  const messageDiv = createAssistantMessage('');
  chatContainer.appendChild(messageDiv);

  const messageContent = messageDiv.querySelector('.message-content');
  let tempDiv = document.createElement('div');
  tempDiv.innerHTML = html;
  let finalText = tempDiv.innerHTML;

  let i = 0;
  const speed = 5; // скорость набора (меньше = быстрее)
  let content = '';

  function type() {
    if (i < finalText.length) {
      content += finalText.charAt(i);
      messageContent.innerHTML = content;
      i++;

      // Прокручиваем к новому контенту
      scrollToBottom();

      // Планируем следующий символ
      setTimeout(type, speed);
    }
  }

  // Запускаем печать
  type();
}

// Обработка отправки сообщения
function sendMessage() {
  const q = questionInput.value.trim();
  if (!q) return;

  // Добавляем сообщение пользователя
  chatContainer.appendChild(createUserMessage(q));
  scrollToBottom();

  // Очищаем поле ввода
  questionInput.value = '';
  questionInput.style.height = 'auto';

  // Показываем индикатор печатания
  const typingIndicator = createTypingIndicator();
  chatContainer.appendChild(typingIndicator);
  scrollToBottom();

  fetch('/ask_docs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question: q })
  })
  .then(r => r.json())
  .then(data => {
    // Удаляем индикатор печатания
    const indicator = document.getElementById('typingIndicator');
    if (indicator) indicator.remove();

    if (data.error) {
      // Создаем сообщение об ошибке
      const errorMessage = createAssistantMessage(`<div style="color: #d9534f;"><i class="fas fa-exclamation-triangle me-2"></i>${data.error}</div>`);
      chatContainer.appendChild(errorMessage);
    } else {
      // Форматируем полученный ответ
      const formattedAnswer = data.answer
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/#### (.*?)\n/g, '<h5>$1</h5>')
        .replace(/### (.*?)\n/g, '<h4>$1</h4>')
        .replace(/## (.*?)\n/g, '<h3>$1</h3>')
        .replace(/- (.*?)\n/g, '<li>$1</li>')
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>');

      // Имитация постепенного появления сообщения
      simulateTypingAndAppend(formattedAnswer);
    }
    scrollToBottom();
  })
  .catch(err => {
    // Удаляем индикатор печатания
    const indicator = document.getElementById('typingIndicator');
    if (indicator) indicator.remove();

    // Создаем сообщение об ошибке
    const errorMessage = createAssistantMessage(`<div style="color: #d9534f;"><i class="fas fa-exclamation-triangle me-2"></i>Произошла ошибка: ${err.message}</div>`);
    chatContainer.appendChild(errorMessage);
    scrollToBottom();
  });
}

// Подключаем обработчики событий
sendBtn.addEventListener('click', sendMessage);

// Адаптивная высота поля ввода
questionInput.addEventListener('input', autoResize);

// Обработка нажатия Enter для отправки
questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendBtn.click();
  }
});

// Инициализация
autoResize();
scrollToBottom();