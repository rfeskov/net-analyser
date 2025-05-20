import telebot
from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
import threading
import uvicorn
from datetime import datetime
import time
import random

# ==== Конфигурация ====
TOKEN = "7517930802:AAFQHZogsvsh2uShM6cGi562G7T9Kvt9csY"
ADMIN_ID = 404051961
API_KEY = "super-secret-key"  # для защиты FastAPI endpoint
MONITORING_INTERVAL = 300  # интервал проверки в секундах (5 минут)

bot = telebot.TeleBot(TOKEN)
app = FastAPI()
subscribers = set()

# ==== Модель запроса для /send ====
class NotificationRequest(BaseModel):
    text: str

# ==== Демонстрационные данные ====
DEMO_POINTS = ['Office-1', 'Office-2', 'Meeting-Room', 'Reception']
DEMO_BANDS = ['2.4 GHz', '5 GHz']
DEMO_CHANNELS = {
    '2.4 GHz': [1, 6, 11],
    '5 GHz': [36, 40, 44, 48]
}
DEMO_ISSUES = [
    "🔴 Высокая нагрузка: {}%",
    "⚠️ Низкая стабильность: {}%",
    "📦 Большая потеря пакетов: {}",
    "📶 Слабый сигнал: {} dBm"
]

def generate_random_notification():
    """Генерирует случайное уведомление о проблемах"""
    num_points = random.randint(1, 3)  # Случайное количество точек с проблемами
    all_issues = []
    
    for _ in range(num_points):
        point = random.choice(DEMO_POINTS)
        band = random.choice(DEMO_BANDS)
        channel = random.choice(DEMO_CHANNELS[band])
        
        # Генерируем случайные значения для проблем
        load = random.randint(75, 95)
        stability = random.randint(40, 65)
        packets = random.randint(25, 50)
        signal = random.randint(-90, -75)
        
        # Выбираем случайное количество проблем для точки
        num_issues = random.randint(1, 3)
        point_issues = random.sample([
            DEMO_ISSUES[0].format(load),
            DEMO_ISSUES[1].format(stability),
            DEMO_ISSUES[2].format(packets),
            DEMO_ISSUES[3].format(signal)
        ], num_issues)
        
        # Формируем сообщение для точки
        all_issues.append(f"\n🔹 Точка {point} ({band}, канал {channel}):")
        all_issues.extend([f"  {issue}" for issue in point_issues])
    
    return "⚡️ Обнаружены проблемы в сети:\n" + "\n".join(all_issues)

def monitor_demo():
    """Имитирует мониторинг, отправляя случайные уведомления"""
    while True:
        try:
            # С вероятностью 70% генерируем уведомление
            if random.random() < 0.7:
                message = generate_random_notification()
                bot.send_message(ADMIN_ID, message)
            
        except Exception as e:
            print(f"Ошибка в мониторинге: {e}")
        
        time.sleep(MONITORING_INTERVAL)

# ==== Команды бота ====
@bot.message_handler(commands=['start'])
def start_handler(message):
    bot.send_message(message.chat.id, "Напишите /subscribe чтобы получать уведомления.")

@bot.message_handler(commands=['subscribe'])
def subscribe_handler(message):
    subscribers.add(message.chat.id)
    bot.send_message(message.chat.id, "Вы подписались на уведомления.")

@bot.message_handler(commands=['unsubscribe'])
def unsubscribe_handler(message):
    subscribers.discard(message.chat.id)
    bot.send_message(message.chat.id, "Вы отписались от уведомлений.")

@bot.message_handler(commands=['notify'])
def notify_handler(message):
    if message.from_user.id != ADMIN_ID:
        bot.send_message(message.chat.id, "Нет доступа.")
        return
    text = message.text.replace("/notify", "").strip()
    if not text:
        bot.send_message(message.chat.id, "Добавь текст после команды.")
        return
    send_notification(text)
    bot.send_message(message.chat.id, "Уведомление отправлено.")

# ==== Рассылка подписчикам ====
def send_notification(text):
    for user_id in subscribers:
        try:
            bot.send_message(user_id, f"📡 Уведомление:\n{text}")
        except Exception as e:
            print(f"Ошибка при отправке пользователю {user_id}: {e}")

# ==== FastAPI endpoint ====
@app.post("/send")
async def send_notification_api(
    data: NotificationRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Неверный API ключ")
    send_notification(data.text)
    return {"status": "ok", "message": "Уведомление отправлено"}

# ==== Запуск FastAPI в отдельном потоке ====
def start_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Запускаем все сервисы в отдельных потоках
threading.Thread(target=start_fastapi, daemon=True).start()
threading.Thread(target=monitor_demo, daemon=True).start()

# ==== Запуск бота ====
bot.infinity_polling()