import telebot
from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
import threading
import uvicorn
from datetime import datetime

# ==== Конфигурация ====
TOKEN = "7517930802:AAFQHZogsvsh2uShM6cGi562G7T9Kvt9csY"
ADMIN_ID = 404051961
API_KEY = "super-secret-key"  # для защиты FastAPI endpoint

bot = telebot.TeleBot(TOKEN)
app = FastAPI()
subscribers = set()

# ==== Модель запроса для /send ====
class NotificationRequest(BaseModel):
    text: str

# ==== Вспомогательные функции ====
def minutes_to_time(minutes):
    """Конвертирует минуты в формат HH:MM"""
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"

def format_channel_table(data, band):
    """Форматирует данные о каналах в текстовую таблицу"""
    if not data.get('time_periods'):
        return f"Нет данных для диапазона {band}"
    
    # Фильтруем периоды для указанного диапазона
    periods = [p for p in data['time_periods'] if p.get('band') == band]
    if not periods:
        return f"Нет данных для диапазона {band}"
    
    # Формируем таблицу
    table = f"📊 Таблица смены каналов ({band}):\n\n"
    table += "Время | Канал\n"
    table += "------|-------\n"
    
    for period in periods:
        start_time = minutes_to_time(period['start_time'])
        end_time = minutes_to_time(period['end_time'])
        channel = period['channel']
        table += f"{start_time}-{end_time} | {channel}\n"
    
    return table

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

@bot.message_handler(commands=['channels'])
def channels_handler(message):
    try:
        # Здесь должен быть код для получения данных о каналах
        # Пока используем тестовые данные
        test_data = {
            'time_periods': [
                {
                    'band': '2.4 GHz',
                    'start_time': 480,  # 8:00
                    'end_time': 720,    # 12:00
                    'channel': 1
                },
                {
                    'band': '2.4 GHz',
                    'start_time': 720,  # 12:00
                    'end_time': 1080,   # 18:00
                    'channel': 6
                },
                {
                    'band': '5 GHz',
                    'start_time': 480,  # 8:00
                    'end_time': 1080,   # 18:00
                    'channel': 36
                }
            ]
        }
        
        # Форматируем и отправляем таблицы для обоих диапазонов
        table_24 = format_channel_table(test_data, '2.4 GHz')
        table_5 = format_channel_table(test_data, '5 GHz')
        
        response = f"{table_24}\n\n{table_5}"
        bot.send_message(message.chat.id, response)
    except Exception as e:
        bot.send_message(message.chat.id, f"Ошибка при получении данных о каналах: {e}")

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

threading.Thread(target=start_fastapi, daemon=True).start()

# ==== Запуск бота ====
bot.infinity_polling()