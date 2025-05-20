import telebot
from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
import threading
import uvicorn
from datetime import datetime
import json
import os

# ==== Конфигурация ====
TOKEN = "7517930802:AAFQHZogsvsh2uShM6cGi562G7T9Kvt9csY"
ADMIN_ID = 404051961
API_KEY = "super-secret-key"  # для защиты FastAPI endpoint
ANALYSIS_FILE = os.path.join(os.path.dirname(__file__), 'analysis_results.json')

bot = telebot.TeleBot(TOKEN)
app = FastAPI()
subscribers = set()

# ==== Модель запроса для /send ====
class NotificationRequest(BaseModel):
    text: str

# ==== Вспомогательные функции ====
def load_analysis_data():
    """Загружает данные анализа из JSON файла"""
    try:
        with open(ANALYSIS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Ошибка при чтении файла анализа: {e}")
        return None

def get_point_names():
    """Получает список имен точек доступа"""
    data = load_analysis_data()
    if data and 'point_analyses' in data:
        return list(data['point_analyses'].keys())
    return []

def get_point_data(point_name):
    """Получает данные для конкретной точки"""
    data = load_analysis_data()
    if data and 'point_analyses' in data:
        return data['point_analyses'].get(point_name)
    return None

def minutes_to_time(minutes):
    """Конвертирует минуты в формат HH:MM"""
    hours = int(minutes) // 60
    mins = int(minutes) % 60
    return f"{hours:02d}:{mins:02d}"

def format_channel_table(data, band):
    """Форматирует данные о каналах в текстовую таблицу"""
    if not data or not data.get('time_periods'):
        return f"Нет данных для диапазона {band}"
    
    # Фильтруем периоды для указанного диапазона
    periods = [p for p in data['time_periods'] if p.get('band') == band]
    if not periods:
        return f"Нет данных для диапазона {band}"
    
    # Формируем таблицу
    table = f"📊 Таблица смены каналов ({band}):\n\n"
    table += "Время | Канал | Нагрузка | Стабильность\n"
    table += "-------|--------|-----------|-------------\n"
    
    for period in periods:
        start_time = minutes_to_time(period['start_time'])
        end_time = minutes_to_time(period['end_time'])
        channel = period['channel']
        load = f"{period['load_score']:.1f}%"
        stability = f"{period['stability'] * 100:.0f}%"
        table += f"{start_time}-{end_time} | {channel} | {load} | {stability}\n"
    
    return table

def format_point_info(point_name, data):
    """Форматирует информацию о точке доступа"""
    if not data or not data.get('time_periods'):
        return f"Нет данных для точки {point_name}"
    
    info = f"📡 Точка доступа: {point_name}\n\n"
    
    # Добавляем таблицы для обоих диапазонов
    table_24 = format_channel_table(data, '2.4 GHz')
    table_5 = format_channel_table(data, '5 GHz')
    
    return f"{info}{table_24}\n\n{table_5}"

# ==== Команды бота ====
@bot.message_handler(commands=['start'])
def start_handler(message):
    bot.send_message(message.chat.id, "Напишите /subscribe чтобы получать уведомления.\n"
                    "Используйте /channels для просмотра информации о каналах.")

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
        # Получаем список точек
        points = get_point_names()
        if not points:
            bot.send_message(message.chat.id, "Нет доступных точек доступа.")
            return
        
        # Если указан параметр с именем точки
        args = message.text.split()
        if len(args) > 1:
            point_name = args[1]
            if point_name in points:
                # Получаем и отправляем данные для конкретной точки
                point_data = get_point_data(point_name)
                response = format_point_info(point_name, point_data)
                bot.send_message(message.chat.id, response)
            else:
                bot.send_message(message.chat.id, f"Точка {point_name} не найдена.\n"
                               f"Доступные точки: {', '.join(points)}")
        else:
            # Если точка не указана, показываем список доступных точек
            response = "Доступные точки доступа:\n\n"
            for point in points:
                response += f"• {point}\n"
            response += "\nИспользуйте команду '/channels имя_точки' для просмотра информации"
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