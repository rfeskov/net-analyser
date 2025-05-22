import telebot
from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel, Field
from typing import List, Optional
import threading
import uvicorn
from datetime import datetime
import time
import random
from storage import Storage

# ==== Конфигурация ====
TOKEN = "7517930802:AAFQHZogsvsh2uShM6cGi562G7T9Kvt9csY"
ADMIN_ID = 404051961
API_KEY = "super-secret-key"  # для защиты FastAPI endpoint
MONITORING_INTERVAL = 60  # интервал проверки в секундах (1 минута)

bot = telebot.TeleBot(TOKEN)
app = FastAPI(
    title="WiFi Monitor Bot API",
    description="API для отправки уведомлений через Telegram бота",
    version="1.0.0"
)

# Инициализация хранилища
storage = Storage()

# Глобальные переменные для управления тестовым режимом
test_mode = False
monitor_thread = None
monitor_active = False

# ==== Модели запросов для API ====
class Issue(BaseModel):
    type: str = Field(..., description="Тип проблемы (load/stability/packets/signal)")
    value: float = Field(..., description="Значение метрики")
    
    class Config:
        schema_extra = {
            "example": {
                "type": "load",
                "value": 85.5
            }
        }

class AccessPoint(BaseModel):
    name: str = Field(..., description="Название точки доступа")
    band: str = Field(..., description="Диапазон частот (2.4 GHz/5 GHz)")
    channel: int = Field(..., description="Номер канала")
    issues: List[Issue] = Field(..., description="Список проблем")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Office-1",
                "band": "2.4 GHz",
                "channel": 6,
                "issues": [
                    {"type": "load", "value": 85.5},
                    {"type": "signal", "value": -85}
                ]
            }
        }

class NotificationRequest(BaseModel):
    text: Optional[str] = Field(None, description="Простой текст уведомления")
    points: Optional[List[AccessPoint]] = Field(None, description="Список точек доступа с проблемами")
    
    class Config:
        schema_extra = {
            "example": {
                "points": [
                    {
                        "name": "Office-1",
                        "band": "2.4 GHz",
                        "channel": 6,
                        "issues": [
                            {"type": "load", "value": 85.5},
                            {"type": "signal", "value": -85}
                        ]
                    },
                    {
                        "name": "Meeting-Room",
                        "band": "5 GHz",
                        "channel": 36,
                        "issues": [
                            {"type": "stability", "value": 45},
                            {"type": "packets", "value": 25}
                        ]
                    }
                ]
            }
        }

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

def format_issue(issue: Issue) -> str:
    """Форматирует проблему в текстовый формат"""
    if issue.type == "load":
        return f"🔴 Высокая нагрузка: {issue.value}%"
    elif issue.type == "stability":
        return f"⚠️ Низкая стабильность: {issue.value}%"
    elif issue.type == "packets":
        return f"📦 Большая потеря пакетов: {issue.value}"
    elif issue.type == "signal":
        return f"📶 Слабый сигнал: {issue.value} dBm"
    return f"❗️ {issue.type}: {issue.value}"

def format_notification(data: NotificationRequest) -> str:
    """Форматирует данные уведомления в текстовый формат"""
    if data.text:
        return data.text
        
    if not data.points:
        raise ValueError("Необходимо указать текст или список точек")
        
    message = "⚡️ Обнаружены проблемы в сети:\n"
    
    for point in data.points:
        message += f"\n🔹 Точка {point.name} ({point.band}, канал {point.channel}):\n"
        for issue in point.issues:
            message += f"  {format_issue(issue)}\n"
    
    return message.strip()

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
    global monitor_active
    monitor_active = True
    
    while monitor_active:
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
    if message.from_user.id == ADMIN_ID:
        bot.send_message(message.chat.id, 
            "Админ-команды:\n"
            "/testmode_on - включить тестовый режим\n"
            "/testmode_off - выключить тестовый режим\n"
            "/status - статус тестового режима\n"
            "/notify - отправить уведомление всем\n"
            "/pending - просмотр заявок на подписку\n"
            "/approve <user_id> - одобрить подписчика\n"
            "/reject <user_id> - отклонить подписчика\n\n"
            "Общие команды:\n"
            "/subscribe - подать заявку на подписку\n"
            "/unsubscribe - отписаться от уведомлений")
    else:
        bot.send_message(message.chat.id, "Напишите /subscribe чтобы подать заявку на получение уведомлений.")

@bot.message_handler(commands=['testmode_on'])
def testmode_on_handler(message):
    global test_mode, monitor_thread, monitor_active
    
    if message.from_user.id != ADMIN_ID:
        bot.send_message(message.chat.id, "Нет доступа.")
        return
    
    if test_mode:
        bot.send_message(message.chat.id, "Тестовый режим уже включен.")
        return
    
    test_mode = True
    monitor_active = True
    monitor_thread = threading.Thread(target=monitor_demo, daemon=True)
    monitor_thread.start()
    
    bot.send_message(message.chat.id, 
        "✅ Тестовый режим включен\n"
        "Уведомления будут приходить каждую минуту\n"
        "Для отключения используйте /testmode_off")

@bot.message_handler(commands=['testmode_off'])
def testmode_off_handler(message):
    global test_mode, monitor_active
    
    if message.from_user.id != ADMIN_ID:
        bot.send_message(message.chat.id, "Нет доступа.")
        return
    
    if not test_mode:
        bot.send_message(message.chat.id, "Тестовый режим уже выключен.")
        return
    
    test_mode = False
    monitor_active = False
    bot.send_message(message.chat.id, "❌ Тестовый режим выключен")

@bot.message_handler(commands=['status'])
def status_handler(message):
    if message.from_user.id != ADMIN_ID:
        bot.send_message(message.chat.id, "Нет доступа.")
        return
    
    status = "✅ Включен" if test_mode else "❌ Выключен"
    bot.send_message(message.chat.id, f"Статус тестового режима: {status}")

@bot.message_handler(commands=['subscribe'])
def subscribe_handler(message):
    user_id = message.chat.id
    user_info = {
        'name': message.from_user.first_name,
        'username': message.from_user.username or 'Нет username',
        'last_name': message.from_user.last_name or ''
    }
    
    if storage.add_pending(user_id, user_info):
        bot.send_message(user_id, "Ваша заявка на подписку отправлена администратору. Ожидайте подтверждения.")
        # Формируем информативное сообщение для админа
        admin_msg = f"Новая заявка на подписку:\n"
        admin_msg += f"👤 Имя: {user_info['name']} {user_info['last_name']}\n"
        admin_msg += f"🔗 Username: @{user_info['username']}\n"
        admin_msg += f"🆔 ID: {user_id}"
        bot.send_message(ADMIN_ID, admin_msg)
    else:
        if user_id in storage.get_subscribers():
            bot.send_message(user_id, "Вы уже подписаны на уведомления.")
        else:
            bot.send_message(user_id, "Ваша заявка уже находится на рассмотрении.")

@bot.message_handler(commands=['unsubscribe'])
def unsubscribe_handler(message):
    user_id = message.chat.id
    if storage.remove_subscriber(user_id):
        bot.send_message(user_id, "Вы отписались от уведомлений.")
    else:
        bot.send_message(user_id, "Вы не были подписаны на уведомления.")

@bot.message_handler(commands=['pending'])
def pending_handler(message):
    if message.from_user.id != ADMIN_ID:
        bot.send_message(message.chat.id, "Нет доступа.")
        return
    
    pending = storage.get_pending_subscribers()
    if not pending:
        bot.send_message(message.chat.id, "Нет ожидающих подтверждения заявок.")
        return
    
    message_text = "Ожидающие подтверждения заявки:\n\n"
    for user_id, user_info in pending.items():
        message_text += f"👤 {user_info['name']} {user_info['last_name']}\n"
        message_text += f"🔗 @{user_info['username']}\n"
        message_text += f"🆔 ID: {user_id}\n\n"
    message_text += "Используйте /approve <ID> или /reject <ID>"
    bot.send_message(message.chat.id, message_text)

@bot.message_handler(commands=['approve'])
def approve_handler(message):
    if message.from_user.id != ADMIN_ID:
        bot.send_message(message.chat.id, "Нет доступа.")
        return
    
    try:
        user_id = int(message.text.split()[1])
        if storage.approve_subscriber(user_id):
            bot.send_message(message.chat.id, f"Пользователь {user_id} одобрен.")
            bot.send_message(user_id, "Ваша заявка на подписку одобрена! Теперь вы будете получать уведомления.")
        else:
            bot.send_message(message.chat.id, f"Пользователь {user_id} не найден в списке ожидающих.")
    except (IndexError, ValueError):
        bot.send_message(message.chat.id, "Использование: /approve <user_id>")

@bot.message_handler(commands=['reject'])
def reject_handler(message):
    if message.from_user.id != ADMIN_ID:
        bot.send_message(message.chat.id, "Нет доступа.")
        return
    
    try:
        user_id = int(message.text.split()[1])
        if storage.reject_subscriber(user_id):
            bot.send_message(message.chat.id, f"Заявка пользователя {user_id} отклонена.")
            bot.send_message(user_id, "Ваша заявка на подписку была отклонена администратором.")
        else:
            bot.send_message(message.chat.id, f"Пользователь {user_id} не найден в списке ожидающих.")
    except (IndexError, ValueError):
        bot.send_message(message.chat.id, "Использование: /reject <user_id>")

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
    subscribers = storage.get_subscribers()
    for user_id in subscribers:
        try:
            bot.send_message(user_id, f"📡 Уведомление:\n{text}")
        except Exception as e:
            print(f"Ошибка при отправке пользователю {user_id}: {e}")

# ==== FastAPI endpoints ====
@app.post("/send", 
    summary="Отправить уведомление",
    description="Отправляет уведомление всем подписчикам бота. Можно отправить простой текст или структурированные данные о проблемах.")
async def send_notification_api(
    data: NotificationRequest,
    x_api_key: str = Header(None, description="API ключ для аутентификации")
):
    """
    Отправляет уведомление через бота.
    
    - Можно отправить простой текст в поле `text`
    - Или структурированные данные о проблемах в поле `points`
    - Требуется указать API ключ в заголовке `x-api-key`
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Неверный API ключ")
    
    try:
        message = format_notification(data)
        send_notification(message)
        return {"status": "ok", "message": "Уведомление отправлено"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при отправке: {e}")

# ==== Запуск FastAPI в отдельном потоке ====
def start_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Запускаем FastAPI в отдельном потоке
threading.Thread(target=start_fastapi, daemon=True).start()

# ==== Запуск бота ====
bot.infinity_polling()