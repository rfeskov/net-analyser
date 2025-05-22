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

def check_access(message) -> bool:
    """Проверяет, имеет ли пользователь доступ"""
    return message.from_user.id == ADMIN_ID or storage.is_subscriber(message.from_user.id)

def send_no_access(message):
    """Отправляет сообщение об отсутствии доступа"""
    bot.send_message(message.chat.id, "У вас нет доступа к боту. Обратитесь к администратору.")

# ==== Команды бота ====
@bot.message_handler(commands=['start'])
def start_handler(message):
    if message.from_user.id == ADMIN_ID:
        markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
        users_btn = telebot.types.KeyboardButton(
            "Выбрать пользователя", 
            request_users=telebot.types.KeyboardButtonRequestUsers(
                request_id=1,
                user_is_bot=False,
                user_is_premium=None
            )
        )
        markup.add(users_btn)
        
        bot.send_message(message.chat.id, 
            "Админ-команды:\n"
            "/testmode_on - включить тестовый режим\n"
            "/testmode_off - выключить тестовый режим\n"
            "/status - статус тестового режима\n"
            "/notify - отправить уведомление всем\n"
            "/list_subs - список подписчиков\n"
            "/remove_sub <id> - удалить подписчика\n\n"
            "Для добавления подписчика:\n"
            "1. Перешлите мне любое сообщение от пользователя\n"
            "2. Или нажмите кнопку 'Выбрать пользователя' и выберите из списка",
            reply_markup=markup)
    else:
        send_no_access(message)

@bot.message_handler(content_types=['users_shared'])
def handle_users_shared(message):
    if message.from_user.id != ADMIN_ID:
        send_no_access(message)
        return

    print(f"Debug: users_shared data: {message.users_shared.__dict__}")
    print(f"Debug: user_ids data: {message.users_shared.user_ids}")
    
    shared_user = message.users_shared.user_ids[0]
    print(f"Debug: first shared_user data: {shared_user}")
    
    # Если shared_user это словарь, берем user_id из него
    user_id = shared_user.get('user_id') if isinstance(shared_user, dict) else shared_user
    print(f"Debug: extracted user_id: {user_id}")
    
    if storage.add_subscriber(user_id):
        name = shared_user.get('first_name', 'Неизвестный') if isinstance(shared_user, dict) else 'Пользователь'
        bot.reply_to(message, f"{name} (ID: {user_id}) добавлен в подписчики.")
        try:
            bot.send_message(user_id, "Вам предоставлен доступ к уведомлениям.")
        except Exception as e:
            bot.reply_to(message, f"Предупреждение: не удалось отправить сообщение пользователю {user_id}")
            print(f"Debug: send message error: {str(e)}")
    else:
        bot.reply_to(message, f"Пользователь (ID: {user_id}) уже является подписчиком.")

@bot.message_handler(func=lambda message: True)
def handle_all_messages(message):
    """Обработчик всех остальных сообщений"""
    if message.from_user.id == ADMIN_ID:
        # Проверяем все возможные варианты пересланного сообщения
        forwarded_user_id = None
        
        if message.forward_from:
            forwarded_user_id = message.forward_from.id
        elif message.forward_sender_name:
            # Пользователь скрыл свой профиль
            bot.reply_to(message, "Не могу добавить этого пользователя, так как его профиль скрыт настройками приватности.")
            return
            
        if forwarded_user_id:
            if storage.add_subscriber(forwarded_user_id):
                bot.reply_to(message, f"Пользователь {forwarded_user_id} добавлен в подписчики.")
                try:
                    bot.send_message(forwarded_user_id, "Вам предоставлен доступ к уведомлениям.")
                except Exception as e:
                    bot.reply_to(message, f"Предупреждение: не удалось отправить сообщение пользователю {forwarded_user_id}")
            else:
                bot.reply_to(message, f"Пользователь {forwarded_user_id} уже является подписчиком.")
            return
    
    if not check_access(message):
        send_no_access(message)

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
    uvicorn.run(app, host="127.0.0.1", port=8000)

# Запускаем FastAPI в отдельном потоке
threading.Thread(target=start_fastapi, daemon=True).start()

# ==== Запуск бота ====
bot.infinity_polling()

@bot.message_handler(commands=['testmode_on'])
def testmode_on_handler(message):
    if message.from_user.id != ADMIN_ID:
        send_no_access(message)
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
    if message.from_user.id != ADMIN_ID:
        send_no_access(message)
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
        send_no_access(message)
        return
    
    status = "✅ Включен" if test_mode else "❌ Выключен"
    bot.send_message(message.chat.id, f"Статус тестового режима: {status}")

@bot.message_handler(commands=['remove_sub'])
def remove_subscriber_handler(message):
    if message.from_user.id != ADMIN_ID:
        send_no_access(message)
        return
    
    try:
        user_id = int(message.text.split()[1])
        if storage.remove_subscriber(user_id):
            bot.send_message(message.chat.id, f"Пользователь {user_id} удален из подписчиков.")
            try:
                bot.send_message(user_id, "Ваш доступ к уведомлениям отключен.")
            except Exception:
                pass
        else:
            bot.send_message(message.chat.id, f"Пользователь {user_id} не является подписчиком.")
    except (IndexError, ValueError):
        bot.send_message(message.chat.id, "Использование: /remove_sub <user_id>")

@bot.message_handler(commands=['list_subs'])
def list_subscribers_handler(message):
    if message.from_user.id != ADMIN_ID:
        send_no_access(message)
        return
    
    subscribers = storage.get_subscribers()
    if not subscribers:
        bot.send_message(message.chat.id, "Список подписчиков пуст.")
        return
    
    message_text = "Список подписчиков:\n\n"
    for user_id in subscribers:
        message_text += f"ID: {user_id}\n"
    bot.send_message(message.chat.id, message_text)

@bot.message_handler(commands=['notify'])
def notify_handler(message):
    if message.from_user.id != ADMIN_ID:
        send_no_access(message)
        return
    
    text = message.text.replace("/notify", "").strip()
    if not text:
        bot.send_message(message.chat.id, "Добавь текст после команды.")
        return
    send_notification(text)
    bot.send_message(message.chat.id, "Уведомление отправлено.")