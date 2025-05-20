import telebot
from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
import threading
import uvicorn

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

# ==== Команды бота ====
@bot.message_handler(commands=['start'])
def start_handler(message):
    bot.send_message(message.chat.id, " Напишите /subscribe чтобы получать уведомления.")

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

threading.Thread(target=start_fastapi, daemon=True).start()

# ==== Запуск бота ====
bot.infinity_polling()