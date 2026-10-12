import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters
import os
# 👉 dán token của bạn vào đây
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# 👉 địa chỉ API bạn đang chạy
API_URL = "http://127.0.0.1:8000/summarize"

# khi user gửi tin nhắn
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text

    try:
        # gửi text sang API
        response = requests.post(API_URL, json={
            "text": text
        })

        result = response.json()

        summary = result["summary"]

        # trả lại cho user
        await update.message.reply_text(summary)

    except Exception as e:
        await update.message.reply_text(f"Loi: {str(e)}")


# chạy bot
def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    print("Bot đang chạy...")
    app.run_polling()


if __name__ == "__main__":
    main()