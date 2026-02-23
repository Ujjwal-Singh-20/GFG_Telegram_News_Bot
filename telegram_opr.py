import requests
import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    """Sends a message to the configured Telegram group/channel."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID or "YOUR_" in TELEGRAM_BOT_TOKEN:
        print("Telegram bot not configured. Skipping notification.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("Telegram message sent successfully!")
        return True
    except Exception as e:
        print(f"Error sending Telegram message: {e}")
        return False
