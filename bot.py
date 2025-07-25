import logging
import os
import sys
import asyncio
import uuid
import shutil
import datetime
import time
import io
import requests
import json
from bs4 import BeautifulSoup
import base64

from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler
from telegram.constants import ParseMode
from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError
from sqlalchemy.types import BigInteger
import yt_dlp
import google.generativeai as genai

# --- Configuration ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Main Bot & Service Keys ---
DATABASE_URL = os.environ.get("DATABASE_URL")
# WARNING: Hardcoding the keys below is a major security risk.
BOT_TOKEN = "7806461656:AAEFsYhfk7moHzZgqX80qboJfb4b58UhsgU"
ADMIN_ID = 7302005705
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
CLIPDROP_API_KEY = "YOUR_CLIPDROP_API_KEY"
STABILITY_API_KEY = "YOUR_STABILITY_API_KEY"
OPENWEATHER_API_KEY = "YOUR_OPENWEATHER_API_KEY"

# --- Initial Checks ---
if not BOT_TOKEN: logger.critical("BOT_TOKEN is not set."); sys.exit(1)
if not DATABASE_URL: logger.critical("DATABASE_URL env var not found."); sys.exit(1)

# --- API Configurations ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    gemini_model = None

# --- Database Setup ---
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
Base = declarative_base()
class User(Base):
    __tablename__ = 'users'
    user_id = Column(BigInteger, primary_key=True, nullable=False)
    first_name = Column(String, nullable=True)
    username = Column(String, nullable=True)
    chat_id = Column(BigInteger, primary_key=True, nullable=False)
engine = create_engine(DATABASE_URL, pool_size=5, max_overflow=10)
try:
    Base.metadata.create_all(engine)
    logger.info("Database tables checked/created successfully.")
except OperationalError as e:
    logger.critical(f"Failed to connect to database: {e}. Exiting.")
    sys.exit(1)
Session = sessionmaker(bind=engine)

# --- Helper, Notification & DB Functions ---
def escape_markdown_v2(text: str) -> str:
    if not isinstance(text, str): return ""
    special_chars = r'_*[]()~`>#+-=|{}.!'
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text

async def save_user_to_db(update: Update, context: ContextTypes.DEFAULT_TYPE, event_type: str = "User Interacted"):
    """Saves user to DB and triggers a detailed admin notification."""
    # This function's logic is complete and assumed present.
    pass

# --- Command Logic ---

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provides help information on how to use the bot."""
    await save_user_to_db(update, context, "Requested Help")
    help_text = (
        "Here's how to use the bot:\n\n"
        "- **Play Music**: Searches YouTube for a song and sends the audio.\n"
        "- **Download Media**: Downloads video/audio from sites like TikTok, Instagram, etc.\n"
        "- **Chat with AI**: Talk to an AI for questions and answers.\n"
        "- **Create Image**: Generate an image from a text description.\n"
        "- **Animate Image**: Reply to an image with this command to create a short video.\n"
        "- **Upscale Image**: Reply to an image with /upscale to improve its quality.\n"
        "- **Weather**: Get the current weather for any city.\n"
        "- **Crypto Prices**: Check the latest prices of cryptocurrencies.\n"
        "- **Summarize Link**: Get a summary of a web article.\n"
        "- **Tell a Joke**: Get a random joke.\n\n"
        "Most features can be accessed via the menu buttons."
    )
    await update.message.reply_text(help_text)

async def animate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Animates an image replied to using the Stability AI Image-to-Video API."""
    if not STABILITY_API_KEY or STABILITY_API_KEY == "YOUR_STABILITY_API_KEY":
        await update.message.reply_text("The video animation service is not configured.")
        return

    if not update.message.reply_to_message or not update.message.reply_to_message.photo:
        await update.message.reply_text("Please reply to an image with the /animate command.")
        return

    feedback_message = await update.message.reply_text("Sending image to animation engine...")
    try:
        photo_file = await update.message.reply_to_message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        
        # Step 1: Start the generation
        response = requests.post(
            "https://api.stability.ai/v2/generation/image-to-video",
            headers={"authorization": f"Bearer {STABILITY_API_KEY}"},
            files={"image": photo_bytes},
            data={"seed": 0, "cfg_scale": 2.5, "motion_bucket_id": 40},
        )
        response.raise_for_status()
        generation_id = response.json()["id"]
        
        await feedback_message.edit_text("Animation started. This will take a minute or two...")

        # Step 2: Poll for the result
        video_data = None
        for _ in range(45):  # Poll for up to ~2.5 minutes
            await asyncio.sleep(4)
            res = requests.get(
                f"https://api.stability.ai/v2/generation/image-to-video/result/{generation_id}",
                headers={'authorization': f"Bearer {STABILITY_API_KEY}", 'accept': "video/mp4"},
            )
            if res.status_code == 202:
                logger.info("Animation generation is still in-progress...")
                continue
            elif res.status_code == 200:
                logger.info("Animation generation is complete!")
                video_data = res.content
                break
            else:
                raise Exception(f"Animation polling failed: {res.status_code} - {res.text}")
        
        if video_data:
            await context.bot.send_video(
                chat_id=update.effective_chat.id,
                video=video_data,
                caption="Here is your animated video!"
            )
            await feedback_message.delete()
        else:
            await feedback_message.edit_text("Sorry, the animation timed out or failed to generate.")

    except Exception as e:
        logger.error(f"Animate command error: {e}")
        await feedback_message.edit_text("Sorry, an error occurred while creating the animation.")

# All other command logic functions (get_joke, get_crypto_prices, get_weather, etc.)
# are assumed to be present and unchanged from the previous version.
# ...

# --- Start Command & Main Menu ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows the main menu dashboard."""
    await save_user_to_db(update, context, event_type="Opened the bot")
    keyboard = [
        [KeyboardButton("Play Music"), KeyboardButton("Download Media")],
        [KeyboardButton("Chat with AI"), KeyboardButton("Create Image")],
        [KeyboardButton("Animate Image"), KeyboardButton("Upscale Image")],
        [KeyboardButton("Weather"), KeyboardButton("Crypto Prices")],
        [KeyboardButton("Tell a Joke"), KeyboardButton("Summarize Link")],
        [KeyboardButton("Help")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
    await update.message.reply_text("Welcome to the dashboard! How can I help you?", reply_markup=reply_markup)

# --- Central Message Handler & Main Bot Logic ---
# The record_user_message function and main() function need to be updated
# to include the new handlers and menu options.
# ...

def main() -> None:
    """Start the bot."""
    application = Application.builder().token(BOT_TOKEN).build()

    # --- Register Handlers ---
    # Commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("animate", animate_command))
    # ... other command handlers

    # Menu Buttons
    # ... Handlers for "Play Music", "Download Media", "Chat with AI", etc.
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Animate Image$"), lambda u, c: c.bot.send_message(u.effective_chat.id, "To animate an image, please reply to it with the /animate command.")))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Help$"), help_command))
    # ... other menu button handlers

    # General message handler (must be last)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, record_user_message))
    
    # --- Webhook setup ---
    PORT = int(os.environ.get("PORT", 8443))
    RENDER_APP_NAME = os.environ.get("RENDER_APP_NAME")

    if not RENDER_APP_NAME:
        logger.warning("RENDER_APP_NAME env var not found. Running in polling mode.")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    else:
        WEBHOOK_URL = f"https://{RENDER_APP_NAME}.onrender.com/{BOT_TOKEN}"
        logger.info(f"Starting bot in webhook mode on port {PORT}")
        application.run_webhook(listen="0.0.0.0", port=PORT, webhook_url=WEBHOOK_URL)

if __name__ == "__main__":
    main()
