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
from bs4 import BeautifulSoup # For Summarize
import base64 # For Imagine

from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler
from telegram.constants import ParseMode
from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError
from sqlalchemy.types import BigInteger
import yt_dlp
import google.generativeai as genai

# --- Configuration & Hardcoded Variables ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Main Bot & Service Keys ---
# WARNING: Hardcoding keys is not secure. Use environment variables in production.
BOT_TOKEN = "7806461656:AAEFsYhfk7moHzZgqX80qboJfb4b58UhsgU"
DATABASE_URL = "YOUR_DATABASE_URL" # ⚠️ Replace with your actual Render Postgres URL
ADMIN_ID = 7302005705
GEMINI_API_KEY = "AIzaSyDsvDWz-lOhuGyQV5rL-uumbtlNamXqfWM" # ⚠️ Replace with your secret Gemini API key
CLIPDROP_API_KEY = "YOUR_CLIPDROP_API_KEY" # For Upscale
STABILITY_API_KEY = "YOUR_STABILITY_API_KEY" # For Imagine (Text-to-Image)
OPENWEATHER_API_KEY = "YOUR_OPENWEATHER_API_KEY" # For Weather

# --- API Configurations ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    gemini_model = None

# --- Database, Helper Functions, Notifications (Omitted for Brevity) ---
# The database models, escape_markdown_v2, save_user_to_db, and 
# send_notification_to_admin functions are unchanged from the last version.
# Make sure they are present in your final file.
# ...

# --- NEW Command Logic ---

async def get_joke(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fetches and sends a random joke."""
    await save_user_to_db(update, context, event_type="Requested a joke")
    try:
        response = requests.get("https://v2.jokeapi.dev/joke/Any?blacklistFlags=nsfw,racist,sexist,political,religious,explicit&type=twopart")
        response.raise_for_status()
        joke_data = response.json()
        if not joke_data['error']:
            await update.message.reply_text(joke_data['setup'])
            await asyncio.sleep(2)
            await update.message.reply_text(joke_data['delivery'])
        else:
            await update.message.reply_text("Sorry, I couldn't think of a joke right now.")
    except Exception as e:
        logger.error(f"Joke API error: {e}")
        await update.message.reply_text("Sorry, the joke service seems to be down.")

async def get_crypto_prices(update: Update, context: ContextTypes.DEFAULT_TYPE, crypto_ids: str) -> None:
    """Fetches and displays crypto prices."""
    ids = [s.strip().lower() for s in crypto_ids.split(',')]
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(ids)}&vs_currencies=usd,ngn"
    try:
        response = requests.get(url)
        response.raise_for_status()
        prices = response.json()
        if not prices:
            await update.message.reply_text("Couldn't find prices for the requested crypto. Please use the full name (e.g., bitcoin, ethereum).")
            return
        
        message = "💰 **Latest Crypto Prices**:\n\n"
        for coin, data in prices.items():
            usd_price = data.get('usd', 0)
            ngn_price = data.get('ngn', 0)
            message += f"🪙 **{coin.title()}**\n  - `${usd_price:,.2f}`\n  - `₦{ngn_price:,.2f}`\n\n"
        
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Crypto API error: {e}")
        await update.message.reply_text("Sorry, I couldn't fetch the crypto prices right now.")

async def get_weather(update: Update, context: ContextTypes.DEFAULT_TYPE, city: str) -> None:
    """Fetches and displays the weather for a given city."""
    if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "YOUR_OPENWEATHER_API_KEY":
        await update.message.reply_text("The weather service is not configured.")
        return

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        main = data['weather'][0]['main']
        description = data['weather'][0]['description']
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        
        weather_icons = {"Clear": "☀️", "Clouds": "☁️", "Rain": "🌧️", "Drizzle": "💧", "Thunderstorm": "⚡", "Snow": "❄️", "Mist": "🌫️"}
        icon = weather_icons.get(main, "🌍")

        message = (
            f"{icon} **Weather in {data['name']}**\n\n"
            f"**Condition:** {description.title()}\n"
            f"**Temperature:** {temp}°C\n"
            f"**Humidity:** {humidity}%"
        )
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            await update.message.reply_text(f"Sorry, I couldn't find the city '{city}'. Please check the spelling.")
        else:
            await update.message.reply_text("Sorry, an error occurred with the weather service.")
    except Exception as e:
        logger.error(f"Weather API error: {e}")
        await update.message.reply_text("Sorry, I couldn't fetch the weather right now.")

async def summarize_url(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str) -> None:
    """Summarizes an article from a URL using Gemini."""
    if not gemini_model:
        await update.message.reply_text("The AI summarizer is not configured.")
        return

    feedback = await update.message.reply_text("Reading the article...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])

        if len(article_text) < 100:
            await feedback.edit_text("Couldn't extract enough text from the article to summarize.")
            return

        await feedback.edit_text("Summarizing with AI...")
        prompt = f"Please provide a concise summary of the following article text:\n\n{article_text[:8000]}"
        ai_response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        
        await feedback.edit_text(f"**Summary of the article:**\n\n{ai_response.text}", parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        logger.error(f"Summarize error for URL {url}: {e}")
        await feedback.edit_text("Sorry, I couldn't read or summarize that URL. Please ensure it's a valid article link.")

async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str) -> None:
    """Generates an image from a text prompt using Stability AI."""
    if not STABILITY_API_KEY or STABILITY_API_KEY == "YOUR_STABILITY_API_KEY":
        await update.message.reply_text("The image generation service is not configured.")
        return
        
    feedback = await update.message.reply_text("🎨 Creating your image... this may take a moment.")
    try:
        url = "https://api.stability.ai/v1/generation/stable-diffusion-v1-6/text-to-image"
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {STABILITY_API_KEY}",
        }
        payload = {
            "steps": 40, "width": 1024, "height": 1024, "samples": 1, "cfg_scale": 7,
            "text_prompts": [{"text": prompt, "weight": 1}],
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        image_b64 = data["artifacts"][0]["base64"]
        image_bytes = base64.b64decode(image_b64)
        
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=image_bytes,
            caption=f"✨ Here is your creation:\n\n`{prompt}`",
            parse_mode=ParseMode.MARKDOWN
        )
        await feedback.delete()
    except Exception as e:
        logger.error(f"Stability AI error: {e}")
        await feedback.edit_text("Sorry, I couldn't create the image. The service might be down or the prompt was blocked.")

# --- Menu Prompt Handlers ---
async def prompt_for_input(update: Update, context: ContextTypes.DEFAULT_TYPE, state: str, message: str, event: str) -> None:
    """A generic function to ask for user input and set state."""
    await save_user_to_db(update, context, event_type=event)
    context.user_data['state'] = state
    await update.message.reply_text(message)

# --- Start Command & Main Menu ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows the main menu dashboard."""
    await save_user_to_db(update, context, event_type="Opened the bot")
    keyboard = [
        [KeyboardButton("▶️ Play Music"), KeyboardButton("⬇️ Download Media")],
        [KeyboardButton("🤖 Chat with AI"), KeyboardButton("🎨 Create Image")],
        [KeyboardButton("🌦️ Weather"), KeyboardButton("💰 Crypto Prices")],
        [KeyboardButton("😂 Tell a Joke"), KeyboardButton("📰 Summarize Link")],
        [KeyboardButton("🖼️ Upscale Image"), KeyboardButton("❓ Help")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("Welcome to your dashboard! What would you like to do?", reply_markup=reply_markup)

# --- Central Message Handler ---
async def record_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles all text messages and routes them based on user state."""
    if not update.message or not update.message.text: return
    state = context.user_data.pop('state', None) # Pop state to handle it once

    if state == 'awaiting_song_name':
        await search_and_play_song(update, context, update.message.text)
    elif state == 'awaiting_url':
        await download_content_from_url(update, context, context.user_data.pop('platform'), update.message.text)
    elif state == 'awaiting_gemini_prompt':
        await gemini_command(update, context, prompt_text=update.message.text)
    elif state == 'awaiting_imagine_prompt':
        await generate_image(update, context, update.message.text)
    elif state == 'awaiting_city':
        await get_weather(update, context, update.message.text)
    elif state == 'awaiting_crypto_symbols':
        await get_crypto_prices(update, context, update.message.text)
    elif state == 'awaiting_summary_url':
        await summarize_url(update, context, update.message.text)
    else:
        # Default interaction if no state matches
        await save_user_to_db(update, context)

# --- Main Bot Logic ---
def main() -> None:
    """Start the bot."""
    application = Application.builder().token(BOT_TOKEN).build()

    # --- Register Handlers ---
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command)) # Assuming you have a help_command
    
    # Direct command handlers for power users
    application.add_handler(CommandHandler("joke", get_joke))
    application.add_handler(CommandHandler("weather", get_weather))
    # ... etc for other commands

    # --- Menu Button Handlers ---
    # Row 1
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^▶️ Play Music$"), lambda u,c: prompt_for_input(u,c, 'awaiting_song_name', "🎵 What song would you like to listen to?", "Pressed 'Play Music'")))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^⬇️ Download Media$"), handle_keyboard_download_button))
    # Row 2
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^🤖 Chat with AI$"), lambda u,c: prompt_for_input(u,c, 'awaiting_gemini_prompt', "🤔 What's on your mind?", "Pressed 'Chat with AI'")))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^🎨 Create Image$"), lambda u,c: prompt_for_input(u,c, 'awaiting_imagine_prompt', "🖼️ Describe the image you want to create.", "Pressed 'Create Image'")))
    # Row 3
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^🌦️ Weather$"), lambda u,c: prompt_for_input(u,c, 'awaiting_city', "🌍 Please enter a city name.", "Pressed 'Weather'")))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^💰 Crypto Prices$"), lambda u,c: prompt_for_input(u,c, 'awaiting_crypto_symbols', "🪙 Enter crypto names, comma-separated (e.g., bitcoin, solana).", "Pressed 'Crypto Prices'")))
    # Row 4
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^😂 Tell a Joke$"), get_joke))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^📰 Summarize Link$"), lambda u,c: prompt_for_input(u,c, 'awaiting_summary_url', "🔗 Send me the link to the article.", "Pressed 'Summarize'")))
    # Row 5
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^🖼️ Upscale Image$"), lambda u,c: u.message.reply_text("Please reply to an image with the `/upscale` command.")))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^❓ Help$"), help_command))

    application.add_handler(CommandHandler("upscale", upscale_image_command))

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

