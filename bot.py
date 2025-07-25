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
import openai

from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler
from telegram.constants import ParseMode
from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError, IntegrityError
from sqlalchemy.types import BigInteger
import yt_dlp

# --- Configuration ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Environment Variables ---
BOT_TOKEN = os.environ.get("BOT_TOKEN")
DATABASE_URL = os.environ.get("DATABASE_URL")
ADMIN_ID = os.environ.get("ADMIN_ID")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GPT_MODEL = os.environ.get("GPT_MODEL", "gpt-3.5-turbo").strip()
CLIPDROP_API_KEY = os.environ.get("CLIPDROP_API_KEY")
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")

# --- Initial Checks ---
if not all([BOT_TOKEN, DATABASE_URL, ADMIN_ID]):
    logger.critical("One or more critical environment variables are missing. Exiting.")
    sys.exit(1)
else:
    ADMIN_ID = int(ADMIN_ID)

# --- API Configurations ---
if OPENAI_API_KEY:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI (GPT) client configured.")
else:
    openai_client = None
    logger.warning("OPENAI_API_KEY not found. AI commands will be disabled.")

# --- Constants ---
DOWNLOAD_DIR = "downloads"; os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# --- Database Setup ---
if DATABASE_URL.startswith("postgres://"): DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
Base = declarative_base()
class User(Base):
    __tablename__ = 'users'
    user_id = Column(BigInteger, primary_key=True, nullable=False); first_name = Column(String, nullable=True)
    username = Column(String, nullable=True); chat_id = Column(BigInteger, primary_key=True, nullable=False)
engine = create_engine(DATABASE_URL, pool_size=5, max_overflow=10)
try: Base.metadata.create_all(engine)
except OperationalError as e: logger.critical(f"Failed to connect to database: {e}. Exiting."); sys.exit(1)
Session = sessionmaker(bind=engine)

# --- Helper, Notification & DB Functions ---
async def send_notification_to_admin(context: ContextTypes.DEFAULT_TYPE, user_info: dict, event_type: str):
    user_id = user_info.get('user_id')
    if user_id == ADMIN_ID: return
    
    first_name = user_info.get('first_name', 'N/A')
    username = user_info.get('username', 'N/A')
    
    message = (
        f"New Interaction: {event_type}\n"
        f"User: {first_name} (ID: `{user_id}`)\n"
        f"Username: @{username}" if username else "Username: Not set"
    )
    try:
        await context.bot.send_message(chat_id=ADMIN_ID, text=message, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Failed to send admin notification: {e}")

async def save_user_to_db(update: Update, context: ContextTypes.DEFAULT_TYPE, event_type: str = "User Interacted"):
    if not hasattr(update, 'effective_user') or not update.effective_user: return
    user = update.effective_user
    chat_id = update.effective_chat.id
    
    session = Session()
    try:
        existing_user = session.query(User).filter_by(user_id=user.id, chat_id=chat_id).first()
        user_info = {'user_id': user.id, 'first_name': user.first_name, 'username': user.username}

        if not existing_user:
            new_user = User(user_id=user.id, first_name=user.first_name, username=user.username, chat_id=chat_id)
            session.add(new_user)
            session.commit()
            await send_notification_to_admin(context, user_info, "New User Added")
        else:
            await send_notification_to_admin(context, user_info, event_type)
    except IntegrityError:
        session.rollback()
    except Exception as e:
        logger.error(f"DB Error for user {user.id}: {e}")
        session.rollback()
    finally:
        session.close()

# --- MAIN MENU & SUB-MENUS ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"Handling /start command or 'Back' button for user {update.effective_user.id}")
    keyboard = [
        [KeyboardButton("AI Tools"), KeyboardButton("Media Tools")],
        [KeyboardButton("Utilities"), KeyboardButton("Help")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("Main Menu:", reply_markup=reply_markup)

async def show_ai_tools_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [KeyboardButton("Chat with AI"), KeyboardButton("Create Image")],
        [KeyboardButton("Animate Image"), KeyboardButton("Upscale Image")],
        [KeyboardButton("Summarize Link"), KeyboardButton("Back to Main Menu")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("AI Tools:", reply_markup=reply_markup)

async def show_media_tools_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [KeyboardButton("Play Music"), KeyboardButton("Download Media")],
        [KeyboardButton("Back to Main Menu")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("Media Tools:", reply_markup=reply_markup)

async def show_utilities_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [KeyboardButton("Weather"), KeyboardButton("Crypto Prices")],
        [KeyboardButton("Translate Text"), KeyboardButton("Tell a Joke")],
        [KeyboardButton("Back to Main Menu")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("Utilities:", reply_markup=reply_markup)

# --- Command Logic Functions ---
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await save_user_to_db(update, context, "Requested Help")
    help_text = (
        "**Here's how to use the bot:**\n\n"
        "**Play Music**: Searches YouTube for a song.\n"
        "**Download Media**: Downloads video/audio from sites like TikTok, Facebook, etc.\n"
        "**Chat with AI**: Talk to an AI for questions.\n"
        "**Create Image**: Generate an image from a text description.\n"
        "**Animate Image**: Reply to an image with /animate to create a short video.\n"
        "**Upscale Image**: Reply to an image with /upscale to improve its quality.\n"
        "**Weather**: Get the current weather for any city.\n"
        "**Crypto Prices**: Check the latest prices of cryptocurrencies.\n"
        "**Summarize Link**: Get a summary of a web article.\n"
        "**Tell a Joke**: Get a random joke."
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

async def gpt_command(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt_text: str = None) -> None:
    if not openai_client: await update.message.reply_text("AI service is not configured. (Missing OPENAI_API_KEY)."); return
    if not prompt_text: prompt_text = " ".join(context.args)
    if not prompt_text: await update.message.reply_text("Please provide a prompt."); return
    
    feedback = await update.message.reply_text("Thinking...")
    try:
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt_text}]
        )
        reply_text = response.choices[0].message.content
        await feedback.edit_text(reply_text)
    except Exception as e:
        logger.error(f"OpenAI API error: {e}"); await feedback.edit_text("Sorry, an error occurred with the AI.")

async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE, text_to_translate: str = None) -> None:
    if not openai_client: await update.message.reply_text("Translate service is not configured. (Missing OPENAI_API_KEY)"); return
    
    if not text_to_translate:
        args = context.args
        if len(args) < 2:
            await update.message.reply_text("Usage: /translate <language> <text>\nExample: /translate Spanish Hello world")
            return
        target_lang, text = args[0], " ".join(args[1:])
    else:
        parts = text_to_translate.split(" ", 1)
        if len(parts) < 2:
            await update.message.reply_text("Format: language text\nExample: Spanish Hello world")
            return
        target_lang, text = parts

    feedback = await update.message.reply_text(f"Translating to {target_lang}...")
    try:
        prompt = f"Translate the following text to {target_lang}: {text}"
        response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        reply_text = response.choices[0].message.content
        await feedback.edit_text(reply_text)
    except Exception as e:
        logger.error(f"Translate API error: {e}"); await feedback.edit_text("Sorry, an error occurred during translation.")

async def summarize_url(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str) -> None:
    if not openai_client: await update.message.reply_text("AI summarizer is not configured. (Missing OPENAI_API_KEY)"); return
    feedback = await update.message.reply_text("Reading article...")
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        article_text = ' '.join([p.get_text() for p in soup.find_all('p')])
        if len(article_text) < 100: await feedback.edit_text("Couldn't extract enough text to summarize."); return
        await feedback.edit_text("Summarizing with AI...")
        prompt = f"Please provide a concise summary of the following article text:\n\n{article_text[:8000]}"
        ai_response = await asyncio.to_thread(
            openai_client.chat.completions.create,
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        reply_text = ai_response.choices[0].message.content
        await feedback.edit_text(f"**Summary:**\n{reply_text}", parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Summarize error for URL {url}: {e}"); await feedback.edit_text("Sorry, I couldn't read or summarize that URL.")

async def get_joke(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        response = requests.get("https://v2.jokeapi.dev/joke/Any?blacklistFlags=nsfw,racist,sexist&type=twopart", timeout=5).json()
        if not response['error']:
            await update.message.reply_text(response['setup']); await asyncio.sleep(2)
            await update.message.reply_text(response['delivery'])
    except Exception as e:
        logger.error(f"Joke API error: {e}"); await update.message.reply_text("Sorry, the joke service is unavailable.")

async def get_crypto_prices(update: Update, context: ContextTypes.DEFAULT_TYPE, crypto_ids: str) -> None:
    ids = [s.strip().lower() for s in crypto_ids.split(',')]
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(ids)}&vs_currencies=usd,ngn"
    try:
        prices = requests.get(url, timeout=5).json()
        if not prices: await update.message.reply_text("Couldn't find prices. Use full coin IDs from CoinGecko (e.g., bitcoin, ethereum)."); return
        message = "**Latest Crypto Prices (from CoinGecko):**\n\n"
        for coin, data in prices.items():
            message += f"**{coin.title()}**\n  - ${data.get('usd', 0):,.2f}\n  - N{data.get('ngn', 0):,.2f}\n\n"
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Crypto API error: {e}"); await update.message.reply_text("Sorry, I couldn't fetch crypto prices.")

async def get_weather(update: Update, context: ContextTypes.DEFAULT_TYPE, city: str) -> None:
    if not OPENWEATHER_API_KEY: await update.message.reply_text("Weather service not configured. (Missing OPENWEATHER_API_KEY)."); return
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        data = requests.get(url, timeout=5).json()
        if data["cod"] != 200: await update.message.reply_text(f"Sorry, couldn't find city '{city}'."); return
        message = f"**Weather in {data['name']}**\n- Condition: {data['weather'][0]['description'].title()}\n- Temperature: {data['main']['temp']}°C"
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Weather API error: {e}"); await update.message.reply_text("Sorry, I couldn't fetch the weather.")

async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str) -> None:
    if not STABILITY_API_KEY: await update.message.reply_text("Image generation service not configured. (Missing STABILITY_API_KEY)."); return
    feedback = await update.message.reply_text("Creating your image...")
    try:
        response = requests.post("https://api.stability.ai/v1/generation/stable-diffusion-v1-6/text-to-image", headers={"Authorization": f"Bearer {STABILITY_API_KEY}", "Accept": "application/json"}, json={"text_prompts": [{"text": prompt}]}, timeout=90)
        response.raise_for_status()
        image_b64 = response.json()["artifacts"][0]["base64"]
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=base64.b64decode(image_b64), caption=f"Creation: `{prompt}`", parse_mode=ParseMode.MARKDOWN)
        await feedback.delete()
    except Exception as e:
        logger.error(f"Stability AI error: {e}"); await feedback.edit_text("Sorry, I couldn't create the image.")

async def upscale_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not CLIPDROP_API_KEY: await update.message.reply_text("Image upscaling service not configured. (Missing CLIPDROP_API_KEY)."); return
    if not update.message.reply_to_message or not update.message.reply_to_message.photo: await update.message.reply_text("Please reply to an image with /upscale."); return
    feedback = await update.message.reply_text("Upscaling your image...")
    try:
        photo_file = await update.message.reply_to_message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        response = requests.post('https://clipdrop-api.co/image-upscaling/v1/upscale', files={'image_file': photo_bytes}, headers={'x-api-key': CLIPDROP_API_KEY}, timeout=90)
        response.raise_for_status()
        await context.bot.send_document(chat_id=update.effective_chat.id, document=response.content, filename='upscaled.png', caption='Here is your upscaled image!')
        await feedback.delete()
    except Exception as e:
        logger.error(f"ClipDrop API Error: {e}"); await feedback.edit_text("Sorry, an error occurred while upscaling.")

async def animate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not STABILITY_API_KEY: await update.message.reply_text("Video animation service not configured. (Missing STABILITY_API_KEY)."); return
    if not update.message.reply_to_message or not update.message.reply_to_message.photo: await update.message.reply_text("Please reply to an image with /animate."); return
    feedback = await update.message.reply_text("Sending image to animation engine...")
    try:
        photo_file = await update.message.reply_to_message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        response = requests.post("https://api.stability.ai/v2/generation/image-to-video", headers={"authorization": f"Bearer {STABILITY_API_KEY}"}, files={"image": photo_bytes}, data={"motion_bucket_id": 40}, timeout=30)
        response.raise_for_status()
        generation_id = response.json()["id"]
        await feedback.edit_text("Animation started. This may take a minute...")
        for _ in range(45):
            await asyncio.sleep(4)
            res = requests.get(f"https://api.stability.ai/v2/generation/image-to-video/result/{generation_id}", headers={'authorization': f"Bearer {STABILITY_API_KEY}", 'accept': "video/mp4"}, timeout=20)
            if res.status_code == 202: continue
            elif res.status_code == 200:
                await context.bot.send_video(chat_id=update.effective_chat.id, video=res.content, caption="Here is your animated video!")
                await feedback.delete(); return
        await feedback.edit_text("Sorry, the animation timed out.")
    except Exception as e:
        logger.error(f"Animate command error: {e}"); await feedback.edit_text("Sorry, an error occurred while creating the animation.")

async def search_and_play_song(update: Update, context: ContextTypes.DEFAULT_TYPE, song_name: str) -> None:
    feedback = await update.message.reply_text(f"Searching for '{song_name}'...")
    try:
        ydl_opts = {'noplaylist': True, 'quiet': True}
        if os.path.exists('cookies_youtube.txt'): ydl_opts['cookiefile'] = 'cookies_youtube.txt'
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: info = ydl.extract_info(f"ytsearch1:{song_name}", download=False)
        if not info.get('entries'): await feedback.edit_text("Sorry, couldn't find any results."); return
        video_info = info['entries'][0]; title = video_info.get('title', 'Unknown Title'); video_id = video_info.get('id')
        keyboard = [[InlineKeyboardButton("Yes, Download", callback_data=f"play_confirm:{video_id}"), InlineKeyboardButton("No, Cancel", callback_data="play_cancel")]]
        await feedback.edit_text(f"I found: '{title}'.\n\nIs this the correct song?", reply_markup=InlineKeyboardMarkup(keyboard))
    except Exception as e:
        logger.error(f"Play search error for '{song_name}': {e}"); await feedback.edit_text("An error occurred while searching. YouTube may require authentication (cookies_youtube.txt).")

async def handle_play_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query; await query.answer()
    video_id = query.data.split(":")[1]
    temp_dir = os.path.join(DOWNLOAD_DIR, str(uuid.uuid4())); os.makedirs(temp_dir)
    await query.edit_message_text("Downloading audio...")
    try:
        audio_opts = {'format': 'bestaudio/best', 'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'), 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}], 'noplaylist': True, 'quiet': True}
        if os.path.exists('cookies_youtube.txt'): audio_opts['cookiefile'] = 'cookies_youtube.txt'
        with yt_dlp.YoutubeDL(audio_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
        audio_path = os.path.join(temp_dir, os.listdir(temp_dir)[0])
        await query.edit_message_text("Sending audio...")
        with open(audio_path, 'rb') as audio_file:
            await context.bot.send_audio(chat_id=query.effective_chat.id, audio=audio_file, title=info.get('title'), duration=info.get('duration'))
        await query.delete_message()
    except Exception as e:
        logger.error(f"Play download error for ID {video_id}: {e}"); await query.edit_message_text("An error occurred while downloading.")
    finally:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

async def handle_play_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query; await query.answer()
    await query.edit_message_text("Search cancelled.")

async def download_content_from_url(update: Update, context: ContextTypes.DEFAULT_TYPE, platform: str, content_url: str) -> None:
    feedback = await update.message.reply_text(f"Starting download for {platform} link...")
    temp_dir = os.path.join(DOWNLOAD_DIR, str(uuid.uuid4())); os.makedirs(temp_dir, exist_ok=True)
    try:
        ydl_opts = {'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'), 'noplaylist': True, 'quiet': True, 'http_headers': {'User-Agent': 'Mozilla/5.0'}}
        if platform.lower() == 'youtube' and os.path.exists('cookies_youtube.txt'): ydl_opts['cookiefile'] = 'cookies_youtube.txt'
        elif os.path.exists('cookies.txt'): ydl_opts['cookiefile'] = 'cookies.txt'
        ydl_opts['format'] = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([content_url])
        downloaded_file = os.path.join(temp_dir, os.listdir(temp_dir)[0])
        await feedback.edit_text(f"Download complete. Uploading to Telegram...")
        with open(downloaded_file, 'rb') as f:
            await context.bot.send_video(chat_id=update.effective_chat.id, video=f, supports_streaming=True)
        await feedback.delete()
    except Exception as e:
        logger.error(f"Download error for {content_url}: {e}")
        await feedback.edit_text("Download failed. The link may be private or invalid.")
    finally:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

async def handle_keyboard_download_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await save_user_to_db(update, context, event_type="Pressed 'Download Media'")
    await show_download_options(update, context)

async def show_download_options(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [[InlineKeyboardButton("Facebook", callback_data="dl:Facebook"), InlineKeyboardButton("Instagram", callback_data="dl:Instagram")], [InlineKeyboardButton("YouTube", callback_data="dl:YouTube"), InlineKeyboardButton("TikTok", callback_data="dl:TikTok")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    text = "Please choose a platform to download from:"
    if update.callback_query: await update.callback_query.edit_message_text(text, reply_markup=reply_markup)
    else: await update.message.reply_text(text, reply_markup=reply_markup)

async def handle_download_platform_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query; await query.answer()
    platform_name = query.data.split(":")[1]
    context.user_data['state'] = 'awaiting_url'; context.user_data['platform'] = platform_name
    await query.edit_message_text(f"Please send me the full URL for the {platform_name} content.")

async def prompt_for_input(update: Update, context: ContextTypes.DEFAULT_TYPE, state: str, message: str, event: str) -> None:
    await save_user_to_db(update, context, event_type=event)
    context.user_data['state'] = state
    await update.message.reply_text(message)

async def record_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text: return
    state = context.user_data.pop('state', None)
    state_handlers = {
        'awaiting_song_name': search_and_play_song,
        'awaiting_url': lambda u, c, t: download_content_from_url(u, c, c.user_data.pop('platform'), t),
        'awaiting_gpt_prompt': lambda u, c, t: gpt_command(u, c, prompt_text=t),
        'awaiting_imagine_prompt': generate_image,
        'awaiting_city': get_weather,
        'awaiting_crypto_symbols': get_crypto_prices,
        'awaiting_summary_url': summarize_url,
        'awaiting_translation_text': lambda u,c,t: translate_command(u,c,text_to_translate=t)
    }
    handler = state_handlers.get(state)
    if handler: await handler(update, context, update.message.text)
    else: await save_user_to_db(update, context)

def main() -> None:
    application = Application.builder().token(BOT_TOKEN).build()
    # Command Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("upscale", upscale_image_command))
    application.add_handler(CommandHandler("animate", animate_command))
    application.add_handler(CommandHandler("translate", translate_command))

    # Main Menu Handlers
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^AI Tools$"), show_ai_tools_menu))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Media Tools$"), show_media_tools_menu))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Utilities$"), show_utilities_menu))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Back to Main Menu$"), start))
    
    # Sub-Menu Button Handlers
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Play Music$"), lambda u,c: prompt_for_input(u,c,'awaiting_song_name', "What song?","Pressed 'Play Music'")))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Download Media$"), handle_keyboard_download_button))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Chat with AI$"), lambda u,c: prompt_for_input(u,c,'awaiting_gpt_prompt', "What's on your mind?","Pressed 'Chat with AI'")))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Create Image$"), lambda u,c: prompt_for_input(u,c,'awaiting_imagine_prompt', "Describe the image.","Pressed 'Create Image'")))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Animate Image$"), lambda u,c: u.message.reply_text("Reply to an image with /animate.")))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Upscale Image$"), lambda u,c: u.message.reply_text("Reply to an image with /upscale.")))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Weather$"), lambda u,c: prompt_for_input(u,c,'awaiting_city', "Enter a city name.","Pressed 'Weather'")))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Crypto Prices$"), lambda u,c: prompt_for_input(u,c,'awaiting_crypto_symbols', "Enter coin IDs from CoinGecko (e.g., bitcoin).","Pressed 'Crypto'")))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Tell a Joke$"), get_joke))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Summarize Link$"), lambda u,c: prompt_for_input(u,c,'awaiting_summary_url', "Send the article link.","Pressed 'Summarize'")))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Help$"), help_command))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Translate Text$"), lambda u,c: prompt_for_input(u,c,'awaiting_translation_text', "Format: language text (e.g., Spanish Hello)","Pressed 'Translate'")))

    # Callback Query Handlers
    application.add_handler(CallbackQueryHandler(handle_download_platform_selection, pattern="^dl:"))
    application.add_handler(CallbackQueryHandler(handle_play_confirmation, pattern="^play_confirm:"))
    application.add_handler(CallbackQueryHandler(handle_play_cancel, pattern="^play_cancel"))

    # General message handler
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, record_user_message))
    
    # Webhook setup
    PORT = int(os.environ.get("PORT", 8443))
    RENDER_APP_NAME = os.environ.get("RENDER_APP_NAME")
    if not RENDER_APP_NAME:
        logger.warning("RENDER_APP_NAME env var not found. Running in polling mode.")
        application.run_polling()
    else:
        WEBHOOK_URL = f"https://{RENDER_APP_NAME}.onrender.com/{BOT_TOKEN}"
        application.run_webhook(listen="0.0.0.0", port=PORT, url_path=BOT_TOKEN, webhook_url=WEBHOOK_URL)

if __name__ == "__main__":
    main()
