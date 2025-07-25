import logging
import os
import sys
import asyncio
import uuid
import shutil
import requests
import json
from bs4 import BeautifulSoup
import base64
import fitz  # PyMuPDF
import google.generativeai as genai

from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler
from telegram.constants import ParseMode
from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import sessionmaker, declarative_base
# ✅ FIX: Added missing IntegrityError for database operations
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
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
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
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Gemini AI client configured.")
    else:
        gemini_model = None
        logger.warning("GEMINI_API_KEY not found. AI commands will be disabled.")
except Exception as e:
    gemini_model = None
    logger.error(f"Failed to configure Gemini API: {e}")

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

# --- Helper Functions ---
# ✅ FIX: Implemented the full database saving and notification logic
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
            if user.id != ADMIN_ID:
                await send_notification_to_admin(context, user_info, "New User Added")
        else:
             if user.id != ADMIN_ID:
                await send_notification_to_admin(context, user_info, event_type)
    except IntegrityError:
        session.rollback()
    except Exception as e:
        logger.error(f"DB Error for user {user.id}: {e}")
        session.rollback()
    finally:
        session.close()

async def send_notification_to_admin(context: ContextTypes.DEFAULT_TYPE, user_info: dict, event_type: str):
    first_name = user_info.get('first_name', 'N/A')
    username = f"@{user_info.get('username')}" if user_info.get('username') else "Not set"
    message = (
        f"Interaction: {event_type}\n"
        f"User: {first_name} (ID: `{user_info.get('user_id')}`)\n"
        f"Username: {username}"
    )
    try:
        await context.bot.send_message(chat_id=ADMIN_ID, text=message, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Failed to send admin notification: {e}")

# --- MAIN MENU & SUB-MENUS ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [KeyboardButton("AI Tools"), KeyboardButton("Media Tools")],
        [KeyboardButton("Utilities"), KeyboardButton("Help")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
    await update.message.reply_text("Main Menu:", reply_markup=reply_markup)

async def show_ai_tools_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [KeyboardButton("Chat with AI"), KeyboardButton("Create Image")],
        [KeyboardButton("Animate Image"), KeyboardButton("Upscale Image")],
        [KeyboardButton("Summarize Link"), KeyboardButton("Summarize File")],
        [KeyboardButton("Back to Main Menu")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
    await update.message.reply_text("AI Tools:", reply_markup=reply_markup)

async def show_media_tools_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [KeyboardButton("Play Music / Video"), KeyboardButton("Download Media")],
        [KeyboardButton("Back to Main Menu")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
    await update.message.reply_text("Media Tools:", reply_markup=reply_markup)

async def show_utilities_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [KeyboardButton("Weather"), KeyboardButton("Crypto Prices")],
        [KeyboardButton("Translate Text"), KeyboardButton("Tell a Joke")],
        [KeyboardButton("Back to Main Menu")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
    await update.message.reply_text("Utilities:", reply_markup=reply_markup)

# --- Command Logic Functions ---
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "**Bot Commands Guide:**\n\n"
        "**AI Tools**\n"
        "- `Chat with AI`: Talk to an AI. You can also reply to an image to ask questions about it.\n"
        "- `Create Image`: Generate an image from text.\n"
        "- `Animate Image`: Reply to an image with /animate to create a short video.\n"
        "- `Upscale Image`: Reply to an image with /upscale to improve its quality.\n"
        "- `Summarize Link`: Get a summary of a web article.\n"
        "- `Summarize File`: Reply to an image or PDF with /summarize_file.\n\n"
        "**Media Tools**\n"
        "- `Play Music / Video`: Searches YouTube for a song or video.\n"
        "- `Download Media`: Downloads video/audio from sites like TikTok, etc.\n\n"
        "**Utilities**\n"
        "- `Weather`, `Crypto Prices`, `Translate Text`, `Tell a Joke`."
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

# ✅ FIX: Modified function to accept text from buttons OR /gemini command
async def gemini_command(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt_text: str = None) -> None:
    if not gemini_model:
        await update.message.reply_text("AI service is not configured (Missing GEMINI_API_KEY).")
        return

    prompt = prompt_text if prompt_text is not None else " ".join(context.args)
    replied_message = update.message.reply_to_message
    image_parts = []

    if replied_message and replied_message.photo:
        photo_file = await replied_message.photo[-1].get_file()
        photo_bytes = await photo_file.download_as_bytearray()
        image_parts.append({"mime_type": "image/jpeg", "data": photo_bytes})
        if not prompt:
            prompt = "Describe this image in detail."
    
    if not prompt:
        await update.message.reply_text("Please provide a prompt, or reply to an image with a prompt.")
        return
        
    feedback = await update.message.reply_text("Thinking...")
    try:
        full_prompt = [prompt] + image_parts
        response = await asyncio.to_thread(gemini_model.generate_content, full_prompt)
        await feedback.edit_text(response.text)
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        await feedback.edit_text("Sorry, an error occurred with the AI.")

async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE, text_to_translate: str = None) -> None:
    if not gemini_model: await update.message.reply_text("Translate service is not configured. (Missing GEMINI_API_KEY)"); return
    
    if not text_to_translate:
        args = context.args
        if len(args) < 2: await update.message.reply_text("Usage: /translate <language> <text>"); return
        target_lang, text = args[0], " ".join(args[1:])
    else:
        parts = text_to_translate.split(" ", 1)
        if len(parts) < 2: await update.message.reply_text("Format: language text"); return
        target_lang, text = parts

    feedback = await update.message.reply_text(f"Translating to {target_lang}...")
    try:
        prompt = f"Translate the following text to {target_lang}: {text}"
        response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        await feedback.edit_text(response.text)
    except Exception as e:
        logger.error(f"Translate API error: {e}"); await feedback.edit_text("Sorry, an error occurred during translation.")

async def summarize_url(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str) -> None:
    if not gemini_model: await update.message.reply_text("AI summarizer is not configured. (Missing GEMINI_API_KEY)"); return
    feedback = await update.message.reply_text("Reading article...")
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        article_text = ' '.join([p.get_text() for p in soup.find_all('p')])
        if len(article_text) < 100: await feedback.edit_text("Couldn't extract enough text to summarize."); return
        await feedback.edit_text("Summarizing with AI...")
        prompt = f"Please provide a concise summary of the following article text:\n\n{article_text[:12000]}"
        ai_response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        await feedback.edit_text(f"**Summary:**\n{ai_response.text}", parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Summarize error for URL {url}: {e}"); await feedback.edit_text("Sorry, I couldn't read or summarize that URL.")

async def summarize_file_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not gemini_model: await update.message.reply_text("AI service is not configured (Missing GEMINI_API_KEY)."); return
    if not update.message.reply_to_message:
        await update.message.reply_text("Please reply to an image or a PDF file with /summarize_file."); return

    replied_message = update.message.reply_to_message
    feedback_message = await replied_message.reply_text("Processing file...")

    try:
        if replied_message.photo:
            photo_file = await replied_message.photo[-1].get_file()
            photo_bytes = await photo_file.download_as_bytearray()
            image_part = {"mime_type": "image/jpeg", "data": photo_bytes}
            prompt = "Describe this image in detail."
            response = await asyncio.to_thread(gemini_model.generate_content, [prompt, image_part])
            summary = response.text

        elif replied_message.document and replied_message.document.mime_type == 'application/pdf':
            pdf_file = await replied_message.document.get_file()
            pdf_bytes = await pdf_file.download_as_bytearray()
            
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            full_text = "".join([page.get_text() for page in doc])
            doc.close()

            if not full_text.strip(): await feedback_message.edit_text("Could not extract any text from this PDF."); return

            await feedback_message.edit_text("Extracted text from PDF. Summarizing...")
            prompt = f"Please provide a detailed summary of the following document text:\n\n{full_text[:12000]}"
            response = await asyncio.to_thread(gemini_model.generate_content, prompt)
            summary = response.text
        
        else:
            await feedback_message.edit_text("This command only works when you reply to an image or a PDF file."); return

        await feedback_message.edit_text(f"**Summary:**\n\n{summary}", parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"File summarization error: {e}")
        await feedback_message.edit_text("Sorry, an error occurred while processing the file.")

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
        url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
        headers = {"Authorization": f"Bearer {STABILITY_API_KEY}", "Accept": "application/json"}
        payload = {"text_prompts": [{"text": prompt}]}
        response = requests.post(url, headers=headers, json=payload, timeout=90)
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
        keyboard = [[InlineKeyboardButton("Download Audio", callback_data=f"play_audio:{video_id}"), InlineKeyboardButton("Download Video", callback_data=f"play_video:{video_id}")], [InlineKeyboardButton("Cancel", callback_data="play_cancel")]]
        await feedback.edit_text(f"I found: '{title}'.\n\nChoose your desired format:", reply_markup=InlineKeyboardMarkup(keyboard))
    except Exception as e:
        logger.error(f"Play search error for '{song_name}': {e}"); await feedback.edit_text("An error occurred while searching. YouTube may be blocking requests. Ensure `cookies_youtube.txt` is valid.")

async def handle_play_audio_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query; await query.answer()
    video_id = query.data.split(":")[1]
    temp_dir = os.path.join(DOWNLOAD_DIR, str(uuid.uuid4())); os.makedirs(temp_dir)
    await query.edit_message_text("Downloading audio...")
    try:
        audio_opts = {'format': 'bestaudio/best', 'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'), 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}], 'noplaylist': True, 'quiet': True}
        if os.path.exists('cookies_youtube.txt'): audio_opts['cookiefile'] = 'cookies_youtube.txt'
        # ✅ FIX: Using the video_id directly is the most reliable way.
        with yt_dlp.YoutubeDL(audio_opts) as ydl: info = ydl.extract_info(video_id, download=True)
        audio_path = os.path.join(temp_dir, os.listdir(temp_dir)[0])
        await query.edit_message_text("Sending audio...")
        with open(audio_path, 'rb') as audio_file:
            await context.bot.send_audio(chat_id=query.message.chat_id, audio=audio_file, title=info.get('title'), duration=info.get('duration'))
        await query.delete_message()
    except Exception as e:
        logger.error(f"Play audio download error for ID {video_id}: {e}"); await query.edit_message_text("An error occurred while downloading.")
    finally:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

async def handle_play_video_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query; await query.answer()
    video_id = query.data.split(":")[1]
    temp_dir = os.path.join(DOWNLOAD_DIR, str(uuid.uuid4())); os.makedirs(temp_dir)
    await query.edit_message_text("Downloading video...")
    try:
        video_opts = {'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', 'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'), 'noplaylist': True, 'quiet': True}
        if os.path.exists('cookies_youtube.txt'): video_opts['cookiefile'] = 'cookies_youtube.txt'
        # ✅ FIX: Using the video_id directly is the most reliable way.
        with yt_dlp.YoutubeDL(video_opts) as ydl: ydl.download([video_id])
        video_path = os.path.join(temp_dir, os.listdir(temp_dir)[0])
        await query.edit_message_text("Sending video...")
        with open(video_path, 'rb') as video_file:
            await context.bot.send_video(chat_id=query.message.chat_id, video=video_file, supports_streaming=True)
        await query.delete_message()
    except Exception as e:
        logger.error(f"Play video download error for ID {video_id}: {e}"); await query.edit_message_text("An error occurred while downloading.")
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
    if update.message and update.message.reply_to_message and update.message.reply_to_message.photo and not context.user_data.get('state'):
        context.args = update.message.text.split() if update.message.text else []
        await gemini_command(update, context)
        return
    if not update.message or not update.message.text: return
    state = context.user_data.pop('state', None)
    
    # Using a dictionary to map states to their handler functions
    state_handlers = {
        'awaiting_song_name': lambda u, c, t: search_and_play_song(u, c, song_name=t),
        'awaiting_url': lambda u, c, t: download_content_from_url(u, c, c.user_data.pop('platform'), t),
        'awaiting_gemini_prompt': lambda u, c, t: gemini_command(u, c, prompt_text=t),
        'awaiting_imagine_prompt': lambda u, c, t: generate_image(u, c, prompt=t),
        'awaiting_city': lambda u, c, t: get_weather(u, c, city=t),
        'awaiting_crypto_symbols': lambda u, c, t: get_crypto_prices(u, c, crypto_ids=t),
        'awaiting_summary_url': lambda u, c, t: summarize_url(u, c, url=t),
        'awaiting_translation_text': lambda u, c, t: translate_command(u, c, text_to_translate=t)
    }
    
    handler = state_handlers.get(state)
    if handler:
        await handler(update, context, update.message.text)
    else:
        # Default behavior for any text not matching a state
        await save_user_to_db(update, context, "Sent a message")


def main() -> None:
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Command Handlers for direct commands like /start
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("upscale", upscale_image_command))
    application.add_handler(CommandHandler("animate", animate_command))
    application.add_handler(CommandHandler("translate", translate_command))
    application.add_handler(CommandHandler("summarize_file", summarize_file_command))
    application.add_handler(CommandHandler("gemini", gemini_command))

    # Message Handlers for the main menu buttons
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^AI Tools$"), show_ai_tools_menu))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Media Tools$"), show_media_tools_menu))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Utilities$"), show_utilities_menu))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Back to Main Menu$"), start))
    
    # Message Handlers for sub-menu buttons that prompt for user input
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Play Music / Video$"), lambda u,c: prompt_for_input(u,c,'awaiting_song_name', "What song or video would you like to search for?","Pressed 'Play'")))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Download Media$"), handle_keyboard_download_button))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Chat with AI$"), lambda u,c: prompt_for_input(u,c,'awaiting_gemini_prompt', "What's on your mind? You can also reply to an image.","Pressed 'Chat with AI'")))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Create Image$"), lambda u,c: prompt_for_input(u,c,'awaiting_imagine_prompt', "Describe the image you want to create.","Pressed 'Create Image'")))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Animate Image$"), lambda u,c: u.message.reply_text("Please reply to an image with the /animate command.")))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Upscale Image$"), lambda u,c: u.message.reply_text("Please reply to an image with the /upscale command.")))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Weather$"), lambda u,c: prompt_for_input(u,c,'awaiting_city', "Please enter a city name.","Pressed 'Weather'")))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Crypto Prices$"), lambda u,c: prompt_for_input(u,c,'awaiting_crypto_symbols', "Enter coin IDs separated by commas (e.g., bitcoin,ethereum).","Pressed 'Crypto'")))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Tell a Joke$"), get_joke))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Summarize Link$"), lambda u,c: prompt_for_input(u,c,'awaiting_summary_url', "Please send the full article link.","Pressed 'Summarize Link'")))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Summarize File$"), lambda u,c: u.message.reply_text("Please reply to an image or a PDF file with the /summarize_file command.")))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Help$"), help_command))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Translate Text$"), lambda u,c: prompt_for_input(u,c,'awaiting_translation_text', "Please enter the text to translate in the format: language text (e.g., Spanish Hello world)","Pressed 'Translate'")))

    # Callback Query Handlers for inline buttons (like download options)
    application.add_handler(CallbackQueryHandler(handle_download_platform_selection, pattern="^dl:"))
    application.add_handler(CallbackQueryHandler(handle_play_audio_confirmation, pattern="^play_audio:"))
    application.add_handler(CallbackQueryHandler(handle_play_video_confirmation, pattern="^play_video:"))
    application.add_handler(CallbackQueryHandler(handle_play_cancel, pattern="^play_cancel"))

    # General message handler to process user input after a prompt
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, record_user_message))
    
    # Webhook setup for deployment
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
