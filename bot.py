# -*- coding: utf-8 -*-

# --- Imports ---
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
import openai # For DALL-E image creation
import pytesseract # For OCR
from PIL import Image # For OCR
import io # For OCR
import smtplib
from email.message import EmailMessage
import shlex # For smart command parsing
import re # Added for robust button handling
import math # Added for video splitting calculation

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler, AIORateLimiter
from telegram.constants import ParseMode, ChatAction
from telegram.helpers import escape_markdown

from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError, IntegrityError
from sqlalchemy.types import BigInteger
import yt_dlp

# Load environment variables from .env file for local development
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Environment Variables & API Configurations ---
BOT_TOKEN = os.environ.get("BOT_TOKEN")
DATABASE_URL = os.environ.get("DATABASE_URL")
ADMIN_ID = os.environ.get("ADMIN_ID")
# AI Keys
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") # For DALL-E only
# Email Keys
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")
# Other API Keys
CLIPDROP_API_KEY = os.environ.get("CLIPDROP_API_KEY")
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
SCREENSHOT_API_KEY = os.environ.get("SCREENSHOT_API_KEY")
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")

# --- Initial Checks ---
if not all([BOT_TOKEN, DATABASE_URL, ADMIN_ID]):
    logger.critical("Critical environment variables are missing. Exiting.")
    sys.exit(1)
else:
    ADMIN_ID = int(ADMIN_ID)

# --- API Configurations ---
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Gemini AI client configured.")
    else: gemini_model = None; logger.warning("GEMINI_API_KEY not found.")
except Exception as e: gemini_model = None; logger.error(f"Failed to configure Gemini API: {e}")

try:
    if OPENAI_API_KEY:
        openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client configured for DALL-E.")
    else: openai_client = None; logger.warning("OPENAI_API_KEY not found.")
except Exception as e: openai_client = None; logger.error(f"Failed to configure OpenAI API: {e}")


# --- Constants & Database Setup ---
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
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
except OperationalError as e:
    logger.critical(f"Failed to connect to database: {e}. Exiting.")
    sys.exit(1)
Session = sessionmaker(bind=engine)

# --- User & Notification Functions ---
async def save_user_to_db(update: Update, context: ContextTypes.DEFAULT_TYPE, event_type: str = "User Interacted"):
    if not hasattr(update, 'effective_user') or not update.effective_user: return
    user, chat_id = update.effective_user, update.effective_chat.id
    session = Session()
    try:
        if not session.query(User).filter_by(user_id=user.id, chat_id=chat_id).first():
            new_user = User(user_id=user.id, first_name=user.first_name, username=user.username, chat_id=chat_id)
            session.add(new_user); session.commit()
            if user.id != ADMIN_ID: await send_notification_to_admin(context, {'user_id': user.id, 'first_name': user.first_name, 'username': user.username}, "New User Added")
        elif user.id != ADMIN_ID: await send_notification_to_admin(context, {'user_id': user.id, 'first_name': user.first_name, 'username': user.username}, event_type)
    except IntegrityError: session.rollback()
    except Exception as e: logger.error(f"DB Error for user {user.id}: {e}"); session.rollback()
    finally: session.close()

async def send_notification_to_admin(context: ContextTypes.DEFAULT_TYPE, user_info: dict, event_type: str):
    first_name = user_info.get('first_name', 'N/A')
    username = f"@{user_info.get('username')}" if user_info.get('username') else "Not set"
    message = (f"Interaction: {event_type}\n" f"User: {first_name} (ID: `{user_info.get('user_id')}`)\n" f"Username: {username}")
    try: await context.bot.send_message(chat_id=ADMIN_ID, text=message, parse_mode=ParseMode.MARKDOWN)
    except Exception as e: logger.error(f"Failed to send admin notification: {e}")

# --- Menu Functions ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # MODIFIED: Added Suggestion button
    keyboard = [
        [KeyboardButton("AI Tools"), KeyboardButton("Media Tools")],
        [KeyboardButton("Utilities"), KeyboardButton("Help")],
        [KeyboardButton("Send Suggestion")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("Welcome! How can I help you today?\nSelect an option from the menu below.", reply_markup=reply_markup)

async def show_ai_tools_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [KeyboardButton("Chat with AI"), KeyboardButton("Create Image")],
        [KeyboardButton("Read Text from Image"), KeyboardButton("Animate Image")],
        [KeyboardButton("Upscale Image"), KeyboardButton("Summarize Link")],
        [KeyboardButton("Summarize File"), KeyboardButton("Back to Main Menu")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("AI Tools:", reply_markup=reply_markup)

async def show_media_tools_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [KeyboardButton("Play Music / Video"), KeyboardButton("Download Media")],
        [KeyboardButton("Search Movie")],
        [KeyboardButton("Back to Main Menu")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("Media Tools:", reply_markup=reply_markup)

async def show_utilities_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [KeyboardButton("Weather"), KeyboardButton("Crypto Prices")],
        [KeyboardButton("Translate Text"), KeyboardButton("Tell a Joke")],
        [KeyboardButton("Ask a Riddle"), KeyboardButton("Take Screenshot")]
    ]
    if update.effective_user.id == ADMIN_ID:
        keyboard.append([KeyboardButton("Send Email (Admin)")])
    keyboard.append([KeyboardButton("Convert Video to Audio")])
    keyboard.append([KeyboardButton("Back to Main Menu")])
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("Utilities:", reply_markup=reply_markup)

# --- Feature Functions ---
# ... (All previous functions from help_command to the media download section remain the same)
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "**Bot Commands Guide:**\n\n"
        "You can use the menu buttons or the following commands:\n\n"
        "**/gemini <prompt>**: Ask the AI a question.\n"
        "**/create <prompt>**: Generate an image from text.\n"
        "**/movie <title>**: Get information about a movie.\n"
        "**/play <song name>**: Search and download a song or video.\n"
        "**/mp4**: Reply to a video to convert it to an MP3 audio file.\n"
        "**/gmail**: Send an email (Admin only)."
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

async def start_ai_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await save_user_to_db(update, context, event_type="Started AI Chat")
    context.user_data['state'] = 'continuous_chat'
    context.user_data['gemini_history'] = []
    chat_keyboard = [[KeyboardButton("End Chat")]]
    reply_markup = ReplyKeyboardMarkup(chat_keyboard, resize_keyboard=True)
    await update.message.reply_text("You are now in a continuous chat with the AI.\n\nSend your message, or press 'End Chat' to return to the main menu.", reply_markup=reply_markup)

async def end_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.pop('state', None)
    context.user_data.pop('gemini_history', None)
    await update.message.reply_text("Chat ended. Returning to the main menu.")
    await start(update, context)

async def gemini_command(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt_text: str = None) -> None:
    if not gemini_model:
        await update.message.reply_text("AI service is not configured.")
        return
    is_continuous_chat = context.user_data.get('state') == 'continuous_chat'
    history = context.user_data.get('gemini_history', []) if is_continuous_chat else []
    if not prompt_text:
        prompt_text = " ".join(context.args) if not is_continuous_chat else update.message.text
    if not prompt_text:
        await update.message.reply_text("Please provide a prompt.")
        return
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    try:
        chat_session = gemini_model.start_chat(history=history)
        response = await asyncio.to_thread(chat_session.send_message, prompt_text)
        if is_continuous_chat:
            context.user_data['gemini_history'] = chat_session.history
        safe_reply = escape_markdown(response.text, version=2)
        await update.message.reply_text(safe_reply, parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e:
        logger.error(f"Gemini command error: {e}")
        await update.message.reply_text("Sorry, an error occurred with the AI. The response may contain formatting I can't send.")

async def gmail_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("Sorry, this command is for the admin only.")
        return
    if not os.environ.get("GMAIL_ADDRESS") or not os.environ.get("GMAIL_APP_PASSWORD"):
        await update.message.reply_text("The email service is not configured on the server.")
        return
    context.user_data['state'] = 'awaiting_email_address'
    await update.message.reply_text("Step 1 of 3: Please enter the recipient's email address.")

async def movie_command(update: Update, context: ContextTypes.DEFAULT_TYPE, title: str = None) -> None:
    if not TMDB_API_KEY:
        await update.message.reply_text("Movie search service is not configured.")
        return
    if not title:
        if not context.args:
            await update.message.reply_text("Please provide a movie title. Usage: `/movie The Matrix`")
            return
        title = " ".join(context.args)
    feedback = await update.message.reply_text(f"Searching for '{title}'...")
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie"
        search_params = {"api_key": TMDB_API_KEY, "query": title}
        response = requests.get(search_url, params=search_params, timeout=10)
        response.raise_for_status()
        search_results = response.json()
        if not search_results['results']:
            await feedback.edit_text("Sorry, I couldn't find any movie with that title.")
            return
        movie_id = search_results['results'][0]['id']
        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        details_params = {"api_key": TMDB_API_KEY}
        response = requests.get(details_url, params=details_params, timeout=10)
        response.raise_for_status()
        details = response.json()
        movie_title = details.get('title', 'N/A')
        release_year = details.get('release_date', '----').split('-')[0]
        rating = f"{details.get('vote_average', 0):.1f}/10"
        overview = details.get('overview', 'No summary available.')
        genres = ", ".join([genre['name'] for genre in details.get('genres', [])])
        poster_path = details.get('poster_path')
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
        caption = (
            f"🎬 *{escape_markdown(movie_title, version=2)} ({release_year})*\n\n"
            f"⭐ *Rating:* {rating}\n"
            f"🎭 *Genres:* {genres}\n\n"
            f"_{escape_markdown(overview, version=2)}_"
        )
        if poster_url:
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=poster_url, caption=caption, parse_mode=ParseMode.MARKDOWN_V2)
            await feedback.delete()
        else:
            await feedback.edit_text(caption, parse_mode=ParseMode.MARKDOWN_V2)
    except requests.exceptions.RequestException as e:
        logger.error(f"TMDb API request failed: {e}")
        await feedback.edit_text("An error occurred while contacting the movie database.")
    except Exception as e:
        logger.error(f"Movie command failed: {e}")
        await feedback.edit_text("An unexpected error occurred.")

async def screenshot_command(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str = None) -> None:
    if not SCREENSHOT_API_KEY:
        await update.message.reply_text("Screenshot service is not configured.")
        return
    if not url:
        if not context.args:
            await update.message.reply_text("Please provide a URL. Usage: `/screenshot google.com`")
            return
        url = context.args[0]
    if not url.startswith('http://') and not url.startswith('https://'):
        url = 'http://' + url
    feedback = await update.message.reply_text(f"Capturing screenshot for `{url}`...", parse_mode=ParseMode.MARKDOWN)
    try:
        api_url = "https://shot.screenshotapi.net/screenshot"
        params = {"token": SCREENSHOT_API_KEY, "url": url, "full_page": "false", "fresh": "true", "output": "image", "file_type": "png", "wait_for_event": "load"}
        response = requests.get(api_url, params=params, timeout=45)
        if response.status_code == 200:
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=response.content, caption=f"Screenshot of `{url}`", parse_mode=ParseMode.MARKDOWN)
            await feedback.delete()
        else:
            logger.error(f"Screenshot API failed with status {response.status_code}: {response.text}")
            await feedback.edit_text("Sorry, failed to capture the screenshot. The URL may be invalid or the service is down.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Screenshot request failed: {e}")
        await feedback.edit_text("An error occurred while trying to reach the screenshot service.")
    except Exception as e:
        logger.error(f"Screenshot command failed: {e}")
        await feedback.edit_text("An unexpected error occurred.")

async def read_text_from_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (function is unchanged)
    pass

async def summarize_url(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str) -> None:
    # ... (function is unchanged)
    pass

async def summarize_file_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (function is unchanged)
    pass

async def create_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str = None) -> None:
    # ... (function is unchanged)
    pass

async def upscale_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (function is unchanged)
    pass

async def animate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (function is unchanged)
    pass

async def convert_video_to_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (function is unchanged)
    pass

async def play_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (function is unchanged)
    pass

async def search_and_play_song(update: Update, context: ContextTypes.DEFAULT_TYPE, song_name: str) -> None:
    # ... (function is unchanged)
    pass

async def handle_play_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (function is unchanged)
    pass

async def handle_audio_download(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (function is unchanged)
    pass


# --- MODIFIED: Upgraded Video Download Function ---
async def handle_video_download(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query; await query.answer()
    video_id = query.data.split(":")[1]
    temp_dir = os.path.join(DOWNLOAD_DIR, str(uuid.uuid4())); os.makedirs(temp_dir)
    info = None # Define info to be accessible in the whole scope
    
    try:
        await query.edit_message_text("Downloading video...")
        
        # First, get video info without downloading to check duration
        with yt_dlp.YoutubeDL({'noplaylist': True, 'quiet': True}) as ydl:
            info = ydl.extract_info(video_id, download=False)

        video_opts = {'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', 'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'), 'noplaylist': True, 'quiet': True, 'cookiefile': 'cookies_youtube.txt' if os.path.exists('cookies_youtube.txt') else None}
        with yt_dlp.YoutubeDL(video_opts) as ydl: ydl.download([video_id])
        
        video_path = os.path.join(temp_dir, os.listdir(temp_dir)[0])
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

        if file_size_mb <= 49: # If file is small enough, send directly
            await query.edit_message_text("Sending video...")
            with open(video_path, 'rb') as video_file:
                await context.bot.send_video(chat_id=query.message.chat_id, video=video_file, supports_streaming=True)
            await query.delete_message()
            return

        # --- NEW: Splitting logic for large files ---
        await query.edit_message_text(f"Video is large ({file_size_mb:.2f} MB). Splitting into parts...")
        
        # Calculate number of parts needed, aiming for ~48MB chunks
        num_parts = math.ceil(file_size_mb / 48)
        total_duration = info.get('duration')
        if not total_duration:
            await query.edit_message_text("Cannot split video: duration not found.")
            return
            
        part_duration = math.ceil(total_duration / num_parts)

        split_cmd = [
            'ffmpeg', '-i', video_path, '-c', 'copy', '-map', '0', 
            '-segment_time', str(part_duration), '-f', 'segment', 
            '-reset_timestamps', '1', os.path.join(temp_dir, 'part_%03d.mp4')
        ]
        
        process = await asyncio.create_subprocess_exec(*split_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await process.communicate()

        if process.returncode != 0:
            logger.error(f"FFmpeg split failed for {video_id}.")
            await query.edit_message_text("Sorry, I failed to split the video.")
            return

        # Send the split parts
        parts = sorted([f for f in os.listdir(temp_dir) if f.startswith('part_')])
        for i, part in enumerate(parts):
            part_path = os.path.join(temp_dir, part)
            await query.message.reply_text(f"Sending part {i+1} of {len(parts)}...")
            with open(part_path, 'rb') as part_file:
                await context.bot.send_document(chat_id=query.message.chat_id, document=part_file)
        
        await query.delete_message()

    except Exception as e:
        logger.error(f"Video download/split error: {e}"); await query.edit_message_text("An error occurred during the download.")
    finally:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)


async def handle_play_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query; await query.answer()
    await query.edit_message_text("Search cancelled.")

async def show_download_platform_options(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await save_user_to_db(update, context, event_type="Pressed 'Download Media'")
    keyboard = [[InlineKeyboardButton("TikTok", callback_data="dl_platform:TikTok"), InlineKeyboardButton("Instagram", callback_data="dl_platform:Instagram")], [InlineKeyboardButton("Facebook", callback_data="dl_platform:Facebook"), InlineKeyboardButton("YouTube", callback_data="dl_platform:YouTube")]]
    await update.message.reply_text("Please choose a platform to download from:", reply_markup=InlineKeyboardMarkup(keyboard))

async def handle_platform_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query; await query.answer()
    platform_name = query.data.split(":")[1]
    context.user_data['state'] = 'awaiting_download_url'; context.user_data['platform'] = platform_name
    await query.edit_message_text(text=f"Okay, send me the full URL for the {platform_name} content.")

async def download_content_from_url(update: Update, context: ContextTypes.DEFAULT_TYPE, content_url: str, platform: str) -> None:
    # This function should also ideally have the splitting logic from handle_video_download.
    # For now, it will fail on large videos from these sources.
    feedback = await update.message.reply_text(f"Starting download from {platform}...")
    temp_dir = os.path.join(DOWNLOAD_DIR, str(uuid.uuid4())); os.makedirs(temp_dir, exist_ok=True)
    try:
        ydl_opts = {'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'), 'noplaylist': True, 'quiet': True, 'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', 'http_headers': {'User-Agent': 'Mozilla/5.0'}}
        if platform.lower() == 'youtube' and os.path.exists('cookies_youtube.txt'): ydl_opts['cookiefile'] = 'cookies_youtube.txt'
        elif os.path.exists('cookies.txt'): ydl_opts['cookiefile'] = 'cookies.txt'
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([content_url])
        downloaded_file_path = os.path.join(temp_dir, os.listdir(temp_dir)[0])
        await feedback.edit_text("Uploading to Telegram...")
        with open(downloaded_file_path, 'rb') as f:
            await context.bot.send_video(chat_id=update.effective_chat.id, video=f, supports_streaming=True)
        await feedback.delete()
    except Exception as e:
        logger.error(f"Download error for {content_url}: {e}")
        await feedback.edit_text("Download failed. The link may be private or invalid.")
    finally:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

async def get_joke(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        response = requests.get("https://v2.jokeapi.dev/joke/Any?blacklistFlags=nsfw,racist,sexist&type=twopart", timeout=5).json()
        if not response['error']: await update.message.reply_text(f"{response['setup']}\n\n...{response['delivery']}")
    except Exception: await update.message.reply_text("Sorry, couldn't get a joke.")

async def get_riddle(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        response = requests.get("https://riddles-api.vercel.app/random", timeout=7).json()
        riddle, answer = response.get('riddle'), response.get('answer')
        if not riddle or not answer: await update.message.reply_text("Sorry, I couldn't think of a riddle right now."); return
        await update.message.reply_text(f"Here is your riddle:\n\n*_{riddle}_*", parse_mode=ParseMode.MARKDOWN)
        await asyncio.sleep(8) 
        await update.message.reply_text(f"**Answer:**\n||{escape_markdown(answer, version=2)}||", parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e:
        logger.error(f"Riddle API error: {e}")
        await update.message.reply_text("Sorry, the riddle service is currently unavailable.")

async def get_crypto_prices(update: Update, context: ContextTypes.DEFAULT_TYPE, crypto_ids: str) -> None:
    ids = ','.join(s.strip().lower() for s in crypto_ids.split(','))
    try:
        prices = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd,ngn", timeout=5).json()
        if not prices: await update.message.reply_text("Couldn't find prices."); return
        message = "**Latest Crypto Prices:**\n" + "\n".join(f"• **{c.title()}**: ${d.get('usd', 0):,.2f} / ₦{d.get('ngn', 0):,.2f}" for c, d in prices.items())
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
    except Exception: await update.message.reply_text("Sorry, couldn't fetch crypto prices.")

async def get_weather(update: Update, context: ContextTypes.DEFAULT_TYPE, city: str) -> None:
    if not OPENWEATHER_API_KEY: await update.message.reply_text("Weather service not configured."); return
    try:
        data = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric", timeout=5).json()
        if data["cod"] != 200: await update.message.reply_text(f"Sorry, couldn't find city '{city}'."); return
        message = f"**Weather in {data['name']}**: {data['weather'][0]['description'].title()} at {data['main']['temp']}°C"
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
    except Exception: await update.message.reply_text("Sorry, couldn't fetch the weather.")

async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE, text_to_translate: str = None) -> None:
    if not gemini_model: await update.message.reply_text("Translate service is not configured."); return
    if not text_to_translate: text_to_translate = " ".join(context.args)
    parts = text_to_translate.split(" ", 1)
    if len(parts) < 2: await update.message.reply_text("Format: `<language> <text>`"); return
    target_lang, text = parts
    prompt = f"Translate the following text to {target_lang}: {text}"
    response = await asyncio.to_thread(gemini_model.generate_content, prompt)
    await update.message.reply_text(response.text)

# --- Message Routing ---
async def prompt_for_input(update: Update, context: ContextTypes.DEFAULT_TYPE, state: str, message: str, event: str) -> None:
    await save_user_to_db(update, context, event_type=event)
    context.user_data['state'] = state
    await update.message.reply_text(message)

async def record_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text: return
    state = context.user_data.get('state')
    
    if state == 'continuous_chat': 
        await gemini_command(update, context)
        return

    if state == 'awaiting_email_address':
        context.user_data['email_to'] = update.message.text
        context.user_data['state'] = 'awaiting_email_subject'
        await update.message.reply_text("Step 2 of 3: Great. Now, what should the subject be?")
        return
    
    if state == 'awaiting_email_subject':
        context.user_data['email_subject'] = update.message.text
        context.user_data['state'] = 'awaiting_email_body'
        await update.message.reply_text("Step 3 of 3: Perfect. Please enter the message body.")
        return
        
    if state == 'awaiting_email_body':
        to_address, subject, body = context.user_data.pop('email_to'), context.user_data.pop('email_subject'), update.message.text
        context.user_data.pop('state', None)
        feedback = await update.message.reply_text(f"Sending email to {to_address}...")
        msg = EmailMessage(); msg.set_content(body); msg['Subject'] = subject; msg['From'] = os.environ.get("GMAIL_ADDRESS"); msg['To'] = to_address
        def send_email_sync():
            try:
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
                    s.login(os.environ.get("GMAIL_ADDRESS"), os.environ.get("GMAIL_APP_PASSWORD"))
                    s.send_message(msg)
                return True
            except Exception as e: logger.error(f"Failed to send email: {e}"); return False
        success = await asyncio.to_thread(send_email_sync)
        if success: await feedback.edit_text("Email sent successfully!")
        else: await feedback.edit_text("Failed to send email. Check server logs.")
        return

    if state == 'awaiting_suggestion':
        user = update.effective_user
        suggestion_text = update.message.text
        context.user_data.pop('state', None) # Clean up state
        
        # Format the suggestion to send to the admin
        admin_message = (
            f"📩 *New Suggestion Received*\n\n"
            f"*From*: {user.first_name} (`{user.id}`)\n"
            f"*Username*: @{user.username}\n\n"
            f"*Suggestion*:\n{suggestion_text}"
        )
        await context.bot.send_message(chat_id=ADMIN_ID, text=admin_message, parse_mode=ParseMode.MARKDOWN)
        await update.message.reply_text("Thank you! Your suggestion has been sent to the admin.")
        return

    state = context.user_data.pop('state', None)
    if not state: await save_user_to_db(update, context, "Sent a message"); return
    text = update.message.text
    state_handlers = {
        'awaiting_gemini_prompt': lambda: gemini_command(update, context, prompt_text=text),
        'awaiting_create_prompt': lambda: create_image_command(update, context, prompt=text),
        'awaiting_song_name': lambda: search_and_play_song(update, context, song_name=text),
        'awaiting_city': lambda: get_weather(update, context, city=text),
        'awaiting_crypto_symbols': lambda: get_crypto_prices(update, context, crypto_ids=text),
        'awaiting_summary_url': lambda: summarize_url(update, context, url=text),
        'awaiting_translation_text': lambda: translate_command(update, context, text_to_translate=text),
        'awaiting_download_url': lambda: download_content_from_url(update, context, content_url=text, platform=context.user_data.pop('platform', 'unknown')),
        'awaiting_screenshot_url': lambda: screenshot_command(update, context, url=text),
        'awaiting_movie_title': lambda: movie_command(update, context, title=text),
    }
    if state in state_handlers: await state_handlers[state]()

# --- Main Application Setup ---
def main() -> None:
    application = Application.builder().token(BOT_TOKEN).rate_limiter(AIORateLimiter()).build()
    
    cmd_handlers = [
        CommandHandler("start", start), CommandHandler("help", help_command),
        CommandHandler("gemini", gemini_command), CommandHandler("create", create_image_command),
        CommandHandler("upscale", upscale_image_command), CommandHandler("animate", animate_command),
        CommandHandler("summarize_file", summarize_file_command), CommandHandler("readtext", read_text_from_image_command),
        CommandHandler("play", play_command), CommandHandler("mp4", convert_video_to_audio),
        CommandHandler("riddle", get_riddle), CommandHandler("gmail", gmail_command),
        CommandHandler("screenshot", screenshot_command), CommandHandler("movie", movie_command),
    ]
    
    menu_button_texts = {
        "AI Tools": show_ai_tools_menu, "Media Tools": show_media_tools_menu,
        "Utilities": show_utilities_menu, "Help": help_command,
        "Back to Main Menu": start, "End Chat": end_chat,
        "Chat with AI": start_ai_chat, "Tell a Joke": get_joke,
        "Ask a Riddle": get_riddle,
        "Send Suggestion": lambda u,c: prompt_for_input(u,c,'awaiting_suggestion', "Please type your suggestion or feedback. I will forward it to the admin.","Pressed 'Suggestion'"),
        "Create Image": lambda u,c: prompt_for_input(u,c,'awaiting_create_prompt', "Describe the image to create.","Pressed 'Create Image'"),
        "Read Text from Image": lambda u,c: u.message.reply_text("Please reply to an image with /readtext to use this feature."),
        "Upscale Image": lambda u,c: u.message.reply_text("Please reply to an image with /upscale."),
        "Animate Image": lambda u,c: u.message.reply_text("Please reply to an image with /animate."),
        "Summarize File": lambda u,c: u.message.reply_text("Please reply to an image or PDF with /summarize_file."),
        "Summarize Link": lambda u,c: prompt_for_input(u,c,'awaiting_summary_url', "Please send the article link.","Pressed 'Summarize Link'"),
        "Play Music / Video": lambda u,c: prompt_for_input(u,c,'awaiting_song_name', "What song or video would you like?","Pressed 'Play'"),
        "Download Media": show_download_platform_options,
        "Search Movie": lambda u,c: prompt_for_input(u,c,'awaiting_movie_title', "What movie are you looking for?","Pressed 'Search Movie'"),
        "Weather": lambda u,c: prompt_for_input(u,c,'awaiting_city', "Please enter a city name.","Pressed 'Weather'"),
        "Crypto Prices": lambda u,c: prompt_for_input(u,c,'awaiting_crypto_symbols', "Enter coin IDs separated by commas (e.g., bitcoin,ethereum).","Pressed 'Crypto'"),
        "Translate Text": lambda u,c: prompt_for_input(u,c,'awaiting_translation_text', "Enter text to translate in the format: <language> <text>","Pressed 'Translate'"),
        "Convert Video to Audio": lambda u,c: u.message.reply_text("To use this feature, please reply to a video with the /mp4 command."),
        "Take Screenshot": lambda u,c: prompt_for_input(u,c,'awaiting_screenshot_url', "Please enter the website URL to capture.","Pressed 'Take Screenshot'"),
        "Send Email (Admin)": gmail_command,
    }
    
    escaped_patterns = [re.escape(p) for p in menu_button_texts.keys()]
    msg_handlers = [MessageHandler(filters.TEXT & filters.Regex(f"^{re.escape(pattern)}$"), func) for pattern, func in menu_button_texts.items()]
    
    callback_handlers = [
        CallbackQueryHandler(handle_play_confirmation, pattern="^play_confirm:"),
        CallbackQueryHandler(handle_play_cancel, pattern="^play_cancel"),
        CallbackQueryHandler(handle_audio_download, pattern="^dl_audio:"),
        CallbackQueryHandler(handle_video_download, pattern="^dl_video:"),
        CallbackQueryHandler(handle_platform_selection, pattern="^dl_platform:"),
    ]

    application.add_handlers(cmd_handlers + msg_handlers + callback_handlers)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.Regex(f"^({'|'.join(escaped_patterns)})$"), record_user_message))
    
    PORT = int(os.environ.get("PORT", 8443))
    RENDER_APP_NAME = os.environ.get("RENDER_APP_NAME")
    if not RENDER_APP_NAME:
        logger.info("Running in polling mode.")
        application.run_polling()
    else:
        WEBHOOK_URL = f"https://{RENDER_APP_NAME}.onrender.com/{BOT_TOKEN}"
        logger.info(f"Running in webhook mode. URL: {WEBHOOK_URL}")
        application.run_webhook(listen="0.0.0.0", port=PORT, url_path=BOT_TOKEN, webhook_url=WEBHOOK_URL)

if __name__ == "__main__":
    main()
