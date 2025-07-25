import logging
import os
import sys
import asyncio
import uuid
import shutil
import datetime
import time

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
# As requested, variables are hardcoded.
# WARNING: This is NOT recommended for security. Use environment variables in production.
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = "7806461656:AAEFsYhfk7moHzZgqX80qboJfb4b58UhsgU"
DATABASE_URL = "YOUR_DATABASE_URL" # ⚠️ Replace with your actual Render Postgres URL
ADMIN_ID = 7302005705

# --- Gemini API Configuration ---
# ⚠️ WARNING: Do NOT hardcode your API key. Use environment variables.
# I have replaced your key with a placeholder for your safety.
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY" # ⚠️ Replace with your new, secret Gemini API key
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
    logger.info("Gemini API configured successfully.")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {e}")
    gemini_model = None

# --- Bot Constants ---
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
TELEGRAM_VIDEO_LIMIT_BYTES = 50 * 1024 * 1024
ADMIN_NOTIFICATION_COOLDOWN = 300
last_admin_notification_time = {}

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
    logger.critical(f"Failed to connect to database or create tables: {e}. Exiting.")
    sys.exit(1)
Session = sessionmaker(bind=engine)

# --- Helper, Database & Notification Functions (largely unchanged) ---
def escape_markdown_v2(text: str) -> str:
    if not isinstance(text, str): return ""
    special_chars = r'_*[]()~`>#+-=|{}.!'
    for char in special_chars:
        text = text.replace(char, f'\\{char}')
    return text

async def save_user_to_db(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # This function's logic remains the same as before.
    # It saves the user to the database upon interaction.
    pass # The full code for this function is lengthy and unchanged.

# --- NEW Gemini Command ---
async def gemini_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /gemini command to interact with Google's Gemini AI."""
    if not gemini_model:
        await update.message.reply_text("Sorry, the AI service is not configured correctly.")
        return

    prompt = " ".join(context.args)
    if not prompt:
        await update.message.reply_text("Please provide a prompt after the `/gemini` command.\n\nExample: `/gemini What is the capital of Nigeria?`")
        return

    thinking_message = await update.message.reply_text("🤔 Thinking...")
    try:
        response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        await thinking_message.edit_text(response.text)
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        await thinking_message.edit_text("Sorry, I encountered an error while contacting the AI. Please try again later.")

# --- NEW Play Music Command & Flow ---
async def play_music_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Asks the user for a song name."""
    context.user_data['state'] = 'awaiting_song_name'
    await update.message.reply_text("🎵 Okay, what song would you like to listen to?")

async def search_and_play_song(update: Update, context: ContextTypes.DEFAULT_TYPE, song_name: str) -> None:
    """Searches for a song, downloads it, and sends it as an audio file."""
    feedback_message = await update.message.reply_text(f" searching for '{song_name}'...")
    
    temp_dir = os.path.join(DOWNLOAD_DIR, str(uuid.uuid4()))
    os.makedirs(temp_dir)

    try:
        # 1. Search for the video URL
        search_opts = {'default_search': 'ytsearch', 'noplaylist': True, 'quiet': True}
        with yt_dlp.YoutubeDL(search_opts) as ydl:
            info = ydl.extract_info(song_name, download=False)
            if not info.get('entries'):
                await feedback_message.edit_text("Sorry, I couldn't find any results for that song.")
                return
            video_info = info['entries'][0]
            video_url = video_info['webpage_url']
            title = video_info.get('title', 'Unknown Title')
            duration = video_info.get('duration', 0)
        
        # Limit duration to 10 minutes (600 seconds) to prevent abuse
        if duration > 600:
            await feedback_message.edit_text("Sorry, the requested song is too long. Please choose a song under 10 minutes.")
            return

        # 2. Download the audio from the URL
        await feedback_message.edit_text(f"⬇️ Downloading '{title}'...")
        audio_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
            'noplaylist': True, 'quiet': True
        }
        with yt_dlp.YoutubeDL(audio_opts) as ydl:
            ydl.download([video_url])
        
        downloaded_files = os.listdir(temp_dir)
        if not downloaded_files:
            raise FileNotFoundError("Audio file not found after download.")
        
        audio_path = os.path.join(temp_dir, downloaded_files[0])

        # 3. Send the audio file
        await feedback_message.edit_text(f"⬆️ Sending '{title}'...")
        with open(audio_path, 'rb') as audio_file:
            await context.bot.send_audio(
                chat_id=update.effective_chat.id,
                audio=audio_file,
                title=title,
                duration=duration,
                filename=os.path.basename(audio_path)
            )
        await feedback_message.delete()

    except Exception as e:
        logger.error(f"Error in play command for '{song_name}': {e}")
        await feedback_message.edit_text("An error occurred while trying to get your song. Please try again.")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# --- Existing Commands (Start, Help, Download, etc.) ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /start command and shows the main menu."""
    await save_user_to_db(update, context)
    keyboard = [
        [KeyboardButton("Download Videos/Audio"), KeyboardButton("Play Music")], # <-- UPDATED
        [KeyboardButton("Help")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text(
        "Hi! I can download media, play music, or chat with an AI. Use the buttons below or type `/gemini` to start.",
        reply_markup=reply_markup
    )

# The other command functions (help_command, tag_all, download_content_from_url, etc.)
# remain the same as the previous full code. I'm omitting them here for brevity
# but they should be in your final script.

async def download_content_from_url(update: Update, context: ContextTypes.DEFAULT_TYPE, platform: str, content_url: str):
    # This function's logic remains the same as before.
    # It handles downloading from TikTok, Instagram, etc.
    pass # The full code for this is lengthy and unchanged.

# --- Message Handlers ---
async def record_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles general text messages, routing them based on user state."""
    if not update.message or not update.message.text: return
    await save_user_to_db(update, context)

    state = context.user_data.get('state')
    
    # Route to download flow
    if state == 'awaiting_url' and context.user_data.get('platform'):
        platform = context.user_data.pop('platform')
        context.user_data.pop('state')
        await download_content_from_url(update, context, platform, update.message.text)
    
    # NEW: Route to play music flow
    elif state == 'awaiting_song_name':
        context.user_data.pop('state')
        await search_and_play_song(update, context, update.message.text)

# --- Main Bot Logic ---
def main() -> None:
    """Start the bot."""
    application = Application.builder().token(BOT_TOKEN).build()

    # --- Register Handlers ---
    # Commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("gemini", gemini_command)) # <-- NEW
    # Add other command handlers like help, tag, etc. here

    # Keyboard buttons (from /start menu)
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Download Videos/Audio$"), show_download_options))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Play Music$"), play_music_prompt)) # <-- NEW
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Help$"), help_command))

    # Inline buttons (from download flow)
    application.add_handler(CallbackQueryHandler(handle_download_platform_selection, pattern="^dl:"))
    
    # General message handler (must be last to catch state-based replies)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, record_user_message))
    
    # --- Webhook setup for Render ---
    PORT = int(os.environ.get("PORT", 8443))
    RENDER_APP_NAME = os.environ.get("RENDER_APP_NAME")

    if not RENDER_APP_NAME:
        logger.warning("RENDER_APP_NAME env var not found. Running in polling mode for local dev.")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    else:
        WEBHOOK_URL = f"https://{RENDER_APP_NAME}.onrender.com/{BOT_TOKEN}"
        logger.info(f"Starting bot in webhook mode on port {PORT}")
        application.run_webhook(listen="0.0.0.0", port=PORT, webhook_url=WEBHOOK_URL)

if __name__ == "__main__":
    main()
