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
import openai # For DALL-E 3
import pytesseract # For OCR
from PIL import Image # For OCR
import io # For OCR
import azapi # For fetching lyrics
from googlesearch import search # For Browse google

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler, AIORateLimiter
from telegram.constants import ParseMode, ChatAction
from telegram.helpers import escape_markdown

from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError, IntegrityError
from sqlalchemy.types import BigInteger
import yt_dlp

# --- Configuration ---
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Environment Variables & API Configurations ---
BOT_TOKEN = os.environ.get("BOT_TOKEN")
DATABASE_URL = os.environ.get("DATABASE_URL")
ADMIN_ID = os.environ.get("ADMIN_ID")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
CLIPDROP_API_KEY = os.environ.get("CLIPDROP_API_KEY")
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY")
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not all([BOT_TOKEN, DATABASE_URL, ADMIN_ID]):
    logger.critical("Critical environment variables are missing. Exiting."); sys.exit(1)
else:
    ADMIN_ID = int(ADMIN_ID)

try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction="You are a helpful and friendly assistant.")
        logger.info("Gemini AI client configured.")
    else: gemini_model = None; logger.warning("GEMINI_API_KEY not found.")
except Exception as e: gemini_model = None; logger.error(f"Failed to configure Gemini API: {e}")

try:
    if OPENAI_API_KEY:
        openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client configured for DALL-E 3.")
    else: openai_client = None; logger.warning("OPENAI_API_KEY not found.")
except Exception as e: openai_client = None; logger.error(f"Failed to configure OpenAI API: {e}")

# --- Constants & Database Setup ---
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

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

# --- User & Notification Functions ---
async def save_user_to_db(update: Update, context: ContextTypes.DEFAULT_TYPE, event_type: str = "User Interacted"):
    if not hasattr(update, 'effective_user') or not update.effective_user: return
    user = update.effective_user; chat_id = update.effective_chat.id
    session = Session()
    try:
        if not session.query(User).filter_by(user_id=user.id, chat_id=chat_id).first():
            new_user = User(user_id=user.id, first_name=user.first_name, username=user.username, chat_id=chat_id)
            session.add(new_user); session.commit()
            if user.id != ADMIN_ID: await send_notification_to_admin(context, {'user_id': user.id, 'first_name': user.first_name, 'username': user.username}, "New User Added")
        else:
             if user.id != ADMIN_ID: await send_notification_to_admin(context, {'user_id': user.id, 'first_name': user.first_name, 'username': user.username}, event_type)
    except IntegrityError: session.rollback()
    except Exception as e: logger.error(f"DB Error for user {user.id}: {e}"); session.rollback()
    finally: session.close()

async def send_notification_to_admin(context: ContextTypes.DEFAULT_TYPE, user_info: dict, event_type: str):
    first_name = user_info.get('first_name', 'N/A')
    username = f"@{user_info.get('username')}" if user_info.get('username') else "Not set"
    message = (f"Interaction: {event_type}\n" f"User: {first_name} (ID: `{user_info.get('user_id')}`)\n" f"Username: {username}")
    try: await context.bot.send_message(chat_id=ADMIN_ID, text=escape_markdown(message, version=2), parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e: logger.error(f"Failed to send admin notification: {e}")

# --- Menu Functions ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [[KeyboardButton("AI Tools"), KeyboardButton("Media Tools")], [KeyboardButton("Utilities"), KeyboardButton("Help")]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("Main Menu:", reply_markup=reply_markup)

async def show_ai_tools_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [[KeyboardButton("Chat with AI"), KeyboardButton("Create Image")], [KeyboardButton("Read Text from Image"), KeyboardButton("Animate Image")], [KeyboardButton("Upscale Image"), KeyboardButton("Summarize Link")], [KeyboardButton("Summarize File"), KeyboardButton("Back to Main Menu")]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True); await update.message.reply_text("AI Tools:", reply_markup=reply_markup)

async def show_media_tools_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [[KeyboardButton("Play Music / Video"), KeyboardButton("Download Media")], [KeyboardButton("Back to Main Menu")]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True); await update.message.reply_text("Media Tools:", reply_markup=reply_markup)

async def show_utilities_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [[KeyboardButton("Weather"), KeyboardButton("Crypto Prices")], [KeyboardButton("Translate Text"), KeyboardButton("Tell a Joke")], [KeyboardButton("Back to Main Menu")]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True); await update.message.reply_text("Utilities:", reply_markup=reply_markup)

# --- Feature Functions ---
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = ("*Bot Commands Guide:*\n\n- *Chat with AI*: Talk to an AI with conversation memory\.\n- `/readtext`: Reply to an image to read text from it\.\n- *Create Image*: Generate an image from text\.\n- `/animate`: Reply to an image to create a video\.\n- `/upscale`: Reply to an image to improve quality\.\n- And many more in the menu\!")
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN_V2)

async def start_ai_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await save_user_to_db(update, context, event_type="Started AI Chat")
    context.user_data['state'] = 'continuous_chat'
    context.user_data['gemini_history'] = []
    chat_keyboard = [[KeyboardButton("End Chat")]]; reply_markup = ReplyKeyboardMarkup(chat_keyboard, resize_keyboard=True)
    await update.message.reply_text("You are now in a continuous chat with the AI.\n\nSend your message, or press 'End Chat' to return to the main menu.", reply_markup=reply_markup)

async def end_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.pop('state', None); context.user_data.pop('gemini_history', None)
    await update.message.reply_text("Chat ended. Returning to the main menu."); await start(update, context)

async def gemini_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not gemini_model: await update.message.reply_text("AI service is not configured."); return
    is_continuous_chat = context.user_data.get('state') == 'continuous_chat'
    history = context.user_data.get('gemini_history', []) if is_continuous_chat else []
    prompt = update.message.text if is_continuous_chat else " ".join(context.args)
    if not prompt and not is_continuous_chat:
        await update.message.reply_text("Please provide a prompt after the command, e.g., `/gemini Who are you?`"); return
    image_parts = []
    if update.message.reply_to_message and update.message.reply_to_message.photo:
        photo_file = await update.message.reply_to_message.photo[-1].get_file()
        image_parts.append({"mime_type": "image/jpeg", "data": await photo_file.download_as_bytearray()})
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    try:
        chat_session = gemini_model.start_chat(history=history)
        response = await asyncio.to_thread(chat_session.send_message, [prompt] + image_parts)
        if is_continuous_chat: context.user_data['gemini_history'] = chat_session.history
        escaped_text = escape_markdown(response.text, version=2)
        await update.message.reply_text(escaped_text, parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e: logger.error(f"Gemini chat error: {e}"); await update.message.reply_text("Sorry, an error occurred with the AI.")

async def read_text_from_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message.reply_to_message or not update.message.reply_to_message.photo:
        await update.message.reply_text("Please reply to an image with the /readtext command."); return
    await save_user_to_db(update, context, event_type="Used OCR")
    feedback = await update.message.reply_text("Reading text from image...")
    try:
        photo_file = await update.message.reply_to_message.photo[-1].get_file()
        image = Image.open(io.BytesIO(await photo_file.download_as_bytearray()))
        text = await asyncio.to_thread(pytesseract.image_to_string, image)
        if text and not text.isspace():
            await feedback.edit_text(f"*Extracted Text:*\n\n{escape_markdown(text, version=2)}", parse_mode=ParseMode.MARKDOWN_V2)
        else:
            await feedback.edit_text("Couldn't find any readable text in the image.")
    except Exception as e:
        logger.error(f"OCR Error: {e}")
        if "Tesseract is not installed" in str(e) or "command not found" in str(e).lower():
             await feedback.edit_text("OCR processing failed. The Tesseract engine is not installed on the server.")
        else:
            await feedback.edit_text("Sorry, an error occurred while processing the image.")

async def summarize_url(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str) -> None:
    if not gemini_model: await update.message.reply_text("AI summarizer is not configured."); return
    feedback = await update.message.reply_text("Analyzing link...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}; head_response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        content_type = head_response.headers.get('content-type', '').lower()
        if 'image' in content_type:
            await feedback.edit_text("Image link detected. Analyzing image...")
            image_response = requests.get(url, headers=headers, timeout=20); image_response.raise_for_status()
            image_part = {"mime_type": content_type, "data": image_response.content}
            prompt = "You are a visual analyst. Describe this image in great detail."
            ai_response = await asyncio.to_thread(gemini_model.generate_content, [prompt, image_part])
            summary = f"*Image Analysis:*\n{ai_response.text}"
        elif 'text/html' in content_type:
            await feedback.edit_text("Article link detected. Reading content...")
            response = requests.get(url, headers=headers, timeout=20); response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            article_text = ' '.join(p.get_text() for p in soup.find_all('p'))
            if len(article_text) < 100: await feedback.edit_text("Couldn't extract enough text to summarize."); return
            await feedback.edit_text("Content extracted. Summarizing with AI...")
            prompt = f"You are an expert analyst. Provide a detailed, structured summary of the following text, using MarkdownV2:\n\n*Key Takeaways:*\n- (3-5 bullet points)\n\n*Detailed Summary:*\n- (A comprehensive paragraph)\n\n*Critical Analysis/Context:*\n- (Deeper insights)\n\n--- TEXT ---\n{article_text[:12000]}"
            ai_response = await asyncio.to_thread(gemini_model.generate_content, prompt); summary = ai_response.text
        else: await feedback.edit_text(f"Unsupported link type: '{content_type}'."); return
        await feedback.edit_text(escape_markdown(summary, version=2), parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e: logger.error(f"Summarize error for URL {url}: {e}"); await feedback.edit_text("Sorry, I couldn't read or summarize that URL.")

async def summarize_file_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not gemini_model: await update.message.reply_text("AI service is not configured."); return
    if not update.message.reply_to_message: await update.message.reply_text("Please reply to an image or a PDF file with /summarize_file."); return
    replied_message, feedback_message = update.message.reply_to_message, await update.message.reply_to_message.reply_text("Processing file...")
    try:
        summary = ""
        if replied_message.photo:
            photo_file = await replied_message.photo[-1].get_file()
            image_part = {"mime_type": "image/jpeg", "data": await photo_file.download_as_bytearray()}
            prompt = "You are a visual analyst. Describe this image in great detail."
            response = await asyncio.to_thread(gemini_model.generate_content, [prompt, image_part])
            summary = f"*Image Analysis:*\n{response.text}"
        elif replied_message.document and replied_message.document.mime_type == 'application/pdf':
            pdf_file = await replied_message.document.get_file()
            with fitz.open(stream=await pdf_file.download_as_bytearray(), filetype="pdf") as doc: full_text = "".join(page.get_text() for page in doc)
            if not full_text.strip(): await feedback_message.edit_text("Could not extract any text from this PDF."); return
            await feedback_message.edit_text("Extracted text. Summarizing...")
            prompt = f"You are an expert analyst. Provide a detailed, structured summary of the following document, using MarkdownV2:\n\n*Key Takeaways:*\n- (3-5 bullet points)\n\n*Detailed Summary:*\n- (A comprehensive paragraph)\n\n*Critical Analysis/Context:*\n- (Deeper insights)\n\n--- DOCUMENT ---\n{full_text[:12000]}"
            response = await asyncio.to_thread(gemini_model.generate_content, prompt); summary = response.text
        else: await feedback_message.edit_text("This command only works on an image or PDF file."); return
        await feedback_message.edit_text(escape_markdown(summary, version=2), parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e: logger.error(f"File summarization error: {e}"); await feedback_message.edit_text("Sorry, an error occurred.")

async def generate_image(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str) -> None:
    if not openai_client and not STABILITY_API_KEY: await update.message.reply_text("No image generation services are configured."); return
    feedback = await update.message.reply_text("Accessing image generation services...")
    if openai_client:
        try:
            await feedback.edit_text("Attempting to create image with DALL-E 3...")
            response = await openai_client.images.generate(model="dall-e-3", prompt=prompt, n=1, size="1024x1024", quality="standard")
            await context.bot.send_photo(update.effective_chat.id, photo=response.data[0].url, caption=f"Created with DALL-E 3: `{escape_markdown(prompt, version=2)}`", parse_mode=ParseMode.MARKDOWN_V2)
            await feedback.delete(); return
        except Exception as e: logger.error(f"DALL-E 3 API error: {e}"); await feedback.edit_text("DALL-E 3 failed. Trying backup service..."); await asyncio.sleep(2)
    if STABILITY_API_KEY:
        try:
            await feedback.edit_text("Attempting to create image with Stability AI...")
            response = requests.post("https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image", headers={"Authorization": f"Bearer {STABILITY_API_KEY}", "Accept": "application/json"}, json={"text_prompts": [{"text": prompt}]}, timeout=90)
            response.raise_for_status()
            await context.bot.send_photo(update.effective_chat.id, photo=base64.b64decode(response.json()["artifacts"][0]["base64"]), caption=f"Created with Stability AI: `{escape_markdown(prompt, version=2)}`", parse_mode=ParseMode.MARKDOWN_V2)
            await feedback.delete(); return
        except Exception as e: logger.error(f"Stability AI error: {e}"); await feedback.edit_text("Sorry, the backup image service also failed."); return
    await feedback.edit_text("Sorry, I couldn't create the image with any available service.")

async def upscale_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not CLIPDROP_API_KEY: await update.message.reply_text("Image upscaling service not configured."); return
    if not update.message.reply_to_message or not update.message.reply_to_message.photo: await update.message.reply_text("Please reply to an image with /upscale."); return
    feedback = await update.message.reply_text("Upscaling your image...")
    try:
        photo_file = await update.message.reply_to_message.photo[-1].get_file()
        response = requests.post('https://clipdrop-api.co/image-upscaling/v1/upscale', files={'image_file': await photo_file.download_as_bytearray()}, headers={'x-api-key': CLIPDROP_API_KEY}, timeout=90)
        response.raise_for_status()
        await context.bot.send_document(update.effective_chat.id, response.content, filename='upscaled.png', caption='Here is your upscaled image!')
        await feedback.delete()
    except Exception as e: logger.error(f"ClipDrop API Error: {e}"); await feedback.edit_text("Sorry, an error occurred while upscaling.")

async def animate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not STABILITY_API_KEY: await update.message.reply_text("Video animation service not configured."); return
    if not update.message.reply_to_message or not update.message.reply_to_message.photo: await update.message.reply_text("Please reply to an image with /animate."); return
    feedback = await update.message.reply_text("Sending image to animation engine...")
    try:
        photo_file = await update.message.reply_to_message.photo[-1].get_file()
        api_url = "https://api.stability.ai/v2beta/image-to-video"
        response = requests.post(api_url, headers={"authorization": f"Bearer {STABILITY_API_KEY}"}, files={"image": await photo_file.download_as_bytearray()}, data={"motion_bucket_id": 40}, timeout=30)
        response.raise_for_status(); generation_id = response.json()["id"]
        await feedback.edit_text("Animation started. This may take a minute...")
        result_url = f"{api_url}/result/{generation_id}"
        for _ in range(45):
            await asyncio.sleep(4)
            res = requests.get(result_url, headers={'authorization': f"Bearer {STABILITY_API_KEY}", 'accept': "video/mp4"}, timeout=20)
            if res.status_code == 200:
                await context.bot.send_video(update.effective_chat.id, res.content, caption="Here is your animated video!"); await feedback.delete(); return
            elif res.status_code != 202: logger.error(f"Animation failed with status {res.status_code}: {res.text}"); break
        await feedback.edit_text("Sorry, the animation timed out or failed.")
    except Exception as e: logger.error(f"Animate command error: {e}"); await feedback.edit_text(f"Sorry, an error occurred: {e}")

async def play_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    song_name = " ".join(context.args)
    if not song_name:
        await update.message.reply_text("Please provide a song name after the command, like this: /play Burna Boy - Last Last")
        return
    await search_and_play_song(update, context, song_name)

async def search_and_play_song(update: Update, context: ContextTypes.DEFAULT_TYPE, song_name: str) -> None:
    message_to_reply = update.message or update.callback_query.message
    feedback = await message_to_reply.reply_text(f"Searching for '{song_name}'...")
    try:
        ydl_opts = {'noplaylist': True, 'quiet': True, 'default_search': 'ytsearch1', 'cookiefile': 'cookies_youtube.txt' if os.path.exists('cookies_youtube.txt') else None}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: info = ydl.extract_info(song_name, download=False)
        if not info.get('entries'): await feedback.edit_text("Sorry, couldn't find any results."); return
        video = info['entries'][0]
        title = video.get('title', 'Unknown Title'); video_id = video.get('id')
        message_text = f"Found: '{title}'.\n\nChoose a format:"
        escaped_text = escape_markdown(message_text, version=2)
        keyboard = [[InlineKeyboardButton("Audio", callback_data=f"play_audio:{video_id}"), InlineKeyboardButton("Video", callback_data=f"select_quality:{video_id}")], [InlineKeyboardButton("Cancel", callback_data="play_cancel")]]
        await feedback.edit_text(text=escaped_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e: logger.error(f"Play search error: {e}"); await feedback.edit_text("An error occurred while searching. YouTube may be blocking your server. Please make sure `cookies_youtube.txt` is present and valid.")

async def show_quality_options_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query; await query.answer()
    video_id = query.data.split(":")[1]
    await query.edit_message_text("Checking available video qualities...")
    try:
        ydl_opts = {'quiet': True, 'cookiefile': 'cookies_youtube.txt' if os.path.exists('cookies_youtube.txt') else None}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: info = ydl.extract_info(video_id, download=False)
        formats = info.get('formats', []); quality_buttons = []; supported_heights = [1080, 720, 480, 360]; found_heights = set()
        for f in reversed(formats):
            if f.get('height') in supported_heights and f['height'] not in found_heights and f.get('vcodec') != 'none' and f.get('acodec') != 'none' and f.get('ext') == 'mp4':
                height = f['height']; filesize_mb = f.get('filesize') or f.get('filesize_approx')
                size_str = f"({filesize_mb / (1024*1024):.1f} MB)" if filesize_mb else ""
                quality_buttons.append([InlineKeyboardButton(f"{height}p {size_str}", callback_data=f"play_video:{video_id}:{f['format_id']}")]); found_heights.add(height)
        if not quality_buttons: await query.edit_message_text("Couldn't find any suitable pre-merged MP4 formats to download."); return
        quality_buttons.append([InlineKeyboardButton("Cancel", callback_data="play_cancel")])
        await query.edit_message_text("Please select a video quality to download:", reply_markup=InlineKeyboardMarkup(quality_buttons))
    except Exception as e: logger.error(f"Error fetching video qualities: {e}"); await query.edit_message_text("Sorry, an error occurred while checking video qualities.")

async def handle_play_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE, media_type: str) -> None:
    query = update.callback_query; await query.answer()
    data_parts = query.data.split(":"); video_id = data_parts[1]; format_id = data_parts[2] if len(data_parts) > 2 else None
    temp_dir = os.path.join(DOWNLOAD_DIR, str(uuid.uuid4())); os.makedirs(temp_dir)
    await query.edit_message_text(f"Downloading {media_type}...")
    try:
        opts = {'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'), 'noplaylist': True, 'quiet': True, 'cookiefile': 'cookies_youtube.txt' if os.path.exists('cookies_youtube.txt') else None}
        if media_type == 'audio': opts.update({'format': 'bestaudio/best', 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}]})
        else:
            if not format_id: raise ValueError("Video format ID is missing.")
            opts['format'] = format_id
        with yt_dlp.YoutubeDL(opts) as ydl: info = ydl.extract_info(video_id, download=True)
        file_path = os.path.join(temp_dir, os.listdir(temp_dir)[0])
        await query.edit_message_text(f"Sending {media_type}...")
        with open(file_path, 'rb') as file:
            if media_type == 'audio':
                artist = info.get('artist') or info.get('uploader', 'Unknown Artist'); title = info.get('title', 'Unknown Title')
                safe_artist = ''.join(c for c in artist if c.isalnum() or c in ' -_')[:20]; safe_title = ''.join(c for c in title if c.isalnum() or c in ' -_')[:20]
                keyboard = [[InlineKeyboardButton("Get Lyrics", callback_data=f"get_lyrics:{safe_artist}|{safe_title}")]]; reply_markup = InlineKeyboardMarkup(keyboard)
                await context.bot.send_audio(query.message.chat_id, file, title=info.get('title'), duration=info.get('duration'), reply_markup=reply_markup)
            else: await context.bot.send_video(query.message.chat_id, file, supports_streaming=True)
        await query.delete_message()
    except Exception as e: logger.error(f"Download error for {video_id}: {e}"); await query.edit_message_text("An error occurred during download. The video might be too large or the format is invalid.")
    finally:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

async def handle_play_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query; await query.answer(); await query.edit_message_text("Search cancelled.")

async def scrape_azlyrics(url: str) -> str:
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=15); response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # This selector is more specific to the div containing lyrics on AZLyrics
        lyrics_div = soup.find('div', class_='ringtone').find_next_sibling('div')
        if lyrics_div:
            return lyrics_div.get_text(separator='\n').strip()
    except Exception as e: logger.error(f"Failed to scrape AZLyrics URL {url}: {e}")
    return None

async def get_lyrics_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query; await query.answer()
    try: _, data = query.data.split(':', 1); artist, title = data.split('|', 1)
    except (ValueError, IndexError): await query.message.reply_text("Could not get song info from the button."); return
    await query.edit_message_reply_markup(None)
    feedback = await query.message.reply_text(f"Searching for lyrics for '{title}' by '{artist}'...")
    lyrics = None
    try:
        await feedback.edit_text("Searching Google for a lyrics source...")
        search_query = f"{artist} {title} lyrics azlyrics"
        search_results = await asyncio.to_thread(search, search_query, num=5, stop=5, pause=1)
        azlyrics_url = next((url for url in search_results if "azlyrics.com/lyrics/" in url), None)
        if azlyrics_url:
            await feedback.edit_text("Found a source on AZLyrics. Scraping lyrics...")
            lyrics = await scrape_azlyrics(azlyrics_url)
    except Exception as e: logger.warning(f"Google search/scrape method failed: {e}. Trying next method.")
    if not lyrics:
        try:
            await feedback.edit_text("Direct scrape failed. Trying azapi library...")
            az_api = azapi.AZlyrics('google', accuracy=0.5); az_api.artist = artist; az_api.title = title
            lyrics = await asyncio.to_thread(az_api.getLyrics, save=False)
        except Exception as e: logger.warning(f"azapi library failed: {e}. Trying final fallback.")
    if not lyrics and gemini_model:
        try:
            await feedback.edit_text("All other methods failed. Asking the AI for the lyrics...")
            prompt = f"Please act as a lyric-finding expert. Search reliable sources like AZLyrics, Genius, or Musixmatch for the full, accurate lyrics of the song '{title}' by '{artist}'. Present the lyrics clearly. If you cannot find them, state that you were unable to locate the lyrics after searching."
            response = await asyncio.to_thread(gemini_model.generate_content, prompt)
            lyrics = response.text
        except Exception as e: logger.error(f"Gemini lyrics fallback failed: {e}")
    if lyrics and "sorry, i could not find" not in lyrics.lower():
        full_message = f"*Lyrics for {title} by {artist}:*\n\n{lyrics}"
        if len(full_message) > 4000:
            await feedback.edit_text(f"{escape_markdown(full_message[:4000], version=2)}\n\n(Lyrics truncated)", parse_mode=ParseMode.MARKDOWN_V2)
        else:
            await feedback.edit_text(escape_markdown(full_message, version=2), parse_mode=ParseMode.MARKDOWN_V2)
    else:
        await feedback.edit_text("Sorry, I couldn't find the lyrics for this song using any available method.")

async def get_joke(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        response = requests.get("https://v2.jokeapi.dev/joke/Any?blacklistFlags=nsfw,racist,sexist&type=twopart", timeout=5).json()
        if not response['error']: await update.message.reply_text(f"{response['setup']}\n\n...{response['delivery']}")
    except Exception as e: logger.error(f"Joke API error: {e}"); await update.message.reply_text("Sorry, couldn't get a joke.")

async def get_crypto_prices(update: Update, context: ContextTypes.DEFAULT_TYPE, crypto_ids: str) -> None:
    ids = ','.join(s.strip().lower() for s in crypto_ids.split(','))
    try:
        prices = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd,ngn", timeout=5).json()
        if not prices: await update.message.reply_text("Couldn't find prices."); return
        message = "*Latest Crypto Prices:*\n" + "\n".join(f"• *{c.title()}*: ${d.get('usd', 0):,.2f} / ₦{d.get('ngn', 0):,.2f}" for c, d in prices.items())
        await update.message.reply_text(escape_markdown(message, version=2), parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e: logger.error(f"Crypto API error: {e}"); await update.message.reply_text("Sorry, couldn't fetch crypto prices.")

async def get_weather(update: Update, context: ContextTypes.DEFAULT_TYPE, city: str) -> None:
    if not OPENWEATHER_API_KEY: await update.message.reply_text("Weather service not configured."); return
    try:
        data = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric", timeout=5).json()
        if data["cod"] != 200: await update.message.reply_text(f"Sorry, couldn't find city '{city}'."); return
        message = f"*Weather in {data['name']}*: {data['weather'][0]['description'].title()} at {data['main']['temp']}°C"
        await update.message.reply_text(escape_markdown(message, version=2), parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e: logger.error(f"Weather API error: {e}"); await update.message.reply_text("Sorry, couldn't fetch the weather.")

# --- Message Routing ---
async def prompt_for_input(update: Update, context: ContextTypes.DEFAULT_TYPE, state: str, message: str, event: str) -> None:
    await save_user_to_db(update, context, event_type=event); context.user_data['state'] = state; await update.message.reply_text(message)

async def record_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text: return
    BUTTON_TEXTS = [
        "AI Tools", "Media Tools", "Utilities", "Help", "Back to Main Menu", "Chat with AI",
        "Create Image", "Read Text from Image", "Animate Image", "Upscale Image", "Summarize Link",
        "Summarize File", "Play Music / Video", "Download Media", "Weather", "Crypto Prices",
        "Translate Text", "Tell a Joke", "End Chat"
    ]
    if update.message.text in BUTTON_TEXTS: return
    state = context.user_data.get('state')
    if state == 'continuous_chat': await gemini_command(update, context); return
    state = context.user_data.pop('state', None)
    if not state: await save_user_to_db(update, context, "Sent a message"); return
    text = update.message.text
    state_handlers = {
        'awaiting_song_name': lambda: search_and_play_song(update, context, song_name=text),
        'awaiting_imagine_prompt': lambda: generate_image(update, context, prompt=text),
        'awaiting_city': lambda: get_weather(update, context, city=text),
        'awaiting_crypto_symbols': lambda: get_crypto_prices(update, context, crypto_ids=text),
        'awaiting_summary_url': lambda: summarize_url(update, context, url=text),
        'awaiting_translation_text': lambda: translate_command(update, context, text_to_translate=text),
    }
    if state in state_handlers: await state_handlers[state]()

# --- Main Application Setup ---
def main() -> None:
    application = Application.builder().token(BOT_TOKEN).rate_limiter(AIORateLimiter()).build()
    
    cmd_handlers = [CommandHandler(cmd, func) for cmd, func in [
        ("start", start), ("help", help_command), ("upscale", upscale_image_command),
        ("animate", animate_command), ("summarize_file", summarize_file_command),
        ("gemini", gemini_command), ("readtext", read_text_from_image_command),
        ("play", play_command)
    ]]
    msg_handlers = [MessageHandler(filters.TEXT & filters.Regex(f"^{pattern}$"), func) for pattern, func in [
        ("AI Tools", show_ai_tools_menu), ("Media Tools", show_media_tools_menu),
        ("Utilities", show_utilities_menu), ("Back to Main Menu", start),
        ("Help", help_command), ("Chat with AI", start_ai_chat), ("End Chat", end_chat),
        ("Read Text from Image", lambda u,c: u.message.reply_text("Please reply to an image with /readtext.")),
        ("Play Music / Video", lambda u,c: prompt_for_input(u,c,'awaiting_song_name', "What song or video?","Pressed 'Play'")),
        ("Create Image", lambda u,c: prompt_for_input(u,c,'awaiting_imagine_prompt', "Describe the image.","Pressed 'Create Image'")),
        ("Weather", lambda u,c: prompt_for_input(u,c,'awaiting_city', "Enter a city name.","Pressed 'Weather'")),
        ("Crypto Prices", lambda u,c: prompt_for_input(u,c,'awaiting_crypto_symbols', "Enter coin IDs (e.g., bitcoin).","Pressed 'Crypto'")),
        ("Tell a Joke", get_joke),
        ("Summarize Link", lambda u,c: prompt_for_input(u,c,'awaiting_summary_url', "Send the article or image link.","Pressed 'Summarize'")),
        ("Translate Text", lambda u,c: prompt_for_input(u,c,'awaiting_translation_text', "Format: language text (e.g., Spanish Hello)","Pressed 'Translate'")),
        ("Animate Image", lambda u,c: u.message.reply_text("Reply to an image with /animate.")),
        ("Upscale Image", lambda u,c: u.message.reply_text("Reply to an image with /upscale.")),
        ("Summarize File", lambda u,c: u.message.reply_text("Reply to an image or PDF with /summarize_file."))
    ]]
    callback_handlers = [
        CallbackQueryHandler(lambda u,c: handle_play_confirmation(u,c,'audio'), pattern="^play_audio:"),
        CallbackQueryHandler(lambda u,c: handle_play_confirmation(u,c,'video'), pattern="^play_video:"),
        CallbackQueryHandler(show_quality_options_callback, pattern="^select_quality:"),
        CallbackQueryHandler(handle_play_cancel, pattern="^play_cancel"),
        CallbackQueryHandler(get_lyrics_callback, pattern="^get_lyrics:")
    ]
    application.add_handlers(cmd_handlers + msg_handlers + callback_handlers)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, record_user_message), group=1)
    
    PORT = int(os.environ.get("PORT", 8443))
    RENDER_APP_NAME = os.environ.get("RENDER_APP_NAME")
    if not RENDER_APP_NAME: application.run_polling()
    else: application.run_webhook(listen="0.0.0.0", port=PORT, url_path=BOT_TOKEN, webhook_url=f"https://{RENDER_APP_NAME}.onrender.com/{BOT_TOKEN}")

if __name__ == "__main__":
    main()
