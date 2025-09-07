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
import re # For robust button handling
from gtts import gTTS
import random
import subprocess # NEW: For running the Node.js script

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup
# MODIFICATION: Added JobQueue for the auto-ping feature
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler, AIORateLimiter, JobQueue
from telegram.constants import ParseMode, ChatAction
from telegram.helpers import escape_markdown

from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError, IntegrityError
from sqlalchemy.types import BigInteger
import yt_dlp

# NEW: For table formatting
from prettytable import PrettyTable

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
try: Base.metadata.create_all(engine)
except OperationalError as e: logger.critical(f"Failed to connect to database: {e}. Exiting."); sys.exit(1)
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
        [KeyboardButton("Read Text from Image"), KeyboardButton("Text to Speech")],
        [KeyboardButton("Animate Image"), KeyboardButton("Upscale Image")],
        [KeyboardButton("Summarize Link"), KeyboardButton("Summarize File")],
        [KeyboardButton("Back to Main Menu")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("AI Tools:", reply_markup=reply_markup)

async def show_media_tools_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [KeyboardButton("Play Music / Video"), KeyboardButton("Download Media")],
        [KeyboardButton("Search TikTok"), KeyboardButton("Youtube")],
        [KeyboardButton("Search Movie"), KeyboardButton("Back to Main Menu")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("Media Tools:", reply_markup=reply_markup)

async def show_utilities_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [KeyboardButton("Weather"), KeyboardButton("Crypto Prices")],
        [KeyboardButton("Translate Text"), KeyboardButton("Tell a Joke")],
        [KeyboardButton("Ask a Riddle"), KeyboardButton("Take Screenshot")],
    ]
    if update.effective_user.id == ADMIN_ID:
        keyboard.append([KeyboardButton("Send Email (Admin)")])
    keyboard.append([KeyboardButton("Convert Video to Audio")])
    keyboard.append([KeyboardButton("Back to Main Menu")])
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("Utilities:", reply_markup=reply_markup)

# --- Feature Functions ---
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "**Bot Commands Guide:**\n\n"
        "You can use the menu buttons or the following commands:\n\n"
        "**/gemini <prompt>**: Ask the AI a question.\n"
        "**/create <prompt>**: Generate an image from text.\n"
        "**/movie <title>**: Get information about a movie.\n"
        "**/tiktoksearch <query>**: Search for TikTok videos.\n"
        "**/ytsearch <query>**: Search for YouTube videos.\n"
        "**/play <song name>**: Search and download a song or video.\n"
        "**/tts <text>**: Convert text to speech.\n"
        "**/mp4**: Reply to a video to convert it to an MP3 audio file.\n"
        "**/gmail**: Send an email (Admin only).\n"
        "**/connect +<number>**: Connect to a WhatsApp account using a pairing code."
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

# NEW: Admin command to view database contents
async def db_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /db command to view database table contents."""
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("Sorry, this command is for the admin only.")
        return
    
    await save_user_to_db(update, context, event_type="Used /db command")
    context.user_data['state'] = 'awaiting_db_table_name'
    await update.message.reply_text("Please enter the name of the database table you want to view (e.g., `users`).")

async def view_db_table(update: Update, context: ContextTypes.DEFAULT_TYPE, table_name: str) -> None:
    """Fetches and formats a database table's contents."""
    session = Session()
    feedback = await update.message.reply_text(f"Fetching data from table `{table_name}`...", parse_mode=ParseMode.MARKDOWN)

    try:
        # Check if table exists (a simple way)
        check_query = f"SELECT 1 FROM {table_name} LIMIT 1;"
        session.execute(check_query)
        
        # Build and execute the main query
        query = f"SELECT * FROM {table_name};"
        result = session.execute(query).fetchall()
        
        if not result:
            await feedback.edit_text(f"Table `{table_name}` is empty or does not exist.")
            return

        # Get column names from the result object
        column_names = session.execute(query).keys()

        # Create a PrettyTable object
        table = PrettyTable()
        table.field_names = column_names
        
        for row in result:
            table.add_row(row)
            
        # Format the table string with Markdown
        formatted_table = "```\n" + str(table) + "\n```"

        # Telegram markdown limit is 4096 characters.
        if len(formatted_table) > 4000:
            await feedback.edit_text("The table content is too large to display. Please view it directly from the database.")
            return

        await feedback.edit_text(formatted_table, parse_mode=ParseMode.MARKDOWN_V2)

    except OperationalError as e:
        logger.error(f"Database error while viewing table {table_name}: {e}")
        await feedback.edit_text(f"Database error: The table `{table_name}` may not exist or an invalid query was executed.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while viewing table {table_name}: {e}")
        await feedback.edit_text("An unexpected error occurred while trying to view the table.")
    finally:
        session.close()

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
    if not update.message.reply_to_message or not update.message.reply_to_message.photo:
        await update.message.reply_text("Please reply to an image with /readtext or use the menu button.")
        return
    await save_user_to_db(update, context, event_type="Used OCR")
    feedback = await update.message.reply_text("Reading text from image...")
    try:
        photo_file = await update.message.reply_to_message.photo[-1].get_file()
        image = Image.open(io.BytesIO(await photo_file.download_as_bytearray()))
        text = await asyncio.to_thread(pytesseract.image_to_string, image)
        if text and not text.isspace():
            await feedback.edit_text(f"**Extracted Text:**\n\n{text}", parse_mode=ParseMode.MARKDOWN)
        else:
            await feedback.edit_text("Couldn't find any readable text in the image.")
    except Exception as e:
        logger.error(f"OCR Error: {e}")
        await feedback.edit_text("Sorry, an error occurred while processing the image.")

async def summarize_url(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str) -> None:
    if not gemini_model: await update.message.reply_text("AI summarizer is not configured."); return
    feedback = await update.message.reply_text("Analyzing link...")
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
        article_text = ' '.join(p.get_text() for p in soup.find_all('p'))
        if len(article_text) < 100: await feedback.edit_text("Couldn't extract enough text to summarize."); return
        await feedback.edit_text("Content extracted. Summarizing with AI...")
        prompt = f"Please provide a concise but comprehensive summary of the following article text:\n\n{article_text[:15000]}"
        ai_response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        await feedback.edit_text(ai_response.text, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Summarize URL error for {url}: {e}")
        await feedback.edit_text("Sorry, I couldn't read or summarize that URL.")

async def summarize_file_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not gemini_model: await update.message.reply_text("AI service is not configured."); return
    if not update.message.reply_to_message: await update.message.reply_text("Please reply to an image or a PDF file with /summarize_file."); return
    replied_message = update.message.reply_to_message
    feedback_message = await replied_message.reply_text("Processing file...")
    summary = ""
    try:
        if replied_message.photo:
            await feedback_message.edit_text("Analyzing image...")
            photo_file = await replied_message.photo[-1].get_file()
            photo_bytes = await photo_file.download_as_bytearray()
            image_part = {"mime_type": "image/jpeg", "data": photo_bytes}
            prompt = "Describe this image in detail."
            response = await asyncio.to_thread(gemini_model.generate_content, [prompt, image_part])
            summary = response.text
        elif replied_message.document and replied_message.document.mime_type == 'application/pdf':
            pdf_file = await replied_message.document.get_file()
            with fitz.open(stream=await pdf_file.download_as_bytearray(), filetype="pdf") as doc: full_text = "".join(page.get_text() for page in doc)
            if not full_text.strip(): await feedback_message.edit_text("Could not extract any text from this PDF."); return
            await feedback_message.edit_text("Extracted text. Summarizing...")
            prompt = f"Please provide a detailed summary of the following document:\n\n{full_text[:15000]}"
            response = await asyncio.to_thread(gemini_model.generate_content, prompt)
            summary = response.text
        else:
            await feedback_message.edit_text("This command only works on an image or PDF file."); return
        await feedback_message.edit_text(f"**Summary:**\n\n{summary}", parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"File summarization error: {e}")
        await feedback_message.edit_text("Sorry, an error occurred while processing the file.")

async def create_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str = None) -> None:
    if not openai_client: await update.message.reply_text("Image generation service (OpenAI) is not configured."); return
    if not prompt: prompt = " ".join(context.args)
    if not prompt: await update.message.reply_text("Please describe the image you want to create."); return
    feedback = await update.message.reply_text("Creating your image with DALL-E 3...")
    try:
        response = await openai_client.images.generate(model="dall-e-3", prompt=prompt, n=1, size="1024x1024", quality="standard")
        await context.bot.send_photo(update.effective_chat.id, photo=response.data[0].url, caption=f"Creation: `{prompt}`", parse_mode=ParseMode.MARKDOWN)
        await feedback.delete()
    except Exception as e:
        logger.error(f"DALL-E 3 API error: {e}")
        await feedback.edit_text("Sorry, I couldn't create the image.")

async def upscale_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Upscales an image using the Stability AI API."""
    # Check if the Stability AI API key is configured, which is also used for animations.
    if not STABILITY_API_KEY:
        await update.message.reply_text("Image upscaling service (Stability AI) is not configured.")
        return

    # Ensure the user is replying to a photo.
    if not update.message.reply_to_message or not update.message.reply_to_message.photo:
        await update.message.reply_text("Please reply to an image with the `/upscale` command to use this feature.")
        return

    feedback = await update.message.reply_text("🚀 Upscaling your image with Stability AI... this can take a moment.")
    try:
        # Download the image file from Telegram.
        photo_file = await update.message.reply_to_message.photo[-1].get_file()
        image_bytes = await photo_file.download_as_bytearray()

        # Make the API call to Stability AI's upscale endpoint.
        response = requests.post(
            "https://api.stability.ai/v1/generation/esr-v1/image-to-image/upscale",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {STABILITY_API_KEY}"
            },
            files={
                "image": image_bytes
            },
            timeout=90  # Set a generous timeout for the API call.
        )
        # Raise an exception for bad status codes (4xx or 5xx).
        response.raise_for_status()

        data = response.json()
        
        # Check if the response contains the expected image data.
        if "artifacts" in data and len(data["artifacts"]) > 0:
            # Decode the base64 encoded image.
            base64_image = data["artifacts"][0]["base64"]
            upscaled_image_data = base64.b64decode(base64_image)

            # Send the upscaled image back to the user as a document to preserve quality.
            await context.bot.send_document(
                chat_id=update.effective_chat.id,
                document=upscaled_image_data,
                filename='upscaled_image.png',
                caption='✨ Here is your upscaled image!'
            )
            # Clean up the feedback message.
            await feedback.delete()
        else:
            # Handle cases where the API returns a success status but no image.
            logger.error("Stability AI API did not return an image artifact.")
            await feedback.edit_text("The upscaling service did not return an image. Please try again later.")

    except requests.exceptions.HTTPError as http_err:
        # Handle specific HTTP errors from the API.
        error_message = "An API error occurred."
        try:
            # Try to parse a more specific error from the API's JSON response.
            error_details = http_err.response.json().get('message', 'No details provided.')
            error_message = f"API Error: {error_details}"
        except json.JSONDecodeError:
            error_message = f"API returned a non-JSON error (Status {http_err.response.status_code})."
        
        logger.error(f"Stability AI upscale request failed: {error_message}")
        await feedback.edit_text(f"Sorry, I couldn't upscale the image. {error_message}")
    except Exception as e:
        # Catch any other unexpected errors.
        logger.error(f"Unexpected error in upscale_command: {e}")
        await feedback.edit_text("An unexpected error occurred while upscaling the image.")


async def animate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not STABILITY_API_KEY: await update.message.reply_text("Video animation service not configured."); return
    if not update.message.reply_to_message or not update.message.reply_to_message.photo: await update.message.reply_text("Please reply to an image with /animate."); return
    feedback = await update.message.reply_text("Sending image to animation engine...")
    try:
        photo_file = await update.message.reply_to_message.photo[-1].get_file()
        response = requests.post("https://api.stability.ai/v2/generation/image-to-video", headers={"authorization": f"Bearer {STABILITY_API_KEY}"}, files={"image": await photo_file.download_as_bytearray()}, data={"motion_bucket_id": 40}, timeout=30)
        response.raise_for_status()
        generation_id = response.json()["id"]
        await feedback.edit_text("Animation started. This may take a minute...")
        for _ in range(45):
            await asyncio.sleep(4)
            res = requests.get(f"https://api.stability.ai/v2/generation/image-to-video/result/{generation_id}", headers={'authorization': f"Bearer {STABILITY_API_KEY}", 'accept': "video/mp4"}, timeout=20)
            if res.status_code == 200:
                await context.bot.send_video(update.effective_chat.id, video=res.content, caption="Here is your animated video!")
                await feedback.delete()
                return
        await feedback.edit_text("Sorry, the animation timed out.")
    except Exception as e:
        logger.error(f"Animate command error: {e}")
        await feedback.edit_text("Sorry, an error occurred while creating the animation.")

async def convert_video_to_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message.reply_to_message or not update.message.reply_to_message.video:
        await update.message.reply_text("To convert a video to audio, please reply to the video with the `/mp4` command.")
        return
    await save_user_to_db(update, context, event_type="Used /mp4 command")
    feedback_message = await update.message.reply_text("Downloading video...")
    temp_dir = os.path.join(DOWNLOAD_DIR, str(uuid.uuid4()))
    os.makedirs(temp_dir, exist_ok=True)
    try:
        video = update.message.reply_to_message.video
        video_file = await video.get_file()
        original_video_path = os.path.join(temp_dir, f"{video.file_unique_id}.mp4")
        await video_file.download_to_drive(original_video_path)
        await feedback_message.edit_text("Converting to audio...")
        output_audio_path = os.path.join(temp_dir, f"{video.file_unique_id}.mp3")
        ffmpeg_command = ['ffmpeg', '-i', original_video_path, '-vn', '-q:a', '0', '-map', 'a', output_audio_path]
        process = await asyncio.create_subprocess_exec(*ffmpeg_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            logger.error(f"FFmpeg conversion failed. Stderr: {stderr.decode()}")
            await feedback_message.edit_text("Sorry, the conversion failed.")
            return
        await feedback_message.edit_text("Uploading audio...")
        with open(output_audio_path, 'rb') as audio_file:
            await context.bot.send_audio(chat_id=update.effective_chat.id, audio=audio_file, title=video.file_name or "Converted Audio", duration=video.duration)
        await feedback_message.delete()
    except Exception as e:
        logger.error(f"Error in /mp4 command: {e}")
        await feedback_message.edit_text("An unexpected error occurred.")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

async def tts_command(update: Update, context: ContextTypes.DEFAULT_TYPE, text_to_speak: str = None) -> None:
    if not text_to_speak:
        if context.args:
            text_to_speak = " ".join(context.args)
        elif update.message.reply_to_message and update.message.reply_to_message.text:
            text_to_speak = update.message.reply_to_message.text
    if not text_to_speak:
        await update.message.reply_text("Usage: `/tts <text>` or reply to a message with `/tts`.")
        return
    if len(text_to_speak) > 1000:
        await update.message.reply_text("Text is too long (max 1000 characters).")
        return
    feedback = await update.message.reply_text("Generating audio...")
    temp_audio_path = os.path.join(DOWNLOAD_DIR, f"{uuid.uuid4()}.mp3")
    try:
        def generate_audio_sync():
            tts = gTTS(text=text_to_speak, lang='en', slow=False)
            tts.save(temp_audio_path)
        await asyncio.to_thread(generate_audio_sync)
        with open(temp_audio_path, 'rb') as audio_file:
            await context.bot.send_audio(chat_id=update.effective_chat.id, audio=audio_file, title="Generated Speech")
        await feedback.delete()
    except Exception as e:
        logger.error(f"gTTS Error: {e}")
        await feedback.edit_text("Sorry, an error occurred while generating the audio.")
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

async def tiktok_search_command(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str = None, count: int = 5) -> None:
    # --- FIX: Correctly determine chat_id from either an Update or a CallbackQuery ---
    chat_id = update.effective_chat.id if hasattr(update, 'effective_chat') and update.effective_chat else update.message.chat_id

    if not query:
        if not context.args:
            await context.bot.send_message(chat_id, "Please provide a search term.")
            return
        query = " ".join(context.args)
        await ask_for_tiktok_count(update, context, query)
        return
    
    feedback = await context.bot.send_message(chat_id=chat_id, text=f"Searching TikTok for '{query}'...")
    try:
        api_url = 'https://tikwm.com/api/feed/search'
        headers = {"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8", "Cookie": "current_language=en", "User-Agent": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Mobile Safari/537.36"}
        data = {'keywords': query, 'count': '20'}
        response = requests.post(api_url, headers=headers, data=data, timeout=20)
        response.raise_for_status()
        videos = response.json().get('data', {}).get('videos', [])
        
        if not videos:
            await feedback.edit_text("No TikTok videos found for that search.")
            return
            
        await feedback.edit_text(f"Found {len(videos)} videos. Sending {count} of them...")
        random.shuffle(videos)
        
        for video in videos[:count]:
            video_url, caption = video.get('play'), video.get('title', 'No caption')
            if video_url:
                try:
                    await context.bot.send_video(chat_id=chat_id, video=video_url, caption=caption, read_timeout=60, write_timeout=60)
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Failed to send TikTok video {video_url}: {e}")
                    await context.bot.send_message(chat_id, f"Could not send one of the videos.")
    except Exception as e:
        logger.error(f"TikTok search failed: {e}")
        await feedback.edit_text("An unexpected error occurred.")

async def ask_for_tiktok_count(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str):
    context.user_data['tiktok_query'] = query
    keyboard = [[
        InlineKeyboardButton("3 Videos", callback_data="tiktok_count:3"),
        InlineKeyboardButton("5 Videos", callback_data="tiktok_count:5"),
        InlineKeyboardButton("10 Videos", callback_data="tiktok_count:10")
    ]]
    await update.message.reply_text(f"How many videos for '{query}' would you like?", reply_markup=InlineKeyboardMarkup(keyboard))

async def handle_tiktok_count_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    count = int(query.data.split(":")[1])
    search_query = context.user_data.pop('tiktok_query', None)
    if not search_query:
        await query.edit_message_text("Sorry, your search expired. Please try again.")
        return
    await query.edit_message_text("Great! Starting your search...")
    await tiktok_search_command(query, context, query=search_query, count=count)

async def Youtube_command(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str = None) -> None:
    if not query:
        if not context.args:
            await update.message.reply_text("Please provide a search term. Usage: `/ytsearch <query>`")
            return
        query = " ".join(context.args)
    feedback = await update.message.reply_text(f"Searching YouTube for '{query}'...")
    try:
        ydl_opts = {
            'noplaylist': True, 'quiet': True, 'default_search': 'ytsearch5',
            'cookiefile': 'cookies_youtube.txt' if os.path.exists('cookies_youtube.txt') else None
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)
        if not info.get('entries'):
            await feedback.edit_text("Sorry, couldn't find any results."); return
        keyboard = []
        for video in info['entries']:
            title, video_id = video.get('title', 'Unknown Title'), video.get('id')
            button_text = (title[:60] + '..') if len(title) > 60 else title
            keyboard.append([InlineKeyboardButton(button_text, callback_data=f"play_confirm:{video_id}")])
        await feedback.edit_text("Here are the top 5 results:", reply_markup=InlineKeyboardMarkup(keyboard))
    except Exception as e:
        logger.error(f"Youtube error: {e}")
        await feedback.edit_text("An error occurred during the search. Make sure your `cookies_youtube.txt` is valid.")

async def play_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    song_name = " ".join(context.args)
    if not song_name:
        await update.message.reply_text("Please provide a song name, e.g., `/play Burna Boy - Last Last`")
        return
    await search_and_play_song(update, context, song_name)

async def search_and_play_song(update: Update, context: ContextTypes.DEFAULT_TYPE, song_name: str) -> None:
    message_to_reply_to = update.callback_query.message if update.callback_query else update.message
    feedback = await message_to_reply_to.reply_text(f"Searching for '{song_name}'...")
    try:
        ydl_opts = {'noplaylist': True, 'quiet': True, 'default_search': 'ytsearch1', 'cookiefile': 'cookies_youtube.txt' if os.path.exists('cookies_youtube.txt') else None}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: info = ydl.extract_info(song_name, download=False)
        if not info.get('entries'):
            await feedback.edit_text("Sorry, couldn't find any results."); return
        video = info['entries'][0]
        title, video_id = video.get('title', 'Unknown Title'), video.get('id')
        keyboard = [[InlineKeyboardButton("Yes, that's it!", callback_data=f"play_confirm:{video_id}")], [InlineKeyboardButton("No, cancel", callback_data="play_cancel")]]
        await feedback.edit_text(f"I found: **{title}**\n\nIs this correct?", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Play search error: {e}")
        await feedback.edit_text("An error occurred while searching. YouTube may be blocking requests.")

async def handle_play_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query; await query.answer()
    video_id = query.data.split(":")[1]
    keyboard = [[InlineKeyboardButton("Audio", callback_data=f"dl_audio:{video_id}"), InlineKeyboardButton("Video", callback_data=f"dl_video:{video_id}")]]
    await query.edit_message_text("Choose your desired format:", reply_markup=InlineKeyboardMarkup(keyboard))

async def handle_audio_download(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query; await query.answer()
    video_id = query.data.split(":")[1]
    temp_dir = os.path.join(DOWNLOAD_DIR, str(uuid.uuid4())); os.makedirs(temp_dir)
    await query.edit_message_text("Downloading audio...")
    info = None
    try:
        audio_opts = {'format': 'bestaudio/best', 'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'), 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}], 'noplaylist': True, 'quiet': True, 'cookiefile': 'cookies_youtube.txt' if os.path.exists('cookies_youtube.txt') else None}
        with yt_dlp.YoutubeDL(audio_opts) as ydl: info = ydl.extract_info(video_id, download=True)
        downloaded_files = os.listdir(temp_dir)
        if not downloaded_files: raise Exception("yt-dlp download failed silently.")
        audio_path = os.path.join(temp_dir, downloaded_files[0])
        await query.edit_message_text("Sending audio...")
        with open(audio_path, 'rb') as audio_file:
            await context.bot.send_audio(chat_id=query.message.chat_id, audio=audio_file, title=info.get('title'), duration=info.get('duration'))
        await query.delete_message()
    except Exception as e:
        logger.error(f"Audio download error for ID {video_id}: {e}")
        await query.edit_message_text("An error occurred during download. The video might be protected.")
    finally:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

async def handle_video_download(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query; await query.answer()
    video_id = query.data.split(":")[1]
    temp_dir = os.path.join(DOWNLOAD_DIR, str(uuid.uuid4())); os.makedirs(temp_dir)
    info = None
    await query.edit_message_text("Checking video details...")
    try:
        ydl_opts = {'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', 'quiet': True, 'cookiefile': 'cookies_youtube.txt' if os.path.exists('cookies_youtube.txt') else None}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_id, download=False)
        filesize = info.get('filesize') or info.get('filesize_approx', 0)
        file_size_mb = filesize / (1024 * 1024) if filesize else 0
        if file_size_mb <= 49:
            await query.edit_message_text(f"Downloading video ({file_size_mb:.2f} MB)...")
            ydl_opts_download = {'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'), **ydl_opts}
            with yt_dlp.YoutubeDL(ydl_opts_download) as ydl_dl: ydl_dl.download([video_id])
            downloaded_files = os.listdir(temp_dir)
            if not downloaded_files: raise Exception("yt-dlp download failed silently.")
            video_path = os.path.join(temp_dir, downloaded_files[0])
            await query.edit_message_text("Sending video...")
            with open(video_path, 'rb') as video_file:
                await context.bot.send_video(chat_id=query.message.chat_id, video=video_file, supports_streaming=True)
            await query.delete_message()
        else:
            video_url = info.get('url')
            if video_url:
                keyboard = [[InlineKeyboardButton("Download Full Video (Direct Link)", url=video_url)]]
                await query.edit_message_text(f"This video is too large to send ({file_size_mb:.2f} MB).\n\nYou can download it directly using this link:", reply_markup=InlineKeyboardMarkup(keyboard))
            else:
                await query.edit_message_text("The video is too large and I couldn't get a direct download link.")
    except Exception as e:
        logger.error(f"Video download error for ID {video_id}: {e}")
        error_message = "An error occurred. The video may be private or protected."
        keyboard = None
        if info and info.get('url'):
            error_message += "\n\nYou can try watching or downloading it directly:"
            keyboard = [[InlineKeyboardButton("Watch/Download Video (Direct Link)", url=info.get('url'))]]
        await query.edit_message_text(error_message, reply_markup=InlineKeyboardMarkup(keyboard) if keyboard else None)
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
    if platform.lower() == 'tiktok':
        feedback = await update.message.reply_text("Analyzing TikTok link...")
        try:
            api_url = f"https://tikwm.com/api/?url={content_url}"
            response = requests.get(api_url, timeout=15).json()
            if response.get('code') == 0 and 'data' in response:
                data = response['data']
                if 'images' in data and data['images']:
                    await feedback.edit_text(f"Found {len(data['images'])} photos. Sending them now...")
                    for image_url in data['images']:
                        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_url)
                    await feedback.delete()
                    return
                else:
                    await feedback.edit_text("It's a video. Using the standard downloader...")
            else:
                 await feedback.edit_text("Could not fetch details. Trying standard downloader...")
        except Exception as e:
            logger.error(f"TikTok API error for {content_url}: {e}")
            await feedback.edit_text("Could not fetch details. Trying standard downloader...")

    feedback = await update.message.reply_text(f"Starting download from {platform}...")
    temp_dir = os.path.join(DOWNLOAD_DIR, str(uuid.uuid4())); os.makedirs(temp_dir, exist_ok=True)
    try:
        ydl_opts = {'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'), 'noplaylist': True, 'quiet': True, 'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', 'http_headers': {'User-Agent': 'Mozilla/5.0'}}
        if platform.lower() == 'youtube' and os.path.exists('cookies_youtube.txt'): ydl_opts['cookiefile'] = 'cookies_youtube.txt'
        elif os.path.exists('cookies.txt'): ydl_opts['cookiefile'] = 'cookies.txt'
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: ydl.download([content_url])
        downloaded_files = os.listdir(temp_dir)
        if not downloaded_files: raise Exception("Download failed silently")
        downloaded_file_path = os.path.join(temp_dir, downloaded_files[0])
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

# NEW FUNCTION: This job will run on a schedule to keep the bot alive
async def ping_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Periodically pings the bot's webhook to prevent it from spinning down on free hosting."""
    webhook_url = context.job.data.get("webhook_url")
    if not webhook_url:
        logger.warning("Webhook URL not found in job context for ping job. Skipping.")
        return
    try:
        # Use a timeout to avoid hanging forever
        response = await asyncio.to_thread(requests.get, webhook_url, timeout=20)
        logger.info(f"Auto-ping successful. Status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Auto-ping failed: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during auto-ping: {e}")

# --- WhatsApp Integration ---
async def connect_whatsapp(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /connect command to link a WhatsApp account."""
    if not context.args or len(context.args) < 1:
        await update.message.reply_text("Please provide a WhatsApp number. Usage: `/connect +<number>` (e.g., /connect +12345678901).")
        return

    phone_number = context.args[0].replace('+', '')
    if not phone_number.isdigit():
        await update.message.reply_text("Invalid phone number format. Please use digits only.")
        return

    user_id = update.effective_user.id
    session_name = f"Ultar_{user_id}"
    session_path = os.path.join(DOWNLOAD_DIR, session_name)
    
    # Check if a session already exists for this user
    if os.path.exists(session_path):
        await update.message.reply_text("A session for your account already exists. Please delete it or use the previous one.")
        return

    feedback_msg = await update.message.reply_text("Generating pairing code...")

    try:
        # Run the Node.js script as a subprocess
        process = await asyncio.create_subprocess_exec(
            "node", "baileys_handler.js", session_name, phone_number,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()
        stdout = stdout.decode('utf-8').strip()
        
        if "PAIRING_CODE:" in stdout:
            code = stdout.split("PAIRING_CODE:")[1].strip()
            await feedback_msg.edit_text(f"Your WhatsApp pairing code is: `{code}`\n\nOpen WhatsApp, go to `Linked Devices` > `Link with phone number` and enter this code.")
            
            # Wait for the connection to be established
            for _ in range(60): # Wait up to 5 minutes for connection
                await asyncio.sleep(5)
                if os.path.exists(os.path.join(session_path, 'creds.json')):
                    await update.message.reply_text("Connection successful! Sending your session file.")
                    
                    # Package the session folder into a zip file
                    shutil.make_archive(session_path, 'zip', session_path)
                    
                    with open(f"{session_path}.zip", "rb") as session_file:
                        await context.bot.send_document(
                            chat_id=update.effective_chat.id,
                            document=session_file,
                            filename=f"{session_name}.zip",
                            caption="Your WhatsApp session is ready! Keep this file secure."
                        )
                    shutil.rmtree(session_path) # Clean up folder
                    os.remove(f"{session_path}.zip") # Clean up zip
                    return

            await update.message.reply_text("Connection timed out. Please try again.")

        else:
            await feedback_msg.edit_text("Failed to get pairing code. Check the server logs for errors.")
            logger.error(f"Baileys script error: {stderr.decode('utf-8')}")
            
    except FileNotFoundError:
        # This is a key fix to provide better user feedback
        await update.message.reply_text("The Node.js environment or `baileys_handler.js` script is not set up correctly. Please ensure both `node` and `baileys_handler.js` are in your bot's directory and executable.")
        logger.error("FileNotFoundError: The 'node' executable or 'baileys_handler.js' script was not found. Please verify your server setup.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during WhatsApp connection: {e}")
        await feedback_msg.edit_text("An unexpected error occurred.")


# --- Message Routing ---
async def prompt_for_input(update: Update, context: ContextTypes.DEFAULT_TYPE, state: str, message: str, event: str) -> None:
    await save_user_to_db(update, context, event_type=event)
    context.user_data['state'] = state
    await update.message.reply_text(message)

async def record_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text: return
    state = context.user_data.get('state')
    if not state:
        await save_user_to_db(update, context, "Sent a message")
        return
    text = update.message.text
    if state == 'continuous_chat':
        await gemini_command(update, context, prompt_text=text)
        return
    
    popped_state = context.user_data.pop('state')
    
    if popped_state == 'awaiting_email_address':
        context.user_data['email_to'] = text; context.user_data['state'] = 'awaiting_email_subject'
        await update.message.reply_text("Step 2 of 3: Great. Now, what should the subject be?")
        return
    elif popped_state == 'awaiting_email_subject':
        context.user_data['email_subject'] = text; context.user_data['state'] = 'awaiting_email_body'
        await update.message.reply_text("Step 3 of 3: Perfect. Please enter the message body.")
        return
    elif popped_state == 'awaiting_email_body':
        to_address = context.user_data.pop('email_to', 'N/A')
        subject = context.user_data.pop('email_subject', 'N/A')
        feedback = await update.message.reply_text(f"Sending email to {to_address}...")
        msg = EmailMessage(); msg.set_content(text); msg['Subject'] = subject; msg['From'] = os.environ.get("GMAIL_ADDRESS"); msg['To'] = to_address
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
    elif popped_state == 'awaiting_tiktok_query':
        await ask_for_tiktok_count(update, context, text)
        return
    # NEW: Handle the database table name input
    elif popped_state == 'awaiting_db_table_name':
        table_name = text.strip()
        await view_db_table(update, context, table_name)
        return

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
        'awaiting_tts_text': lambda: tts_command(update, context, text_to_speak=text),
        'awaiting_ytsearch_query': lambda: Youtube_command(update, context, query=text),
        'awaiting_suggestion': lambda: handle_suggestion(update, context, suggestion=text)
    }
    
    if popped_state in state_handlers: await state_handlers[popped_state]()

async def handle_suggestion(update: Update, context: ContextTypes.DEFAULT_TYPE, suggestion: str):
    user = update.effective_user
    admin_message = (f"📩 New Suggestion Received\n\n"
                     f"From: {user.first_name} (`{user.id}`)\n"
                     f"Username: @{user.username or 'N/A'}\n\n"
                     f"Suggestion:\n{suggestion}")
    await context.bot.send_message(chat_id=ADMIN_ID, text=admin_message, parse_mode=ParseMode.MARKDOWN)
    await update.message.reply_text("Thank you! Your suggestion has been sent to the admin.")

# --- Main Application Setup ---
def main() -> None:
    # MODIFICATION: Added .job_queue() to the builder
    application = Application.builder().token(BOT_TOKEN).rate_limiter(AIORateLimiter()).job_queue(JobQueue()).build()
    
    cmd_handlers = [
        CommandHandler("start", start), 
        CommandHandler("help", help_command),
        CommandHandler("gemini", gemini_command), 
        CommandHandler("create", create_image_command),
        CommandHandler("upscale", upscale_image_command), 
        CommandHandler("animate", animate_command),
        CommandHandler("summarize_file", summarize_file_command), 
        CommandHandler("readtext", read_text_from_image_command),
        CommandHandler("play", play_command), 
        CommandHandler("mp4", convert_video_to_audio),
        CommandHandler("riddle", get_riddle), 
        CommandHandler("gmail", gmail_command),
        CommandHandler("screenshot", screenshot_command), 
        CommandHandler("movie", movie_command),
        CommandHandler("tts", tts_command), 
        CommandHandler("tiktoksearch", tiktok_search_command),
        CommandHandler("ytsearch", Youtube_command),
        # NEW: Handler for WhatsApp connection
        CommandHandler("connect", connect_whatsapp),
        # NEW: Handler for the /db command
        CommandHandler("db", db_command)
    ]
    
    menu_button_texts = {
        "AI Tools": show_ai_tools_menu, "Media Tools": show_media_tools_menu,
        "Utilities": show_utilities_menu, "Help": help_command,
        "Back to Main Menu": start, "End Chat": end_chat,
        "Chat with AI": start_ai_chat, "Tell a Joke": get_joke,
        "Ask a Riddle": get_riddle,
        "Send Suggestion": lambda u,c: prompt_for_input(u,c,'awaiting_suggestion', "Please type your suggestion or feedback and I will forward it to the admin.","Pressed 'Suggestion'"),
        "Create Image": lambda u,c: prompt_for_input(u,c,'awaiting_create_prompt', "Describe the image to create.","Pressed 'Create Image'"),
        "Read Text from Image": lambda u,c: u.message.reply_text("Please reply to an image with /readtext to use this feature."),
        "Text to Speech": lambda u,c: prompt_for_input(u,c,'awaiting_tts_text', "What text should I convert to speech?","Pressed 'TTS'"),
        "Upscale Image": lambda u,c: u.message.reply_text("Please reply to an image with /upscale."),
        "Animate Image": lambda u,c: u.message.reply_text("Please reply to an image with /animate."),
        "Summarize File": lambda u,c: u.message.reply_text("Please reply to an image or PDF with /summarize_file."),
        "Summarize Link": lambda u,c: prompt_for_input(u,c,'awaiting_summary_url', "Please send the article link.","Pressed 'Summarize Link'"),
        "Play Music / Video": lambda u,c: prompt_for_input(u,c,'awaiting_song_name', "What song or video would you like?","Pressed 'Play'"),
        "Download Media": show_download_platform_options,
        "Search Movie": lambda u,c: prompt_for_input(u,c,'awaiting_movie_title', "What movie are you looking for?","Pressed 'Search Movie'"),
        "Search TikTok": lambda u,c: prompt_for_input(u,c,'awaiting_tiktok_query', "What do you want to search for on TikTok?","Pressed 'Search TikTok'"),
        "Youtube": lambda u,c: prompt_for_input(u,c,'awaiting_ytsearch_query', "What do you want to search for on YouTube?","Pressed 'Youtube'"),
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
        CallbackQueryHandler(handle_tiktok_count_selection, pattern="^tiktok_count:"),
    ]

    application.add_handlers(cmd_handlers + msg_handlers + callback_handlers)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.Regex(f"^({'|'.join(escaped_patterns)})$"), record_user_message))
    
    PORT = int(os.environ.get("PORT", 8443))
    RENDER_APP_NAME = os.environ.get("RENDER_APP_NAME")

    # MODIFICATION: Schedule the ping job only in webhook mode
    if not RENDER_APP_NAME:
        logger.info("Running in polling mode.")
        application.run_polling()
    else:
        WEBHOOK_URL = f"https://{RENDER_APP_NAME}.onrender.com/{BOT_TOKEN}"
        logger.info(f"Running in webhook mode. URL: {WEBHOOK_URL}")
        
        # Schedule the auto-ping job
        job_queue = application.job_queue
        job_queue.run_repeating(
            callback=ping_job,
            interval=600,  # 10 minutes (in seconds)
            first=10,      # Start after 10 seconds
            name="auto_ping_job",
            data={"webhook_url": WEBHOOK_URL}
        )
        logger.info("Auto-ping job scheduled to run every 10 minutes.")
        
        application.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=BOT_TOKEN,
            webhook_url=WEBHOOK_URL
        )

if __name__ == "__main__":
    main()
