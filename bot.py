# -*- coding: utf-8 -*-

# --- Imports ---
import logging
import os
import sys
import asyncio
import uuid
import shutil
import time
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
import subprocess # For running the Node.js script
import datetime # For the recurring email job

from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler, AIORateLimiter, JobQueue
from telegram.constants import ParseMode, ChatAction
from telegram.helpers import escape_markdown

from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError, IntegrityError
from sqlalchemy.types import BigInteger
import yt_dlp

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
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN") 
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
        [KeyboardButton("Download Novel"), KeyboardButton("Search TikTok")],
        [KeyboardButton("Youtube"), KeyboardButton("Search Movie")],
        [KeyboardButton("Back to Main Menu")]
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
        "**/novel <title>**: Search for a novel to download.\n"
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

async def db_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("Sorry, this command is for the admin only.")
        return
    await save_user_to_db(update, context, event_type="Used /db command")
    context.user_data['state'] = 'awaiting_db_table_name'
    await update.message.reply_text("Please enter the name of the database table you want to view (e.g., `users`).")

async def view_db_table(update: Update, context: ContextTypes.DEFAULT_TYPE, table_name: str) -> None:
    session = Session()
    feedback = await update.message.reply_text(f"Fetching data from table `{table_name}`...", parse_mode=ParseMode.MARKDOWN)
    try:
        query = f"SELECT * FROM {table_name};"
        result_proxy = session.execute(query)
        result = result_proxy.fetchall()
        if not result:
            await feedback.edit_text(f"Table `{table_name}` is empty or does not exist.")
            return
        column_names = result_proxy.keys()
        table = PrettyTable()
        table.field_names = column_names
        for row in result:
            table.add_row(row)
        formatted_table = f"```\n{str(table)}\n```"
        if len(formatted_table) > 4000:
            await feedback.edit_text("The table content is too large to display.")
            return
        await feedback.edit_text(formatted_table, parse_mode=ParseMode.MARKDOWN_V2)
    except OperationalError:
        await feedback.edit_text(f"Database error: The table `{table_name}` may not exist.")
    except Exception as e:
        logger.error(f"Error viewing table {table_name}: {e}")
        await feedback.edit_text("An unexpected error occurred.")
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
        await update.message.reply_text("Sorry, an error occurred with the AI.")

async def gmail_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_user.id != ADMIN_ID:
        await update.message.reply_text("Sorry, this command is for the admin only.")
        return
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        await update.message.reply_text("The email service is not configured.")
        return
    context.user_data['state'] = 'awaiting_email_address'
    await update.message.reply_text("Step 1 of 3: Please enter the recipient's email address.\n\nFor multiple recipients, separate them with commas.")

# --- Functions for recurring email feature ---

async def resend_email_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    """The job function that resends the email."""
    job = context.job
    data = job.data
    
    # weekday() returns 0 for Monday and 6 for Sunday.
    if datetime.datetime.now().weekday() == data['stop_day_index']:
        logger.info(f"Stopping recurring email job {job.name} as it has reached the stop day.")
        await context.bot.send_message(
            chat_id=data['chat_id'], 
            text=f"Recurring email to *{escape_markdown(data['to'], version=2)}* has now stopped as scheduled\.", 
            parse_mode=ParseMode.MARKDOWN_V2
        )
        job.schedule_removal()
        return

    logger.info(f"Running recurring email job {job.name} to {data['to']}")
    msg = EmailMessage()
    msg.set_content(data['body'])
    msg['Subject'] = data['subject']
    msg['From'] = GMAIL_ADDRESS
    msg['To'] = data['to'] # This string can contain comma-separated addresses
    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            s.send_message(msg)
        logger.info(f"Successfully resent email via job {job.name}")
    except Exception as e:
        logger.error(f"Failed to send email via job {job.name}: {e}")
        await context.bot.send_message(
            chat_id=data['chat_id'], 
            text=f"⚠️ Failed to send recurring email to *{escape_markdown(data['to'], version=2)}*\.", 
            parse_mode=ParseMode.MARKDOWN_V2
        )

async def handle_resend_interval_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the user's selection of the resend interval."""
    query = update.callback_query
    await query.answer()

    _, interval_str, job_id = query.data.split(':', 2)
    interval = int(interval_str)

    if job_id not in context.chat_data:
        await query.edit_message_text("This request has expired. Please try sending the email again.")
        return

    context.chat_data[job_id]['interval'] = interval

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    keyboard = []
    days_buttons = [InlineKeyboardButton(day, callback_data=f"resend_stop:{i}:{job_id}") for i, day in enumerate(days)]
    keyboard.append(days_buttons[:4])
    keyboard.append(days_buttons[4:])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        f"Great\! The email will be resent every {interval // 60} minutes\.\n\nOn which day should it stop?", 
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN_V2
    )

async def handle_resend_stop_day_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the user's selection of the stop day and schedules the job."""
    query = update.callback_query
    await query.answer()

    _, stop_day_index_str, job_id = query.data.split(':', 2)
    stop_day_index = int(stop_day_index_str)

    if job_id not in context.chat_data or 'interval' not in context.chat_data.get(job_id, {}):
        await query.edit_message_text("This request has expired or is incomplete. Please try sending the email again.")
        return
        
    email_data = context.chat_data.pop(job_id)
    
    job_data = {
        'chat_id': query.message.chat_id,
        'to': email_data['to'],
        'subject': email_data['subject'],
        'body': email_data['body'],
        'stop_day_index': stop_day_index
    }
    
    interval = email_data['interval']
    
    context.job_queue.run_repeating(
        resend_email_job,
        interval=interval,
        first=interval,
        data=job_data,
        name=f"email_{job_id}"
    )
    
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    stop_day_name = days[stop_day_index]
    
    await query.edit_message_text(
        f"✅ All set\! I will resend the email to *{escape_markdown(email_data['to'], version=2)}* every {interval // 60} minutes\. This will stop on *{stop_day_name}*\.", 
        parse_mode=ParseMode.MARKDOWN_V2
    )

async def handle_resend_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Cancels the resend setup."""
    query = update.callback_query
    await query.answer()
    
    try:
        _, job_id = query.data.split(':', 1)
        if job_id in context.chat_data:
            del context.chat_data[job_id]
    except (ValueError, KeyError):
        pass
        
    await query.edit_message_text("Okay, no recurring email will be sent.")

# --- END of recurring email functions ---

async def movie_command(update: Update, context: ContextTypes.DEFAULT_TYPE, title: str = None) -> None:
    if not TMDB_API_KEY:
        await update.message.reply_text("Movie search service is not configured.")
        return
    if not title:
        title = " ".join(context.args)
        if not title:
            await update.message.reply_text("Please provide a movie title. Usage: `/movie The Matrix`")
            return
    feedback = await update.message.reply_text(f"Searching for '{title}'...")
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": title}
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json().get('results')
        if not results:
            await feedback.edit_text("Sorry, I couldn't find any movie with that title.")
            return
        movie = results[0]
        caption = (
            f"🎬 *{escape_markdown(movie.get('title', 'N/A'), version=2)} ({movie.get('release_date', '----').split('-')[0]})*\n\n"
            f"⭐ *Rating:* {movie.get('vote_average', 0):.1f}/10\n\n"
            f"_{escape_markdown(movie.get('overview', 'No summary available.'), version=2)}_"
        )
        poster_path = movie.get('poster_path')
        if poster_path:
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=poster_url, caption=caption, parse_mode=ParseMode.MARKDOWN_V2)
            await feedback.delete()
        else:
            await feedback.edit_text(caption, parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e:
        logger.error(f"Movie command failed: {e}")
        await feedback.edit_text("An unexpected error occurred.")

async def screenshot_command(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str = None) -> None:
    if not SCREENSHOT_API_KEY:
        await update.message.reply_text("Screenshot service is not configured.")
        return
    if not url:
        url = " ".join(context.args)
        if not url:
            await update.message.reply_text("Please provide a URL. Usage: `/screenshot google.com`")
            return
    if not url.startswith('http'):
        url = 'http://' + url
    feedback = await update.message.reply_text(f"Capturing screenshot for `{url}`...")
    try:
        api_url = f"https://shot.screenshotapi.net/screenshot?token={SCREENSHOT_API_KEY}&url={url}&full_page=true&fresh=true&output=image&file_type=png"
        response = requests.get(api_url, timeout=45)
        if response.status_code == 200:
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=response.content, caption=f"Screenshot of `{url}`", parse_mode=ParseMode.MARKDOWN)
            await feedback.delete()
        else:
            await feedback.edit_text("Sorry, failed to capture the screenshot.")
    except Exception as e:
        logger.error(f"Screenshot command failed: {e}")
        await feedback.edit_text("An unexpected error occurred.")

async def read_text_from_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message.reply_to_message or not update.message.reply_to_message.photo:
        await update.message.reply_text("Please reply to an image with /readtext.")
        return
    feedback = await update.message.reply_text("Reading text from image...")
    try:
        photo_file = await update.message.reply_to_message.photo[-1].get_file()
        image = Image.open(io.BytesIO(await photo_file.download_as_bytearray()))
        text = await asyncio.to_thread(pytesseract.image_to_string, image)
        await feedback.edit_text(f"**Extracted Text:**\n\n{text}" if text.strip() else "Couldn't find any readable text.")
    except Exception as e:
        logger.error(f"OCR Error: {e}")
        await feedback.edit_text("Sorry, an error occurred while processing the image.")

async def summarize_url(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str) -> None:
    if not gemini_model:
        await update.message.reply_text("AI summarizer is not configured.")
        return
    feedback = await update.message.reply_text("Analyzing link...")
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
        article_text = ' '.join(p.get_text() for p in soup.find_all('p'))
        if len(article_text) < 100:
            await feedback.edit_text("Couldn't extract enough text to summarize.")
            return
        await feedback.edit_text("Content extracted. Summarizing with AI...")
        prompt = f"Please provide a concise but comprehensive summary of the following article text:\n\n{article_text[:15000]}"
        ai_response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        await feedback.edit_text(ai_response.text, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Summarize URL error: {e}")
        await feedback.edit_text("Sorry, I couldn't read or summarize that URL.")

async def summarize_file_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not gemini_model:
        await update.message.reply_text("AI service is not configured.")
        return
    if not update.message.reply_to_message:
        await update.message.reply_text("Please reply to an image or a PDF file with /summarize_file.")
        return
    replied_message = update.message.reply_to_message
    feedback = await replied_message.reply_text("Processing file...")
    try:
        text_to_summarize = ""
        if replied_message.photo:
            photo_file = await replied_message.photo[-1].get_file()
            photo_bytes = await photo_file.download_as_bytearray()
            image_part = {"mime_type": "image/jpeg", "data": photo_bytes}
            response = await asyncio.to_thread(gemini_model.generate_content, ["Describe this image in detail.", image_part])
            text_to_summarize = response.text
        elif replied_message.document and replied_message.document.mime_type == 'application/pdf':
            pdf_file = await replied_message.document.get_file()
            pdf_bytes = await pdf_file.download_as_bytearray()
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                text_to_summarize = "".join(page.get_text() for page in doc)
            if not text_to_summarize.strip():
                await feedback.edit_text("Could not extract any text from this PDF.")
                return
            prompt = f"Please provide a detailed summary of the following document:\n\n{text_to_summarize[:15000]}"
            response = await asyncio.to_thread(gemini_model.generate_content, prompt)
            text_to_summarize = response.text
        else:
            await feedback.edit_text("This command only works on an image or PDF file.")
            return
        await feedback.edit_text(f"**Summary:**\n\n{text_to_summarize}", parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"File summarization error: {e}")
        await feedback.edit_text("Sorry, an error occurred while processing the file.")

async def create_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE, prompt: str = None) -> None:
    if not openai_client:
        await update.message.reply_text("Image generation service (OpenAI) is not configured.")
        return
    if not prompt:
        prompt = " ".join(context.args)
    if not prompt:
        await update.message.reply_text("Please describe the image you want to create.")
        return
    feedback = await update.message.reply_text("Creating your image with DALL-E 3...")
    try:
        response = await openai_client.images.generate(model="dall-e-3", prompt=prompt, n=1, size="1024x1024", quality="standard")
        await context.bot.send_photo(update.effective_chat.id, photo=response.data[0].url, caption=f"Creation: `{prompt}`", parse_mode=ParseMode.MARKDOWN)
        await feedback.delete()
    except Exception as e:
        logger.error(f"DALL-E 3 API error: {e}")
        await feedback.edit_text("Sorry, I couldn't create the image.")

async def upscale_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not REPLICATE_API_TOKEN:
        await update.message.reply_text("Image upscaling service (Replicate) is not configured.")
        return
    if not update.message.reply_to_message or not update.message.reply_to_message.photo:
        await update.message.reply_text("Please reply to an image with the `/upscale` command.")
        return
    feedback = await update.message.reply_text("🚀 Sending image to Replicate for upscaling...")
    try:
        photo_file = await update.message.reply_to_message.photo[-1].get_file()
        image_bytes = await photo_file.download_as_bytearray()
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:image/png;base64,{b64_image}"
        start_response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers={"Authorization": f"Token {REPLICATE_API_TOKEN}", "Content-Type": "application/json"},
            json={
                "version": "42fed1c4974146d4d2414e2be2c5236e7a8c90531b54541756316998143e4034",
                "input": {"img": data_uri, "scale": 4},
            },
        )
        start_response.raise_for_status()
        prediction_url = start_response.json()["urls"]["get"]
        await feedback.edit_text("⏳ Upscaling job started... this may take a minute.")
        result_data = {}
        for _ in range(60):
            await asyncio.sleep(2)
            poll_response = requests.get(prediction_url, headers={"Authorization": f"Token {REPLICATE_API_TOKEN}"})
            poll_response.raise_for_status()
            result_data = poll_response.json()
            if result_data["status"] == "succeeded":
                break
            elif result_data["status"] in ["failed", "canceled"]:
                raise Exception(f"Replicate job failed: {result_data.get('error', 'Unknown')}")
        if result_data.get("status") != "succeeded" or not result_data.get("output"):
            raise Exception("Upscaling job timed out or did not succeed.")
        final_image_url = result_data["output"]
        await context.bot.send_document(
            chat_id=update.effective_chat.id,
            document=final_image_url,
            filename='upscaled_replicate.png',
            caption='✨ Here is your upscaled image, powered by Replicate!'
        )
        await feedback.delete()
    except Exception as e:
        logger.error(f"Upscale command error: {e}")
        await feedback.edit_text(f"An unexpected error occurred: {e}")

async def animate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not STABILITY_API_KEY:
        await update.message.reply_text("Video animation service not configured.")
        return
    if not update.message.reply_to_message or not update.message.reply_to_message.photo:
        await update.message.reply_text("Please reply to an image with /animate.")
        return
    feedback = await update.message.reply_text("Sending image to animation engine...")
    try:
        photo_file = await update.message.reply_to_message.photo[-1].get_file()
        response = requests.post(
            "https://api.stability.ai/v2/generation/image-to-video",
            headers={"authorization": f"Bearer {STABILITY_API_KEY}"},
            files={"image": await photo_file.download_as_bytearray()},
            data={"motion_bucket_id": 40},
            timeout=30,
        )
        response.raise_for_status()
        generation_id = response.json()["id"]
        await feedback.edit_text("Animation started. This may take a minute...")
        for _ in range(45):
            await asyncio.sleep(4)
            res = requests.get(
                f"https://api.stability.ai/v2/generation/image-to-video/result/{generation_id}",
                headers={'authorization': f"Bearer {STABILITY_API_KEY}", 'accept': "video/mp4"},
                timeout=20,
            )
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
        await update.message.reply_text("Please reply to a video with the `/mp4` command.")
        return
    feedback = await update.message.reply_text("Downloading video...")
    temp_dir = os.path.join(DOWNLOAD_DIR, str(uuid.uuid4()))
    os.makedirs(temp_dir)
    try:
        video_file = await update.message.reply_to_message.video.get_file()
        video_path = os.path.join(temp_dir, "video.mp4")
        await video_file.download_to_drive(video_path)
        await feedback.edit_text("Converting to audio...")
        audio_path = os.path.join(temp_dir, "audio.mp3")
        ffmpeg_cmd = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path]
        process = await asyncio.create_subprocess_exec(*ffmpeg_cmd)
        await process.wait()
        if process.returncode != 0:
            await feedback.edit_text("Sorry, the conversion failed.")
            return
        await feedback.edit_text("Uploading audio...")
        with open(audio_path, 'rb') as f:
            await context.bot.send_audio(chat_id=update.effective_chat.id, audio=f)
        await feedback.delete()
    except Exception as e:
        logger.error(f"Error in /mp4 command: {e}")
        await feedback.edit_text("An unexpected error occurred.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

async def tts_command(update: Update, context: ContextTypes.DEFAULT_TYPE, text_to_speak: str = None) -> None:
    text = text_to_speak or " ".join(context.args)
    if not text and update.message.reply_to_message:
        text = update.message.reply_to_message.text
    if not text:
        await update.message.reply_text("Usage: `/tts <text>` or reply to a message.")
        return
    if len(text) > 1000:
        await update.message.reply_text("Text is too long (max 1000 characters).")
        return
    feedback = await update.message.reply_text("Generating audio...")
    temp_audio_path = os.path.join(DOWNLOAD_DIR, f"{uuid.uuid4()}.mp3")
    try:
        await asyncio.to_thread(gTTS(text=text, lang='en').save, temp_audio_path)
        with open(temp_audio_path, 'rb') as f:
            await context.bot.send_audio(chat_id=update.effective_chat.id, audio=f)
        await feedback.delete()
    except Exception as e:
        logger.error(f"gTTS Error: {e}")
        await feedback.edit_text("Sorry, an error occurred while generating the audio.")
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

# --- NEW: Corrected and Improved TikTok Search Functions ---

async def tiktok_search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles the /tiktoksearch command. 
    It extracts the query from the user's message and proceeds to ask for the video count.
    """
    query = " ".join(context.args)
    if not query:
        # If the user just types /tiktoksearch, prompt them for a query via the state handler.
        await prompt_for_input(
            update, 
            context, 
            state='awaiting_tiktok_query', 
            message="What would you like to search for on TikTok?", 
            event="Used /tiktoksearch"
        )
        return
    
    # If a query was provided (e.g., /tiktoksearch funny cats), ask for the count directly.
    await ask_for_tiktok_count(update, context, query)

async def ask_for_tiktok_count(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str) -> None:
    """
    Asks the user how many videos they want to receive for a given search query.
    """
    context.user_data['tiktok_query'] = query
    
    keyboard = [[
        InlineKeyboardButton("3", callback_data="tiktok_count:3"),
        InlineKeyboardButton("5", callback_data="tiktok_count:5"),
        InlineKeyboardButton("10", callback_data="tiktok_count:10")
    ]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(f"How many videos for '{query}'?", reply_markup=reply_markup)

async def handle_tiktok_count_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles the button press for the video count, performs the API call, and sends the videos.
    """
    cb_query = update.callback_query
    await cb_query.answer()
    
    count = int(cb_query.data.split(":")[1])
    search_query = context.user_data.pop('tiktok_query', None)
    
    if not search_query:
        await cb_query.edit_message_text("This search has expired. Please try again.")
        return

    await cb_query.edit_message_text(f"Searching TikTok for '{search_query}'...")
    
    try:
        response = requests.post(
            'https://tikwm.com/api/feed/search',
            headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"},
            data={'keywords': search_query, 'count': '20'},
            timeout=20,
        )
        response.raise_for_status()
        videos = response.json().get('data', {}).get('videos', [])
        
        if not videos:
            await cb_query.edit_message_text("No TikTok videos found for that search.")
            return

        await cb_query.edit_message_text(f"Found {len(videos)} videos. Sending the top {count} of them...")
        
        random.shuffle(videos)
        sent_count = 0
        for video in videos:
            if sent_count >= count:
                break
            if video.get('play'):
                try:
                    await context.bot.send_video(
                        chat_id=cb_query.message.chat_id, 
                        video=video['play'], 
                        caption=video.get('title', '')
                    )
                    sent_count += 1
                except Exception as send_error:
                    logger.error(f"Failed to send a TikTok video: {send_error}")
        
        if sent_count == 0:
             await cb_query.edit_message_text("Found videos, but was unable to send any of them. They may be protected.")

    except Exception as e:
        logger.error(f"TikTok search failed: {e}")
        await cb_query.edit_message_text("An unexpected error occurred while searching TikTok.")
        
# --- END NEW TikTok Functions ---

async def youtube_command(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str = None) -> None:
    if not query:
        query = " ".join(context.args)
        if not query:
            await update.message.reply_text("Please provide a search term.")
            return
    feedback = await update.message.reply_text(f"Searching YouTube for '{query}'...")
    try:
        ydl_opts = {'noplaylist': True, 'quiet': True, 'default_search': 'ytsearch5'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)
        if not info.get('entries'):
            await feedback.edit_text("Sorry, couldn't find any results.")
            return
        keyboard = [
            [InlineKeyboardButton(
                (v['title'][:60] + '..') if len(v['title']) > 60 else v['title'],
                callback_data=f"play_confirm:{v['id']}"
            )] for v in info['entries']
        ]
        await feedback.edit_text("Top 5 results:", reply_markup=InlineKeyboardMarkup(keyboard))
    except Exception as e:
        logger.error(f"YouTube error: {e}")
        await feedback.edit_text("An error occurred during the search.")

async def play_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    song_name = " ".join(context.args)
    if not song_name:
        await update.message.reply_text("Please provide a song name.")
        return
    await search_and_play_song(update, context, song_name)

async def search_and_play_song(update: Update, context: ContextTypes.DEFAULT_TYPE, song_name: str) -> None:
    msg_obj = update.callback_query.message if update.callback_query else update.message
    feedback = await msg_obj.reply_text(f"Searching for '{song_name}'...")
    try:
        ydl_opts = {'noplaylist': True, 'quiet': True, 'default_search': 'ytsearch1'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(song_name, download=False)
        if not info.get('entries'):
            await feedback.edit_text("Sorry, couldn't find any results.")
            return
        video = info['entries'][0]
        keyboard = [
            [InlineKeyboardButton("Yes, that's it!", callback_data=f"play_confirm:{video['id']}")],
            [InlineKeyboardButton("No, cancel", callback_data="play_cancel")]
        ]
        await feedback.edit_text(f"Found: **{video['title']}**\n\nIs this correct?", reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Play search error: {e}")
        await feedback.edit_text("An error occurred while searching.")

async def handle_play_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    video_id = query.data.split(":")[1]
    keyboard = [[
        InlineKeyboardButton("Audio", callback_data=f"dl_audio:{video_id}"),
        InlineKeyboardButton("Video", callback_data=f"dl_video:{video_id}")
    ]]
    await query.edit_message_text("Choose your format:", reply_markup=InlineKeyboardMarkup(keyboard))

async def handle_audio_download(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    video_id = query.data.split(":")[1]
    temp_dir = os.path.join(DOWNLOAD_DIR, str(uuid.uuid4()))
    os.makedirs(temp_dir)
    await query.edit_message_text("Downloading audio...")
    try:
        audio_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
            'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
            'noplaylist': True,
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(audio_opts) as ydl:
            info = ydl.extract_info(video_id, download=True)
        audio_path = os.path.join(temp_dir, os.listdir(temp_dir)[0])
        await query.edit_message_text("Sending audio...")
        with open(audio_path, 'rb') as f:
            await context.bot.send_audio(
                chat_id=query.message.chat_id,
                audio=f,
                title=info.get('title'),
                duration=info.get('duration')
            )
        await query.delete_message()
    except Exception as e:
        logger.error(f"Audio download error: {e}")
        await query.edit_message_text("An error occurred during download.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

async def handle_video_download(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    video_id = query.data.split(":")[1]
    temp_dir = os.path.join(DOWNLOAD_DIR, str(uuid.uuid4()))
    os.makedirs(temp_dir)
    await query.edit_message_text("Checking video details...")
    try:
        ydl_opts = {'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_id, download=False)
        filesize_mb = (info.get('filesize') or info.get('filesize_approx', 0)) / (1024 * 1024)
        if filesize_mb <= 49:
            await query.edit_message_text(f"Downloading video ({filesize_mb:.2f} MB)...")
            ydl_opts['outtmpl'] = os.path.join(temp_dir, '%(title)s.%(ext)s')
            with yt_dlp.YoutubeDL(ydl_opts) as ydl_dl:
                ydl_dl.download([video_id])
            video_path = os.path.join(temp_dir, os.listdir(temp_dir)[0])
            await query.edit_message_text("Sending video...")
            with open(video_path, 'rb') as f:
                await context.bot.send_video(chat_id=query.message.chat_id, video=f, supports_streaming=True)
            await query.delete_message()
        else:
            await query.edit_message_text(f"Video is too large to send ({filesize_mb:.2f} MB).")
    except Exception as e:
        logger.error(f"Video download error: {e}")
        await query.edit_message_text("An error occurred during download.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

async def handle_play_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Search cancelled.")

async def show_download_platform_options(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [[
        InlineKeyboardButton("TikTok", callback_data="dl_platform:TikTok"),
        InlineKeyboardButton("Instagram", callback_data="dl_platform:Instagram")
    ], [
        InlineKeyboardButton("Facebook", callback_data="dl_platform:Facebook"),
        InlineKeyboardButton("YouTube", callback_data="dl_platform:YouTube")
    ]]
    await update.message.reply_text("Choose a platform:", reply_markup=InlineKeyboardMarkup(keyboard))

async def handle_platform_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    platform = query.data.split(":")[1]
    context.user_data['state'] = 'awaiting_download_url'
    context.user_data['platform'] = platform
    await query.edit_message_text(f"Send me the {platform} URL.")

async def download_content_from_url(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str, platform: str) -> None:
    feedback = await update.message.reply_text(f"Starting download from {platform}...")
    temp_dir = os.path.join(DOWNLOAD_DIR, str(uuid.uuid4()))
    os.makedirs(temp_dir)
    try:
        ydl_opts = {
            'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
            'noplaylist': True,
            'quiet': True,
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        file_path = os.path.join(temp_dir, os.listdir(temp_dir)[0])
        await feedback.edit_text("Uploading to Telegram...")
        with open(file_path, 'rb') as f:
            await context.bot.send_video(chat_id=update.effective_chat.id, video=f)
        await feedback.delete()
    except Exception as e:
        logger.error(f"Download error for {url}: {e}")
        await feedback.edit_text("Download failed. The link may be private or invalid.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

async def get_joke(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        response = requests.get("https://v2.jokeapi.dev/joke/Any?blacklistFlags=nsfw,racist,sexist&type=twopart", timeout=5).json()
        if not response.get('error'):
            await update.message.reply_text(f"{response['setup']}\n\n...{response['delivery']}")
    except Exception:
        await update.message.reply_text("Sorry, couldn't get a joke.")

async def get_riddle(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        response = requests.get("https://riddles-api.vercel.app/random", timeout=7).json()
        await update.message.reply_text(f"Riddle:\n*_{response['riddle']}_*", parse_mode=ParseMode.MARKDOWN)
        await asyncio.sleep(8)
        await update.message.reply_text(f"**Answer:**\n||{escape_markdown(response['answer'], version=2)}||", parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e:
        logger.error(f"Riddle API error: {e}")
        await update.message.reply_text("Sorry, the riddle service is unavailable.")

async def get_crypto_prices(update: Update, context: ContextTypes.DEFAULT_TYPE, crypto_ids: str) -> None:
    ids = ','.join(s.strip().lower() for s in crypto_ids.split(','))
    try:
        prices = requests.get(f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd,ngn", timeout=5).json()
        if not prices:
            await update.message.reply_text("Couldn't find prices.")
            return
        message = "**Latest Crypto Prices:**\n" + "\n".join(f"• **{c.title()}**: ${d.get('usd', 0):,.2f} / ₦{d.get('ngn', 0):,.2f}" for c, d in prices.items())
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text("Sorry, couldn't fetch crypto prices.")

async def get_weather(update: Update, context: ContextTypes.DEFAULT_TYPE, city: str) -> None:
    if not OPENWEATHER_API_KEY:
        await update.message.reply_text("Weather service not configured.")
        return
    try:
        data = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric", timeout=5).json()
        if data.get("cod") != 200:
            await update.message.reply_text(f"Sorry, couldn't find city '{city}'.")
            return
        message = f"**Weather in {data['name']}**: {data['weather'][0]['description'].title()} at {data['main']['temp']}°C"
        await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text("Sorry, couldn't fetch the weather.")

async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE, text_to_translate: str = None) -> None:
    if not gemini_model:
        await update.message.reply_text("Translate service is not configured.")
        return
    text = text_to_translate or " ".join(context.args)
    if not text:
        await update.message.reply_text("Format: `<language> <text>`")
        return
    target_lang, text_to_trans = text.split(" ", 1)
    prompt = f"Translate the following text to {target_lang}: {text_to_trans}"
    response = await asyncio.to_thread(gemini_model.generate_content, prompt)
    await update.message.reply_text(response.text)

async def ping_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    webhook_url = context.job.data.get("webhook_url")
    if not webhook_url:
        logger.warning("Webhook URL not found for ping job.")
        return
    try:
        await asyncio.to_thread(requests.get, webhook_url, timeout=20)
        logger.info(f"Auto-ping successful.")
    except Exception as e:
        logger.warning(f"Auto-ping failed: {e}")

async def novel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await prompt_for_input(update, context, 'awaiting_novel_title', "What novel would you like to search for?", "Pressed 'Download Novel'")
        return
    query = " ".join(context.args)
    await search_for_novel(update.message, context, query)

async def search_for_novel(message, context: ContextTypes.DEFAULT_TYPE, query: str) -> None:
    feedback = await message.reply_text(f"📚 Searching for '{query}'...")
    try:
        search_url = f"https://www.pdfroom.com/search?q={requests.utils.quote(query)}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(search_url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        results = soup.find_all('div', class_='book-card', limit=5)
        if not results:
            await feedback.edit_text("Sorry, I couldn't find any novels matching that title.")
            return
        keyboard = []
        for book in results:
            title = book.find('h5').get_text(strip=True)
            book_page_url = f"https://www.pdfroom.com{book.find('a')['href']}"
            button_text = (title[:60] + '..') if len(title) > 60 else title
            keyboard.append([InlineKeyboardButton(button_text, callback_data=f"novel_dl:{book_page_url}")])
        await feedback.edit_text("Top results. Choose one to download:", reply_markup=InlineKeyboardMarkup(keyboard))
    except Exception as e:
        logger.error(f"Novel search failed: {e}")
        await feedback.edit_text("An error occurred during the search.")

async def handle_novel_download(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    book_page_url = query.data.split(":", 1)[1]
    feedback = await query.edit_message_text("📥 Preparing download...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(book_page_url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        download_link = f"https://www.pdfroom.com{soup.find('a', id='download-button')['href']}"
        await feedback.edit_text("Downloading PDF... this may take a while.")
        pdf_response = requests.get(download_link, headers=headers, timeout=120)
        pdf_response.raise_for_status()
        filename = download_link.split('/')[-1] or "novel.pdf"
        if len(pdf_response.content) > 50 * 1024 * 1024:
            await feedback.edit_text(f"File is too large for Telegram. Download directly: {download_link}")
            return
        await context.bot.send_document(
            chat_id=query.message.chat_id,
            document=pdf_response.content,
            filename=filename,
            caption="Here is your novel!"
        )
        await feedback.delete()
    except Exception as e:
        logger.error(f"Novel download failed: {e}")
        await feedback.edit_text(f"An error occurred during download: {e}")

async def prompt_for_input(update: Update, context: ContextTypes.DEFAULT_TYPE, state: str, message: str, event: str) -> None:
    await save_user_to_db(update, context, event_type=event)
    context.user_data['state'] = state
    await update.message.reply_text(message)

async def record_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text: return
    state = context.user_data.get('state')
    if not state:
        await save_user_to_db(update, context)
        return
    text = update.message.text
    if state == 'continuous_chat':
        await gemini_command(update, context, prompt_text=text)
        return
    popped_state = context.user_data.pop('state')
    if popped_state == 'awaiting_email_address':
        context.user_data['email_to'] = text
        context.user_data['state'] = 'awaiting_email_subject'
        await update.message.reply_text("Step 2 of 3: What's the subject?")
    elif popped_state == 'awaiting_email_subject':
        context.user_data['email_subject'] = text
        context.user_data['state'] = 'awaiting_email_body'
        await update.message.reply_text("Step 3 of 3: What's the message?")
    elif popped_state == 'awaiting_email_body':
        to, subject = context.user_data.pop('email_to'), context.user_data.pop('email_subject')
        feedback = await update.message.reply_text(f"Sending email to {to}...")
        msg = EmailMessage()
        msg.set_content(text)
        msg['Subject'], msg['From'], msg['To'] = subject, GMAIL_ADDRESS, to
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
                s.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
                s.send_message(msg)
            await feedback.edit_text("Email sent successfully!")

            # --- Recurring Email Feature ---
            job_id = str(uuid.uuid4())
            context.chat_data[job_id] = {'to': to, 'subject': subject, 'body': text}

            keyboard = [
                [
                    InlineKeyboardButton("5 mins", callback_data=f"resend_interval:300:{job_id}"),
                    InlineKeyboardButton("10 mins", callback_data=f"resend_interval:600:{job_id}"),
                    InlineKeyboardButton("15 mins", callback_data=f"resend_interval:900:{job_id}"),
                ],
                [InlineKeyboardButton("No, thanks", callback_data=f"resend_cancel:{job_id}")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                "Would you like to resend this email periodically?",
                reply_markup=reply_markup
            )

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            await feedback.edit_text("Failed to send email.")
    else:
        state_handlers = {
            'awaiting_gemini_prompt': lambda: gemini_command(update, context, prompt_text=text),
            'awaiting_create_prompt': lambda: create_image_command(update, context, prompt=text),
            'awaiting_novel_title': lambda: search_for_novel(update.message, context, query=text),
            'awaiting_song_name': lambda: search_and_play_song(update, context, song_name=text),
            'awaiting_city': lambda: get_weather(update, context, city=text),
            'awaiting_crypto_symbols': lambda: get_crypto_prices(update, context, crypto_ids=text),
            'awaiting_summary_url': lambda: summarize_url(update, context, url=text),
            'awaiting_translation_text': lambda: translate_command(update, context, text_to_translate=text),
            'awaiting_download_url': lambda: download_content_from_url(update, context, url=text, platform=context.user_data.pop('platform', 'unknown')),
            'awaiting_screenshot_url': lambda: screenshot_command(update, context, url=text),
            'awaiting_movie_title': lambda: movie_command(update, context, title=text),
            'awaiting_tts_text': lambda: tts_command(update, context, text_to_speak=text),
            'awaiting_ytsearch_query': lambda: youtube_command(update, context, query=text),
            'awaiting_suggestion': lambda: handle_suggestion(update, context, suggestion=text),
            'awaiting_db_table_name': lambda: view_db_table(update, context, table_name=text),
            # NEW: This line fixes the TikTok search loop by wiring the user's text input to the next step.
            'awaiting_tiktok_query': lambda: ask_for_tiktok_count(update, context, query=text),
        }
        if popped_state in state_handlers:
            await state_handlers[popped_state]()

async def handle_suggestion(update: Update, context: ContextTypes.DEFAULT_TYPE, suggestion: str):
    user = update.effective_user
    admin_message = f"📩 Suggestion from {user.first_name} (`{user.id}`):\n\n{suggestion}"
    await context.bot.send_message(chat_id=ADMIN_ID, text=admin_message, parse_mode=ParseMode.MARKDOWN)
    await update.message.reply_text("Thank you! Your suggestion has been sent.")

def main() -> None:
    application = Application.builder().token(BOT_TOKEN).rate_limiter(AIORateLimiter()).job_queue(JobQueue()).build()
    
    cmd_handlers = [
        CommandHandler("start", start), CommandHandler("help", help_command),
        CommandHandler("gemini", gemini_command), CommandHandler("create", create_image_command),
        CommandHandler("upscale", upscale_image_command), CommandHandler("animate", animate_command),
        CommandHandler("summarize_file", summarize_file_command), CommandHandler("readtext", read_text_from_image_command),
        CommandHandler("play", play_command), CommandHandler("mp4", convert_video_to_audio),
        CommandHandler("novel", novel_command), CommandHandler("riddle", get_riddle), 
        CommandHandler("gmail", gmail_command), CommandHandler("screenshot", screenshot_command),
        CommandHandler("movie", movie_command), CommandHandler("tts", tts_command),
        CommandHandler("tiktoksearch", tiktok_search_command), CommandHandler("ytsearch", youtube_command),
        CommandHandler("db", db_command)
    ]
    
    menu_button_texts = {
        "AI Tools": show_ai_tools_menu, "Media Tools": show_media_tools_menu,
        "Utilities": show_utilities_menu, "Help": help_command,
        "Back to Main Menu": start, "End Chat": end_chat,
        "Chat with AI": start_ai_chat, "Tell a Joke": get_joke, "Ask a Riddle": get_riddle,
        "Send Suggestion": lambda u,c: prompt_for_input(u,c,'awaiting_suggestion', "Please type your suggestion...", "Pressed 'Suggestion'"),
        "Create Image": lambda u,c: prompt_for_input(u,c,'awaiting_create_prompt', "Describe the image...", "Pressed 'Create Image'"),
        "Read Text from Image": lambda u,c: u.message.reply_text("Reply to an image with /readtext."),
        "Text to Speech": lambda u,c: prompt_for_input(u,c,'awaiting_tts_text', "What text to speak?", "Pressed 'TTS'"),
        "Upscale Image": lambda u,c: u.message.reply_text("Reply to an image with /upscale."),
        "Animate Image": lambda u,c: u.message.reply_text("Reply to an image with /animate."),
        "Summarize File": lambda u,c: u.message.reply_text("Reply to an image or PDF with /summarize_file."),
        "Summarize Link": lambda u,c: prompt_for_input(u,c,'awaiting_summary_url', "Send the article link.", "Pressed 'Summarize Link'"),
        "Play Music / Video": lambda u,c: prompt_for_input(u,c,'awaiting_song_name', "What song or video?", "Pressed 'Play'"),
        "Download Novel": lambda u,c: prompt_for_input(u,c,'awaiting_novel_title', "What novel to search for?", "Pressed 'Download Novel'"),
        "Download Media": show_download_platform_options,
        "Search Movie": lambda u,c: prompt_for_input(u,c,'awaiting_movie_title', "What movie?", "Pressed 'Search Movie'"),
        "Search TikTok": lambda u,c: prompt_for_input(u,c,'awaiting_tiktok_query', "Search TikTok for?", "Pressed 'Search TikTok'"),
        "Youtube": lambda u,c: prompt_for_input(u,c,'awaiting_ytsearch_query', "Search YouTube for?", "Pressed 'Youtube'"),
        "Weather": lambda u,c: prompt_for_input(u,c,'awaiting_city', "Enter a city name.", "Pressed 'Weather'"),
        "Crypto Prices": lambda u,c: prompt_for_input(u,c,'awaiting_crypto_symbols', "Enter coin IDs (e.g., bitcoin,ethereum).", "Pressed 'Crypto'"),
        "Translate Text": lambda u,c: prompt_for_input(u,c,'awaiting_translation_text', "Format: <language> <text>", "Pressed 'Translate'"),
        "Convert Video to Audio": lambda u,c: u.message.reply_text("Reply to a video with /mp4."),
        "Take Screenshot": lambda u,c: prompt_for_input(u,c,'awaiting_screenshot_url', "Enter the website URL.", "Pressed 'Take Screenshot'"),
        "Send Email (Admin)": gmail_command,
    }
    
    msg_handlers = [MessageHandler(filters.TEXT & filters.Regex(f"^{re.escape(p)}$"), f) for p, f in menu_button_texts.items()]
    
    callback_handlers = [
        CallbackQueryHandler(handle_play_confirmation, pattern="^play_confirm:"),
        CallbackQueryHandler(handle_play_cancel, pattern="^play_cancel"),
        CallbackQueryHandler(handle_audio_download, pattern="^dl_audio:"),
        CallbackQueryHandler(handle_video_download, pattern="^dl_video:"),
        CallbackQueryHandler(handle_platform_selection, pattern="^dl_platform:"),
        CallbackQueryHandler(handle_tiktok_count_selection, pattern="^tiktok_count:"),
        CallbackQueryHandler(handle_novel_download, pattern="^novel_dl:"),
        # --- HANDLERS for recurring email ---
        CallbackQueryHandler(handle_resend_interval_selection, pattern="^resend_interval:"),
        CallbackQueryHandler(handle_resend_stop_day_selection, pattern="^resend_stop:"),
        CallbackQueryHandler(handle_resend_cancel, pattern="^resend_cancel:"),
    ]

    application.add_handlers(cmd_handlers)
    application.add_handlers(msg_handlers)
    application.add_handlers(callback_handlers)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, record_user_message))
    
    PORT = int(os.environ.get("PORT", 8443))
    RENDER_APP_NAME = os.environ.get("RENDER_APP_NAME")

    if not RENDER_APP_NAME:
        logger.info("Running in polling mode.")
        application.run_polling()
    else:
        WEBHOOK_URL = f"https://{RENDER_APP_NAME}.onrender.com/{BOT_TOKEN}"
        logger.info(f"Running in webhook mode. URL: {WEBHOOK_URL}")
        job_queue = application.job_queue
        job_queue.run_repeating(callback=ping_job, interval=600, first=10, name="auto_ping", data={"webhook_url": WEBHOOK_URL})
        logger.info("Auto-ping job scheduled.")
        application.run_webhook(listen="0.0.0.0", port=PORT, url_path=BOT_TOKEN, webhook_url=WEBHOOK_URL)

if __name__ == "__main__":
    main()
