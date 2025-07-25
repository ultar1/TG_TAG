import logging
import os
import sys
import asyncio
import uuid # For unique filenames
import shutil # For removing directories
import datetime # Import for timestamp in notification
import time # Import for time-based notification throttling

from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler
from telegram.constants import ParseMode
from sqlalchemy import create_engine, Column, String, UniqueConstraint
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.types import BigInteger
import yt_dlp

# --- Configuration ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Environment Variables ---
BOT_TOKEN = os.environ.get("BOT_TOKEN")
DATABASE_URL = os.environ.get("DATABASE_URL")
ADMIN_ID = os.environ.get("ADMIN_ID")

# Define a temporary directory for downloads
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Telegram's direct video upload limit (50 MB)
TELEGRAM_VIDEO_LIMIT_MB = 50
TELEGRAM_VIDEO_LIMIT_BYTES = TELEGRAM_VIDEO_LIMIT_MB * 1024 * 1024

# Admin Notification Throttling
ADMIN_NOTIFICATION_COOLDOWN = 300 # seconds (5 minutes)
last_admin_notification_time = {}

# --- Initial Checks ---
if not BOT_TOKEN:
    logger.critical("BOT_TOKEN environment variable not set. Exiting.")
    sys.exit(1)

if not DATABASE_URL:
    logger.critical("DATABASE_URL environment variable not set. Exiting.")
    sys.exit(1)
    
if not ADMIN_ID:
    logger.critical("ADMIN_ID environment variable not set. Exiting.")
    sys.exit(1)
else:
    try:
        ADMIN_ID = int(ADMIN_ID)
    except ValueError:
        logger.critical("ADMIN_ID is not a valid integer. Exiting.")
        sys.exit(1)

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# --- Database Setup ---
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    user_id = Column(BigInteger, primary_key=True, nullable=False)
    first_name = Column(String, nullable=True)
    username = Column(String, nullable=True)
    chat_id = Column(BigInteger, primary_key=True, nullable=False)

    def __repr__(self):
        return (f"<User(id={self.user_id}, first_name='{self.first_name}', "
                f"username='{self.username}', chat_id={self.chat_id})>")

# Use a smaller pool size for free-tier databases
engine = create_engine(DATABASE_URL, pool_size=5, max_overflow=10)

try:
    Base.metadata.create_all(engine)
    logger.info("Database tables checked/created successfully.")
except OperationalError as e:
    logger.critical(f"Failed to connect to database or create tables: {e}. Exiting.")
    sys.exit(1)

Session = sessionmaker(bind=engine)

# --- Constants ---
MAX_MENTIONS_PER_MESSAGE = 50
MAX_MESSAGE_LENGTH = 4096

# --- Helper Functions ---
def escape_markdown_v2(text: str) -> str:
    """Escapes common MarkdownV2 special characters."""
    if not isinstance(text, str):
        return ""
    special_chars = r'_*[]()~`>#+-=|{}.!'
    escaped_text = text
    for char in special_chars:
        escaped_text = escaped_text.replace(char, f'\\{char}')
    return escaped_text

async def send_notification_to_admin(context: ContextTypes.DEFAULT_TYPE, user_info: dict, event_type: str) -> None:
    """Sends a notification message to the admin."""
    user_id = user_info.get('user_id', 'N/A')
    
    current_time = time.time()
    if user_id in last_admin_notification_time and \
       (current_time - last_admin_notification_time[user_id]) < ADMIN_NOTIFICATION_COOLDOWN:
        logger.info(f"Admin notification for user {user_id} throttled.")
        return

    first_name = user_info.get('first_name', 'N/A')
    username = user_info.get('username', 'N/A')
    chat_id = user_info.get('chat_id', 'N/A')
    chat_type = user_info.get('chat_type', 'N/A')
    message_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    escaped_event_type = escape_markdown_v2(event_type)
    escaped_first_name = escape_markdown_v2(first_name)
    escaped_username = escape_markdown_v2(f"@{username}") if username != 'N/A' else 'N/A'
    escaped_chat_type = escape_markdown_v2(chat_type)
    
    notification_message = (
        f"🔔 *New User Interaction* 🔔\n\n"
        f"*Event*: {escaped_event_type}\n"
        f"*User ID*: `{user_id}`\n"
        f"*First Name*: {escaped_first_name}\n"
        f"*Username*: {escaped_username}\n"
        f"*Chat ID*: `{chat_id}`\n"
        f"*Chat Type*: {escaped_chat_type}\n"
        f"*Time*: `{escape_markdown_v2(message_time)}`"
    )
    
    try:
        await context.bot.send_message(
            chat_id=ADMIN_ID,
            text=notification_message,
            parse_mode=ParseMode.MARKDOWN_V2
        )
        last_admin_notification_time[user_id] = current_time
    except Exception as e:
        logger.error(f"Failed to send admin notification for user {user_id}: {e}")

async def save_user_to_db(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Saves or updates user information in the database."""
    if not update.message or not update.message.from_user:
        return

    user = update.message.from_user
    chat_id = update.message.chat_id
    chat_type = update.message.chat.type

    if chat_type not in ["group", "supergroup", "private"]:
        return

    session = Session()
    try:
        existing_user = session.query(User).filter_by(user_id=user.id, chat_id=chat_id).first()
        user_info = {
            'user_id': user.id, 'first_name': user.first_name,
            'username': user.username, 'chat_id': chat_id, 'chat_type': chat_type
        }

        if not existing_user:
            new_user = User(
                user_id=user.id, first_name=user.first_name,
                username=user.username, chat_id=chat_id
            )
            session.add(new_user)
            session.commit()
            logger.info(f"DB: Added user {user.id} for chat {chat_id}")
            await send_notification_to_admin(context, user_info, "New User Added")
        else:
            if existing_user.first_name != user.first_name or existing_user.username != user.username:
                existing_user.first_name = user.first_name
                existing_user.username = user.username
                session.commit()
                logger.info(f"DB: Updated user {user.id} for chat {chat_id}")
                await send_notification_to_admin(context, user_info, "User Info Updated")
            else:
                await send_notification_to_admin(context, user_info, "User Interacted")
    except Exception as e:
        session.rollback()
        logger.error(f"DB: Error saving user {user.id} for chat {chat_id}: {e}")
    finally:
        session.close()

# --- Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /start command."""
    if update.message and update.message.from_user:
        await save_user_to_db(update, context)
        keyboard = [
            [KeyboardButton("Download Videos/Audio")],
            [KeyboardButton("Help")]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        await update.message.reply_text(
            escape_markdown_v2("Hi! I can download media and tag group members. Use the buttons below to start."),
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=reply_markup
        )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /help command."""
    if update.message and update.message.from_user:
        await save_user_to_db(update, context)
    help_text = (
        "How to use the Bot:\n\n"
        "*To Download Media:*\n"
        "- Tap 'Download Videos/Audio' below.\n"
        "- Choose a platform.\n"
        "- Send me the full URL.\n\n"
        "*To Tag Group Members:*\n"
        "1. Make me an Administrator in your group.\n"
        "2. Turn off Group Privacy via @BotFather so I can see members.\n"
        "3. Reply to any message with `/tag` to mention all known non-admin members."
    )
    await update.message.reply_text(escape_markdown_v2(help_text), parse_mode=ParseMode.MARKDOWN_V2)

async def tag_all(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Tags all known non-admin members in a group."""
    if not update.message.reply_to_message:
        await update.message.reply_text("Please reply to a message with `/tag` to use this command.")
        return
    
    chat_id = update.message.chat_id
    if update.message.chat.type not in ['group', 'supergroup']:
        await update.message.reply_text("This command only works in groups.")
        return

    await save_user_to_db(update, context)
    feedback_message = await update.message.reply_text("Fetching user list...")

    session = Session()
    try:
        all_known_users = session.query(User).filter_by(chat_id=chat_id).all()
        admins = await context.bot.get_chat_administrators(chat_id)
        admin_ids = {admin.user.id for admin in admins}
    except Exception as e:
        logger.error(f"Error fetching users/admins for chat {chat_id}: {e}")
        await feedback_message.edit_text("An error occurred. Make sure I am an admin in this group.")
        return
    finally:
        session.close()

    members_to_tag_links = [
        f"[{escape_markdown_v2(user.first_name or 'User')}](tg://user?id={user.user_id})"
        for user in all_known_users if user.user_id not in admin_ids
    ]

    if not members_to_tag_links:
        await feedback_message.edit_text("No known non-admin members to tag. Users must send a message first.")
        return

    replied_message_text = escape_markdown_v2(update.message.reply_to_message.text or update.message.reply_to_message.caption or "")
    
    # Split mentions into chunks
    for i in range(0, len(members_to_tag_links), MAX_MENTIONS_PER_MESSAGE):
        chunk = members_to_tag_links[i:i + MAX_MENTIONS_PER_MESSAGE]
        message_text = f"{replied_message_text}\n\n" + " ".join(chunk)
        try:
            await context.bot.send_message(
                chat_id=chat_id, text=message_text, parse_mode=ParseMode.MARKDOWN_V2
            )
            await asyncio.sleep(0.5) # Avoid rate limits
        except Exception as e:
            logger.error(f"Failed to send tag message chunk to chat {chat_id}: {e}")
    
    await feedback_message.delete()

# --- Download Flow Handlers ---
async def show_download_options(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows platform selection buttons for downloading."""
    keyboard = [
        [InlineKeyboardButton("TikTok", callback_data="dl:TikTok"), InlineKeyboardButton("Facebook", callback_data="dl:Facebook")],
        [InlineKeyboardButton("Instagram", callback_data="dl:Instagram"), InlineKeyboardButton("YouTube", callback_data="dl:YouTube")],
        [InlineKeyboardButton("Twitter", callback_data="dl:Twitter"), InlineKeyboardButton("SoundCloud", callback_data="dl:SoundCloud")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    message_source = update.callback_query or update.message
    text = "Please choose a platform to download from:"
    
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.edit_message_text(text, reply_markup=reply_markup)
    else:
        await message_source.reply_text(text, reply_markup=reply_markup)

async def handle_download_platform_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles platform selection and prompts for a URL."""
    query = update.callback_query
    await query.answer()
    platform_name = query.data.split(":")[1]
    
    context.user_data['state'] = 'awaiting_url'
    context.user_data['platform'] = platform_name

    await query.edit_message_text(f"Please send me the full URL for the {platform_name} content.")

async def download_content_from_url(update: Update, context: ContextTypes.DEFAULT_TYPE, platform: str, content_url: str) -> None:
    """Downloads content from a given URL using yt-dlp and sends it."""
    temp_dir_name = os.path.join(DOWNLOAD_DIR, str(uuid.uuid4()))
    os.makedirs(temp_dir_name, exist_ok=True)
    
    loading_emojis = ["🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘", "🕙", "🕚", "🕛"]
    loading_message_text = escape_markdown_v2(f"Getting your {platform} content, please wait... ")
    processing_message = await update.message.reply_text(loading_message_text + loading_emojis[0], parse_mode=ParseMode.MARKDOWN_V2)

    animation_running = True
    async def animate_loading():
        index = 0
        while animation_running:
            try:
                await processing_message.edit_text(loading_message_text + loading_emojis[index % len(loading_emojis)], parse_mode=ParseMode.MARKDOWN_V2)
                index += 1
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.debug(f"Loading animation error: {e}")
                break

    animation_task = asyncio.create_task(animate_loading())
    
    try:
        ydl_opts = {
            'outtmpl': os.path.join(temp_dir_name, '%(title)s.%(ext)s'),
            'noplaylist': True, 'verbose': False, 'logger': logger, 'no_warnings': True,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'
            },
        }

        if os.path.exists('cookies.txt'):
            ydl_opts['cookiefile'] = 'cookies.txt'
            logger.info("Using cookies.txt for download.")
        else:
            logger.info("No cookies.txt found. Downloads for Instagram/Facebook may fail.")

        platform_key = platform.lower()
        if platform_key == "soundcloud":
            ydl_opts['format'] = 'bestaudio/best'
            ydl_opts['postprocessors'] = [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}]
        else:
            ydl_opts['format'] = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Attempting to download {content_url} from {platform}")
            ydl.download([content_url])

        downloaded_files = [f for f in os.listdir(temp_dir_name) if os.path.isfile(os.path.join(temp_dir_name, f))]
        if not downloaded_files:
            raise FileNotFoundError("yt-dlp did not download any file.")
        
        downloaded_file_path = os.path.join(temp_dir_name, downloaded_files[0])
        file_size = os.path.getsize(downloaded_file_path)
        file_title = os.path.splitext(os.path.basename(downloaded_file_path))[0]
        content_title = escape_markdown_v2(file_title)

        animation_running = False
        await animation_task

        upload_message = f"Sending {platform} {'audio' if platform_key == 'soundcloud' else 'video'} ({file_size / (1024*1024):.2f} MB)..."
        await processing_message.edit_text(escape_markdown_v2(upload_message), parse_mode=ParseMode.MARKDOWN_V2)

        with open(downloaded_file_path, 'rb') as file_data:
            caption = f"Here's your {platform} content:\n{content_title}"
            if platform_key == "soundcloud" or file_size > TELEGRAM_VIDEO_LIMIT_BYTES:
                await context.bot.send_document(
                    chat_id=update.message.chat_id, document=InputFile(file_data, filename=os.path.basename(downloaded_file_path)),
                    caption=caption, parse_mode=ParseMode.MARKDOWN_V2, read_timeout=300, write_timeout=300
                )
            else:
                await context.bot.send_video(
                    chat_id=update.message.chat_id, video=InputFile(file_data, filename=os.path.basename(downloaded_file_path)),
                    caption=caption, parse_mode=ParseMode.MARKDOWN_V2, supports_streaming=True, read_timeout=300, write_timeout=300
                )
        await processing_message.delete()

    except yt_dlp.utils.DownloadError as e:
        logger.error(f"yt-dlp download error for {content_url}: {e}")
        user_facing_error = "Download failed. The content may be private or region-restricted."
        if "login" in str(e).lower():
            user_facing_error = "Login required. The bot's cookies might be expired or invalid."
        animation_running = False; await animation_task
        await processing_message.edit_text(escape_markdown_v2(f"Error: {user_facing_error}"), parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e:
        logger.error(f"General download error for {content_url}: {e}", exc_info=True)
        animation_running = False; await animation_task
        await processing_message.edit_text(escape_markdown_v2("An unexpected server error occurred."), parse_mode=ParseMode.MARKDOWN_V2)
    finally:
        animation_running = False
        if 'animation_task' in locals() and not animation_task.done():
            animation_task.cancel()
        if os.path.exists(temp_dir_name):
            shutil.rmtree(temp_dir_name)

# --- Message Handlers ---
async def record_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles general text messages, including URLs for download."""
    if not update.message or not update.message.text:
        return

    await save_user_to_db(update, context) 
    
    user_data = context.user_data
    if user_data.get('state') == 'awaiting_url' and user_data.get('platform'):
        platform_in_state = user_data.pop('platform')
        user_data.pop('state')
        await download_content_from_url(update, context, platform_in_state, update.message.text)

# --- Main Bot Logic ---
def main() -> None:
    """Start the bot."""
    application = Application.builder().token(BOT_TOKEN).build()

    # --- Register Handlers ---
    # Commands
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("tag", tag_all))

    # Keyboard buttons (from /start)
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Download Videos/Audio$"), show_download_options))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Help$"), help_command))

    # Inline buttons (from download flow)
    application.add_handler(CallbackQueryHandler(handle_download_platform_selection, pattern="^dl:"))
    
    # General message handler (must be last)
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
        application.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            webhook_url=WEBHOOK_URL
        )

if __name__ == "__main__":
    main()
