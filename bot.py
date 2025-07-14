import logging
import os
import sys
import asyncio
import uuid # For unique filenames
import shutil # For removing directories
import datetime # Import for timestamp in notification
import time # Import for time-based notification throttling

from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton # Added ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler
from telegram.constants import ParseMode
from sqlalchemy import create_engine, Column, String, UniqueConstraint
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.types import BigInteger # NEW: Import BigInteger for larger IDs
import yt_dlp # Make sure yt-dlp is installed: pip install yt-dlp

# --- Configuration ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# 1. Hardcoded Bot Token
BOT_TOKEN = "7806461656:AAEpUb79cc1vmH75N1fc00fYuqSuE4JrW0Y"

DATABASE_URL = os.environ.get("DATABASE_URL")

# Define a temporary directory for downloads
# On Heroku, this will be in the ephemeral filesystem
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True) # Create if it doesn't exist

# Telegram's direct video upload limit (50 MB)
TELEGRAM_VIDEO_LIMIT_MB = 50
TELEGRAM_VIDEO_LIMIT_BYTES = TELEGRAM_VIDEO_LIMIT_MB * 1024 * 1024

# --- Admin ID for notifications ---
ADMIN_ID = 7302005705 # Your specified admin ID

# 2. Admin Notification Throttling
ADMIN_NOTIFICATION_COOLDOWN = 300 # seconds (5 minutes)
last_admin_notification_time = {} # Dictionary to store last notification time per user

if not BOT_TOKEN:
    logger.critical("BOT_TOKEN environment variable not set. Exiting.")
    sys.exit(1)

if not DATABASE_URL:
    logger.critical("DATABASE_URL environment variable not set. Please add Heroku Postgres add-on. Exiting.")
    sys.exit(1)

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# --- Database Setup ---
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    # Changed Integer to BigInteger for user_id and chat_id to prevent NumericValueOutOfRange
    user_id = Column(BigInteger, primary_key=True, nullable=False)
    first_name = Column(String, nullable=True)
    username = Column(String, nullable=True)
    chat_id = Column(BigInteger, primary_key=True, nullable=False)

    def __repr__(self):
        return (f"<User(id={self.user_id}, first_name='{self.first_name}', "
                f"username='{self.username}', chat_id={self.chat_id})>")

engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)

try:
    Base.metadata.create_all(engine)
    logger.info("Database tables checked/created successfully.")
except OperationalError as e:
    logger.critical(f"Failed to connect to database or create tables: {e}. Exiting.")
    sys.exit(1)

Session = sessionmaker(bind=engine)
# --- End Database Setup ---

# Constants for message splitting
MAX_MENTIONS_PER_MESSAGE = 50
MAX_MESSAGE_LENGTH = 4096

# --- Helper Functions ---
def escape_markdown_v2(text: str) -> str:
    """Escapes common MarkdownV2 special characters."""
    if not isinstance(text, str):
        return ""
    
    # Correct list of special characters for MarkdownV2, including backslash itself to be handled first.
    special_chars = r'_*[]()~`>#+-=|{}.!'
    
    # Process backslashes first, then other special characters
    escaped_text = text.replace('\\', '\\\\')
    for char in special_chars:
        escaped_text = escaped_text.replace(char, f'\\{char}')
    return escaped_text

async def send_notification_to_admin(context: ContextTypes.DEFAULT_TYPE, user_info: dict, event_type: str) -> None:
    """Sends a notification message to the admin."""
    user_id = user_info.get('user_id', 'N/A')
    
    # Check cooldown before sending notification
    current_time = time.time()
    if user_id in last_admin_notification_time and \
       (current_time - last_admin_notification_time[user_id]) < ADMIN_NOTIFICATION_COOLDOWN:
        logger.info(f"Admin notification for user {user_id} throttled. Last sent {current_time - last_admin_notification_time[user_id]:.2f}s ago.")
        return

    first_name = user_info.get('first_name', 'N/A')
    username = user_info.get('username', 'N/A')
    chat_id = user_info.get('chat_id', 'N/A')
    chat_type = user_info.get('chat_type', 'N/A')
    
    # Use current time of interaction for the message content
    message_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Apply escape_markdown_v2 to all variables that go into the message
    # before concatenating them.
    escaped_event_type = escape_markdown_v2(event_type)
    escaped_first_name = escape_markdown_v2(first_name)
    escaped_username = escape_markdown_v2(username) if username != 'N/A' else 'N/A'
    escaped_chat_type = escape_markdown_v2(chat_type)
    
    notification_message = (
        f"🔔 New User Interaction\\! 🔔\n\n"
        f"Event Type: *{escaped_event_type}*\n"
        f"User ID: `{user_id}`\n"
        f"First Name: *{escaped_first_name}*\n"
        f"Username: `@{escaped_username}`" if escaped_username != 'N/A' else f"Username: `N/A`\n"
        f"Chat ID: `{chat_id}`\n"
        f"Chat Type: *{escaped_chat_type}*\n"
        f"Time: `{escape_markdown_v2(message_time)}`"
    )
    
    try:
        await context.bot.send_message(
            chat_id=ADMIN_ID,
            text=notification_message,
            parse_mode=ParseMode.MARKDOWN_V2
        )
        last_admin_notification_time[user_id] = current_time # Update last sent time
        logger.info(f"Admin notification sent for user {user_id} (event: {event_type}).")
    except Exception as e:
        logger.error(f"Failed to send admin notification for user {user_id}: {e}")


async def save_user_to_db(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Saves or updates user information in the database and sends admin notification."""
    # This function expects update.message to be present
    if not update.message or not update.message.from_user:
        logger.warning("save_user_to_db called without a valid message or user object.")
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
            'user_id': user.id,
            'first_name': user.first_name,
            'username': user.username,
            'chat_id': chat_id,
            'chat_type': chat_type
        }

        if not existing_user:
            new_user = User(
                user_id=user.id,
                first_name=user.first_name,
                username=user.username,
                chat_id=chat_id
            )
            session.add(new_user)
            session.commit()
            logger.info(f"DB: Added user {user.id} ({user.first_name}) for chat {chat_id}")
            await send_notification_to_admin(context, user_info, "New User Added")
        else:
            # Check if any info changed to avoid unnecessary DB writes and notifications
            info_changed = False
            if existing_user.first_name != user.first_name:
                existing_user.first_name = user.first_name
                info_changed = True
            if existing_user.username != user.username:
                existing_user.username = user.username
                info_changed = True
            
            if info_changed:
                session.commit()
                logger.info(f"DB: Updated user {user.id} ({user.first_name}) for chat {chat_id}")
                await send_notification_to_admin(context, user_info, "User Info Updated")
            else:
                logger.info(f"DB: User {user.id} ({user.first_name}) already exists and no info changed for chat {chat_id}. Not updating.")
                # Send a general interaction notification even if user info didn't change
                await send_notification_to_admin(context, user_info, "User Interacted")

    except IntegrityError:
        session.rollback()
        logger.warning(f"DB: User {user.id} already exists (race condition) for chat {chat_id}. Rolling back.")
        # Still send notification if it's a known user interacting for the first time in this session
        await send_notification_to_admin(context, user_info, "User Interacted (Race Condition)")
    except Exception as e:
        session.rollback()
        logger.error(f"DB: Error saving/updating user {user.id} for chat {chat_id}: {e}")
        user_info['error'] = str(e)
        await send_notification_to_admin(context, user_info, f"Database Error: {e}")
    finally:
        session.close()

# --- Command Handlers ---
async def record_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handler for messages that are NOT commands, NOT specific keyboard button texts,
    and are potentially URLs for download based on user state.
    It also saves user info for general interactions.
    """
    if update.message and update.message.from_user:
        # Save user info for any message that passes through here
        await save_user_to_db(update, context) 
    
    # Check if the user is in a state awaiting a URL
    if update.message and update.message.text:
        user_data = context.user_data
        if user_data.get('state') == 'awaiting_url' and user_data.get('platform'):
            platform_in_state = user_data['platform']
            url = update.message.text
            
            # Clear the state immediately so it doesn't try to process future messages
            user_data['state'] = None
            user_data['platform'] = None

            # Call the universal download function
            await download_content_from_url(update, context, platform_in_state, url)
            return # Consume the message if it's a URL for download

    # If it's not a URL for download, it's just a regular message.
    # We could add a default response here if needed, but for now, it just records.
    # Example: await update.message.reply_text(escape_markdown_v2("I didn't understand that. Please use the buttons or commands."))


async def handle_keyboard_download_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the 'Download Videos/Audio' keyboard button press."""
    if update.message and update.message.from_user:
        await save_user_to_db(update, context)

    # Call the existing show_download_options logic.
    # show_download_options is designed to work with both CallbackQuery and direct Message updates.
    await show_download_options(update, context)

async def handle_keyboard_help_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the 'Help' keyboard button press."""
    if update.message and update.message.from_user:
        await save_user_to_db(update, context)
    
    # Call the existing help_command logic
    await help_command(update, context)


async def tag_all(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Resends the replied message and tags all *known* non-admin members.
    Mentions use custom text (first name) and user ID for notifications.
    Splits messages if too many mentions.
    """
    if not update.message.reply_to_message:
        await update.message.reply_text(
            escape_markdown_v2("Please reply to a message with `/tag` to use this command."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    chat_id = update.message.chat_id
    replied_message = update.message.reply_to_message
    replied_message_text = replied_message.text or replied_message.caption

    if not replied_message_text:
        await update.message.reply_text(
            escape_markdown_v2("The replied message does not contain any text or caption to resend."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    # Call save_user_to_db here to ensure the command user is recorded and notification sent
    if update.message and update.message.from_user:
        await save_user_to_db(update, context)

    feedback_message = await update.message.reply_text(
        escape_markdown_v2("Fetching user list and preparing mentions, please wait..."),
        parse_mode=ParseMode.MARKDOWN_V2
    )

    session = Session()
    all_known_users_in_chat = []
    try:
        all_known_users_in_chat = session.query(User).filter_by(chat_id=chat_id).all()
        logger.info(f"DB: Found {len(all_known_users_in_chat)} known users for chat {chat_id}.")
    except Exception as e:
        logger.error(f"DB: Error querying users for chat {chat_id}: {e}")
        await feedback_message.edit_text(escape_markdown_v2("An error occurred while fetching known users from the database. Please try again later."))
        return
    finally:
        session.close()

    members_to_tag_links = []
    current_chat_administrators_ids = set()

    try:
        administrators = await context.bot.get_chat_administrators(chat_id)
        for admin in administrators:
            current_chat_administrators_ids.add(admin.user.id)
            if admin.user.id == context.bot.id: 
                current_chat_administrators_ids.add(admin.user.id)
        logger.info(f"Telegram API: Found {len(current_chat_administrators_ids)} administrators for chat {chat_id}.")
    except Exception as e:
        logger.warning(f"Telegram API: Could not retrieve chat administrators for chat {chat_id}: {e}. Admins might not be excluded from tagging if this fails or bot is not admin.")

    for user_obj in all_known_users_in_chat:
        if user_obj.user_id not in current_chat_administrators_ids:
            mention_name = user_obj.first_name if user_obj.first_name else "A User"
            escaped_mention_name = escape_markdown_v2(mention_name)
            members_to_tag_links.append(f"[{escaped_mention_name}](tg://user?id={user_obj.user_id})")

    if not members_to_tag_links:
        await feedback_message.edit_text(
            escape_markdown_v2("No known non-admin members to tag in this group. Users must send a message first to be added to the tag list, and ensure bot privacy is off."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    logger.info(f"Prepared {len(members_to_tag_links)} members for tagging in chat {chat_id}.")

    # --- Prepare the message content ---
    final_replied_message_content = escape_markdown_v2(replied_message_text)
    
    initial_command_text = ""
    if update.message.text and ' ' in update.message.text:
        initial_command_text = escape_markdown_v2(update.message.text.split(' ', 1)[1])
    
    full_message_content_start = ""
    if initial_command_text:
        full_message_content_start += f"{initial_command_text}\n\n"
    full_message_content_start += f"{final_replied_message_content}\n\n"


    # --- Pagination Logic for Mentions ---
    messages_to_send = []
    current_mentions_group = []
    
    # Use len() on the encoded string to approximate byte length for Telegram's character limit
    current_message_base_length_bytes = len(full_message_content_start.encode('utf-8')) + len("\n ".encode('utf-8'))
    
    for mention_link in members_to_tag_links:
        mention_length_bytes = len(mention_link.encode('utf-8'))
        
        if (current_message_base_length_bytes + sum(len(m.encode('utf-8')) + 1 for m in current_mentions_group) + mention_length_bytes > MAX_MESSAGE_LENGTH or
            len(current_mentions_group) >= MAX_MENTIONS_PER_MESSAGE):
            
            messages_to_send.append(full_message_content_start + " " + " ".join(current_mentions_group)) 
            
            current_mentions_group = []
            full_message_content_start = "" 
            current_message_base_length_bytes = len(" ".encode('utf-8')) 

        current_mentions_group.append(mention_link)

    if current_mentions_group:
        if not messages_to_send and full_message_content_start: 
             messages_to_send.append(full_message_content_start + " " + " ".join(current_mentions_group))
        elif messages_to_send: 
             messages_to_send.append(" " + " ".join(current_mentions_group))
        else: 
             messages_to_send.append(" " + " ".join(current_mentions_group))


    await feedback_message.edit_text(escape_markdown_v2(f"Sending {len(messages_to_send)} messages with mentions..."))

    successful_sends = 0
    for i, message_text in enumerate(messages_to_send):
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=message_text,
                parse_mode=ParseMode.MARKDOWN_V2,
                disable_notification=(i > 0 and len(messages_to_send) > 1) 
            )
            successful_sends += 1
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Telegram API: Failed to send tagged message part {i+1}/{len(messages_to_tag_links)} for chat {chat_id}: {e}")

    if successful_sends == len(messages_to_send):
        await feedback_message.edit_text(
            escape_markdown_v2(f"Successfully sent {successful_sends} messages with mentions for {len(members_to_tag_links)} members."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
    else:
        await feedback_message.edit_text(
            escape_markdown_v2(f"Completed sending messages. Sent {successful_sends} out of {len(messages_to_send)} parts. Some errors occurred while sending mentions. Check logs for details and ensure bot privacy is off."),
            parse_mode=ParseMode.MARKDOWN_V2
        )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message and saves the user."""
    if update.message and update.message.from_user:
        await save_user_to_db(update, context) # Pass context here

        # Regular Keyboard (persistent, appears above message input)
        keyboard = [
            [KeyboardButton("Download Videos/Audio")], # Text for the button
            [KeyboardButton("Help")]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=False, resize_keyboard=True)

        # Inline Keyboard (still useful for specific actions or dynamic options)
        inline_keyboard = [
            [InlineKeyboardButton("Download Videos/Audio (Inline)", callback_data="show_download_options")],
            [InlineKeyboardButton("Help (Inline)", callback_data="help_button")]
        ]
        # inline_reply_markup = InlineKeyboardMarkup(inline_keyboard) # Defined but not used directly in reply_text for start

        await update.message.reply_text(
            escape_markdown_v2("Hi there\\! I'm your multimedia download and group tagging bot\\.\n\n"
            "To get started, tap 'Download Videos/Audio' on the keyboard below, or 'Help' for more info\\."),
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=reply_markup # Use the ReplyKeyboardMarkup here
        )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a help message."""
    # Ensure user is saved and notification sent if this is a direct /help command
    if update.message and update.message.from_user:
        await save_user_to_db(update, context)

    # For the help message, you can also include the ReplyKeyboardMarkup
    keyboard = [
        [KeyboardButton("Download Videos/Audio")],
        [KeyboardButton("Help")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=False, resize_keyboard=True)

    await update.message.reply_text(
        escape_markdown_v2("How to use the Bot:\n\n"
        "To Download Videos and Audio:\n"
        "- Tap the 'Download Videos/Audio' button on your keyboard or use the /download command.\n"
        "- Select the platform (TikTok, Facebook, Instagram, Pinterest, Twitter, YouTube, SoundCloud).\n"
        "- Send the full video/audio URL when prompted.\n\n"
        "To Tag Group Members:\n"
        "1. Add me to your group and make me an Administrator (this helps me exclude admins from tags).\n"
        "2. Crucial: Go to @BotFather -> My Bots -> (Your Bot) -> Bot Settings -> Group Privacy -> *Turn off*.\n"
        "   Why? This allows me to see all messages in the group and build a list of members to tag.\n"
        "3. Wait for members to send messages. I can only tag users who have sent a message in the group *after* I've joined and my Group Privacy is off.\n"
        "4. To tag everyone: Reply to any message in the group with the command `/tag`.\n"
        "I will resend the replied message and mention all known non-admin members. "
        "Mentions will show their first name, not their username, ensuring privacy while still notifying them.\n\n"
        "Limitations:\n"
        "- Download functionality relies on `yt-dlp` and may not always work if the content is restricted or the platform changes its API, or if the file size exceeds Telegram's 2GB limit.\n"
        "- Some content might require a logged-in session or be geo-restricted.\n"
        "- I cannot tag users who have never sent a message since I joined.\n"
        "- Very large groups might experience delays or split messages due to Telegram's limits."),
        parse_mode=ParseMode.MARKDOWN_V2,
        reply_markup=reply_markup # Apply ReplyKeyboardMarkup here too
    )

async def show_download_options(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Determine if the update came from a CallbackQuery or a regular Message
    message_source = update.callback_query or update.message

    if not message_source or not message_source.from_user:
        logger.warning("show_download_options called without a valid message or user object.")
        return

    # Extract info from the appropriate source (query.message or update.message)
    from_user = message_source.from_user
    chat_id = message_source.chat_id if hasattr(message_source, 'chat_id') else message_source.chat.id
    chat_type = message_source.chat.type

    # Save user on interaction point
    class DummyMessage:
        def __init__(self, from_user, chat_id, chat_type):
            self.from_user = from_user
            self.chat_id = chat_id
            self.chat = type('Chat', (object,), {'type': chat_type})() # Mock chat object
    
    dummy_message = DummyMessage(from_user, chat_id, chat_type)
    dummy_update = Update(update_id=0, message=dummy_message)
    await save_user_to_db(dummy_update, context)

    keyboard = [
        [
            InlineKeyboardButton("TikTok", callback_data="download_platform:TikTok"),
            InlineKeyboardButton("Facebook", callback_data="download_platform:Facebook"),
            InlineKeyboardButton("Instagram", callback_data="download_platform:Instagram")
        ],
        [
            InlineKeyboardButton("Pinterest", callback_data="download_platform:Pinterest"),
            InlineKeyboardButton("Twitter", callback_data="download_platform:Twitter"),
            InlineKeyboardButton("YouTube", callback_data="download_platform:YouTube")
        ],
        [
            InlineKeyboardButton("SoundCloud", callback_data="download_platform:SoundCloud")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    if update.callback_query:
        # It's an inline button press, edit the existing message
        await update.callback_query.answer() # Acknowledge the inline button press
        await update.callback_query.edit_message_text(
            escape_markdown_v2("Please choose a platform to download from:"),
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=reply_markup
        )
    else: # It's a regular message (from the ReplyKeyboardMarkup button)
        await update.message.reply_text(
            escape_markdown_v2("Please choose a platform to download from:"),
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=reply_markup
        )


async def handle_download_platform_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer() # Acknowledge the button press
    
    # Save user on callback query as well
    if query.message and query.from_user:
        class DummyMessage:
            def __init__(self, from_user, chat_id, chat_type):
                self.from_user = from_user
                self.chat_id = chat_id
                self.chat = type('Chat', (object,), {'type': chat_type})()
        
        dummy_message = DummyMessage(query.from_user, query.message.chat_id, query.message.chat.type)
        dummy_update = type('Update', (object,), {'message': dummy_message})()
        await save_user_to_db(dummy_update, context)

    # Parse the callback data: "download_platform:PLATFORM_NAME"
    platform_name = query.data.split(":")[1]
    
    # Set user state
    context.user_data['state'] = 'awaiting_url'
    context.user_data['platform'] = platform_name

    await query.edit_message_text(
        escape_markdown_v2(f"Please send me the full URL for the {platform_name} content."),
        parse_mode=ParseMode.MARKDOWN_V2
    )


# --- Universal Video/Audio Download Handler ---

async def download_content_from_url(update: Update, context: ContextTypes.DEFAULT_TYPE, platform: str, content_url: str) -> None:
    """
    Downloads content from a given URL using yt-dlp and sends it.
    Args:
        platform (str): The platform (e.g., 'TikTok', 'Facebook', 'Instagram', 'Pinterest', 'Twitter', 'YouTube, 'SoundCloud') for messaging.
        content_url (str): The URL to download.
    """
    # save_user_to_db already called by record_user_message before this function
    
    # Basic URL validation specific to platform
    valid_platforms = {
        "tiktok": "tiktok.com",
        "fb": "facebook.com",
        "facebook": "facebook.com",
        "instagram": "instagram.com",
        "insta": "instagram.com",
        "pinterest": "pinterest.com",
        "twitter": "twitter.com",
        "youtube": ["youtube.com", "youtu.be", "m.youtube.com"], # More comprehensive YouTube domains
        "soundcloud": "soundcloud.com"
    }
    
    # Use lowercase platform key for dictionary lookup
    platform_key = platform.lower()
    if platform_key == "insta": # Handle alias
        platform_key = "instagram"
    if platform_key == "fb": # Handle alias
        platform_key = "facebook"
    
    expected_domains = valid_platforms.get(platform_key)

    is_valid_url = False
    if content_url.startswith("http://") or content_url.startswith("https://"):
        if isinstance(expected_domains, list): # For YouTube, check multiple domains
            if any(domain in content_url for domain in expected_domains):
                is_valid_url = True
        elif expected_domains and expected_domains in content_url:
            is_valid_url = True
    
    if not is_valid_url:
        await update.message.reply_text(
            escape_markdown_v2(f"That doesn't look like a valid {platform} URL. Please provide a full URL."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    # Create a unique temporary directory for this download
    temp_dir_name = os.path.join(DOWNLOAD_DIR, str(uuid.uuid4()))
    os.makedirs(temp_dir_name, exist_ok=True)
    
    # Loading animation setup
    loading_emojis = ["🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘", "🕙", "🕚", "🕛"]
    loading_message_text = escape_markdown_v2(f"Getting your {platform} content, please wait... ")
    processing_message = await update.message.reply_text(
        loading_message_text + loading_emojis[0],
        parse_mode=ParseMode.MARKDOWN_V2
    )

    # Flag to control the animation loop
    animation_running = True
    async def animate_loading():
        index = 0
        while animation_running:
            try:
                await processing_message.edit_text(
                    loading_message_text + loading_emojis[index % len(loading_emojis)],
                    parse_mode=ParseMode.MARKDOWN_V2
                )
                index += 1
                await asyncio.sleep(0.5) # Update every half second
            except Exception as e:
                # Catch exceptions if message is deleted or inaccessible
                logger.debug(f"Loading animation error: {e}")
                break

    animation_task = asyncio.create_task(animate_loading())

    downloaded_file_path = None
    try:
        ydl_opts = {
            'outtmpl': os.path.join(temp_dir_name, '%(id)s.%(ext)s'),
            'noplaylist': True,
            'verbose': False, # Changed to False for cleaner logs in production
            'logger': logger,
            'no_warnings': True,
            'postprocessors': [{ # Use FFmpeg to ensure compatible MP4 for Telegram
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
        }

        # Special handling for SoundCloud (audio only)
        if platform_key == "soundcloud":
            ydl_opts['format'] = 'bestaudio/best' # Prioritize audio
            # Remove video postprocessor for audio only
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }]
        else: # For video platforms
            ydl_opts['format'] = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'

        # Optional: Add cookies.txt for platforms like Instagram/Facebook/YouTube that sometimes require it
        # This requires you to create a 'cookies.txt' file in your bot's root directory
        # containing cookies exported from a logged-in browser session.
        # Use with caution, especially for public bots, due to security/privacy.
        if os.path.exists('cookies.txt'):
            ydl_opts['cookiefile'] = 'cookies.txt'
            logger.info("Using cookies.txt for download.")
        else:
            logger.info("No cookies.txt found. Proceeding without cookies.")


        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(content_url, download=False) # Get info first
            
            if info_dict.get('is_live'):
                animation_running = False # Stop animation
                await animation_task # Ensure the animation task finishes (or is cancelled)
                await processing_message.edit_text(
                    escape_markdown_v2("Sorry, I cannot download live streams. Please provide a link to a completed video."),
                    parse_mode=ParseMode.MARKDOWN_V2
                )
                return
            
            # Use 'actual_ext' if available, otherwise 'ext'
            file_extension = info_dict.get('actual_ext', info_dict.get('ext', 'mp4'))
            # For audio, ensure it's mp3 or m4a if extracted
            if platform_key == "soundcloud" and 'mp3' not in file_extension:
                file_extension = 'mp3' # Force mp3 if yt-dlp might suggest something else

            # Some platforms might return a generic ID, use title if available for filename to be more descriptive
            suggested_filename_base = info_dict.get('title', info_dict.get('id', 'content'))
            # Clean filename from problematic characters (e.g., / \ : * ? " < > |)
            suggested_filename_base = "".join(c for c in suggested_filename_base if c.isalnum() or c in (' ', '.', '_', '-')).strip()
            # Truncate filename if too long for filesystem limits (e.g., 255 chars)
            if len(suggested_filename_base) > 150: # Arbitrary but reasonable limit
                suggested_filename_base = suggested_filename_base[:150]
            
            suggested_filename = f"{suggested_filename_base}.{file_extension}"
            downloaded_file_path_template = os.path.join(temp_dir_name, suggested_filename) # Use this for outtmpl
            
            # Re-configure outtmpl with the cleaned filename template
            ydl_opts['outtmpl'] = downloaded_file_path_template

            logger.info(f"Attempting to download {content_url} from {platform} to {downloaded_file_path_template}")
            
            # Re-initialize YDL with updated opts, or just download if only outtmpl changed
            with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                 ydl_download.download([content_url])
            
            logger.info(f"Download initiated via yt-dlp to temp directory.")

        # Find the actual downloaded file in the temp directory after download
        downloaded_files = [f for f in os.listdir(temp_dir_name) if os.path.isfile(os.path.join(temp_dir_name, f))]
        if not downloaded_files:
            raise FileNotFoundError("yt-dlp did not download any file or file not found in temp directory.")
        
        # Take the first file, which should be the main content
        downloaded_file_path = os.path.join(temp_dir_name, downloaded_files[0])


        file_size = os.path.getsize(downloaded_file_path) # in bytes
        logger.info(f"Downloaded file size: {file_size / (1024*1024):.2f} MB")

        # Get content title for caption
        content_title = escape_markdown_v2(info_dict.get('title', f'{platform} Content'))
        
        # Decide whether to send as video/audio or document based on platform and size
        if platform_key == "soundcloud": # Always send audio as document
            animation_running = False # Stop animation
            await animation_task # Ensure the animation task finishes (or is cancelled)
            await processing_message.edit_text(
                escape_markdown_v2(f"Audio downloaded! Sending as document ({file_size / (1024*1024):.2f} MB). Please wait, this might take a while. "),
                parse_mode=ParseMode.MARKDOWN_V2
            )
            with open(downloaded_file_path, 'rb') as audio_file:
                await context.bot.send_document(
                    chat_id=update.message.chat_id,
                    document=InputFile(audio_file, filename=os.path.basename(downloaded_file_path)),
                    caption=f"Here's your {platform} audio: {content_title}",
                    parse_mode=ParseMode.MARKDOWN_V2,
                    read_timeout=300,
                    write_timeout=300,
                    connect_timeout=300
                )
        elif file_size > TELEGRAM_VIDEO_LIMIT_BYTES:
            # Send as document if larger than 50MB (Telegram's send_video limit)
            animation_running = False # Stop animation
            await animation_task # Ensure the animation task finishes (or is cancelled)
            await processing_message.edit_text(
                escape_markdown_v2(f"Video downloaded! Sending as document due to size ({file_size / (1024*1024):.2f} MB). Please wait, this might take a while. "),
                parse_mode=ParseMode.MARKDOWN_V2
            )
            with open(downloaded_file_path, 'rb') as video_file:
                await context.bot.send_document(
                    chat_id=update.message.chat_id,
                    document=InputFile(video_file, filename=os.path.basename(downloaded_file_path)), # Use actual filename
                    caption=f"Here's your {platform} video: {content_title}",
                    parse_mode=ParseMode.MARKDOWN_V2,
                    read_timeout=300, # Increased timeout for large uploads
                    write_timeout=300, # Increased timeout for large uploads
                    connect_timeout=300
                )
        else:
            # Send as video if smaller than 50MB
            animation_running = False # Stop animation
            await animation_task # Ensure the animation task finishes (or is cancelled)
            await processing_message.edit_text(
                escape_markdown_v2(f"Video downloaded! Sending in high quality ({file_size / (1024*1024):.2f} MB). Please wait. "),
                parse_mode=ParseMode.MARKDOWN_V2
            )
            with open(downloaded_file_path, 'rb') as video_file:
                await context.bot.send_video(
                    chat_id=update.message.chat_id,
                    video=InputFile(video_file, filename=os.path.basename(downloaded_file_path)), # Use actual filename
                    caption=f"Here's your {platform} video: {content_title}",
                    parse_mode=ParseMode.MARKDOWN_V2,
                    supports_streaming=True, # Allows streaming before full download
                    read_timeout=300, 
                    write_timeout=300,
                    connect_timeout=300
                )

        await processing_message.delete() # Remove the "processing" message
        await update.message.reply_text(
            escape_markdown_v2(f"Your {platform} content has been sent successfully! Enjoy!"),
            parse_mode=ParseMode.MARKDOWN_V2
        )

    except yt_dlp.utils.DownloadError as e:
        logger.error(f"yt-dlp download error for {content_url} ({platform}): {e}")
        error_message = str(e)
        user_facing_error = "An unexpected download error occurred."

        if "ERROR: This video is unavailable" in error_message or "TikTok said: Video unavailable" in error_message or "Private video" in error_message or "This content isn't available" in error_message:
            user_facing_error = "The content might be private, removed, or region-restricted."
        elif "Unsupported URL" in error_message:
            user_facing_error = f"This URL is not supported by the downloader for {platform}."
        elif "HTTP Error 404" in error_message:
            user_facing_error = "The link leads to a 404 error (content not found)."
        elif "rate-limit reached" in error_message or "login required" in error_message or "Please sign in" in error_message: # Added "Please sign in" for YouTube
            user_facing_error = f"Download failed due to rate-limiting or login requirement. Some content from {platform} may require a logged-in session or you might have to provide cookies (advanced)."
        elif "network error" in error_message or "Connection reset by peer" in error_message:
            user_facing_error = "A network error occurred during download. Please try again later."
        elif "no suitable formats found" in error_message:
            user_facing_error = "No suitable format found for download. The content might be protected or unusual."
        elif "ffmpeg is not installed" in error_message: # Specific error for FFmpeg
            user_facing_error = "FFmpeg is not installed on the server, which is required to process this video. Please inform the bot administrator."
        
        animation_running = False # Stop animation
        await animation_task # Ensure the animation task finishes (or is cancelled)
        await processing_message.edit_text(
            escape_markdown_v2(f"Failed to download the {platform} content: `{escape_markdown_v2(user_facing_error)}`\n\n"
                               "Please ensure the link is public and valid. Try again later."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
    except FileNotFoundError as e:
        logger.error(f"File system error during {platform} download for {content_url}: {e}")
        animation_running = False # Stop animation
        await animation_task # Ensure the animation task finishes (or is cancelled)
        await processing_message.edit_text(
            escape_markdown_v2(f"A file error occurred: `{escape_markdown_v2(str(e))}`\n\n"
                               "The content might not have been downloaded correctly. Please try again. "),
            parse_mode=ParseMode.MARKDOWN_V2
        )
    except Exception as e:
        logger.error(f"General error processing {platform} download for {content_url}: {e}", exc_info=True)
        animation_running = False # Stop animation
        await animation_task # Ensure the animation task finishes (or is cancelled)
        await processing_message.edit_text(
            escape_markdown_v2(f"An unexpected error occurred while processing your request: `{escape_markdown_v2(str(e))}`\n\n"
                               "Please try again later. If the issue persists, contact support. "),
            parse_mode=ParseMode.MARKDOWN_V2
        )
    finally:
        animation_running = False # Ensure animation stops even if not explicitly stopped in try/except
        if not animation_task.done():
            animation_task.cancel() # Cancel the task if it's still running
            try:
                await animation_task # Await cancellation to avoid RuntimeWarning
            except asyncio.CancelledError:
                pass

        # Clean up the temporary directory
        if os.path.exists(temp_dir_name):
            try:
                shutil.rmtree(temp_dir_name)
                logger.info(f"Cleaned up temporary directory: {temp_dir_name}")
            except OSError as e:
                logger.error(f"Error removing temporary directory {temp_dir_name}: {e}")

# --- Command-specific wrappers for the universal download function (not used directly with buttons) ---
# These are kept for consistency or if you decide to re-introduce /command <url> later
async def tiktok_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # If called directly as /tiktok <URL> this will work.
    # If called via button flow, content_url will be None, and the record_user_message handles it.
    if context.args:
        # Ensure user is saved and notification sent
        await save_user_to_db(update, context)
        await download_content_from_url(update, context, "TikTok", context.args[0])
    else:
        # Ensure user is saved and notification sent even if command is incomplete
        await save_user_to_db(update, context)
        await update.message.reply_text(
            escape_markdown_v2("Please use the 'Download Videos/Audio' button to select a platform, then send the URL.")
        )

async def fb_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        await save_user_to_db(update, context)
        await download_content_from_url(update, context, "Facebook", context.args[0])
    else:
        await save_user_to_db(update, context)
        await update.message.reply_text(
            escape_markdown_v2("Please use the 'Download Videos/Audio' button to select a platform, then send the URL.")
        )

async def insta_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        await save_user_to_db(update, context)
        await download_content_from_url(update, context, "Instagram", context.args[0])
    else:
        await save_user_to_db(update, context)
        await update.message.reply_text(
            escape_markdown_v2("Please use the 'Download Videos/Audio' button to select a platform, then send the URL.")
        )

async def pinterest_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        await save_user_to_db(update, context)
        await download_content_from_url(update, context, "Pinterest", context.args[0])
    else:
        await save_user_to_db(update, context)
        await update.message.reply_text(
            escape_markdown_v2("Please use the 'Download Videos/Audio' button to select a platform, then send the URL.")
        )

async def twitter_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        await save_user_to_db(update, context)
        await download_content_from_url(update, context, "Twitter", context.args[0])
    else:
        await save_user_to_db(update, context)
        await update.message.reply_text(
            escape_markdown_v2("Please use the 'Download Videos/Audio' button to select a platform, then send the URL.")
        )

async def youtube_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        await save_user_to_db(update, context)
        await download_content_from_url(update, context, "YouTube", context.args[0])
    else:
        await save_user_to_db(update, context)
        await update.message.reply_text(
            escape_markdown_v2("Please use the 'Download Videos/Audio' button to select a platform, then send the URL.")
        )

async def soundcloud_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        await save_user_to_db(update, context)
        await download_content_from_url(update, context, "SoundCloud", context.args[0])
    else:
        await save_user_to_db(update, context)
        await update.message.reply_text(
            escape_markdown_v2("Please use the 'Download Videos/Audio' button to select a platform, then send the URL.")
        )


# --- Main Bot Logic and Heroku Integration ---
def main() -> None:
    """Start the bot."""
    application = Application.builder().token(BOT_TOKEN).build()

    # Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("tag", tag_all))
    
    # Register the command handlers for explicit /command <url> usage (if desired, currently prompts)
    application.add_handler(CommandHandler("tiktok", tiktok_command))
    application.add_handler(CommandHandler("fb", fb_command))
    application.add_handler(CommandHandler("insta", insta_command))
    application.add_handler(CommandHandler("pinterest", pinterest_command))
    application.add_handler(CommandHandler("twitter", twitter_command))
    application.add_handler(CommandHandler("youtube", youtube_command))
    application.add_handler(CommandHandler("soundcloud", soundcloud_command))

    # --- IMPORTANT: Order matters here! Specific handlers first. ---

    # 1. Specific Message Handlers for Keyboard Buttons
    # Use filters.Regex to match the exact button text.
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Download Videos/Audio$"), handle_keyboard_download_button))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Help$"), handle_keyboard_help_button))

    # 2. Callback Query Handlers for inline buttons
    application.add_handler(CallbackQueryHandler(show_download_options, pattern="^show_download_options$"))
    application.add_handler(CallbackQueryHandler(handle_download_platform_selection, pattern="^download_platform:"))
    application.add_handler(CallbackQueryHandler(help_command, pattern="^help_button$"))

    # 3. General Message Handler (LAST, to catch everything else, but exclude commands)
    # This handler will also save user info and handle URLs when in 'awaiting_url' state.
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, record_user_message))
    
    logger.info("Running in polling mode. If deployed on Heroku, ensure this is on a WORKER dyno.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()

