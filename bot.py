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
from telegram.constants import ParseMode # Still needed for ParseMode.HTML
from sqlalchemy import create_engine, Column, UniqueConstraint, String, BigInteger # FIX: Import String and BigInteger
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError, OperationalError
import yt_dlp # Make sure yt-dlp is installed: pip install yt-dlp

# --- OpenAI API Integration ---
import openai

# --- Configuration ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- HARDCODED SENSITIVE INFORMATION (LESS SECURE) ---
# I've hardcoded these back as per your explicit request, BUT IT'S STRONGLY RECOMMENDED TO USE ENVIRONMENT VARIABLES.
# WARNING: This is generally NOT recommended for production environments due to security risks.
# For example, on Heroku, set these as Config Vars:
# BOT_TOKEN="7806461656:AAEFsYhfk7moHzZgqX80qboJfb4b58UhsgU"
# OPENAI_API_KEY="sk-proj-YOUR_ACTUAL_OPENAI_API_KEY_HERE"
# ADMIN_ID=7302005705 (if not sensitive, can be hardcoded; otherwise, also use config var)

BOT_TOKEN = "7806461656:AAEFsYhfk7moHzZgqX80qboJfb4b58UhsgU" # Your Bot Token - REPLACE WITH OS.ENVIRON.GET IF USING ENV VARS
OPENAI_API_KEY = "sk-proj-YOUR_ACTUAL_OPENAI_API_KEY_HERE" # Your OpenAI API Key - REPLACE WITH OS.ENVIRON.GET IF USING ENV VARS
ADMIN_ID = 7302005705 # Your specified admin ID

# DATABASE_URL is still loaded from environment variable as it's crucial for Heroku Postgres
DATABASE_URL = os.environ.get("DATABASE_URL")

# --- Validation for hardcoded values (still good practice) ---
if not BOT_TOKEN:
    logger.critical("BOT_TOKEN is not set. Exiting.")
    sys.exit(1)

if not OPENAI_API_KEY:
    logger.critical("OPENAI_API_KEY is not set. Exiting.")
    sys.exit(1)

if not isinstance(ADMIN_ID, int):
    logger.critical("ADMIN_ID must be an integer. Exiting.")
    sys.exit(1)

# Adjust DATABASE_URL for SQLAlchemy 2.0 compatibility with Heroku Postgres
if DATABASE_URL: # Only replace if DATABASE_URL exists
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
else:
    logger.critical("DATABASE_URL environment variable not set. Please add Heroku Postgres add-on. Exiting.")
    sys.exit(1)


# Configure OpenAI API
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Define the GPT model to use
OPENAI_MODEL_NAME = "gpt-4o" # You can choose "gpt-3.5-turbo" for lower cost/faster, or "gpt-4o" for better quality

# Dictionary to store ongoing OpenAI conversations (for stateful chat)
# Stores history as a list of dictionaries in the format [{role: ..., content: ...}]
openai_conversations = {}

# Define a temporary directory for downloads
# On Heroku, this will be in the ephemeral filesystem
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True) # Create if it doesn't exist

# Telegram's direct video upload limit (50 MB)
TELEGRAM_VIDEO_LIMIT_MB = 50
TELEGRAM_VIDEO_LIMIT_BYTES = TELEGRAM_VIDEO_LIMIT_MB * 1024 * 1024

# Admin Notification Throttling
ADMIN_NOTIFICATION_COOLDOWN = 300 # seconds (5 minutes)
last_admin_notification_time = {} # Dictionary to store last notification time per user

# --- Database Setup ---
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    user_id = Column(BigInteger, primary_key=True, nullable=False)
    first_name = Column(String, nullable=True)
    username = Column(String, nullable=True)
    chat_id = Column(BigInteger, primary_key=True, nullable=False) # chat_id as part of composite key
    __table_args__ = (UniqueConstraint('user_id', 'chat_id', name='_user_chat_uc'),) # Ensures unique user per chat

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

# --- User States for conversational flow ---
GPT_STATE = 'gpt_chat'
AWAITING_URL_STATE = 'awaiting_url'

# --- Helper Functions ---
def escape_html(text: str) -> str:
    """Escapes HTML special characters to prevent issues with ParseMode.HTML."""
    if not isinstance(text, str):
        text = str(text)
    return text.replace("&", "&").replace("<", "<").replace(">", ">").replace('"', """)

async def send_notification_to_admin(context: ContextTypes.DEFAULT_TYPE, user_info: dict, event_type: str, button_pressed: str = None, event_details: str = None) -> None:
    """Sends a notification message to the admin, with optional button and details."""
    if ADMIN_ID is None: # Do not send if ADMIN_ID is not set (e.g., if set to None intentionally)
        return
        
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

    # Apply escape_html to ALL dynamic variables that go into the message content
    escaped_event_type = escape_html(event_type)
    escaped_first_name = escape_html(first_name)
    escaped_username = escape_html(username) if username and username != 'N/A' else 'N/A'
    escaped_chat_type = escape_html(chat_type)
    escaped_button_pressed = escape_html(button_pressed) if button_pressed else 'N/A'
    escaped_event_details = escape_html(event_details) if event_details else 'N/A'

    escaped_user_id = escape_html(user_id)
    escaped_chat_id = escape_html(chat_id)
    escaped_message_time = escape_html(message_time)

    notification_message = (
        f"New User Interaction!\n\n"
        f"Event Type: <b>{escaped_event_type}</b>\n"
        f"User ID: <code>{escaped_user_id}</code>\n"
        f"First Name: <b>{escaped_first_name}</b>\n"
        f"Username: <code>@{escaped_username}</code>" if escaped_username != 'N/A' else f"Username: <code>N/A</code>\n"
        f"Chat ID: <code>{escaped_chat_id}</code>\n"
        f"Chat Type: <b>{escaped_chat_type}</b>\n"
        f"Time: <code>{escaped_message_time}</code>\n"
    )
    if button_pressed and button_pressed != 'N/A':
        notification_message += f"Button Pressed: <b>{escaped_button_pressed}</b>\n"
    if event_details and event_details != 'N/A':
        # Truncate details if too long for notification, e.g., first 200 chars
        truncated_details = escaped_event_details[:200] + "..." if len(escaped_event_details) > 200 else escaped_event_details
        notification_message += f"Details: <code>{truncated_details}</code>\n"
    
    try:
        await context.bot.send_message(
            chat_id=ADMIN_ID,
            text=notification_message,
            parse_mode=ParseMode.HTML
        )
        last_admin_notification_time[user_id] = current_time # Update last sent time
        logger.info(f"Admin notification sent for user {user_id} (event: {event_type}).")
    except Exception as e:
        logger.error(f"Failed to send admin notification for user {user_id}: {e}")


async def save_user_to_db(update: Update, context: ContextTypes.DEFAULT_TYPE, button_pressed: str = None, event_details: str = None) -> None:
    """Saves or updates user information in the database and sends admin notification."""
    if update.message:
        user = update.message.from_user
        chat_id = update.message.chat_id
        chat_type = update.message.chat.type
    elif update.callback_query:
        user = update.callback_query.from_user
        chat_id = update.callback_query.message.chat_id
        chat_type = update.callback_query.message.chat.type
    else:
        logger.warning("save_user_to_db called without a valid message or callback_query object.")
        return

    if not user:
        logger.warning("save_user_to_db called without a valid user object.")
        return

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
            await send_notification_to_admin(context, user_info, "New User Added", button_pressed, event_details)
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
                await send_notification_to_admin(context, user_info, "User Info Updated", button_pressed, event_details)
            else:
                logger.info(f"DB: User {user.id} ({user.first_name}) already exists and no info changed for chat {chat_id}. Not updating.")
                # Send a general interaction notification even if user info didn't change
                await send_notification_to_admin(context, user_info, "User Interacted", button_pressed, event_details)

    except IntegrityError:
        session.rollback()
        logger.warning(f"DB: User {user.id} already exists (race condition) for chat {chat_id}. Rolling back.")
        await send_notification_to_admin(context, user_info, "User Interacted (Race Condition)", button_pressed, event_details)
    except Exception as e:
        session.rollback()
        logger.error(f"DB: Error saving/updating user {user.id} for chat {chat_id}: {e}")
        user_info['error'] = str(e)
        await send_notification_to_admin(context, user_info, f"Database Error: {e}", button_pressed, event_details)
    finally:
        session.close()

# --- Command Handlers ---
async def record_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handler for messages that are NOT commands, NOT specific keyboard button texts,
    and are potentially URLs for download based on user state or GPT chat.
    It also saves user info for general interactions.
    """
    if not update.message or not update.message.text:
        return

    if update.message.from_user:
        await save_user_to_db(update, context, event_details=update.message.text)
    
    user_data = context.user_data
    user_id = update.message.from_user.id
    user_message = update.message.text

    if user_data.get('state') == GPT_STATE:
        if user_id not in openai_conversations:
            openai_conversations[user_id] = [{"role": "system", "content": "You are a helpful assistant."}]
            logger.warning(f"GPT conversation not found for user {user_id} in GPT_STATE, initializing new one.")

        openai_conversations[user_id].append({"role": "user", "content": user_message})

        loading_emojis = ["...", ". . .", "...."]
        loading_message_text = escape_html("GPT is thinking")
        processing_message = await update.message.reply_text(
            loading_message_text + loading_emojis[0],
            parse_mode=ParseMode.HTML
        )

        animation_running = True
        async def animate_loading_gpt():
            index = 0
            while animation_running:
                try:
                    await processing_message.edit_text(
                        loading_message_text + loading_emojis[index % len(loading_emojis)],
                        parse_mode=ParseMode.HTML
                    )
                    index += 1
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.debug(f"GPT loading animation error: {e}")
                    break

        animation_task = asyncio.create_task(animate_loading_gpt())

        try:
            response = await asyncio.to_thread(
                openai_client.chat.completions.create,
                model=OPENAI_MODEL_NAME,
                messages=openai_conversations[user_id]
            )
            
            gpt_text_response = response.choices[0].message.content
            
            openai_conversations[user_id].append({"role": "assistant", "content": gpt_text_response})
            
            escaped_gpt_response = escape_html(gpt_text_response)

            animation_running = False
            await animation_task
            
            await processing_message.edit_text(
                escaped_gpt_response,
                parse_mode=ParseMode.HTML
            )
            logger.info(f"GPT replied to user {user_id}: {gpt_text_response[:50]}...")
        except openai.APIErrors.RateLimitError as e:
            logger.warning(f"OpenAI rate limit hit for user {user_id}: {e}")
            animation_running = False
            if not animation_task.done(): animation_task.cancel(); await animation_task
            await processing_message.edit_text(
                escape_html("I'm experiencing high traffic right now. Please try again in a moment."),
                parse_mode=ParseMode.HTML
            )
        except openai.APIErrors.AuthenticationError as e:
            logger.critical(f"OpenAI authentication error for user {user_id}: {e}")
            animation_running = False
            if not animation_task.done(): animation_task.cancel(); await animation_task
            await processing_message.edit_text(
                escape_html("There's an issue with my API key. Please contact the bot administrator."),
                parse_mode=ParseMode.HTML
            )
        except openai.APIErrors.APIError as e:
            logger.error(f"OpenAI API error for user {user_id}: {e}", exc_info=True)
            animation_running = False
            if not animation_task.done(): animation_task.cancel(); await animation_task
            await processing_message.edit_text(
                escape_html("I'm sorry, I encountered an issue with the OpenAI API. Please try again later."),
                parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logger.error(f"Error during GPT chat for user {user_id}: {e}", exc_info=True)
            animation_running = False
            if not animation_task.done():
                animation_task.cancel()
                try: await animation_task
                except asyncio.CancelledError: pass

            await processing_message.edit_text(
                escape_html("I'm sorry, I couldn't process your request with GPT right now. Please try again later."),
                parse_mode=ParseMode.HTML
            )
        return

    if user_data.get('state') == AWAITING_URL_STATE and user_data.get('platform'):
        platform_in_state = user_data['platform']
        url = update.message.text
        
        user_data['state'] = None
        user_data['platform'] = None

        await download_content_from_url(update, context, platform_in_state, url)
        return

    await update.message.reply_text(
        escape_html("I didn't understand that. Please use the buttons below to interact with me, or send a valid URL after selecting a download option."),
        parse_mode=ParseMode.HTML,
        reply_markup=ReplyKeyboardMarkup(
            [[KeyboardButton("Download Videos/Audio"), KeyboardButton("GPT")], [KeyboardButton("Help")]],
            one_time_keyboard=False,
            resize_keyboard=True
        )
    )


async def handle_keyboard_download_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the 'Download Videos/Audio' keyboard button press."""
    if update.message and update.message.from_user:
        await save_user_to_db(update, context, button_pressed="Download Videos/Audio")

    await show_download_options(update, context)

async def handle_keyboard_help_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the 'Help' keyboard button press."""
    if update.message and update.message.from_user:
        await save_user_to_db(update, context, button_pressed="Help")
    
    await help_command(update, context)

async def handle_keyboard_gpt_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the 'GPT' keyboard button press."""
    if update.message and update.message.from_user:
        await save_user_to_db(update, context, button_pressed="GPT")

    user_id = update.message.from_user.id

    context.user_data['state'] = GPT_STATE
    
    if user_id not in openai_conversations:
        openai_conversations[user_id] = [{"role": "system", "content": "You are a helpful assistant."}]
        logger.info(f"Started new GPT conversation for user {user_id}")
    else:
        logger.info(f"Resuming GPT conversation for user {user_id}")
    
    keyboard = [[KeyboardButton("Exit GPT Chat")]]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=False, resize_keyboard=True)

    await update.message.reply_text(
        escape_html("Hello! I'm GPT. I'm ready to chat with you.\n\n"
                           "What's on your mind today? Ask me anything!"),
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup
    )

async def handle_exit_gpt_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the 'Exit GPT Chat' keyboard button press."""
    if update.message and update.message.from_user:
        await save_user_to_db(update, context, button_pressed="Exit GPT Chat")

    user_id = update.message.from_user.id
    context.user_data['state'] = None
    
    if user_id in openai_conversations:
        del openai_conversations[user_id]
        logger.info(f"Ended GPT conversation for user {user_id}")

    keyboard = [
        [KeyboardButton("Download Videos/Audio"), KeyboardButton("GPT")],
        [KeyboardButton("Help")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=False, resize_keyboard=True)

    await update.message.reply_text(
        escape_html("You've exited GPT chat. How can I help you further?"),
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup
    )

async def tag_all(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Resends the replied message and tags all *known* non-admin members.
    Mentions use custom text (first name) and user ID for notifications.
    Splits messages if too many mentions.
    """
    if not update.message.reply_to_message:
        await update.message.reply_text(
            escape_html("Please reply to a message with <code>/tag</code> to use this command."),
            parse_mode=ParseMode.HTML
        )
        return

    chat_id = update.message.chat_id
    replied_message = update.message.reply_to_message
    replied_message_text = replied_message.text or replied_message.caption

    if not replied_message_text:
        await update.message.reply_text(
            escape_html("The replied message does not contain any text or caption to resend."),
            parse_mode=ParseMode.HTML
        )
        return

    if update.message and update.message.from_user:
        await save_user_to_db(update, context, event_details=update.message.text)

    feedback_message = await update.message.reply_text(
        escape_html("Fetching user list and preparing mentions, please wait..."),
        parse_mode=ParseMode.HTML
    )

    session = Session()
    all_known_users_in_chat = []
    try:
        all_known_users_in_chat = session.query(User).filter_by(chat_id=chat_id).all()
        logger.info(f"DB: Found {len(all_known_users_in_chat)} known users for chat {chat_id}.")
    except Exception as e:
        logger.error(f"DB: Error querying users for chat {chat_id}: {e}")
        await feedback_message.edit_text(escape_html("An error occurred while fetching known users from the database. Please try again later."))
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
            escaped_mention_name = escape_html(mention_name)
            members_to_tag_links.append(f"<a href='tg://user?id={user_obj.user_id}'>{escaped_mention_name}</a>")

    if not members_to_tag_links:
        await feedback_message.edit_text(
            escape_html("No known non-admin members to tag in this group. Users must send a message first to be added to the tag list, and ensure bot privacy is off."),
            parse_mode=ParseMode.HTML
        )
        return

    logger.info(f"Prepared {len(members_to_tag_links)} members for tagging in chat {chat_id}.")

    # --- Prepare the message content ---
    final_replied_message_content = escape_html(replied_message_text)
    
    initial_command_text = ""
    if update.message.text and ' ' in update.message.text:
        initial_command_text = escape_html(update.message.text.split(' ', 1)[1])
    
    full_message_content_start = ""
    if initial_command_text:
        full_message_content_start += f"{initial_command_text}\n\n"
    full_message_content_start += f"{final_replied_message_content}\n\n"


    # --- Pagination Logic for Mentions ---
    messages_to_send = []
    current_mentions_group = []
    
    current_message_base_length = len(full_message_content_start) + len("\n ")
    
    for mention_link in members_to_tag_links:
        mention_length = len(mention_link)
        
        if (current_message_base_length + sum(len(m) + 1 for m in current_mentions_group) + mention_length > MAX_MESSAGE_LENGTH or
            len(current_mentions_group) >= MAX_MENTIONS_PER_MESSAGE):
            
            messages_to_send.append(full_message_content_start + " " + " ".join(current_mentions_group)) 
            
            current_mentions_group = []
            full_message_content_start = "" 
            current_message_base_length = len(" ".encode('utf-8')) 

        current_mentions_group.append(mention_link)

    if current_mentions_group:
        if not messages_to_send and full_message_content_start: 
             messages_to_send.append(full_message_content_start + " " + " ".join(current_mentions_group))
        elif messages_to_send: 
             messages_to_send.append(" " + " ".join(current_mentions_group))
        else: 
             messages_to_send.append(" " + " ".join(current_mentions_group))


    await feedback_message.edit_text(escape_html(f"Sending {len(messages_to_send)} messages with mentions..."))

    successful_sends = 0
    for i, message_text in enumerate(messages_to_send):
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=message_text,
                parse_mode=ParseMode.HTML,
                disable_notification=(i > 0 and len(messages_to_send) > 1) 
            )
            successful_sends += 1
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Telegram API: Failed to send tagged message part {i+1}/{len(members_to_tag_links)} for chat {chat_id}: {e}")

    if successful_sends == len(messages_to_send):
        await feedback_message.edit_text(
            escape_html(f"Successfully sent {successful_sends} messages with mentions for {len(members_to_tag_links)} members."),
            parse_mode=ParseMode.HTML
        )
    else:
        await feedback_message.edit_text(
            escape_html(f"Completed sending messages. Sent {successful_sends} out of {len(messages_to_send)} parts. Some errors occurred while sending mentions. Check logs for details and ensure bot privacy is off."),
            parse_mode=ParseMode.HTML
        )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message and saves the user."""
    if update.message and update.message.from_user:
        await save_user_to_db(update, context)

        keyboard = [
            [KeyboardButton("Download Videos/Audio"), KeyboardButton("GPT")],
            [KeyboardButton("Help")]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=False, resize_keyboard=True)

        await update.message.reply_text(
            escape_html("Hi there! I'm your multimedia download and group tagging bot.\n\n"
            "To get started, tap 'Download Videos/Audio' on the keyboard below, or 'Help' for more info. "
            "You can also chat with GPT for anything else you need!"),
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a help message."""
    if update.message:
        await save_user_to_db(update, context, button_pressed="Help (Command)")
    elif update.callback_query:
         await save_user_to_db(update, context, button_pressed="Help (Inline Button)")


    keyboard = [
        [KeyboardButton("Download Videos/Audio"), KeyboardButton("GPT")],
        [KeyboardButton("Help")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=False, resize_keyboard=True)

    message_text = escape_html("How to use the Bot:\n\n"
        "To Download Videos and Audio:\n"
        "- Tap the 'Download Videos/Audio' button on your keyboard or use the /download command.\n"
        "- Select the platform (TikTok, Facebook, Instagram, Pinterest, Twitter, YouTube, SoundCloud).\n"
        "- Send the full video/audio URL when prompted.\n\n"
        "To Chat with GPT (AI):\n"
        "- Tap the 'GPT' button on your keyboard.\n"
        "- Ask me anything you want! I'll try my best to answer.\n"
        "- To exit the conversation, tap the 'Exit GPT Chat' button.\n\n"
        "To Tag Group Members:\n"
        "1. Add me to your group and make me an Administrator (this helps me exclude admins from tags).\n"
        "2. Crucial: Go to @BotFather -> My Bots -> (Your Bot) -> Bot Settings -> Group Privacy -> <b>Turn off</b>.\n"
        "   Why? This allows me to see all messages in the group and build a list of members to tag.\n"
        "3. Wait for members to send messages. I can only tag users who have sent a message in the group <b>after</b> I've joined and my Group Privacy is off.\n"
        "4. To tag everyone: Reply to any message in the group with the command <code>/tag</code>.\n"
        "I will resend the replied message and mention all known non-admin members. "
        "Mentions will show their first name, not their username, ensuring privacy while still notifying them.\n\n"
        "Limitations:\n"
        "- Download functionality relies on <code>yt-dlp</code> and may not always work if the content is restricted or the platform changes its API, or if the file size exceeds Telegram's 2GB limit.\n"
        "- Some content might require a logged-in session or be geo-restricted.\n"
        "- I cannot tag users who have never sent a message since I joined.\n"
        "- Very large groups might experience delays or split messages due to Telegram's limits.")

    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.edit_message_text(
            message_text,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
    else:
        await update.message.reply_text(
            message_text,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )

async def show_download_options(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message_source = update.callback_query or update.message

    if not message_source or not message_source.from_user:
        logger.warning("show_download_options called without a valid message or user object.")
        return

    button_label = "Download Videos/Audio (Inline)" if update.callback_query else "Download Videos/Audio (Keyboard)"
    await save_user_to_db(update, context, button_pressed=button_label)

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
        await update.callback_query.answer()
        await update.callback_query.edit_message_text(
            escape_html("Please choose a platform to download from:"),
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )
    else:
        await update.message.reply_text(
            escape_html("Please choose a platform to download from:"),
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup
        )


async def handle_download_platform_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    
    if query.message and query.from_user:
        platform_name_for_notification = query.data.split(":")[1]
        await save_user_to_db(update, context, button_pressed=f"Selected Platform: {platform_name_for_notification}")

    platform_name = query.data.split(":")[1]
    
    context.user_data['state'] = AWAITING_URL_STATE
    context.user_data['platform'] = platform_name

    await query.edit_message_text(
        escape_html(f"Please send me the full URL for the {platform_name} content."),
        parse_mode=ParseMode.HTML
    )


# --- Universal Video/Audio Download Handler ---

async def download_content_from_url(update: Update, context: ContextTypes.DEFAULT_TYPE, platform: str, content_url: str) -> None:
    """
    Downloads content from a given URL using yt-dlp and sends it.
    Args:
        platform (str): The platform (e.g., 'TikTok', 'Facebook', 'Instagram', 'Pinterest', 'Twitter', 'YouTube', 'SoundCloud') for messaging.
        content_url (str): The URL to download.
    """
    
    valid_platforms = {
        "tiktok": "tiktok.com",
        "fb": "facebook.com",
        "facebook": "facebook.com",
        "instagram": "instagram.com",
        "insta": "instagram.com",
        "pinterest": "pinterest.com",
        "twitter": "twitter.com",
        "youtube": ["youtube.com", "youtu.be", "youtube.com", "youtu.be"], # Added common YouTube domains
        "soundcloud": "soundcloud.com"
    }
    
    platform_key = platform.lower()
    if platform_key == "insta":
        platform_key = "instagram"
    if platform_key == "fb":
        platform_key = "facebook"
    
    expected_domains = valid_platforms.get(platform_key)

    is_valid_url = False
    if content_url.startswith("http://") or content_url.startswith("https://"):
        if isinstance(expected_domains, list):
            if any(domain in content_url for domain in expected_domains):
                is_valid_url = True
        elif expected_domains and expected_domains in content_url:
            is_valid_url = True
    
    if not is_valid_url:
        await update.message.reply_text(
            escape_html(f"That doesn't look like a valid {platform} URL. Please provide a full URL."),
            parse_mode=ParseMode.HTML
        )
        return

    temp_dir_name = os.path.join(DOWNLOAD_DIR, str(uuid.uuid4()))
    os.makedirs(temp_dir_name, exist_ok=True)
    
    loading_emojis = ["...", ". . .", "...."]
    loading_message_text = escape_html(f"Getting your {platform} content, please wait")
    processing_message = await update.message.reply_text(
        loading_message_text + loading_emojis[0],
        parse_mode=ParseMode.HTML
    )

    animation_running = True
    async def animate_loading():
        index = 0
        while animation_running:
            try:
                await processing_message.edit_text(
                    loading_message_text + loading_emojis[index % len(loading_emojis)],
                    parse_mode=ParseMode.HTML
                )
                index += 1
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.debug(f"Loading animation error: {e}")
                break

    animation_task = asyncio.create_task(animate_loading())

    downloaded_file_path = None
    try:
        ydl_opts = {
            'noplaylist': True,
            'verbose': False,
            'logger': logger,
            'no_warnings': True,
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
        }

        if platform_key == "soundcloud":
            ydl_opts['format'] = 'bestaudio/best'
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }]
        else:
            ydl_opts['format'] = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'

        if os.path.exists('cookies.txt'):
            ydl_opts['cookiefile'] = 'cookies.txt'
            logger.info("Using cookies.txt for download.")
        else:
            logger.info("No cookies.txt found. Proceeding without cookies.")


        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(content_url, download=False)
            
            if info_dict.get('is_live'):
                animation_running = False
                if not animation_task.done():
                    animation_task.cancel()
                    try: await animation_task
                    except asyncio.CancelledError: pass

                await processing_message.edit_text(
                    escape_html("Sorry, I cannot download live streams. Please provide a link to a completed video."),
                    parse_mode=ParseMode.HTML
                )
                return
            
            file_extension = info_dict.get('actual_ext', info_dict.get('ext'))
            if platform_key == "soundcloud":
                file_extension = 'mp3'
            elif not file_extension:
                file_extension = 'mp4'

            suggested_filename_base = info_dict.get('title', info_dict.get('id', 'content'))
            suggested_filename_base = "".join(c for c in suggested_filename_base if c.isalnum() or c in (' ', '.', '_', '-')).strip()
            if len(suggested_filename_base) > 150:
                suggested_filename_base = suggested_filename_base[:150]
            
            suggested_filename = f"{suggested_filename_base}.{file_extension}"
            downloaded_file_path_template = os.path.join(temp_dir_name, suggested_filename)
            
            ydl_opts['outtmpl'] = downloaded_file_path_template

            logger.info(f"Attempting to download {content_url} from {platform} to {downloaded_file_path_template}")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                 ydl_download.download([content_url])
            
            logger.info(f"Download initiated via yt-dlp to temp directory.")

        downloaded_files = [f for f in os.listdir(temp_dir_name) if os.path.isfile(os.path.join(temp_dir_name, f))]
        if not downloaded_files:
            raise FileNotFoundError("yt-dlp did not download any file or file not found in temp directory.")
        
        downloaded_file_path = os.path.join(temp_dir_name, downloaded_files[0])


        file_size = os.path.getsize(downloaded_file_path)
        logger.info(f"Downloaded file size: {file_size / (1024*1024):.2f} MB")

        content_title = escape_html(info_dict.get('title', f'{platform} Content'))
        
        if platform_key == "soundcloud":
            animation_running = False
            if not animation_task.done():
                animation_task.cancel()
                try: await animation_task
                except asyncio.CancelledError: pass

            await processing_message.edit_text(
                escape_html(f"Audio downloaded! Sending as document ({file_size / (1024*1024):.2f} MB). Please wait, this might take a while. "),
                parse_mode=ParseMode.HTML
            )
            with open(downloaded_file_path, 'rb') as audio_file:
                await context.bot.send_document(
                    chat_id=update.message.chat_id,
                    document=InputFile(audio_file, filename=os.path.basename(downloaded_file_path)),
                    caption=f"Here's your {platform} audio: {content_title}",
                    parse_mode=ParseMode.HTML,
                    read_timeout=300,
                    write_timeout=300,
                    connect_timeout=300
                )
        elif file_size > TELEGRAM_VIDEO_LIMIT_BYTES:
            animation_running = False
            if not animation_task.done():
                animation_task.cancel()
                try: await animation_task
                except asyncio.CancelledError: pass

            await processing_message.edit_text(
                escape_html(f"Video downloaded! Sending as document due to size ({file_size / (1024*1024):.2f} MB). Please wait, this might take a while. "),
                parse_mode=ParseMode.HTML
            )
            with open(downloaded_file_path, 'rb') as video_file:
                await context.bot.send_document(
                    chat_id=update.message.chat_id,
                    document=InputFile(video_file, filename=os.path.basename(downloaded_file_path)),
                    caption=f"Here's your {platform} video: {content_title}",
                    parse_mode=ParseMode.HTML,
                    read_timeout=300,
                    write_timeout=300,
                    connect_timeout=300
                )
        else:
            animation_running = False
            if not animation_task.done():
                animation_task.cancel()
                try: await animation_task
                except asyncio.CancelledError: pass

            await processing_message.edit_text(
                escape_html(f"Video downloaded! Sending in high quality ({file_size / (1024*1024):.2f} MB). Please wait. "),
                parse_mode=ParseMode.HTML
            )
            with open(downloaded_file_path, 'rb') as video_file:
                await context.bot.send_video(
                    chat_id=update.message.chat_id,
                    video=InputFile(video_file, filename=os.path.basename(downloaded_file_path)),
                    caption=f"Here's your {platform} video: {content_title}",
                    parse_mode=ParseMode.HTML,
                    supports_streaming=True,
                    read_timeout=300, 
                    write_timeout=300,
                    connect_timeout=300
                )

        await processing_message.delete()
        await update.message.reply_text(
            escape_html(f"Your {platform} content has been sent successfully! Enjoy!"),
            parse_mode=ParseMode.HTML
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
        elif "rate-limit reached" in error_message or "login required" in error_message or "Please sign in" in error_message:
            user_facing_error = f"Download failed due to rate-limiting or login requirement. Some content from {platform} may require a logged-in session or you might have to provide cookies (advanced)."
        elif "network error" in error_message or "Connection reset by peer" in error_message:
            user_facing_error = "A network error occurred during download. Please try again later."
        elif "no suitable formats found" in error_message:
            user_facing_error = "No suitable format found for download. The content might be protected or unusual."
        elif "ffmpeg is not installed" in error_message:
            user_facing_error = "FFmpeg is not installed on the server, which is required to process this video. Please inform the bot administrator."
        
        animation_running = False
        if not animation_task.done():
            animation_task.cancel()
            try: await animation_task
            except asyncio.CancelledError: pass

        await processing_message.edit_text(
            escape_html(f"Failed to download the {platform} content: <code>{escape_html(user_facing_error)}</code>\n\n"
                               "Please ensure the link is public and valid. Try again later."),
            parse_mode=ParseMode.HTML
        )
    except FileNotFoundError as e:
        logger.error(f"File system error during {platform} download for {content_url}: {e}")
        animation_running = False
        if not animation_task.done():
            animation_task.cancel()
            try: await animation_task
            except asyncio.CancelledError: pass

        await processing_message.edit_text(
            escape_html(f"A file error occurred: <code>{escape_html(str(e))}</code>\n\n"
                               "The content might not have been downloaded correctly. Please try again. "),
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        logger.error(f"General error processing {platform} download for {content_url}: {e}", exc_info=True)
        animation_running = False
        if not animation_task.done():
            animation_task.cancel()
            try: await animation_task
            except asyncio.CancelledError: pass

        await processing_message.edit_text(
            escape_html(f"An unexpected error occurred while processing your request: <code>{escape_html(str(e))}</code>\n\n"
                               "Please try again later. If the issue persists, contact support. "),
            parse_mode=ParseMode.HTML
        )
    finally:
        animation_running = False
        if not animation_task.done():
            animation_task.cancel()
            try:
                await animation_task
            except asyncio.CancelledError:
                pass

        if os.path.exists(temp_dir_name):
            try:
                shutil.rmtree(temp_dir_name)
                logger.info(f"Cleaned up temporary directory: {temp_dir_name}")
            except OSError as e:
                logger.error(f"Error removing temporary directory {temp_dir_name}: {e}")

async def tiktok_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        await save_user_to_db(update, context, event_details=update.message.text)
        await download_content_from_url(update, context, "TikTok", context.args[0])
    else:
        await save_user_to_db(update, context, event_details=update.message.text)
        await update.message.reply_text(
            escape_html("Please use the 'Download Videos/Audio' button to select a platform, then send the URL.")
        )

async def fb_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        await save_user_to_db(update, context, event_details=update.message.text)
        await download_content_from_url(update, context, "Facebook", context.args[0])
    else:
        await save_user_to_db(update, context, event_details=update.message.text)
        await update.message.reply_text(
            escape_html("Please use the 'Download Videos/Audio' button to select a platform, then send the URL.")
        )

async def insta_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        await save_user_to_db(update, context, event_details=update.message.text)
        await download_content_from_url(update, context, "Instagram", context.args[0])
    else:
        await save_user_to_db(update, context, event_details=update.message.text)
        await update.message.reply_text(
            escape_html("Please use the 'Download Videos/Audio' button to select a platform, then send the URL.")
        )

async def pinterest_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        await save_user_to_db(update, context, event_details=update.message.text)
        await download_content_from_url(update, context, "Pinterest", context.args[0])
    else:
        await save_user_to_db(update, context, event_details=update.message.text)
        await update.message.reply_text(
            escape_html("Please use the 'Download Videos/Audio' button to select a platform, then send the URL.")
        )

async def twitter_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        await save_user_to_db(update, context, event_details=update.message.text)
        await download_content_from_url(update, context, "Twitter", context.args[0])
    else:
        await save_user_to_db(update, context, event_details=update.message.text)
        await update.message.reply_text(
            escape_html("Please use the 'Download Videos/Audio' button to select a platform, then send the URL.")
        )

async def youtube_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        await save_user_to_db(update, context, event_details=update.message.text)
        await download_content_from_url(update, context, "YouTube", context.args[0])
    else:
        await save_user_to_db(update, context, event_details=update.message.text)
        await update.message.reply_text(
            escape_html("Please use the 'Download Videos/Audio' button to select a platform, then send the URL.")
        )

async def soundcloud_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        await save_user_to_db(update, context, event_details=update.message.text)
        await download_content_from_url(update, context, "SoundCloud", context.args[0])
    else:
        await save_user_to_db(update, context, event_details=update.message.text)
        await update.message.reply_text(
            escape_html("Please use the 'Download Videos/Audio' button to select a platform, then send the URL.")
        )


# --- Main Bot Logic and Heroku Integration ---
def main() -> None:
    """Start the bot."""
    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("tag", tag_all))
    
    application.add_handler(CommandHandler("tiktok", tiktok_command))
    application.add_handler(CommandHandler("fb", fb_command))
    application.add_handler(CommandHandler("insta", insta_command))
    application.add_handler(CommandHandler("pinterest", pinterest_command))
    application.add_handler(CommandHandler("twitter", twitter_command))
    application.add_handler(CommandHandler("youtube", youtube_command))
    application.add_handler(CommandHandler(
        "soundcloud", soundcloud_command))

    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Download Videos/Audio$"), handle_keyboard_download_button))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Help$"), handle_keyboard_help_button))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^GPT$"), handle_keyboard_gpt_button))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex("^Exit GPT Chat$"), handle_exit_gpt_button))

    application.add_handler(CallbackQueryHandler(show_download_options, pattern="^show_download_options$"))
    application.add_handler(CallbackQueryHandler(handle_download_platform_selection, pattern="^download_platform:"))
    application.add_handler(CallbackQueryHandler(help_command, pattern="^help_button$"))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, record_user_message))
    
    logger.info("Running in polling mode. If deployed on Heroku, ensure this is on a WORKER dyno.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
