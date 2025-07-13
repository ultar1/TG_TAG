import logging
import os
import sys
import signal
import asyncio # For better async operations and tasks

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.constants import ParseMode
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError, OperationalError
import urllib.parse # For parsing DATABASE_URL

# --- Configuration ---
# Set up logging at the beginning
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Environment variables
BOT_TOKEN = os.environ.get("BOT_TOKEN")
DATABASE_URL = os.environ.get("DATABASE_URL")
HEROKU_APP_NAME = os.environ.get("HEROKU_APP_NAME")

# Validate environment variables
if not BOT_TOKEN:
    logger.critical("BOT_TOKEN environment variable not set. Exiting.")
    sys.exit(1)

if not DATABASE_URL:
    logger.critical("DATABASE_URL environment variable not set. Please add Heroku Postgres add-on. Exiting.")
    sys.exit(1)

# Heroku's DATABASE_URL might be 'postgres://', but SQLAlchemy expects 'postgresql://'
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# --- Database Setup ---
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    user_id = Column(Integer, primary_key=True, unique=True, nullable=False)
    first_name = Column(String, nullable=True)
    username = Column(String, nullable=True) # Storing username, but not using for mentions as per requirement
    chat_id = Column(Integer, nullable=False) # The group ID where the user was last seen

    def __repr__(self):
        return (f"<User(id={self.user_id}, first_name='{self.first_name}', "
                f"username='{self.username}', chat_id={self.chat_id})>")

# Create an engine for PostgreSQL
# Add connection pooling for better performance in a web environment
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)

try:
    Base.metadata.create_all(engine) # Create tables if they don't exist
    logger.info("Database tables checked/created successfully.")
except OperationalError as e:
    logger.critical(f"Failed to connect to database or create tables: {e}. Exiting.")
    sys.exit(1)

Session = sessionmaker(bind=engine)
# --- End Database Setup ---

# Constants for message splitting
MAX_MENTIONS_PER_MESSAGE = 50 # Telegram's effective limit for entities is usually around 100
MAX_MESSAGE_LENGTH = 4096 # Telegram's max message length

# --- Helper Functions ---
def escape_markdown_v2(text: str) -> str:
    """Escapes common MarkdownV2 special characters."""
    if not isinstance(text, str):
        return "" # Handle non-string inputs gracefully
    
    # Characters that need to be escaped in MarkdownV2
    # _, *, [, ], (, ), ~, `, >, #, +, -, =, |, {, }, ., !
    # Backslash itself needs to be escaped
    return (text.replace('\\', '\\\\')
                .replace('_', '\\_')
                .replace('*', '\\*')
                .replace('[', '\\[')
                .replace(']', '\\]')
                .replace('(', '\\(')
                .replace(')', '\\)')
                .replace('~', '\\~')
                .replace('`', '\\`')
                .replace('>', '\\>')
                .replace('#', '\\#')
                .replace('+', '\\+')
                .replace('-', '\\-')
                .replace('=', '\\=')
                .replace('|', '\\|')
                .replace('{', '\\{')
                .replace('}', '\\}')
                .replace('.', '\\.')
                .replace('!', '\\!'))

async def save_user_to_db(update: Update):
    """Saves or updates user information in the database."""
    user = update.message.from_user
    chat_id = update.message.chat_id

    # Avoid processing messages from channels or private chats not meant for user tracking
    if update.message.chat.type not in ["group", "supergroup"]:
        return

    session = Session()
    try:
        existing_user = session.query(User).filter_by(user_id=user.id, chat_id=chat_id).first()
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
        else:
            # Update user info if it changed (e.g., first_name, username)
            if (existing_user.first_name != user.first_name or
                existing_user.username != user.username):
                existing_user.first_name = user.first_name
                existing_user.username = user.username
                session.commit()
                logger.info(f"DB: Updated user {user.id} ({user.first_name}) for chat {chat_id}")

    except IntegrityError:
        session.rollback() # Handle potential race conditions where user is added simultaneously
        logger.warning(f"DB: User {user.id} already exists (race condition) for chat {chat_id}.")
    except Exception as e:
        session.rollback()
        logger.error(f"DB: Error saving/updating user {user.id} for chat {chat_id}: {e}")
    finally:
        session.close()

# --- Command Handlers ---
async def record_user_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for all messages to record user info, excluding commands."""
    if update.message and update.message.from_user:
        await save_user_to_db(update)

async def tag_all(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Resends the replied message and tags all *known* non-admin members.
    Mentions use custom text (first name) and user ID for notifications.
    Splits messages if too many mentions.
    """
    if not update.message.reply_to_message:
        await update.message.reply_text(
            "Please reply to a message with `/tag` to use this command.",
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    chat_id = update.message.chat_id
    replied_message = update.message.reply_to_message
    replied_message_text = replied_message.text or replied_message.caption

    if not replied_message_text:
        await update.message.reply_text(
            "The replied message does not contain any text or caption to resend.",
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    # Acknowledge the command quickly
    feedback_message = await update.message.reply_text(
        "Fetching user list and preparing mentions, please wait...",
        parse_mode=ParseMode.MARKDOWN_V2
    )

    session = Session()
    all_known_users_in_chat = []
    try:
        all_known_users_in_chat = session.query(User).filter_by(chat_id=chat_id).all()
        logger.info(f"Found {len(all_known_users_in_chat)} known users for chat {chat_id}.")
    except Exception as e:
        logger.error(f"DB: Error querying users for chat {chat_id}: {e}")
        await feedback_message.edit_text("An error occurred while fetching known users from the database. Please try again later.")
        return
    finally:
        session.close()

    members_to_tag_links = []
    current_chat_admins_ids = set()

    try:
        administrators = await context.bot.get_chat_administrators(chat_id)
        for admin in administrators:
            current_chat_admins_ids.add(admin.user.id)
            if admin.user.id == context.bot.id: # Exclude the bot itself
                current_chat_admins_ids.add(admin.user.id)
        logger.info(f"Found {len(current_chat_admins_ids)} administrators for chat {chat_id}.")
    except Exception as e:
        logger.warning(f"Telegram API: Could not retrieve chat administrators for chat {chat_id}: {e}. Bot might not be an admin, or API error. Admins will not be excluded from tagging if this fails.")
        # We proceed without admin exclusion if fetching admins fails

    for user_obj in all_known_users_in_chat:
        if user_obj.user_id not in current_chat_admins_ids:
            mention_name = user_obj.first_name if user_obj.first_name else "A User"
            escaped_mention_name = escape_markdown_v2(mention_name)
            members_to_tag_links.append(f"[{escaped_mention_name}](tg://user?id={user_obj.user_id})")

    if not members_to_tag_links:
        await feedback_message.edit_text(
            "No known non-admin members to tag in this group. "
            "Users must send a message first to be added to the tag list.",
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    logger.info(f"Prepared {len(members_to_tag_links)} members for tagging in chat {chat_id}.")

    # Escape the original message text once
    escaped_replied_message_text = escape_markdown_v2(replied_message_text)

    # --- Pagination Logic for Mentions ---
    messages_to_send = []
    current_message_parts = [escaped_replied_message_text, "\n\n📢"]
    mentions_count_in_current_message = 0

    for i, mention_link in enumerate(members_to_tag_links):
        # Check if adding the next mention exceeds limits
        temp_message = " ".join(current_message_parts + [mention_link])
        
        if len(temp_message) > MAX_MESSAGE_LENGTH or mentions_count_in_current_message >= MAX_MENTIONS_PER_MESSAGE:
            messages_to_send.append(" ".join(current_message_parts))
            current_message_parts = ["📢"] # Start new message with tag prefix
            mentions_count_in_current_message = 0
            
        current_message_parts.append(mention_link)
        mentions_count_in_current_message += 1

    # Add the last accumulated message if any
    if current_message_parts:
        messages_to_send.append(" ".join(current_message_parts))

    await feedback_message.edit_text(f"Sending {len(messages_to_send)} messages with mentions...")

    successful_sends = 0
    for i, message_text in enumerate(messages_to_send):
        try:
            await context.bot.send_message(
                chat_id=chat_id,
                text=message_text,
                parse_mode=ParseMode.MARKDOWN_V2,
                # Optional: Disable notification for subsequent messages if many
                disable_notification=(i > 0 and len(messages_to_send) > 1) 
            )
            successful_sends += 1
            # Add a small delay to respect API rate limits for sending multiple messages
            await asyncio.sleep(0.5) 
        except Exception as e:
            logger.error(f"Telegram API: Failed to send tagged message part {i+1}/{len(messages_to_send)} for chat {chat_id}: {e}")
            # Do not stop, try to send remaining parts

    if successful_sends == len(messages_to_send):
        await feedback_message.edit_text(
            f"Successfully sent {successful_sends} messages with mentions for {len(members_to_tag_links)} members.",
            parse_mode=ParseMode.MARKDOWN_V2
        )
    else:
        await feedback_message.edit_text(
            f"Completed sending messages. Sent {successful_sends} out of {len(messages_to_send)} parts. Some errors occurred. "
            "Check logs for details.",
            parse_mode=ParseMode.MARKDOWN_V2
        )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message and saves the user."""
    if update.message and update.message.from_user:
        await save_user_to_db(update) # Record user even if they start bot in private chat
        await update.message.reply_text(
            "Hi there! I'm a group tagging bot. \n"
            "To use me, reply to any message in a group with `/tag`.\n"
            "I'll resend the message and mention all *known* non-admin members "
            "(those who have messaged in the group while I'm active).\n\n"
            "Make sure to turn off Group Privacy for me via @BotFather!",
            parse_mode=ParseMode.MARKDOWN_V2
        )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a help message."""
    await update.message.reply_text(
        "**How to use the Tag Bot:**\n\n"
        "1. **Add me to your group** and **make me an Administrator** (this helps me exclude admins from tags).\n"
        "2. **Crucial:** Go to @BotFather -> My Bots -> (Your Bot) -> Bot Settings -> Group Privacy -> **Turn off**.\n"
        "   *Why?* This allows me to see all messages in the group and build a list of members to tag.\n"
        "3. **Wait for members to send messages.** I can only tag users who have sent a message in the group *after* I've joined and my Group Privacy is off.\n"
        "4. **To tag everyone:** Reply to any message in the group with the command `/tag`.\n"
        "I will resend the replied message and mention all known non-admin members. "
        "Mentions will show their first name, not their username, ensuring privacy while still notifying them.\n\n"
        "**Limitations:**\n"
        "- I cannot tag users who have never sent a message since I joined.\n"
        "- Very large groups might experience delays or split messages due to Telegram's limits.",
        parse_mode=ParseMode.MARKDOWN_V2
    )

# --- Main Bot Logic and Heroku Integration ---
def main() -> None:
    """Start the bot."""
    application = Application.builder().token(BOT_TOKEN).build()

    # Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("tag", tag_all))

    # Message handler to record all users who send messages in a group
    application.add_handler(MessageHandler(filters.ChatType.GROUPS & ~filters.COMMAND, record_user_message))
    
    # Also record users who interact with the bot in private chats
    application.add_handler(MessageHandler(filters.ChatType.PRIVATE & ~filters.COMMAND, record_user_message))

    # Heroku deployment specific logic (webhooks)
    if HEROKU_APP_NAME:
        PORT = int(os.environ.get('PORT', 8443))
        # Ensure the webhook URL is correctly formed with the app name and bot token
        webhook_url = f"https://{HEROKU_APP_NAME}.herokuapp.com/{BOT_TOKEN}"
        
        logger.info(f"Running in webhook mode on port {PORT} for app {HEROKU_APP_NAME}. Webhook URL: {webhook_url}")
        
        # Add graceful shutdown for Heroku
        def shutdown_handler(signum, frame):
            logger.info("Received signal, initiating graceful shutdown...")
            application.stop()
            sys.exit(0)

        signal.signal(signal.SIGTERM, shutdown_handler)
        signal.signal(signal.SIGINT, shutdown_handler) # For local Ctrl+C

        application.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=BOT_TOKEN, # Use token as url_path for security
            webhook_url=webhook_url,
            allowed_updates=Update.ALL_TYPES # Process all update types
        )
    else:
        logger.warning("HEROKU_APP_NAME environment variable not set. Running in polling mode. "
                       "This is suitable for local testing or non-Heroku deployment.")
        application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()

