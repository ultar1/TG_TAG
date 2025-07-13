import logging
import os
import sys
import signal
import asyncio

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.constants import ParseMode
from sqlalchemy import create_engine, Column, Integer, String, UniqueConstraint
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError, OperationalError
import urllib.parse

# --- Configuration ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("BOT_TOKEN")
DATABASE_URL = os.environ.get("DATABASE_URL")
# HEROKU_APP_NAME is no longer strictly needed for polling, but keeping it as an env var is harmless.
# HEROKU_APP_NAME = os.environ.get("HEROKU_APP_NAME") 

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
    user_id = Column(Integer, primary_key=True, nullable=False)
    first_name = Column(String, nullable=True)
    username = Column(String, nullable=True)
    chat_id = Column(Integer, primary_key=True, nullable=False)

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
    
    special_chars = r'_*[]()~`>#+-=|{}.!'
    escaped_text = text.replace('\\', '\\\\')
    for char in special_chars:
        escaped_text = escaped_text.replace(char, f'\\{char}')
    return escaped_text

async def save_user_to_db(update: Update):
    """Saves or updates user information in the database."""
    user = update.message.from_user
    chat_id = update.message.chat_id

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
            if (existing_user.first_name != user.first_name or
                existing_user.username != user.username):
                existing_user.first_name = user.first_name
                existing_user.username = user.username
                session.commit()
                logger.info(f"DB: Updated user {user.id} ({user.first_name}) for chat {chat_id}")

    except IntegrityError:
        session.rollback()
        logger.warning(f"DB: User {user.id} already exists (race condition) for chat {chat_id}. Rolling back.")
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

    feedback_message = await update.message.reply_text(
        "Fetching user list and preparing mentions, please wait...",
        parse_mode=ParseMode.MARKDOWN_V2
    )

    session = Session()
    all_known_users_in_chat = []
    try:
        all_known_users_in_chat = session.query(User).filter_by(chat_id=chat_id).all()
        logger.info(f"DB: Found {len(all_known_users_in_chat)} known users for chat {chat_id}.")
    except Exception as e:
        logger.error(f"DB: Error querying users for chat {chat_id}: {e}")
        await feedback_message.edit_text("An error occurred while fetching known users from the database. Please try again later.")
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
            "No known non-admin members to tag in this group. "
            "Users must send a message first to be added to the tag list, and ensure bot privacy is off.",
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
    
    current_message_base_length_bytes = len(full_message_content_start.encode('utf-8')) + len("\n ".encode('utf-8')) # Removed emoji
    
    for mention_link in members_to_tag_links:
        mention_length_bytes = len(mention_link.encode('utf-8'))
        
        if (current_message_base_length_bytes + sum(len(m.encode('utf-8')) + 1 for m in current_mentions_group) + mention_length_bytes > MAX_MESSAGE_LENGTH or
            len(current_mentions_group) >= MAX_MENTIONS_PER_MESSAGE):
            
            messages_to_send.append(full_message_content_start + " " + " ".join(current_mentions_group)) # Removed emoji
            
            current_mentions_group = []
            full_message_content_start = "" 
            current_message_base_length_bytes = len(" ".encode('utf-8')) # Removed emoji

        current_mentions_group.append(mention_link)

    if current_mentions_group:
        if not messages_to_send and full_message_content_start: 
             messages_to_send.append(full_message_content_start + " " + " ".join(current_mentions_group)) # Removed emoji
        elif messages_to_send: 
             messages_to_send.append(" " + " ".join(current_mentions_group)) # Removed emoji
        else: 
             messages_to_send.append(" " + " ".join(current_mentions_group)) # Removed emoji


    await feedback_message.edit_text(f"Sending {len(messages_to_send)} messages with mentions...")

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
            f"Successfully sent {successful_sends} messages with mentions for {len(members_to_tag_links)} members.",
            parse_mode=ParseMode.MARKDOWN_V2
        )
    else:
        await feedback_message.edit_text(
            f"Completed sending messages. Sent {successful_sends} out of {len(messages_to_send)} parts. "
            "Some errors occurred while sending mentions. Check logs for details and ensure bot privacy is off.",
            parse_mode=ParseMode.MARKDOWN_V2
        )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message and saves the user."""
    if update.message and update.message.from_user:
        await save_user_to_db(update)
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

    # For polling, simply run the application without webhook specifics
    # IMPORTANT: If deploying to Heroku, this should be run on a WORKER dyno, NOT a WEB dyno.
    # Your Procfile should be something like: worker: python bot.py
    logger.info("Running in polling mode. If deployed on Heroku, ensure this is on a WORKER dyno.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
