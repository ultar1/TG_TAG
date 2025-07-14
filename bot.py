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
    
    # Correct list of special characters for MarkdownV2, including backslash itself to be handled first.
    special_chars = r'_*[]()~`>#+-=|{}.!'
    
    # Process backslashes first, then other special characters
    escaped_text = text.replace('\\', '\\\\')
    for char in special_chars:
        escaped_text = escaped_text.replace(char, f'\\{char}')
    return escaped_text

async def save_user_to_db(update: Update):
    """Saves or updates user information in the database."""
    user = update.message.from_user
    chat_id = update.message.chat_id

    if update.message.chat.type not in ["group", "supergroup", "private"]: # Include private chats for /tiktok
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
        # FIXED: Escaped static string
        await update.message.reply_text(
            escape_markdown_v2("Please reply to a message with `/tag` to use this command."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    chat_id = update.message.chat_id
    replied_message = update.message.reply_to_message
    replied_message_text = replied_message.text or replied_message.caption

    if not replied_message_text:
        # FIXED: Escaped static string
        await update.message.reply_text(
            escape_markdown_v2("The replied message does not contain any text or caption to resend."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    # FIXED: Escaped static string
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
        # FIXED: Escaped static string
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
        # FIXED: Escaped static string
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
    
    current_message_base_length_bytes = len(full_message_content_start.encode('utf-8')) + len("\n ".encode('utf-8'))
    
    for mention_link in members_to_tag_links:
        mention_length_bytes = len(mention_link.encode('utf-8'))
        
        if (current_message_base_length_bytes + sum(len(m.encode('utf-8')) + 1 for m in current_mentions_group) + mention_length_bytes > MAX_MESSAGE_LENGTH or
            len(current_mentions_group) >= MAX_MENTIONS_PER_MESSAGE):
            
            # This part should be safe now because full_message_content_start is escaped
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


    # FIXED: Escaped static string
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
        # FIXED: Escaped static string
        await feedback_message.edit_text(
            escape_markdown_v2(f"Successfully sent {successful_sends} messages with mentions for {len(members_to_tag_links)} members."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
    else:
        # FIXED: Escaped static string
        await feedback_message.edit_text(
            escape_markdown_v2(f"Completed sending messages. Sent {successful_sends} out of {len(messages_to_send)} parts. Some errors occurred while sending mentions. Check logs for details and ensure bot privacy is off."),
            parse_mode=ParseMode.MARKDOWN_V2
        )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message and saves the user."""
    if update.message and update.message.from_user:
        await save_user_to_db(update)
        # FIXED: Escaped static string
        await update.message.reply_text(
            escape_markdown_v2("Hi there! I'm a group tagging bot. \n"
            "To use me, reply to any message in a group with `/tag`.\n"
            "I'll resend the message and mention all *known* non-admin members "
            "(those who have messaged in the group while I'm active).\n\n"
            "You can also use `/tiktok <TikTok_URL>` to download TikTok videos! \n\n"
            "Make sure to turn off Group Privacy for me via @BotFather!"),
            parse_mode=ParseMode.MARKDOWN_V2
        )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a help message."""
    # FIXED: Escaped static string. Note: the `**` for bold is Markdown itself, not special char.
    await update.message.reply_text(
        escape_markdown_v2("**How to use the Tag Bot:**\n\n"
        "1. **Add me to your group** and **make me an Administrator** (this helps me exclude admins from tags).\n"
        "2. **Crucial:** Go to @BotFather -> My Bots -> (Your Bot) -> Bot Settings -> Group Privacy -> **Turn off**.\n"
        "   *Why?* This allows me to see all messages in the group and build a list of members to tag.\n"
        "3. **Wait for members to send messages.** I can only tag users who have sent a message in the group *after* I've joined and my Group Privacy is off.\n"
        "4. **To tag everyone:** Reply to any message in the group with the command `/tag`.\n"
        "I will resend the replied message and mention all known non-admin members. "
        "Mentions will show their first name, not their username, ensuring privacy while still notifying them.\n\n"
        "**To Download TikTok Videos:**\n"
        "- Use the command `/tiktok <TikTok_URL>` in any chat (group or private).\n"
        "- Provide the full TikTok video URL after the command.\n\n"
        "**Limitations:**\n"
        "- I cannot tag users who have never sent a message since I joined.\n"
        "- Very large groups might experience delays or split messages due to Telegram's limits.\n"
        "- TikTok download functionality relies on external services and may not always work if the service is down or TikTok changes its API."),
        parse_mode=ParseMode.MARKDOWN_V2
    )

# --- TikTok Download Handler ---

async def tiktok_download(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Downloads a TikTok video given its URL."""
    await save_user_to_db(update) # Ensure user is saved
    
    if not context.args:
        await update.message.reply_text(
            escape_markdown_v2("Please provide a TikTok video URL after the `/tiktok` command\. \n"
                               "Example: `/tiktok https://www.tiktok.com/@username/video/1234567890`"),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    tiktok_url = context.args[0]
    # Basic URL validation (you might want a more robust regex)
    if not (tiktok_url.startswith("http://") or tiktok_url.startswith("https://")) or "tiktok.com" not in tiktok_url:
        await update.message.reply_text(
            escape_markdown_v2("That doesn't look like a valid TikTok URL\. Please provide a full URL\."),
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    processing_message = await update.message.reply_text(
        escape_markdown_v2("⏳ Getting your TikTok video, please wait... This might take a moment\. ⏳"),
        parse_mode=ParseMode.MARKDOWN_V2
    )

    try:
        # --- Placeholder for TikTok Video Download Logic ---
        # This is where you would integrate with a TikTok download API or a tool like yt-dlp.
        # For demonstration purposes, I'm just creating a dummy URL.
        # In a real scenario, you'd make an HTTP request to a service or run a subprocess.

        # Example using a hypothetical API:
        # import httpx # You'd need to install httpx or requests
        # api_url = f"https://your-tiktok-downloader-api.com/download?url={urllib.parse.quote(tiktok_url)}"
        # async with httpx.AsyncClient() as client:
        #     response = await client.get(api_url, timeout=30.0) # Add a timeout
        #     response.raise_for_status()
        #     data = response.json()
        #     video_direct_url = data.get("video_url")
        #     if not video_direct_url:
        #         raise ValueError("Could not get video URL from API response.")

        # For a more robust solution, especially for self-hosting, consider using yt-dlp:
        # You'd need to run yt-dlp as a subprocess and parse its output.
        # Example (requires yt-dlp to be installed on your system/dyno and executable in PATH):
        # process = await asyncio.create_subprocess_exec(
        #     "yt-dlp", "--no-warnings", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        #     "--get-url", tiktok_url,
        #     stdout=asyncio.subprocess.PIPE,
        #     stderr=asyncio.subprocess.PIPE
        # )
        # stdout, stderr = await process.communicate()
        # if process.returncode != 0:
        #     raise RuntimeError(f"yt-dlp failed: {stderr.decode()}")
        # video_direct_url = stdout.decode().strip()

        # Dummy response for demonstration
        video_direct_url = "https://example.com/your_tiktok_video_download_link.mp4" # Replace with actual logic
        
        # Check if the generated URL is valid (not empty and starts with http)
        if not video_direct_url or not (video_direct_url.startswith("http://") or video_direct_url.startswith("https://")):
            raise ValueError("Failed to get a valid direct video download link.")

        await processing_message.edit_text(
            escape_markdown_v2(f"✅ Here's your TikTok video download link: \n\n{video_direct_url}\n\n"
                               "💡 You can usually click this link to open the video in your browser and then download it\."),
            parse_mode=ParseMode.MARKDOWN_V2,
            disable_web_page_preview=False # Allow link preview for the video
        )

    except Exception as e:
        logger.error(f"Error downloading TikTok video for {tiktok_url}: {e}")
        await processing_message.edit_text(
            escape_markdown_v2(f"❌ Failed to download the TikTok video\. Error: `{escape_markdown_v2(str(e))}`\n\n"
                               "Possible reasons:\n"
                               "  - The link is broken or private\n"
                               "  - TikTok changed its format (I need an update)\n"
                               "  - The download service is temporarily unavailable\n"
                               "Please try again later or with a different link\. ✨"),
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
    application.add_handler(CommandHandler("tiktok", tiktok_download)) # New TikTok handler

    # Message handler to record all users who send messages in a group
    application.add_handler(MessageHandler(filters.ChatType.GROUPS & ~filters.COMMAND, record_user_message))
    
    # Also record users who interact with the bot in private chats
    application.add_handler(MessageHandler(filters.ChatType.PRIVATE & ~filters.COMMAND, record_user_message))

    logger.info("Running in polling mode. If deployed on Heroku, ensure this is on a WORKER dyno.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()

