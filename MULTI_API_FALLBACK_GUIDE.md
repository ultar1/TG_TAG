MULTI-API FALLBACK SYSTEM IMPLEMENTATION - November 16, 2025

OVERVIEW:
Implemented robust multi-API fallback systems for both TikTok and Pinterest downloads.
Each platform now has 3+ independent API methods that are tried in sequence until one succeeds.

=== TIKTOK DOWNLOAD SYSTEM ===

Methods (in order of priority):
1. tikwm.com API - Extracts video ID and fetches from tikwm
2. ttdownloader.com API - Alternative downloader service
3. musicaldown.com API - Another downloader API
4. yt-dlp - Direct video extraction fallback

Process Flow:
- User provides TikTok URL
- System attempts Method 1 (tikwm)
  - Extracts video ID from URL
  - Calls tikwm.com/api/feed/video
  - Checks for slideshow (type=2) or direct video
  - If slideshow: sends as media group
  - If video: downloads and sends as video
- If Method 1 fails → Try Method 2 (ttdownloader)
  - POST request to ttdownloader.com/api/download
  - Extracts download URL
  - Downloads and sends video
- If Method 2 fails → Try Method 3 (musicaldown)
  - POST request to musicaldown.com/api/download
  - Extracts video_url or download_url
  - Downloads and sends video
- If Method 3 fails → Try Method 4 (yt-dlp)
  - Direct extraction using yt-dlp
  - Best quality format selection
  - Downloads and sends video
- If all fail → Return error message

Functions:
- download_tiktok_image_post() - Main orchestrator
- try_tikwm_api() - Method 1
- try_ttdownloader_api() - Method 2
- try_musicaldown_api() - Method 3
- send_tiktok_result() - Result handler
- download_content_from_url() - Method 4 fallback

=== PINTEREST DOWNLOAD SYSTEM ===

Methods (in order of priority):
1. yt-dlp - Professional downloader with Pinterest support
2. Direct Pinterest API - fetch pin data from Pinterest API
3. pinterestdownloader.com - External service
4. Manual HTML Scraping - Extract media URLs from page

Process Flow:
- User provides Pinterest URL
- System attempts Method 1 (yt-dlp)
  - Configures yt-dlp with Pinterest options
  - Downloads to temporary directory
  - Detects file type (image/video)
  - Validates file integrity
- If Method 1 fails → Try Method 2 (Direct API)
  - Extracts pin ID from URL
  - Calls Pinterest API
  - Extracts image_url or video url
  - Returns media URL
- If Method 2 fails → Try Method 3 (pinterestdownloader)
  - POST to pinterestdownloader.com
  - Parses HTML response for download links
  - Looks for pinimg or pcdn URLs
- If Method 3 fails → Try Method 4 (Manual Scrape)
  - Fetches Pinterest page HTML
  - Searches for <img> tags with pinimg/pcdn
  - Searches for <video> and <source> tags
  - Extracts highest quality media URL
- If all fail → Return error message

Functions:
- download_pinterest_content() - Main orchestrator
- try_pinterest_ytdlp() - Method 1
- try_pinterest_direct_api() - Method 2
- try_pinterest_downloader_service() - Method 3
- try_pinterest_manual_scrape() - Method 4
- send_pinterest_result() - Result handler

=== ERROR HANDLING ===

Each method includes:
- Try-except blocks with specific error logging
- Timeout handling (15-30 seconds)
- Size validation
- File integrity checks
- Proper cleanup of temporary files

Result types handled:
- Slideshow (TikTok) - Media group of images
- Video - Direct video download
- Image - Photo download
- File - Generic document download
- URL - Download from URL and send

=== FEATURES ===

TikTok:
- Slideshow detection and download
- Video quality optimization
- Multiple fallback methods
- Error-specific messaging

Pinterest:
- Image and video support
- File type detection
- Size validation (max 50MB for Telegram)
- Graceful degradation

General:
- No emoji indicators (text-based status)
- Informative progress updates
- Multiple API redundancy
- Comprehensive error logging
- Temporary file cleanup

=== DEPENDENCIES ===

New/Updated imports:
- requests (for API calls)
- BeautifulSoup (for HTML scraping)
- yt-dlp (for video extraction)
- uuid (for temp directories)
- os, shutil (for file handling)

=== TESTING RECOMMENDATIONS ===

TikTok Tests:
1. Test with slideshow URL - Should extract all images
2. Test with regular video - Should extract video
3. Test with invalid URL - Should show error
4. Simulate API failures - Should fallback correctly

Pinterest Tests:
1. Test with image pin - Should extract image
2. Test with video pin - Should extract video
3. Test with private pin - Should show appropriate error
4. Test large video - Should show size error

Fallback Tests:
1. Disable each method and verify fallback chain
2. Test with slow connections - Should timeout properly
3. Test with region-restricted content - Should fail gracefully

=== KNOWN LIMITATIONS ===

- APIs may rate-limit after multiple requests
- Region-specific restrictions apply
- Private/deleted content cannot be accessed
- Telegram 50MB file size limit
- Some APIs may change URLs without notice

=== FUTURE IMPROVEMENTS ===

- Add caching to reduce API calls
- Implement request retry logic with exponential backoff
- Add quality selection UI
- Support for batch downloads
- Add progress bar for large downloads
