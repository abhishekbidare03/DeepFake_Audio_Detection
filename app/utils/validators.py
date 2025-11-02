import os

# Allowed audio extensions and size limit
ALLOWED_EXTENSIONS = {'.wav', '.flac'}
# 10 MB default max file size for uploads
MAX_FILE_SIZE_BYTES = int(os.environ.get('MAX_AUDIO_UPLOAD_BYTES', 10 * 1024 * 1024))


def is_valid_audio_filename(filename: str) -> bool:
    if not filename:
        return False
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def max_file_size_ok(content: bytes) -> bool:
    return len(content) <= MAX_FILE_SIZE_BYTES


def sanitize_filename(filename: str) -> str:
    # Minimal sanitization: remove path separators
    return os.path.basename(filename)
