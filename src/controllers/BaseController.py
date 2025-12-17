from utils import get_settings, Settings
import os
import random
import string

class BaseController:
    def __init__(self):
        self.settings: Settings = get_settings()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.file_dir = os.path.join(self.base_dir, 'assets', 'files')

    def generate_random_string(self, length: int = 12) -> str:
        letters = string.ascii_letters + string.digits
        return ''.join(random.choice(letters) for i in range(length))

    def get_filename(self, filename: str) -> str:
        """Return the original filename without changes."""
        return filename