import os
from dotenv import load_dotenv

load_dotenv()

api_key: str = os.environ["API_KEY"]
base_url: str = os.environ["BASE_URL"]