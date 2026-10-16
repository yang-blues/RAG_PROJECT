import os

from dotenv import load_dotenv

load_dotenv(override=True)

UNSTRUCTURED_KEY = os.getenv('UNSTRUCTURED_KEY')