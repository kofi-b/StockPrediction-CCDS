from pathlib import Path
import configparser
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Initialize configparser to read the .ini file
config = configparser.ConfigParser()
config.read(PROJ_ROOT / "stock_prediction" / "config.ini")

# Access API keys from the config file
API_KEYS = {
    "alpha_vantage": config["API_KEYS"]["alpha_vantage_api_key"],
    "newsapi": config["API_KEYS"]["newsapi_key"],
    "reddit_client_id": config["API_KEYS"]["reddit_client_id"],
    "reddit_client_secret": config["API_KEYS"]["reddit_client_secret"],
    "reddit_user_agent": config["API_KEYS"]["reddit_user_agent"],
    "finnhub": config["API_KEYS"]["finnhub_api_key"],
}
# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
