import os
import re
import logging
from pathlib import Path
from huggingface_hub import get_token
from dotenv import load_dotenv

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load .env from repo root (two levels up from app/)
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# --- Hugging Face Configuration ---
# These models are confirmed to work with free tier Inference API
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "google/flan-t5-base")

# Fallback models - all confirmed working on free tier
FALLBACK_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/flan-t5-large",
    "mistralai/Mistral-7B-Instruct-v0.2",
]

# Get token from CLI login (huggingface-cli login)
HF_TOKEN = os.getenv("HF_API_TOKEN")
if not HF_TOKEN:
    HF_TOKEN = get_token()
HF_ADVISOR_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
if not HF_TOKEN:
    logger.warning(
        "Hugging Face token not found. "
        "Run 'huggingface-cli login' to enable LLM categorization. "
        "Transactions will be categorized using rules only."
    )
else:
    logger.info(f"✓ Hugging Face token loaded. Using model: {HF_MODEL_ID}")

# Log Clerk key status
clerk_key = os.getenv("CLERK_SECRET_KEY")
if clerk_key:
    logger.info("✓ CLERK_SECRET_KEY loaded")
else:
    logger.error("❌ CLERK_SECRET_KEY not found in environment!")


# --- Categories & Rule-Based Engine ---
VALID_CATEGORIES = [
    "Food & Drinks", "Subscriptions", "Shopping", "Travel", "Groceries", "Education",
    "Investments", "Utilities", "Health", "Entertainment", "Personal Care",
    "Personal & Transfers", "Income", "Miscellaneous"
]
CATEGORIES_LIST_STR = ", ".join(VALID_CATEGORIES)


CATEGORIZATION_RULES = {
    "Food & Drinks": [
        "zomato", "swiggy", "mcdonald's", "dominos", "kfc", "eats", "restaurant", 
        "cafe", "starbucks", "pizza", "burger", "biryani", "food", "dining", "subway",
        "haldiram", "barbeque", "chinese", "pizza hut", "dunkin", "food court"
    ],
    "Groceries": [
        "blinkit", "grofers", "zepto", "bigbasket", "groceries", "supermarket", 
        "dmart", "fresh", "vegetables", "fruits", "milk", "instamart", "jiomart",
        "more", "reliance fresh", "spencer"
    ],
    "Travel": [
        "uber", "ola", "rapido", "irctc", "redbus", "makemytrip", "indigo", 
        "airways", "fuel", "petrol", "goibibo", "train", "flight", "taxi", "cab",
        "parking", "toll", "yatra", "cleartrip", "spicejet", "vistara", "air india"
    ],
    "Shopping": [
        "amazon", "flipkart", "myntra", "ajio", "zara", "h&m", "decathlon", 
        "ikea", "meesho", "snapdeal", "shopping", "store", "mall", "nykaa",
        "lenskart", "cars24", "olx", "quikr", "furniture", "electronics", "lifestyle",
        "westside", "pantaloons", "max fashion"
    ],
    "Subscriptions": [
        "netflix", "spotify", "prime video", "hotstar", "youtube premium", 
        "google one", "icloud", "disney", "subscription", "membership", "amazon prime",
        "sonyliv", "sony", "zee5", "voot", "jiocinema", "sunnxt", "eros", "mx player",
        "gaana", "wynk", "jiosaavn", "apple music", "crunchyroll", "sony pictures"
    ],
    "Utilities": [
        "airtel", "jio", "vodafone", "vi", "bses", "electricity", "broadband", 
        "aws", "google cloud", "bill", "recharge", "mobile", "internet", "gas", "water",
        "tata power", "adani", "reliance", "paytm bill", "phonepe recharge", "power"
    ],
    "Health": [
        "apollo", "pharmacy", "medplus", "netmeds", "practo", "hospital", 
        "clinic", "1mg", "doctor", "medical", "medicine", "pharmeasy", "healthkart",
        "max hospital", "fortis", "diagnostics", "pathology", "lab", "health"
    ],
    "Entertainment": [
        "bookmyshow", "pvr", "inox", "paytm insider", "steam games", 
        "playstation", "xbox", "movie", "game", "cinema", "concert", "theatre",
        "pubg", "gaming", "valorant", "carnival", "entertainment"
    ],
    "Personal Care": [
        "salon", "spa", "parlour", "haircut", "grooming", "beauty", "lakme",
        "looks", "naturals", "hair", "facial", "massage", "wellness", "gym",
        "fitness", "yoga", "hayat", "barber", "nail"
    ],
    "Personal & Transfers": [
        "bank of", "hdfc", "icici", "sbi", "axis", "atm", "transfer", 
        "paytm wallet", "upi", "payment", "kotak", "phonepe", "gpay", "unknown"
    ],
    "Education": [
        "college", "school", "university", "course", "udemy", "coursera", 
        "fees", "tuition", "book", "stationery", "unacademy", "byju", "vedantu",
        "khan academy", "skillshare", "upgrad", "learning", "study"
    ],
    "Investments": [
        "mutual fund", "stocks", "zerodha", "groww", "upstox", "investment", 
        "sip", "trading", "equity", "gold", "coin", "smallcase", "etmoney",
        "angel broking", "5paisa", "paytm money", "broking"
    ],
}