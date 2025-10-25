# --- Regular Imports ---
import os
import re
import asyncio
import httpx
import logging
from datetime import datetime, timedelta
from dateutil import parser as dateutil_parser
from huggingface_hub import get_token
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint


# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# --- FastAPI Configuration ---
app = FastAPI(title="Transaction Processing API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
HF_TOKEN = get_token()
HF_ADVISOR_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Initialize LangChain LLM
llm = None
chat_history_store = {}  # Store chat histories per session

if HF_TOKEN:
    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        
        # Initialize endpoint
        endpoint = HuggingFaceEndpoint(
            repo_id=HF_ADVISOR_MODEL,
            huggingfacehub_api_token=HF_TOKEN,
        )
        
        # Wrap in chat interface
        llm = ChatHuggingFace(llm=endpoint)
        
        logger.info("✓ LangChain Chat LLM initialized")
    except Exception as e:
        logger.error(f"LangChain init failed: {e}")
        llm = None


# LangChain prompt template
chat_template = ChatPromptTemplate([
    ('system', 'You are an expert financial advisor. Provide specific, actionable advice using numbers from the data. Be concise and helpful.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', 'Financial Data:\n{financial_summary}\n\nQuestion: {query}')
])


if not HF_TOKEN:
    logger.warning(
        "Hugging Face token not found. "
        "Run 'huggingface-cli login' to enable LLM categorization. "
        "Transactions will be categorized using rules only."
    )
else:
    logger.info(f"✓ Hugging Face token loaded. Using model: {HF_MODEL_ID}")


# --- Categories & Rule-Based Engine ---
# --- Categories & Rule-Based Engine ---
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


def categorize_by_rules(merchant_name: str) -> str | None:
    """Categorizes a transaction based on keywords. Returns None if no rule matches."""
    if not isinstance(merchant_name, str):
        return None

    merchant_lower = merchant_name.lower()
    
    # Check against all rule-based categories
    for category, keywords in CATEGORIZATION_RULES.items():
        if any(keyword in merchant_lower for keyword in keywords):
            return category

    # Heuristic: All-caps two-word names are often bank transfers
    if merchant_name.isupper() and len(merchant_name.split()) == 2:
        return "Personal & Transfers"
    
    # If merchant name is empty or "Unknown", categorize as transfers
    if not merchant_name or merchant_name.lower() == "unknown":
        return "Personal & Transfers"

    return None




# --- Robust Date Parser ---
class RobustDateParser:
    """
    A robust date parser that handles multiple date formats with fallback strategies
    """
    
    def __init__(self):
        self.date_formats = [
            '%Y-%m-%d',
            '%d/%m/%Y',
            '%m/%d/%Y',
            '%d-%m-%Y',
            '%m-%d-%Y',
            '%Y/%m/%d',
            '%d.%m.%Y',
            '%d %b %Y',
            '%d %B %Y',
            '%b %d, %Y',
            '%B %d, %Y',
            '%Y-%m-%d %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
            '%d-%m-%Y %H:%M',
            '%Y%m%d',
            '%d-%b-%Y',
            '%d/%m/%y',
            '%m/%d/%y',
            '%d %b %Y, %H:%M:%S',
            '%d %B %Y, %H:%M:%S',
        ]
        
        self.parse_success_count = 0
        self.parse_failure_count = 0
        self.failed_dates = []
    
    def clean_date_string(self, date_string):
        if not date_string or not isinstance(date_string, str):
            return None
        
        date_string = date_string.strip()
        date_string = date_string.replace('&nbsp;', ' ')
        date_string = ' '.join(date_string.split())
        date_string = re.sub(r"\bSept\b", "Sep", date_string, flags=re.IGNORECASE)
        
        return date_string
    
    def parse_with_dateutil(self, date_string, dayfirst=False):
        try:
            parsed_date = dateutil_parser.parse(date_string, fuzzy=False, dayfirst=dayfirst)
            return parsed_date
        except (ValueError, TypeError, dateutil_parser.ParserError):
            return None
    
    def parse_with_formats(self, date_string):
        for date_format in self.date_formats:
            try:
                parsed_date = datetime.strptime(date_string, date_format)
                return parsed_date
            except ValueError:
                continue
        return None
    
    def parse_date(self, date_string, context=''):
        cleaned_date = self.clean_date_string(date_string)
        
        if not cleaned_date:
            logger.warning(f"Empty or invalid date string {context}")
            self.parse_failure_count += 1
            return None
        
        parsed_date = self.parse_with_dateutil(cleaned_date, dayfirst=False)
        if parsed_date:
            logger.debug(f"Parsed '{date_string}' as {parsed_date} using dateutil (US) {context}")
            self.parse_success_count += 1
            return parsed_date
        
        parsed_date = self.parse_with_dateutil(cleaned_date, dayfirst=True)
        if parsed_date:
            logger.debug(f"Parsed '{date_string}' as {parsed_date} using dateutil (Int'l) {context}")
            self.parse_success_count += 1
            return parsed_date
        
        parsed_date = self.parse_with_formats(cleaned_date)
        if parsed_date:
            logger.debug(f"Parsed '{date_string}' as {parsed_date} using format matching {context}")
            self.parse_success_count += 1
            return parsed_date
        
        logger.error(f"Failed to parse date: '{date_string}' {context}")
        self.parse_failure_count += 1
        self.failed_dates.append((date_string, context))
        return None
    
    def get_statistics(self):
        total = self.parse_success_count + self.parse_failure_count
        success_rate = (self.parse_success_count / total * 100) if total > 0 else 0
        
        return {
            'total_attempts': total,
            'successful': self.parse_success_count,
            'failed': self.parse_failure_count,
            'success_rate': round(success_rate, 2),
            'failed_dates': self.failed_dates[:10]
        }


# --- HTML Parsing ---
def normalize_entry_text(text: str) -> str:
    t = " ".join(text.split())
    t = re.sub(r"\bSept\b", "Sep", t, flags=re.IGNORECASE)
    t = t.strip().strip(',')
    return t


def parse_gpay_html(html_content: str):
    """Parses GPay HTML content to extract transactions from the last year."""
    soup = BeautifulSoup(html_content, "html.parser")
    blocks = soup.find_all("div", class_="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1")

    logger.info(f"Found {len(blocks)} potential transaction blocks in HTML")

    date_parser = RobustDateParser()

    date_line_re = re.compile(
        r"((?:0?[1-9]|[12][0-9]|3[01])\s+[A-Za-z]{3,9}\s+\d{4}(?:,\s+\d{2}:\d{2}:\d{2})?)" 
        r"\s*(?:GMT[+-]\d{2}:\d{2})?", 
        flags=re.IGNORECASE,
    )
    
    left_re = re.compile(
    r"^(Paid|Sent|Received)\s*₹([\d,.]+)"    # Group 1: Action, Group 2: Amount
    r"(?:\s*(?:to|from)\s*(.*?))?"           # Group 3: The receiver (OPTIONAL)
    r"(?:\s*using\s+Bank\s+Account.*)?$",    # Matches "using Bank Account..." (OPTIONAL)
    flags=re.IGNORECASE,
)

    transactions = []
    one_year_ago = datetime.now() - timedelta(days=365)
    skipped_no_date = 0
    skipped_old_date = 0

    for idx, block in enumerate(blocks):
        raw = block.get_text("\n")
        full_text = normalize_entry_text(raw)
        lines = [normalize_entry_text(l) for l in raw.splitlines() if l.strip()]

        left_line = next((l for l in lines if re.match(r"^(?:Paid|Sent|Received)\b", l, flags=re.IGNORECASE)), None)
        date_match = date_line_re.search(full_text)
        
        if not left_line:
            logger.debug(f"Block {idx}: No action line found")
            continue
            
        if not date_match:
            logger.debug(f"Block {idx}: No date pattern found in text: {full_text}")
            skipped_no_date += 1
            continue

        m = left_re.match(left_line)
        d = date_match
        if not m or not d:
            logger.debug(f"Block {idx}: Regex match failed")
            continue

        try:
            action, amount_str, receiver = m.groups()
            receiver = (receiver or "").strip()
            receiver = re.sub(r"\s*using\s+Bank\s+Account.*", "", receiver, flags=re.IGNORECASE).strip()
            receiver = re.sub(r"^(to|from)\s+", "", receiver, flags=re.IGNORECASE).strip()

            date_str = d.group(1)
            
            date_obj = date_parser.parse_date(date_str, context=f"(block {idx})")
            
            if not date_obj:
                logger.warning(f"Block {idx}: Could not parse date '{date_str}'")
                skipped_no_date += 1
                continue

            if date_obj < one_year_ago:
                skipped_old_date += 1
                continue

            transactions.append({
                "type": action.title(),
                "amount": float(amount_str.replace(",", "")),
                "receiver": receiver if receiver else "Unknown",
                "date": date_obj.strftime("%Y-%m-%d"),
                "date_original": date_str,
            })
        except (ValueError, TypeError) as e:
            logger.error(f"Block {idx}: Error processing transaction: {e}")
            continue

    stats = date_parser.get_statistics()
    logger.info(f"Date parsing statistics: {stats}")
    logger.info(f"Total transactions parsed: {len(transactions)}")
    logger.info(f"Skipped (no date): {skipped_no_date}")
    logger.info(f"Skipped (older than 1 year): {skipped_old_date}")
    
    if stats['failed_dates']:
        logger.warning(f"Failed to parse {len(stats['failed_dates'])} date formats:")
        for failed_date, context in stats['failed_dates']:
            logger.warning(f"  - '{failed_date}' {context}")

    return transactions, stats


# --- LLM Categorization (Simplified for free models) ---
category_cache: dict[str, str] = {}


def _extract_generated_text(result_json) -> str:
    """Extract text from HF response"""
    try:
        if isinstance(result_json, list) and result_json and isinstance(result_json[0], dict):
            return result_json[0].get("generated_text", "") or ""
        if isinstance(result_json, dict):
            return result_json.get("generated_text", "") or ""
    except Exception:
        pass
    return ""


async def _call_hf_inference(model_id: str, prompt: str) -> str | None:
    """Call HF Inference API for a specific model."""
    if not HF_TOKEN:
        return None

    url = f"https://router.huggingface.co/hf-inference/{model_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    # Simplified prompt for smaller models
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 20,
            "temperature": 0.1,
            "do_sample": False
        }
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            
            if resp.status_code == 503:
                logger.info(f"Model {model_id} warming up, retrying...")
                await asyncio.sleep(2.0)
                resp = await client.post(url, headers=headers, json=payload)
            
            if resp.status_code == 404:
                logger.warning(f"Model {model_id} not available")
                return None
                
            if resp.status_code != 200:
                logger.warning(f"Model {model_id} returned status {resp.status_code}")
                return None
                
            resp.raise_for_status()
            return _extract_generated_text(resp.json())
        except Exception as e:
            logger.error(f"Error calling HF API for model {model_id}: {e}")
            return None


async def get_category_from_llm(merchant_name: str) -> str:
    """Categorizes a merchant using HF Inference API with enhanced fallback."""
    if not HF_TOKEN or not merchant_name:
        return "Miscellaneous"

    if merchant_name in category_cache:
        logger.debug(f"Cache hit for '{merchant_name}'")
        return category_cache[merchant_name]

    # Simplified prompt that works better with smaller models
    prompt = (
        f"Classify the merchant '{merchant_name}' into one of the "
        f"following categories: {CATEGORIES_LIST_STR}\n"
        f"Category:"
    )

    models_to_try = [HF_MODEL_ID] + [m for m in FALLBACK_MODELS if m != HF_MODEL_ID]
    generated_category = None

    for mid in models_to_try:
        txt = await _call_hf_inference(mid, prompt)
        if txt:
            generated_category = txt.replace(prompt, "").strip()
            logger.debug(f"LLM categorized '{merchant_name}' as '{generated_category}' using {mid}")
            break

    if not generated_category:
        logger.debug(f"All models failed for '{merchant_name}', defaulting to Miscellaneous")
        category_cache[merchant_name] = "Miscellaneous"
        return "Miscellaneous"

    # Extract category from response
    for category in VALID_CATEGORIES:
        if re.search(r'\b' + re.escape(category) + r'\b', generated_category, re.IGNORECASE):
            category_cache[merchant_name] = category
            return category

    category_cache[merchant_name] = "Miscellaneous"
    return "Miscellaneous"


# --- Main API Endpoint ---
@app.post("/api/process-transactions")
async def process_transactions(file: UploadFile = File(...)):
    """Process uploaded HTML transaction file and return categorized transactions"""
    logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")
    
    if file.content_type != 'text/html':
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an HTML file.")

    try:
        contents = await file.read()
        parsed_transactions, parsing_stats = parse_gpay_html(contents.decode('utf-8'))
    except Exception as e:
        logger.error(f"Error parsing HTML: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error parsing HTML file: {str(e)}")

    if not parsed_transactions:
        error_msg = "No valid transactions found from the last year."
        if parsing_stats['failed'] > 0:
            error_msg += f" Failed to parse {parsing_stats['failed']} dates."
        raise HTTPException(status_code=404, detail=error_msg)

    logger.info(f"Successfully parsed {len(parsed_transactions)} transactions")

    categorized_transactions = []
    transactions_to_llm = []

    # First pass: categorize using rules
    for t in parsed_transactions:
        if t['type'] == 'Received':
            t['category'] = 'Income'
            categorized_transactions.append(t)
            continue

        category = categorize_by_rules(t['receiver'])
        if category:
            t['category'] = category
            categorized_transactions.append(t)
        else:
            transactions_to_llm.append(t)

    logger.info(f"Categorized {len(categorized_transactions)} transactions by rules")
    logger.info(f"Remaining {len(transactions_to_llm)} transactions need LLM categorization")

    # Second pass: Try LLM categorization for remaining transactions
    # If LLM fails, default to Miscellaneous
    if transactions_to_llm:
        if HF_TOKEN:
            names = list({t['receiver'] for t in transactions_to_llm if t['receiver']})
            logger.info(f"Attempting LLM categorization for {len(names)} unique merchants")
            
            # Try LLM but don't fail if it doesn't work
            try:
                results = await asyncio.gather(
                    *(get_category_from_llm(n) for n in names), 
                    return_exceptions=True
                )
                name_to_cat = {n: (r if isinstance(r, str) else "Miscellaneous") for n, r in zip(names, results)}
                
                for transaction in transactions_to_llm:
                    transaction['category'] = name_to_cat.get(transaction['receiver'], "Miscellaneous")
                    categorized_transactions.append(transaction)
                
                logger.info("LLM categorization completed successfully")
            except Exception as e:
                logger.warning(f"LLM categorization failed: {e}, defaulting to Miscellaneous")
                for transaction in transactions_to_llm:
                    transaction['category'] = "Miscellaneous"
                    categorized_transactions.append(transaction)
        else:
            logger.warning("No HF token available, defaulting to Miscellaneous")
            for transaction in transactions_to_llm:
                transaction['category'] = "Miscellaneous"
                categorized_transactions.append(transaction)

    logger.info(f"Final categorized transactions: {len(categorized_transactions)}")

    # Sort by date (most recent first)
    sorted_transactions = sorted(categorized_transactions, key=lambda x: x['date'], reverse=True)

    return {
        "success": True,
        "transaction_count": len(sorted_transactions),
        "parsing_statistics": parsing_stats,
        "llm_used": HF_TOKEN is not None,
        "transactions": sorted_transactions
    }


# --- Health Check Endpoint ---
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "hf_token_available": HF_TOKEN is not None,
        "model_id": HF_MODEL_ID,
        "cache_size": len(category_cache),
        "timestamp": datetime.now().isoformat()
    }


# --- Root Endpoint ---
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Transaction Processing API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "process_transactions": "/api/process-transactions (POST)"
        }
    }

def generate_financial_summary(transactions):
    """Generate concise financial summary for AI"""
    total_spent = sum(t['amount'] for t in transactions if t['type'] in ['Paid', 'Sent'])
    total_income = sum(t['amount'] for t in transactions if t['type'] == 'Received')
    net_flow = total_income - total_spent
    
    # Monthly calculations
    unique_months = len(set(t['date'][:7] for t in transactions))
    months = max(1, unique_months)
    monthly_spent = total_spent / months
    monthly_income = total_income / months
    monthly_savings = net_flow / months
    savings_rate = (monthly_savings/monthly_income*100) if monthly_income > 0 else 0
    
    # Category breakdown
    category_spending = {}
    for t in transactions:
        if t['type'] in ['Paid', 'Sent']:
            cat = t.get('category', 'Miscellaneous')
            category_spending[cat] = category_spending.get(cat, 0) + t['amount']
    
    top_categories = sorted(category_spending.items(), key=lambda x: x[1], reverse=True)[:5]
    
    summary = f"""Monthly Income: Rs.{monthly_income:,.0f}
Monthly Spending: Rs.{monthly_spent:,.0f}
Monthly Savings: Rs.{monthly_savings:,.0f}
Savings Rate: {savings_rate:.1f}%
Top Spending Categories:"""
    
    for i, (cat, amount) in enumerate(top_categories, 1):
        pct = (amount / total_spent * 100) if total_spent > 0 else 0
        summary += f"\n{i}. {cat}: Rs.{amount:,.0f} ({pct:.1f}%)"
    
    return summary


def generate_rule_based_advice(question, transactions):
    """Smart Financial Advisor with Better Intent Recognition"""
    question_lower = question.lower()
    
    # Calculate all metrics
    total_spent = sum(t['amount'] for t in transactions if t['type'] in ['Paid', 'Sent'])
    total_income = sum(t['amount'] for t in transactions if t['type'] == 'Received')
    net_flow = total_income - total_spent
    
    category_spending = {}
    for t in transactions:
        if t['type'] in ['Paid', 'Sent']:
            cat = t.get('category', 'Miscellaneous')
            category_spending[cat] = category_spending.get(cat, 0) + t['amount']
    
    sorted_categories = sorted(category_spending.items(), key=lambda x: x[1], reverse=True)
    top_category = sorted_categories[0] if sorted_categories else ("Unknown", 0)
    
    unique_months = len(set(t['date'][:7] for t in transactions))
    months = max(1, unique_months)
    monthly_spent = total_spent / months
    monthly_income = total_income / months
    monthly_savings = net_flow / months
    
    # PRIORITY 1: Spending/Expense questions (most common)
    spending_keywords = ["spend", "spending", "expense", "expenses", "expenditure", "cost", "costly", "reduce", "cut", "minimize", "less", "going"]
    if any(word in question_lower for word in spending_keywords):
        top_3 = sorted_categories[:3]
        category_tip = get_category_tip(top_category[0])
        
        cat2_name = top_3[1][0] if len(top_3) > 1 else "N/A"
        cat2_amt = top_3[1][1] if len(top_3) > 1 else 0
        cat2_pct = (cat2_amt/total_spent*100) if len(top_3) > 1 and total_spent > 0 else 0
        
        cat3_name = top_3[2][0] if len(top_3) > 2 else "N/A"
        cat3_amt = top_3[2][1] if len(top_3) > 2 else 0
        cat3_pct = (cat3_amt/total_spent*100) if len(top_3) > 2 and total_spent > 0 else 0
        
        return f"""**Reduce Spending: Action Plan**

**Current Monthly Spending:** Rs.{monthly_spent:,.0f}

**Top 3 Expense Categories:**
1. {top_3[0][0]}: Rs.{top_3[0][1]:,.0f} ({top_3[0][1]/total_spent*100:.1f}%) - START HERE
2. {cat2_name}: Rs.{cat2_amt:,.0f} ({cat2_pct:.1f}%)
3. {cat3_name}: Rs.{cat3_amt:,.0f} ({cat3_pct:.1f}%)

**Priority: Reduce {top_category[0]} First**
{category_tip}

**Action Steps:**
1. Cut {top_category[0]} by 20%: Save Rs.{top_category[1] * 0.2:,.0f}
2. Review and cancel unused subscriptions
3. Track daily expenses for 1 month
4. Set weekly spending limits

**Target:** Reduce total spending by 15% = Rs.{monthly_spent * 0.15:,.0f}/month savings"""
    
    # PRIORITY 2: Savings questions
    elif any(word in question_lower for word in ["save", "saving", "savings"]):
        savings_rate = (monthly_savings / monthly_income * 100) if monthly_income > 0 else 0
        target_savings = monthly_income * 0.2
        gap = target_savings - monthly_savings
        category_tip = get_category_tip(top_category[0])
        
        status_msg = "Excellent! You're saving well!" if savings_rate >= 20 else "Let's improve your savings!"
        
        return f"""**Savings Optimization Plan:**

**Current Status:**
- Monthly Savings: Rs.{monthly_savings:,.0f} ({savings_rate:.1f}% of income)
- Target: Rs.{target_savings:,.0f} (20% of income)
- Gap: Rs.{gap:,.0f}

{status_msg}

**Top 3 Ways to Save Rs.{gap:,.0f}/month:**

1. Reduce {top_category[0]} (Rs.{top_category[1]:,.0f})
   - Current: Rs.{top_category[1]:,.0f}
   - Cut 20%: Save Rs.{top_category[1] * 0.2:,.0f}
   - {category_tip}

2. Review Subscriptions
   - Cancel unused services
   - Potential savings: Rs.1,000-2,000

3. Smart Shopping
   - Use discount codes
   - Buy generic brands
   - Potential savings: Rs.1,500-3,000

**Challenge:** Try saving Rs.{gap:,.0f} extra this month!"""
    
    # PRIORITY 3: Investment questions (check AFTER spending)
    elif any(word in question_lower for word in ["invest", "investment", "sip", "mutual fund", "stocks", "portfolio"]):
        emergency_fund = monthly_spent * 6
        monthly_sip = max(3000, monthly_savings * 0.5)
        
        return f"""**Investment Strategy for You:**

**Current Financial Position:**
- Monthly Savings: Rs.{monthly_savings:,.0f}
- Available for Investment: Rs.{monthly_sip:,.0f}/month

**Step-by-Step Investment Plan:**

**Step 1: Emergency Fund (Priority 1)**
- Goal: Rs.{emergency_fund:,.0f} (6 months expenses)
- Keep in: Savings account or liquid mutual fund
- Timeline: Build over 6-12 months

**Step 2: Start Monthly SIP (Rs.{monthly_sip:,.0f})**

**Beginner-Friendly Options:**
1. Index Funds (Recommended for beginners)
   - Nifty 50 Index Fund
   - Sensex Index Fund
   - Low cost, diversified

2. Large Cap Mutual Funds
   - Safer, stable returns
   - Good for long-term

3. Balanced Funds
   - 60% equity + 40% debt
   - Moderate risk

**Asset Allocation:**
- 60% Equity (growth)
- 30% Debt (stability)
- 10% Gold/International (diversification)

**Action Plan:**
1. Open account with Zerodha/Groww
2. Complete KYC
3. Start with Rs.5,000/month
4. Increase by 10% yearly
5. Review quarterly

**Pro Tip:** Don't try to time the market. Start now, even small amounts matter!

**Expected Returns:** 12-15% annually (long-term)"""
    
    # PRIORITY 4: Budget questions
    elif any(word in question_lower for word in ["budget", "plan", "allocate"]):
        current_pct = (monthly_spent/monthly_income*100) if monthly_income > 0 else 0
        savings_pct = (monthly_savings/monthly_income*100) if monthly_income > 0 else 0
        balance_msg = "Great balance!" if (monthly_savings/monthly_income) >= 0.2 else "Try to increase savings to 20%"
        
        return f"""**Your Personalized Budget:**

**Monthly Income:** Rs.{monthly_income:,.0f}

**50/30/20 Budget Plan:**

**Needs (50%)** - Rs.{monthly_income * 0.5:,.0f}
- Rent/EMI
- Groceries
- Utilities
- Transportation
- Insurance

**Wants (30%)** - Rs.{monthly_income * 0.3:,.0f}
- Dining out
- Entertainment
- Shopping
- Hobbies

**Savings (20%)** - Rs.{monthly_income * 0.2:,.0f}
- Emergency fund
- Investments
- Goals

**Your Current Split:**
- Needs + Wants: {current_pct:.0f}%
- Savings: {savings_pct:.0f}%

{balance_msg}"""
    
    # Default: General overview
    else:
        return f"""**Hello! I'm your Financial Advisor.**

**Quick Overview:**
- Monthly Income: Rs.{monthly_income:,.0f}
- Monthly Spending: Rs.{monthly_spent:,.0f}
- Monthly Savings: Rs.{monthly_savings:,.0f}
- Top Expense: {top_category[0]} (Rs.{top_category[1]:,.0f})

**I can help you with:**

- "Where am I spending too much?"
- "How can I save more money?"
- "Should I invest more?"
- "Create a budget for me"
- "How to reduce expenses?"

**What would you like to know?**"""



def get_category_tip(category):
    """Get specific tips for each category"""
    tips = {
        "Food & Drinks": "• Cook at home 4 days/week\n• Pack lunch for work\n• Limit eating out to weekends",
        "Subscriptions": "• Cancel unused services\n• Share family plans\n• Use free alternatives",
        "Shopping": "• Wait 24hrs before buying\n• Use price comparison\n• Buy only during sales",
        "Groceries": "• Make weekly meal plan\n• Buy seasonal produce\n• Avoid shopping hungry",
        "Travel": "• Use public transport\n• Carpool when possible\n• Plan trips efficiently",
    }
    return tips.get(category, f"• Set monthly limits\n• Track every transaction\n• Find alternatives")
@app.post("/api/financial-advisor")
async def financial_advisor_chat(request: dict):
    """LangChain-Powered Financial Advisor with Memory"""
    user_question = request.get("question", "")
    transactions = request.get("transactions", [])
    session_id = request.get("session_id", "default_session")
    
    if not user_question or not transactions:
        raise HTTPException(status_code=400, detail="Missing required fields")
    
    logger.info(f"Question: {user_question}")
    
    # Generate financial summary
    financial_summary = generate_financial_summary(transactions)
    
    # Get or create chat history for this session
    if session_id not in chat_history_store:
        chat_history_store[session_id] = []
    
    chat_history = chat_history_store[session_id]
    
    # Try LangChain LLM first
    if llm:
        try:
            # Create prompt with history
            prompt = chat_template.invoke({
                'chat_history': chat_history[-6:],  # Last 3 exchanges
                'financial_summary': financial_summary,
                'query': user_question
            })
            
            # Get AI response
            ai_response = llm.invoke(prompt)
            
            # Get the text content from the AIMessage object
            ai_answer = ai_response.content.strip() # <-- FIX 1
            
            # Update chat history
            chat_history.append(HumanMessage(content=user_question))
            chat_history.append(ai_response) # <-- FIX 2: Append the actual AIMessage object
            
            # Keep history manageable
            if len(chat_history) > 12:
                chat_history = chat_history[-12:]
            
            chat_history_store[session_id] = chat_history
            
            logger.info("✓ LangChain AI responded")
            return {
                "success": True,
                "question": user_question,
                "answer": ai_answer,  # Return the stripped text string to the user
                "model": "Llama-3.1 (LangChain)",
                "has_memory": True
            }
            
        except Exception as e:
            logger.warning(f"LangChain failed: {e}")
    
    # Fallback to rule-based
    logger.info("Using rule-based fallback")
    rule_answer = generate_rule_based_advice(user_question, transactions)
    
    return {
        "success": True,
        "question": user_question,
        "answer": rule_answer,
        "model": "rule-based",
        "has_memory": False
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Transaction Processing API...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
