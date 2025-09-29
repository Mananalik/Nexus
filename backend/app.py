from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
import re

app = FastAPI()

# --- CORS Configuration ---
# Allows your frontend (running on localhost:3000) to communicate with this backend.
origins = [
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HTML Parsing Logic ---
def parse_transaction_html(html_content: str):
    """
    Parses Google Takeout 'My Activity' (Google Pay) HTML using a robust,
    two-step regex logic to handle variations in transaction strings.
    
    Returns a list of dicts with keys: type, amount, receiver, account, date, time.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    divs = soup.find_all("div", class_="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1")

    primary_pattern = re.compile(
        r"^(Paid|Sent|Received|Used GPay Vouchers|Used Google Pay)\s*"
        r"(?:₹([\d,.]+))?\s*"
        r"(.*?)\s*"
        r"(\d{1,2}\s+\w+\s+\d{4}),\s+(\d{2}:\d{2}:\d{2})"
    )

    transactions = []

    for d in divs:
        text = " ".join(d.get_text().split())
        match = primary_pattern.search(text)
        
        if match:
            action, amount_str, middle_content, date, time = match.groups()
            
            receiver = None
            account = None
            
            # --- START: IMPROVED LOGIC ---
            # Step 1: Find the account details first.
            account_match = re.search(r"using Bank Account\s*(XXXXXX\d+)", middle_content)
            if account_match:
                account = account_match.group(1).strip()
                # Step 2: REMOVE the account string from the middle content.
                # This prevents it from being picked up by the receiver regex.
                middle_content = middle_content.replace(account_match.group(0), "")

            # Step 3: Now, safely find the receiver in the cleaned-up middle content.
            receiver_match = re.search(r"(?:to|from)\s+([A-Za-z0-9\s.'-]+)", middle_content)
            if receiver_match:
                receiver = receiver_match.group(1).strip()
            # --- END: IMPROVED LOGIC ---

            amount = float(amount_str.replace(",", "")) if amount_str else None

            transactions.append({
                "type": action,
                "amount": amount,
                "receiver": receiver,
                "account": account,
                "date": date,
                "time": time
            })

    return transactions


# --- API Endpoint ---
@app.post("/api/process-transactions")
async def process_transactions(file: UploadFile = File(...)):
    """
    Receives an HTML file, parses it, and returns the extracted transaction data.
    """
    if file.content_type != 'text/html':
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an HTML file.")
    
    # Read the file content into memory
    contents = await file.read()
    
    # Process the HTML using our parsing function
    transactions_data = parse_transaction_html(contents)
    
    if not transactions_data:
        raise HTTPException(
            status_code=404, 
            detail="No completed transactions found in the file. Please check the file content and format."
        )

    return transactions_data

