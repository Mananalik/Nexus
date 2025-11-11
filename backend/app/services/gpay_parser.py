import re
import logging
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

from app.utils import RobustDateParser
from app.config import logger


# --- HTML Parsing ---

def normalize_entry_text(text: str) -> str:
    """Cleans and normalizes text extracted from HTML."""
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
        r"(?:\s*using\s+Bank\s+Account.*)?$",   # Matches "using Bank Account..." (OPTIONAL)
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