import re
import logging
from datetime import datetime
from dateutil import parser as dateutil_parser

from .config import logger  # Import the configured logger

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


# --- Hugging Face Response Helper ---

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