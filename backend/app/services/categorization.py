import re
import httpx
import asyncio
import logging
from app.services.advisor import llm
from app.config import (
    logger, 
    CATEGORIZATION_RULES, 
    HF_TOKEN, 
    HF_MODEL_ID, 
    FALLBACK_MODELS, 
    VALID_CATEGORIES, 
    CATEGORIES_LIST_STR
)
from app.utils import _extract_generated_text

# --- Category Cache ---
category_cache: dict[str, str] = {}


# --- Rule-Based Engine ---

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


# --- LLM Categorization (Simplified for free models) ---

async def _call_hf_inference(model_id: str, prompt: str) -> str | None:
    """Call HF Inference API for a specific model."""
    if not HF_TOKEN:
        return None

    url = f"https://api-inference.huggingface.co/models/{model_id}"
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
    """Categorizes a merchant using the pre-configured LangChain LLM."""

    # Check 1: Is the LLM (from advisor.py) even working?
    if not llm:
        logger.warning("Categorization LLM not available. Defaulting to Miscellaneous.")
        return "Miscellaneous"

    # Check 2: No merchant name
    if not merchant_name:
        return "Miscellaneous"

    # Check 3: Cache
    if merchant_name in category_cache:
        logger.debug(f"Cache hit for '{merchant_name}'")
        return category_cache[merchant_name]

    # Simplified prompt for a chat model
    prompt = (
        f"Classify the merchant '{merchant_name}' into exactly one "
        f"of these categories: {CATEGORIES_LIST_STR}\n"
        f"Respond with only the category name."
    )

    generated_category = ""
    try:
        # Use the same .ainvoke() method the chatbot uses
        ai_response = await llm.ainvoke(prompt)
        generated_category = ai_response.content.strip()
        logger.debug(f"LLM categorized '{merchant_name}' as '{generated_category}'")

    except Exception as e:
        logger.error(f"LLM categorization failed for '{merchant_name}': {e}")
        category_cache[merchant_name] = "Miscellaneous" # Cache failure
        return "Miscellaneous"

    if not generated_category:
        logger.debug(f"LLM returned empty response for '{merchant_name}', defaulting to Miscellaneous")
        category_cache[merchant_name] = "Miscellaneous"
        return "Miscellaneous"

    # Extract category from response
    # The model might respond with "Category: Food & Drinks" or just "Food & Drinks"
    for category in VALID_CATEGORIES:
        if re.search(r'\b' + re.escape(category) + r'\b', generated_category, re.IGNORECASE):
            logger.info(f"Successfully categorized '{merchant_name}' as '{category}'")
            category_cache[merchant_name] = category
            return category

    # If the model's response didn't match any valid category
    logger.warning(f"LLM response '{generated_category}' for '{merchant_name}' not in VALID_CATEGORIES.")
    category_cache[merchant_name] = "Miscellaneous"
    return "Miscellaneous"