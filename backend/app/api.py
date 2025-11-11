# app/api.py
import asyncio
from datetime import datetime
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from langchain_core.messages import HumanMessage
from app.config import logger, HF_TOKEN, HF_MODEL_ID
from app.services.gpay_parser import parse_gpay_html
from app.services.categorization import (
    categorize_by_rules,
    get_category_from_llm,
    category_cache
)
from app.services.advisor import (
    llm,
    chat_template,
    chat_history_store,
    generate_financial_summary,
    generate_rule_based_advice
)
from app.services.clerk_auth import get_current_user
from app.db import get_db
from app.services.users import get_or_create_user_from_clerk

router = APIRouter(prefix="/api")


@router.get("/protected")
async def protected_route(current_user=Depends(get_current_user)):
    return {"ok": True, "user": current_user}


@router.post("/process-transactions")
async def process_transactions(
    file: UploadFile = File(...),
    db = Depends(get_db),
    current_user = Depends(get_current_user),
):
    """Process uploaded HTML transaction file and return categorized transactions"""
    # upsert user
    user = await get_or_create_user_from_clerk(db, current_user)

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
        if parsing_stats.get('failed', 0) > 0:
            error_msg += f" Failed to parse {parsing_stats['failed']} dates."
        raise HTTPException(status_code=404, detail=error_msg)

    # Categorization logic (unchanged)...
    categorized_transactions = []
    transactions_to_llm = []

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

    if transactions_to_llm:
        if HF_TOKEN:
            names = list({t['receiver'] for t in transactions_to_llm if t['receiver']})
            try:
                results = await asyncio.gather(
                    *(get_category_from_llm(n) for n in names),
                    return_exceptions=True
                )
                name_to_cat = {n: (r if isinstance(r, str) else "Miscellaneous") for n, r in zip(names, results)}
                for transaction in transactions_to_llm:
                    transaction['category'] = name_to_cat.get(transaction['receiver'], "Miscellaneous")
                    categorized_transactions.append(transaction)
            except Exception as e:
                logger.warning(f"LLM categorization failed: {e}, defaulting to Miscellaneous")
                for transaction in transactions_to_llm:
                    transaction['category'] = "Miscellaneous"
                    categorized_transactions.append(transaction)
        else:
            for transaction in transactions_to_llm:
                transaction['category'] = "Miscellaneous"
                categorized_transactions.append(transaction)

    sorted_transactions = sorted(categorized_transactions, key=lambda x: x['date'], reverse=True)
    return {
        "success": True,
        "transaction_count": len(sorted_transactions),
        "parsing_statistics": parsing_stats,
        "llm_used": HF_TOKEN is not None,
        "transactions": sorted_transactions
    }


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "hf_token_available": HF_TOKEN is not None,
        "model_id": HF_MODEL_ID,
        "cache_size": len(category_cache),
        "timestamp": datetime.now().isoformat()
    }


@router.post("/financial-advisor")
async def financial_advisor_chat(
    request: dict,
    db = Depends(get_db),
    current_user = Depends(get_current_user),
):
    # upsert user
    user = await get_or_create_user_from_clerk(db, current_user)

    user_question = request.get("question", "")
    transactions = request.get("transactions", [])
    session_id = request.get("session_id", user.clerk_user_id or "default_session")

    if not user_question or not transactions:
        raise HTTPException(status_code=400, detail="Missing required fields")

    logger.info(f"Question: {user_question}")

    financial_summary = generate_financial_summary(transactions)

    if session_id not in chat_history_store:
        chat_history_store[session_id] = []

    chat_history = chat_history_store[session_id]

    if llm:
        try:
            prompt = chat_template.invoke({
                "chat_history": chat_history[-6:],
                "financial_summary": financial_summary,
                "query": user_question
            })
            ai_response = await llm.ainvoke(prompt)
            ai_answer = ai_response.content.strip()
            chat_history.append(HumanMessage(content=user_question))
            chat_history.append(ai_response)
            if len(chat_history) > 12:
                chat_history = chat_history[-12:]
            chat_history_store[session_id] = chat_history
            return {
                "success": True,
                "question": user_question,
                "answer": ai_answer,
                "model": "Llama-3.1 (LangChain)",
                "has_memory": True
            }
        except Exception as e:
            logger.warning(f"LangChain failed: {e}")

    rule_answer = generate_rule_based_advice(user_question, transactions)
    return {
        "success": True,
        "question": user_question,
        "answer": rule_answer,
        "model": "rule-based",
        "has_memory": False
    }
