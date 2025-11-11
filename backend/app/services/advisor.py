import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from app.config import logger, HF_TOKEN, HF_ADVISOR_MODEL

# Conditionally import and initialize LangChain components
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

except ImportError:
    logger.warning("langchain_huggingface not installed. LLM advisor will be disabled.")
    llm = None
except Exception as e:
    logger.error(f"LangChain init failed: {e}")
    llm = None


# --- Chat Configuration ---

# Store chat histories per session
chat_history_store = {}

# LangChain prompt template
chat_template = ChatPromptTemplate.from_messages([
    ('system', 'You are an expert financial advisor. Provide specific, actionable advice using numbers from the data. Be concise and helpful.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', 'Financial Data:\n{financial_summary}\n\nQuestion: {query}')
])


# --- Financial Summary & Advice Logic ---

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