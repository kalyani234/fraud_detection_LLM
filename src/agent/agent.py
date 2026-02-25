"""
src/agent/agent.py - Main fraud detection agent using LangGraph

Uses ReAct pattern with single production prompt
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from src.config import llm
from src.prompts import get_system_prompt
from src.agent.tools import tools

# ═══════════════════════════════════════════════════════════════════════════
# AGENT SETUP
# ═══════════════════════════════════════════════════════════════════════════

def create_agent():
    """Create ReAct agent with production configuration."""
    return create_react_agent(model=llm, tools=tools)


# Single cached agent for all analysis
_agent = None

def get_agent():
    """Get or create the production agent."""
    global _agent
    if _agent is None:
        _agent = create_agent()
    return _agent

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def run_fraud_analysis(transaction: dict, mode: str = "production") -> str:
    """
    Analyze a transaction for fraud risk using LangGraph agent.
    
    The system uses a weighted 4-signal framework:
    1. Account Behavior (40%) - Strongest signal
    2. Balance Anomaly (40%) - Secondary signal
    3. Destination Type (10%) - Contextual
    4. Amount Context (10%) - Weak signal
    
    Args:
        transaction: Transaction data dict with fields:
                    type, amount, nameOrig, nameDest, oldbalanceOrg, etc.
        mode: Analysis mode (currently "production" only, others use production)
    
    Returns:
        String with agent analysis, probability, and fraud decision
    """
    
    mode = mode.lower()
    
    # All modes use the same production prompt
    if mode not in ["production", "balanced", "conservative", "aggressive"]:
        raise ValueError(
            f"Invalid mode: {mode}. Use: production, balanced, conservative, or aggressive. "
            f"(All modes currently use the production prompt)"
        )
    
    # Get the production agent
    agent_executor = get_agent()
    
    # Get system prompt (all modes use production prompt)
    system_prompt = get_system_prompt("production")
    
    # Format transaction data
    tx_str = "\n".join(f"• {k}: {v}" for k, v in transaction.items())
    
    # Build analysis task
    analysis_task = f"""Analyze this transaction using the 4-signal weighted framework:

TRANSACTION DATA:
{tx_str}

ANALYSIS STEPS:
1. Check transaction type (RULE 1)
2. If TRANSFER/CASH_OUT, use tools to get:
   - Account history (get_origin_history)
   - Balance anomaly assessment (check_balance_anomaly)
   - Destination type check (is_merchant_account)
3. Calculate 4-signal score
4. Apply decision threshold
5. Provide fraud probability and final decision

OUTPUT EXACTLY IN THIS FORMAT:

Agent Analysis (Step by Step)

1. [Type check and initial assessment]
2. [Signal A - Account behavior score and reasoning]
3. [Signal B - Balance anomaly score and reasoning]
4. [Signal C - Destination type score and reasoning]
5. [Signal D - Amount context score and reasoning]
6. [Total score calculation and threshold explanation]

FRAUD PROBABILITY: XX%
REASON: [2-3 sentence summary of key signals]
FINAL DECISION: [FRAUD / SUSPICIOUS / LEGITIMATE]"""
    
    try:
        # Create messages with system prompt and task
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=analysis_task)
        ]
        
        # Run agent with the production framework
        result = agent_executor.invoke({"messages": messages})
        
        # Extract response
        if result and "messages" in result:
            last_message = result["messages"][-1]
            content = last_message.content.strip() if hasattr(last_message, 'content') else str(last_message)
            return content if content else "No response from agent."
        else:
            return "Error: No response from agent."
    
    except Exception as e:
        # Detailed error response
        import traceback
        tx_summary = "\n".join(f"  {k}: {v}" for k, v in transaction.items())
        return (
            f"❌ Agent Error: {str(e)}\n\n"
            f"Transaction:\n{tx_summary}\n\n"
            f"Traceback:\n{traceback.format_exc()}\n\n"
            f"Troubleshooting:\n"
            f"1. Check OPENAI_API_KEY in .env\n"
            f"2. Verify paysim.csv in data/ folder\n"
            f"3. Ensure all dependencies installed"
        )

# ═══════════════════════════════════════════════════════════════════════════
# BATCH ANALYSIS (For testing)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_batch(transactions: list[dict], mode: str = "production") -> list[dict]:
    """
    Analyze multiple transactions.
    
    Args:
        transactions: List of transaction dicts
        mode: Analysis mode
    
    Returns:
        List of analysis results
    """
    results = []
    for i, tx in enumerate(transactions):
        print(f"  Analyzing transaction {i+1}/{len(transactions)}...")
        analysis = run_fraud_analysis(tx, mode)
        results.append({
            "transaction": tx,
            "analysis": analysis,
            "index": i
        })
    return results

# ═══════════════════════════════════════════════════════════════════════════
# TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example transaction for testing
    example_tx = {
        "step": 100,
        "type": "CASH_OUT",
        "amount": 297988.0,
        "nameOrig": "C1000000001",
        "oldbalanceOrg": 30030.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C1792659267",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 297988.0,
        "isFraud": 0,
        "isFlaggedFraud": 0
    }
    
    print("🔄 Testing fraud detection agent...")
    print("=" * 80)
    print("Using PRODUCTION prompt with 4-signal weighted framework")
    print("=" * 80)
    
    analysis = run_fraud_analysis(example_tx, mode="production")
    print(analysis)