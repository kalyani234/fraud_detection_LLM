"""
src/agent/tools.py - Tools for fraud detection agent

Tools:
  1. get_origin_history - Check sender's transaction history
  2. check_balance_anomaly - Analyze balance vs transaction amount
  3. is_merchant_account - Check if destination is merchant
  4. get_account_statistics - Get overall account stats
  5. compare_to_account_average - Check if amount is typical
"""

from langchain_core.tools import tool
from src.data.loader import load_data
import pandas as pd

# Load data once (reuse across tool calls)
df = load_data(sample_frac=0.01)  # Use 1% sample for performance

# ═══════════════════════════════════════════════════════════════════════════
# TOOL 1: Get Origin History
# ═══════════════════════════════════════════════════════════════════════════

@tool
def get_origin_history(origin_id: str, n: int = 10) -> str:
    """Get recent transaction history for sender account.
    
    Use this to determine:
    - Is this a NEW account? (no history = risky)
    - Does the account normally do TRANSFER/CASH_OUT? (trusted pattern)
    - Has this account committed fraud before? (risky)
    
    Args:
        origin_id: The sender's account ID (nameOrig)
        n: Number of recent transactions to retrieve
    
    Returns:
        String with transaction history and risk assessment
    """
    
    mask = df["nameOrig"] == origin_id
    
    if not mask.any():
        return (
            f"⚠️ Account {origin_id}: NO TRANSACTION HISTORY\n"
            f"   Risk: NEW/UNKNOWN ACCOUNT\n"
            f"   Signal Score: +2 points (risky)\n"
            f"   Recommendation: Apply stricter fraud checks"
        )
    
    # Get recent transactions
    recent = df[mask].sort_values("step", ascending=False).head(n)
    fraud_count = recent["isFraud"].sum()
    fraud_rate = (fraud_count / len(recent)) * 100 if len(recent) > 0 else 0
    
    # Calculate statistics
    transfer_count = (recent["type"] == "TRANSFER").sum()
    cashout_count = (recent["type"] == "CASH_OUT").sum()
    high_risk_txs = transfer_count + cashout_count
    avg_amount = recent["amount"].mean()
    max_amount = recent["amount"].max()
    
    # Get transaction details
    history_text = recent[["step", "type", "amount", "isFraud"]].to_string()
    
    # Risk assessment
    if fraud_rate > 5:
        assessment = "🔴 RISKY ACCOUNT: Past fraud detected"
        signal = "+2"
    elif high_risk_txs >= 5:
        assessment = "🟢 TRUSTED ACCOUNT: Frequent TRANSFER/CASH_OUT (normal pattern)"
        signal = "-2"
    elif len(recent) < 3:
        assessment = "🟡 LIMITED HISTORY: Few prior transactions"
        signal = "+1"
    elif fraud_rate > 0:
        assessment = "🟠 CAUTION: Some fraud in history"
        signal = "+1"
    else:
        assessment = "🟢 NORMAL BEHAVIOR: No fraud history"
        signal = "-1"
    
    return f"""Transaction History for {origin_id}:

Recent Transactions (last {min(n, len(recent))}):
{history_text}

Account Statistics:
  • Total transactions analyzed: {len(recent)}
  • Fraud count: {fraud_count}
  • Fraud rate: {fraud_rate:.1f}%
  • TRANSFER count: {transfer_count}
  • CASH_OUT count: {cashout_count}
  • High-risk transactions: {high_risk_txs}
  • Avg transaction amount: {avg_amount:,.0f}
  • Max transaction amount: {max_amount:,.0f}

Risk Assessment: {assessment}
Signal Score: {signal} points"""


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 2: Check Balance Anomaly
# ═══════════════════════════════════════════════════════════════════════════

@tool
def check_balance_anomaly(amount: float, oldbalanceOrg: float, tx_type: str) -> str:
    """Check if transaction amount exceeds available balance.
    
    IMPORTANT: Balance anomaly is a WEAK signal (appears in 15% of legitimate).
    Use in combination with other signals.
    
    Args:
        amount: Transaction amount
        oldbalanceOrg: Sender's balance before transaction
        tx_type: Transaction type (TRANSFER, CASH_OUT, etc.)
    
    Returns:
        String with anomaly analysis and signal score
    """
    
    # Safe types: skip balance check
    if tx_type in ["PAYMENT", "CASH_IN", "DEBIT"]:
        return (
            f"✓ {tx_type}: Low-risk transaction type\n"
            f"   Balance check not applicable\n"
            f"   Signal Score: 0 points"
        )
    
    # Cannot assess with no balance data
    if oldbalanceOrg <= 0:
        return (
            f"⚠️ No available balance data\n"
            f"   Amount: {amount:,.0f}\n"
            f"   Signal Score: 0 points (cannot assess)"
        )
    
    # Calculate ratio
    ratio = amount / oldbalanceOrg
    
    if ratio > 2.0:
        return (
            f"🔴 SEVERE ANOMALY:\n"
            f"   Amount: {amount:,.0f}\n"
            f"   Balance: {oldbalanceOrg:,.0f}\n"
            f"   Ratio: {ratio:.2f}x\n"
            f"   Assessment: Very unusual. Amount is {ratio:.1f} times the available balance.\n"
            f"   Context: Mobile money systems may have credit/overdraft, but this is high risk.\n"
            f"   Signal Score: +2 points"
        )
    
    elif ratio > 1.5:
        return (
            f"🟠 MODERATE ANOMALY:\n"
            f"   Amount: {amount:,.0f}\n"
            f"   Balance: {oldbalanceOrg:,.0f}\n"
            f"   Ratio: {ratio:.2f}x\n"
            f"   Assessment: Amount exceeds balance.\n"
            f"   Context: This appears in ~15% of legitimate PaySim transactions.\n"
            f"   Signal Score: +1 point"
        )
    
    elif ratio > 1.0:
        return (
            f"🟡 MILD ANOMALY:\n"
            f"   Amount: {amount:,.0f}\n"
            f"   Balance: {oldbalanceOrg:,.0f}\n"
            f"   Ratio: {ratio:.2f}x\n"
            f"   Assessment: Amount slightly exceeds balance.\n"
            f"   Context: Common in PaySim. Accounts may have credit lines.\n"
            f"   Signal Score: +0.5 points"
        )
    
    else:
        return (
            f"✓ NO ANOMALY:\n"
            f"   Amount: {amount:,.0f}\n"
            f"   Balance: {oldbalanceOrg:,.0f}\n"
            f"   Ratio: {ratio:.2f}x\n"
            f"   Assessment: Amount is within available balance. Safe.\n"
            f"   Signal Score: 0 points"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 3: Is Merchant Account
# ═══════════════════════════════════════════════════════════════════════════

@tool
def is_merchant_account(dest_id: str) -> str:
    """Check if destination account is a merchant.
    
    Merchants (account ID starts with 'M'):
    - Usually payment processors or retailers
    - Have oldbalanceDest ≈ 0 (no personal balance)
    - Lower fraud risk
    
    Args:
        dest_id: Destination account ID (nameDest)
    
    Returns:
        String with account type and signal score
    """
    
    if dest_id.startswith("M"):
        return (
            f"✓ MERCHANT ACCOUNT: {dest_id}\n"
            f"   Type: Payment processor or business account\n"
            f"   Risk Profile: LOWER (merchants are generally safe)\n"
            f"   Typical Use: Legitimate purchases, bill payments\n"
            f"   Signal Score: -1 point (safer)"
        )
    else:
        return (
            f"🟡 REGULAR USER ACCOUNT: {dest_id}\n"
            f"   Type: Personal/regular user account\n"
            f"   Risk Profile: NEUTRAL (can be fraud or legitimate)\n"
            f"   Assessment: Requires other signals to assess risk\n"
            f"   Signal Score: 0 points"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 4: Get Account Statistics
# ═══════════════════════════════════════════════════════════════════════════

@tool
def get_account_statistics(account_id: str) -> str:
    """Get comprehensive statistics for an account.
    
    Useful for assessing overall account risk and fraud history.
    
    Args:
        account_id: The account ID (nameOrig)
    
    Returns:
        String with account statistics
    """
    
    mask = df["nameOrig"] == account_id
    
    if not mask.any():
        return f"No transaction history for {account_id}"
    
    txs = df[mask]
    fraud_count = txs["isFraud"].sum()
    fraud_rate = (fraud_count / len(txs)) * 100 if len(txs) > 0 else 0
    
    # Type breakdown
    type_counts = txs["type"].value_counts().to_dict()
    
    return (
        f"Account Statistics for {account_id}:\n"
        f"  • Total transactions: {len(txs)}\n"
        f"  • Fraud count: {fraud_count}\n"
        f"  • Fraud rate: {fraud_rate:.2f}%\n"
        f"  • Avg transaction amount: {txs['amount'].mean():,.0f}\n"
        f"  • Transaction types: {type_counts}\n"
        f"  • Risk level: {'🔴 HIGH (>5% fraud)' if fraud_rate > 5 else '🟢 NORMAL'}"
    )


# ═══════════════════════════════════════════════════════════════════════════
# TOOL 5: Compare to Account Average
# ═══════════════════════════════════════════════════════════════════════════

@tool
def compare_to_account_average(origin_id: str, amount: float, tx_type: str) -> str:
    """Check if transaction amount is typical for this account.
    
    Unusual amounts may indicate fraud or account takeover.
    
    Args:
        origin_id: Sender account ID
        amount: Transaction amount
        tx_type: Transaction type (TRANSFER, CASH_OUT, etc.)
    
    Returns:
        String with comparison and signal score
    """
    
    mask = df["nameOrig"] == origin_id
    
    if not mask.any():
        return (
            f"⚠️ New account {origin_id}\n"
            f"   Cannot compare to account average\n"
            f"   Signal Score: +1 point (unusual - no baseline)"
        )
    
    account_txs = df[mask]
    same_type = account_txs[account_txs["type"] == tx_type]
    
    if len(same_type) == 0:
        return (
            f"⚠️ Account {origin_id} has never done {tx_type} before\n"
            f"   Amount: {amount:,.0f}\n"
            f"   First transaction of this type\n"
            f"   Signal Score: +1 point (unusual behavior)"
        )
    
    avg = same_type["amount"].mean()
    std = same_type["amount"].std()
    upper_bound = avg + 2 * std
    
    if amount > upper_bound:
        return (
            f"🔴 UNUSUAL AMOUNT for {origin_id}:\n"
            f"   Current amount: {amount:,.0f}\n"
            f"   Account average: {avg:,.0f}\n"
            f"   Upper bound (mean + 2σ): {upper_bound:,.0f}\n"
            f"   Assessment: Far above typical for this account\n"
            f"   Signal Score: +1 point"
        )
    else:
        return (
            f"✓ TYPICAL AMOUNT for {origin_id}:\n"
            f"   Current amount: {amount:,.0f}\n"
            f"   Account average: {avg:,.0f}\n"
            f"   Assessment: Within normal range for this account\n"
            f"   Signal Score: -1 point (normal behavior)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# TOOL EXPORTS
# ═══════════════════════════════════════════════════════════════════════════

tools = [
    get_origin_history,
    check_balance_anomaly,
    is_merchant_account,
    get_account_statistics,
    compare_to_account_average,
]

if __name__ == "__main__":
    # Test tools
    print("🔄 Testing tools...")
    print(get_origin_history("C1000000001"))
    print("\n" + "="*80 + "\n")
    print(check_balance_anomaly(100000, 50000, "CASH_OUT"))
    print("\n" + "="*80 + "\n")
    print(is_merchant_account("M123456"))
