#!/usr/bin/env python3
"""
Enhanced Lottery Prediction Bot with Advanced Analytics
"""

import requests
import json
from dotenv import load_dotenv
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import io
from collections import deque
from datetime import datetime, timedelta
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
import asyncio

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('lottery_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Files and API setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIGITS_FILE = os.path.join(BASE_DIR, "collected_digits.json")
LABELS_FILE = os.path.join(BASE_DIR, "collected_labels.json")
ACCURACY_FILE = os.path.join(BASE_DIR, "accuracy.json")
HISTORY_FILE = os.path.join(BASE_DIR, "prediction_history.json")

# API Configuration (Replace with your actual API)
API_URL = "https://6lotteryapi.com/api/webapi/GetNoaverageEmerdList"
API_PAYLOAD = {
    "pageSize": 10,
    "pageNo": 1,
    "typeId": 30,
    "language": 7,
    "random": "4b8d5e47503e4e40b300b0ccf6865e45",
    "signature": "15ED725FB82310BBAFF2B822D79DEA7D",
    "timestamp": 1753620290
}

# Load environment variables
load_dotenv()

# Bot Configuration (Updated to use environment variables)
BOT_TOKEN = os.getenv("BOT_TOKEN", "8371563583:AAEMp4NFClHZSAa0nP_BcBPpiKolccU4oGw")
CHAT_ID = os.getenv("CHAT_ID", "1632891170")
ADMIN_IDS = [os.getenv("ADMIN_ID", "1632891170")]
# Trading Parameters
SAFE_BET_THRESHOLD = 0.68    # Base confidence threshold
MAX_LOSS_STREAK = 3          # Stop after X consecutive losses
INITIAL_BANKROLL = 1000      # Starting balance
MIN_BET_SIZE = 5             # Minimum bet amount
MAX_BET_PERCENTAGE = 0.05    # Max 5% of bankroll per bet

# Model weights for ensemble
MODEL_WEIGHTS = {
    "markov_1": 0.25,
    "markov_2": 0.25, 
    "markov_3": 0.20,
    "frequency": 0.15,
    "pattern": 0.10,
    "anti_streak": 0.05
}

# ==================== UTILITY FUNCTIONS ====================

def load_json_file(filename, default=None):
    """Load JSON file with error handling"""
    if default is None:
        default = []
    
    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {filename}: {e}")
    return default

def save_json_file(filename, data):
    """Save data to JSON file"""
    try:
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save {filename}: {e}")

def digit_to_label(digit):
    """Convert digit to small/big label"""
    return "s" if digit <= 4 else "b"

def get_timestamp():
    """Get current timestamp"""
    return datetime.now().isoformat()

# ==================== PREDICTION MODELS ====================

def predict_markov_next_label_order(labels, order=1):
    """Markov chain prediction with specified order"""
    if len(labels) <= order:
        return None, 0.0
    
    transitions = {}
    counts = {}
    
    # Build transition matrix
    for i in range(len(labels) - order):
        pattern = tuple(labels[i:i + order])
        next_label = labels[i + order]
        
        if pattern not in transitions:
            transitions[pattern] = {"s": 0, "b": 0}
            counts[pattern] = 0
        
        transitions[pattern][next_label] += 1
        counts[pattern] += 1
    
    # Predict next label
    last_pattern = tuple(labels[-order:])
    if last_pattern in transitions and counts[last_pattern] > 0:
        s_prob = transitions[last_pattern]["s"] / counts[last_pattern]
        b_prob = transitions[last_pattern]["b"] / counts[last_pattern]
        
        predicted = "s" if s_prob > b_prob else "b"
        confidence = max(s_prob, b_prob)
        
        # Add smoothing for small sample sizes
        if counts[last_pattern] < 5:
            confidence *= 0.8
        
        return predicted, confidence
    
    return None, 0.0

def predict_frequency(labels, window=50):
    """Frequency-based prediction with sliding window"""
    if not labels:
        return None, 0.0
    
    # Use recent window for frequency analysis
    recent_labels = labels[-window:] if len(labels) > window else labels
    
    freq = {"s": recent_labels.count("s"), "b": recent_labels.count("b")}
    total = len(recent_labels)
    
    if total == 0:
        return None, 0.0
    
    s_prob = freq["s"] / total
    b_prob = freq["b"] / total
    
    # Predict less frequent outcome (regression to mean)
    predicted = "s" if s_prob < b_prob else "b"
    confidence = abs(s_prob - 0.5) + 0.5  # Distance from 50/50
    
    return predicted, min(confidence, 0.75)  # Cap at 75%

def detect_advanced_patterns(labels, window=15):
    """Advanced pattern detection"""
    if len(labels) < window:
        return None, None, 0.0
    
    # 1. Cycle detection
    for cycle_len in range(2, min(len(labels)//3, 7)):
        if len(labels) >= cycle_len * 3:
            pattern = labels[-cycle_len:]
            prev_pattern = labels[-cycle_len*2:-cycle_len]
            
            if pattern == prev_pattern:
                # Pattern repeats, predict next in cycle
                next_in_cycle = labels[-cycle_len*3] if len(labels) > cycle_len*3 else pattern[0]
                return f"cycle_{cycle_len}", next_in_cycle, 0.75
    
    # 2. Alternating pattern detection
    alternating_streak = 0
    for i in range(1, min(len(labels), 8)):
        if labels[-i] != labels[-i-1]:
            alternating_streak += 1
        else:
            break
    
    if alternating_streak >= 4:
        next_pred = "b" if labels[-1] == "s" else "s"
        return "alternating", next_pred, 0.65
    
    # 3. Hot/cold streak analysis
    recent = labels[-window:]
    s_ratio = recent.count("s") / len(recent)
    
    if s_ratio >= 0.75:
        return "hot_b_expected", "b", 0.7
    elif s_ratio <= 0.25:  
        return "hot_s_expected", "s", 0.7
    
    # 4. Consecutive same value detection
    consecutive_count = 1
    for i in range(1, min(len(labels), 6)):
        if labels[-i] == labels[-i-1]:
            consecutive_count += 1
        else:
            break
    
    if consecutive_count >= 4:
        opposite = "b" if labels[-1] == "s" else "s"
        return f"anti_streak_{consecutive_count}", opposite, 0.8
    
    return None, None, 0.0

def ensemble_prediction(labels):
    """Combine multiple models with weighted voting"""
    if len(labels) < 10:
        return None, 0.0, {}
    
    predictions = {}
    model_details = {}
    
    # 1. Markov models
    for order in [1, 2, 3]:
        pred, conf = predict_markov_next_label_order(labels, order)
        if pred:
            key = f"markov_{order}"
            predictions[key] = (pred, conf)
            model_details[key] = f"Order {order}: {pred.upper()} ({conf:.2f})"
    
    # 2. Frequency analysis
    pred, conf = predict_frequency(labels)
    if pred:
        predictions["frequency"] = (pred, conf)
        model_details["frequency"] = f"Frequency: {pred.upper()} ({conf:.2f})"
    
    # 3. Pattern detection
    pattern_type, pred, conf = detect_advanced_patterns(labels)
    if pred:
        predictions["pattern"] = (pred, conf)
        model_details["pattern"] = f"Pattern ({pattern_type}): {pred.upper()} ({conf:.2f})"
    
    if not predictions:
        return None, 0.0, {}
    
    # Weighted ensemble voting
    ensemble_votes = {"s": 0, "b": 0}
    total_weight = 0
    
    for method, (pred, conf) in predictions.items():
        weight = MODEL_WEIGHTS.get(method, 0.1)
        ensemble_votes[pred] += conf * weight
        total_weight += weight
    
    if total_weight == 0:
        return None, 0.0, model_details
    
    # Normalize votes
    final_pred = max(ensemble_votes, key=ensemble_votes.get)
    final_confidence = ensemble_votes[final_pred] / total_weight
    
    return final_pred, final_confidence, model_details

# ==================== RISK MANAGEMENT ====================

def get_dynamic_threshold(accuracy_data):
    """Calculate dynamic confidence threshold based on recent performance"""
    if not accuracy_data.get("recent_results"):
        return SAFE_BET_THRESHOLD
    
    recent_results = list(accuracy_data["recent_results"])
    if len(recent_results) < 10:
        return SAFE_BET_THRESHOLD
    
    recent_accuracy = sum(recent_results[-10:]) / 10
    
    # Adjust threshold based on recent performance
    adjustment = (recent_accuracy - 0.5) * 0.3
    new_threshold = SAFE_BET_THRESHOLD + adjustment
    
    # Keep threshold within reasonable bounds
    return max(0.6, min(0.85, new_threshold))

def kelly_bet_size(confidence, bankroll, win_odds=1.0, loss_odds=1.0):
    """Calculate optimal bet size using Kelly Criterion"""
    if confidence <= 0.5:
        return 0
    
    win_prob = confidence
    lose_prob = 1 - win_prob
    
    # Kelly formula: f = (bp - q) / b
    # where b = odds, p = win prob, q = lose prob
    edge = win_prob - lose_prob
    kelly_fraction = edge / win_odds if edge > 0 else 0
    
    # Apply conservative scaling (usually 25-50% of full Kelly)
    conservative_kelly = kelly_fraction * 0.25
    
    # Calculate bet size
    bet_size = bankroll * conservative_kelly
    
    # Apply limits
    bet_size = max(MIN_BET_SIZE, min(bet_size, bankroll * MAX_BET_PERCENTAGE))
    
    return round(bet_size, 2)

def is_risky_conditions(labels, accuracy_data):
    """Check for risky betting conditions"""
    risks = []
    
    # 1. High loss streak
    if accuracy_data.get("loss_streak", 0) >= MAX_LOSS_STREAK:
        risks.append("🛑 Stop-loss activated")
    
    # 2. Low recent accuracy
    recent_results = list(accuracy_data.get("recent_results", []))
    if len(recent_results) >= 10:
        recent_accuracy = sum(recent_results[-10:]) / 10
        if recent_accuracy < 0.4:
            risks.append("📉 Poor recent performance")
    
    # 3. Extreme streaks in data
    if len(labels) >= 8:
        consecutive_count = 1
        for i in range(1, min(len(labels), 8)):
            if labels[-i] == labels[-i-1]:
                consecutive_count += 1
            else:
                break
        
        if consecutive_count >= 6:
            risks.append(f"⚠️ Extreme streak detected ({consecutive_count})")
    
    # 4. Low data volume
    if len(labels) < 30:
        risks.append("📊 Insufficient historical data")
    
    return risks

# ==================== ACCURACY TRACKING ====================

def load_accuracy():
    """Load accuracy tracking data"""
    default_data = {
        "total": 0,
        "correct": 0,
        "accuracy_percent": 0.0,
        "recent_results": [],
        "bankroll": INITIAL_BANKROLL,
        "loss_streak": 0,
        "win_streak": 0,
        "max_win_streak": 0,
        "max_loss_streak": 0,
        "total_profit": 0.0,
        "best_day": 0.0,
        "worst_day": 0.0,
        "last_updated": get_timestamp()
    }
    
    try:
        with open(ACCURACY_FILE, "r") as f:
            data = json.load(f)
            # Convert recent_results to deque for efficiency
            data["recent_results"] = deque(data.get("recent_results", []), maxlen=100)
            return {**default_data, **data}
    except Exception:
        return default_data

def save_accuracy(data):
    """Save accuracy data"""
    # Convert deque back to list for JSON serialization
    data_copy = data.copy()
    data_copy["recent_results"] = list(data.get("recent_results", []))
    data_copy["last_updated"] = get_timestamp()
    
    save_json_file(ACCURACY_FILE, data_copy)

def update_accuracy(predicted, actual, bet_amount, data):
    """Update accuracy statistics"""
    data["total"] += 1
    correct = int(predicted == actual)
    data["correct"] += correct
    
    # Update recent results
    if "recent_results" not in data:
        data["recent_results"] = deque(maxlen=100)
    data["recent_results"].append(correct)
    
    # Calculate accuracy percentage
    data["accuracy_percent"] = round((data["correct"] / data["total"]) * 100, 2)
    
    # Update streaks
    if correct:
        data["win_streak"] = data.get("win_streak", 0) + 1
        data["loss_streak"] = 0
        data["max_win_streak"] = max(data.get("max_win_streak", 0), data["win_streak"])
    else:
        data["loss_streak"] = data.get("loss_streak", 0) + 1
        data["win_streak"] = 0
        data["max_loss_streak"] = max(data.get("max_loss_streak", 0), data["loss_streak"])
    
    # Update bankroll
    profit_loss = bet_amount if correct else -bet_amount
    data["bankroll"] += profit_loss
    data["total_profit"] = data.get("total_profit", 0) + profit_loss
    
    # Track best/worst day
    data["best_day"] = max(data.get("best_day", 0), profit_loss)
    data["worst_day"] = min(data.get("worst_day", 0), profit_loss)
    
    save_accuracy(data)
    return data

# ==================== TELEGRAM BOT HANDLERS ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler"""
    welcome_text = (
        "🎯 **Enhanced Lottery Prediction Bot**\n\n"
        "🤖 **Available Commands:**\n"
        "• `/predict` - Get AI prediction with risk analysis\n"
        "• `/stats` - View detailed performance statistics\n"
        "• `/history` - Show recent prediction history\n"
        "• `/simulate` - Backtest strategy on historical data\n"
        "• `/settings` - Adjust bot parameters\n"
        "• `/help` - Show detailed help\n\n"
        "📊 **Features:**\n"
        "✅ Multiple AI models (Markov, Pattern, Frequency)\n"
        "✅ Advanced risk management\n"
        "✅ Kelly Criterion position sizing\n"
        "✅ Real-time performance tracking\n"
        "✅ Interactive feedback system\n\n"
        "⚠️ **Disclaimer:** For educational purposes only. "
        "Past performance doesn't guarantee future results."
    )
    
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced prediction with comprehensive analysis"""
    try:
        labels = load_json_file(LABELS_FILE)
        digits = load_json_file(DIGITS_FILE)
        
        if len(labels) < 10:
            await update.message.reply_text(
                "⚠️ **Insufficient Data**\n\n"
                "Need at least 10 historical results for reliable prediction.\n"
                f"Current data points: {len(labels)}"
            )
            return
        
        # Get ensemble prediction
        final_pred, final_confidence, model_details = ensemble_prediction(labels)
        
        if not final_pred:
            await update.message.reply_text("❌ Unable to generate prediction at this time.")
            return
        
        # Risk analysis
        accuracy_data = load_accuracy()
        threshold = get_dynamic_threshold(accuracy_data)
        risks = is_risky_conditions(labels, accuracy_data)
        
        # Betting recommendation
        suggested_bet = kelly_bet_size(final_confidence, accuracy_data["bankroll"])
        
        # Risk level assessment
        if final_confidence >= 0.8 and not risks:
            risk_level = "🟢 **LOW RISK**"
            risk_emoji = "✅"
        elif final_confidence >= 0.65 and len(risks) <= 1:
            risk_level = "🟡 **MEDIUM RISK**"
            risk_emoji = "⚠️"
        else:
            risk_level = "🔴 **HIGH RISK**"
            risk_emoji = "❌"
        
        # Recent performance
        recent_results = list(accuracy_data.get("recent_results", []))
        recent_accuracy = sum(recent_results[-10:]) / 10 if len(recent_results) >= 10 else 0.5
        
        # Build prediction message
        prediction_text = (
            f"🎯 **PREDICTION ANALYSIS**\n\n"
            f"🔮 **Prediction:** {final_pred.upper()} ({'Small (0-4)' if final_pred == 's' else 'Big (5-9)'})\n"
            f"📊 **Confidence:** {final_confidence:.1%}\n"
            f"🎯 **Risk Level:** {risk_level}\n\n"
            f"💰 **BETTING RECOMMENDATION**\n"
            f"{risk_emoji} Confidence threshold: {threshold:.0%}\n"
            f"💵 Suggested bet: ${suggested_bet:.2f} ({suggested_bet/accuracy_data['bankroll']*100:.1f}% of bankroll)\n"
        )
        
        if risks:
            prediction_text += f"\n⚠️ **Risk Warnings:**\n"
            for risk in risks:
                prediction_text += f"• {risk}\n"
        
        prediction_text += (
            f"\n📈 **PERFORMANCE METRICS**\n"
            f"• Recent accuracy (10 games): {recent_accuracy*100:.0f}%\n"
            f"• Current bankroll: ${accuracy_data['bankroll']:.2f}\n"
            f"• Win/Loss streak: {accuracy_data.get('win_streak', 0)}/{accuracy_data.get('loss_streak', 0)}\n\n"
            f"📊 **DATA INSIGHTS**\n"
            f"• Last 5 results: {' '.join(labels[-5:]).upper()}\n"
            f"• S/B ratio (last 20): {labels[-20:].count('s')}/{labels[-20:].count('b')}\n"
            f"• Total historical data: {len(labels)} draws"
        )
        
        # Create interactive buttons
        keyboard = [
            [
                InlineKeyboardButton("✅ Correct", callback_data=f"feedback_correct_{final_pred}_{suggested_bet}"),
                InlineKeyboardButton("❌ Wrong", callback_data=f"feedback_wrong_{final_pred}_{suggested_bet}")
            ],
            [
                InlineKeyboardButton("📊 Model Details", callback_data="show_models"),
                InlineKeyboardButton("📈 Visual Chart", callback_data="show_chart")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            prediction_text, 
            parse_mode='Markdown', 
            reply_markup=reply_markup
        )
        
        # Save prediction to history
        save_prediction_history(final_pred, final_confidence, model_details, suggested_bet)
        
    except Exception as e:
        logger.error(f"Error in predict_command: {e}")
        await update.message.reply_text("❌ An error occurred while generating prediction.")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show comprehensive statistics with visual chart"""
    try:
        accuracy_data = load_accuracy()
        labels = load_json_file(LABELS_FILE)
        
        if accuracy_data["total"] == 0:
            await update.message.reply_text("📊 No prediction history available yet.")
            return
        
        # Create performance chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Lottery Bot Performance Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Accuracy trend
        recent_results = list(accuracy_data["recent_results"])
        if recent_results:
            x = range(len(recent_results))
            cumulative_accuracy = np.cumsum(recent_results) / np.arange(1, len(recent_results) + 1)
            ax1.plot(x, cumulative_accuracy, 'b-', linewidth=2, label='Cumulative Accuracy')
            ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random (50%)')
            ax1.set_title('Prediction Accuracy Trend')
            ax1.set_ylabel('Accuracy')
            ax1.set_xlabel('Prediction #')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Bankroll evolution
        if len(recent_results) > 1:
            # Simulate bankroll changes (simplified)
            bankroll_history = [INITIAL_BANKROLL]
            current_bankroll = INITIAL_BANKROLL
            avg_bet = 50  # Simplified average bet
            
            for result in recent_results:
                profit = avg_bet if result else -avg_bet
                current_bankroll += profit
                bankroll_history.append(current_bankroll)
            
            ax2.plot(range(len(bankroll_history)), bankroll_history, 'g-', linewidth=2)
            ax2.axhline(y=INITIAL_BANKROLL, color='gray', linestyle='--', alpha=0.7)
            ax2.set_title('Bankroll Evolution')
            ax2.set_ylabel('Bankroll ($)')
            ax2.set_xlabel('Prediction #')
            ax2.grid(True, alpha=0.3)
        
        # 3. Win/Loss distribution
        if labels and len(labels) >= 20:
            recent_labels = labels[-50:] if len(labels) > 50 else labels
            s_count = recent_labels.count('s')
            b_count = recent_labels.count('b')
            
            ax3.bar(['Small (0-4)', 'Big (5-9)'], [s_count, b_count], 
                   color=['skyblue', 'lightcoral'], alpha=0.8)
            ax3.set_title('Recent Number Distribution')
            ax3.set_ylabel('Count')
            
            # Add percentage labels
            total = s_count + b_count
            ax3.text(0, s_count + 1, f'{s_count/total*100:.1f}%', ha='center')
            ax3.text(1, b_count + 1, f'{b_count/total*100:.1f}%', ha='center')
        
        # 4. Performance metrics
        metrics_text = (
            f"Overall Accuracy: {accuracy_data['accuracy_percent']:.1f}%\n"
            f"Total Predictions: {accuracy_data['total']}\n"
            f"Current Bankroll: ${accuracy_data['bankroll']:.2f}\n"
            f"Total P&L: ${accuracy_data.get('total_profit', 0):.2f}\n"
            f"Max Win Streak: {accuracy_data.get('max_win_streak', 0)}\n"
            f"Max Loss Streak: {accuracy_data.get('max_loss_streak', 0)}\n"
            f"Best Day: ${accuracy_data.get('best_day', 0):.2f}\n"
            f"Worst Day: ${accuracy_data.get('worst_day', 0):.2f}"
        )
        
        ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax4.set_title('Key Performance Metrics')
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save chart to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Statistics summary text
        recent_10_accuracy = sum(recent_results[-10:]) / 10 if len(recent_results) >= 10 else 0
        
        stats_text = (
            f"📊 **PERFORMANCE DASHBOARD**\n\n"
            f"🎯 **Overall Statistics:**\n"
            f"• Total Predictions: {accuracy_data['total']}\n"
            f"• Correct Predictions: {accuracy_data['correct']}\n"
            f"• Overall Accuracy: {accuracy_data['accuracy_percent']:.1f}%\n\n"
            f"📈 **Recent Performance:**\n"
            f"• Last 10 Games: {recent_10_accuracy*100:.0f}% accuracy\n"
            f"• Current Win Streak: {accuracy_data.get('win_streak', 0)}\n"
            f"• Current Loss Streak: {accuracy_data.get('loss_streak', 0)}\n\n"
            f"💰 **Financial Metrics:**\n"
            f"• Current Bankroll: ${accuracy_data['bankroll']:.2f}\n"
            f"• Total P&L: ${accuracy_data.get('total_profit', 0):+.2f}\n"
            f"• ROI: {accuracy_data.get('total_profit', 0)/INITIAL_BANKROLL*100:+.1f}%\n\n"
            f"📊 **Historical Records:**\n"
            f"• Max Win Streak: {accuracy_data.get('max_win_streak', 0)}\n"
            f"• Max Loss Streak: {accuracy_data.get('max_loss_streak', 0)}\n"
            f"• Best Single Day: ${accuracy_data.get('best_day', 0):+.2f}\n"
            f"• Worst Single Day: ${accuracy_data.get('worst_day', 0):+.2f}"
        )
        
        await update.message.reply_photo(
            photo=buf, 
            caption=stats_text, 
            parse_mode='Markdown'
        )
        
    except Exception as e:
        logger.error(f"Error in stats_command: {e}")
        await update.message.reply_text("❌ Error generating statistics.")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline button interactions"""
    query = update.callback_query
    await query.answer()
    
    try:
        data = query.data
        
        if data.startswith("feedback_"):
            parts = data.split("_")
            result = parts[1]  # correct or wrong
            predicted = parts[2]  # s or b
            bet_amount = float(parts[3]) if len(parts) > 3 else 50.0
            
            accuracy_data = load_accuracy()
            
            if result == "correct":
                actual = predicted
                accuracy_data = update_accuracy(predicted, actual, bet_amount, accuracy_data)
                response_text = (
                    f"✅ **Feedback Received: CORRECT**\n\n"
                    f"🎯 Prediction: {predicted.upper()}\n"
                    f"💰 Profit: +${bet_amount:.2f}\n"
                    f"📊 Updated Accuracy: {accuracy_data['accuracy_percent']:.1f}%\n"
                    f"💵 New Bankroll: ${accuracy_data['bankroll']:.2f}\n"
                    f"🔥 Win Streak: {accuracy_data.get('win_streak', 0)}"
                )
            else:
                actual = "b" if predicted == "s" else "s"
                accuracy_data = update_accuracy(predicted, actual, bet_amount, accuracy_data)
                response_text = (
                    f"❌ **Feedback Received: WRONG**\n\n"
                    f"🎯 Prediction: {predicted.upper()}\n"
                    f"📉 Loss: -${bet_amount:.2f}\n"
                    f"📊 Updated Accuracy: {accuracy_data['accuracy_percent']:.1f}%\n"
                    f"💵 New Bankroll: ${accuracy_data['bankroll']:.2f}\n"
                    f"💔 Loss Streak: {accuracy_data.get('loss_streak', 0)}"
                )
            
            await query.edit_message_text(response_text, parse_mode='Markdown')
        
        elif data == "show_models":
            labels = load_json_file(LABELS_FILE)
            final_pred, final_confidence, model_details = ensemble_prediction(labels)
            
            models_text = (
                f"🧠 **MODEL BREAKDOWN**\n\n"
                f"🎯 **Final Prediction:** {final_pred.upper()} ({final_confidence:.1%})\n\n"
                f"📊 **Individual Models:**\n"
            )
            
            for model, details in model_details.items():
                models_text += f"• {details}\n"
            
            models_text += (
                f"\n⚖️ **Model Weights:**\n"
                f"• Markov Order 1: {MODEL_WEIGHTS['markov_1']*100:.0f}%\n"
                f"• Markov Order 2: {MODEL_WEIGHTS['markov_2']*100:.0f}%\n"
                f"• Markov Order 3: {MODEL_WEIGHTS['markov_3']*100:.0f}%\n"
                f"• Frequency Analysis: {MODEL_WEIGHTS['frequency']*100:.0f}%\n"
                f"• Pattern Detection: {MODEL_WEIGHTS['pattern']*100:.0f}%\n"
                f"• Anti-Streak: {MODEL_WEIGHTS['anti_streak']*100:.0f}%"
            )
            
            await query.edit_message_text(models_text, parse_mode='Markdown')
        
        elif data == "show_chart":
            await show_prediction_chart(query)
    
    except Exception as e:
        logger.error(f"Error in button_callback: {e}")
        await query.edit_message_text("❌ Error processing request.")

async def show_prediction_chart(query):
    """Show visual prediction analysis chart"""
    try:
        labels = load_json_file(LABELS_FILE)
        digits = load_json_file(DIGITS_FILE)
        
        if len(labels) < 20:
            await query.edit_message_text("⚠️ Need more data for chart visualization.")
            return
        
        # Create comprehensive chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Prediction Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Recent results timeline
        recent_labels = labels[-30:] if len(labels) > 30 else labels
        recent_digits = digits[-30:] if len(digits) > 30 else digits
        
        x_pos = range(len(recent_labels))
        colors = ['skyblue' if label == 's' else 'lightcoral' for label in recent_labels]
        
        ax1.bar(x_pos, [1]*len(recent_labels), color=colors, alpha=0.7)
        ax1.set_title('Recent Results Pattern (Blue=Small, Red=Big)')
        ax1.set_ylabel('Frequency')
        ax1.set_xlabel('Draw Number (Most Recent →)')
        
        # Add actual numbers as labels
        for i, digit in enumerate(recent_digits):
            ax1.text(i, 0.5, str(digit), ha='center', va='center', fontweight='bold')
        
        # 2. Distribution analysis
        s_counts = []
        b_counts = []
        window_size = 10
        
        for i in range(window_size, len(labels)):
            window = labels[i-window_size:i]
            s_counts.append(window.count('s'))
            b_counts.append(window.count('b'))
        
        if s_counts and b_counts:
            ax2.plot(range(len(s_counts)), s_counts, 'b-', label='Small Count', linewidth=2)
            ax2.plot(range(len(b_counts)), b_counts, 'r-', label='Big Count', linewidth=2)
            ax2.axhline(y=5, color='gray', linestyle='--', alpha=0.7, label='Expected (50%)')
            ax2.set_title(f'Rolling {window_size}-Draw Distribution')
            ax2.set_ylabel('Count')
            ax2.set_xlabel('Window Position')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Markov transition matrix visualization
        transitions = {"ss": 0, "sb": 0, "bs": 0, "bb": 0}
        for i in range(len(labels)-1):
            state = labels[i] + labels[i+1]
            if state in transitions:
                transitions[state] += 1
        
        total = sum(transitions.values())
        if total > 0:
            transition_probs = {k: v/total for k, v in transitions.items()}
            
            states = list(transition_probs.keys())
            probs = list(transition_probs.values())
            
            bars = ax3.bar(states, probs, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow'])
            ax3.set_title('State Transition Probabilities')
            ax3.set_ylabel('Probability')
            ax3.set_xlabel('State Transition')
            
            # Add probability labels on bars
            for bar, prob in zip(bars, probs):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{prob:.2f}', ha='center', va='bottom')
        
        # 4. Prediction confidence over time
        history = load_json_file(HISTORY_FILE, [])
        if len(history) > 1:
            confidences = [entry.get('confidence', 0) for entry in history[-20:]]
            predictions = [entry.get('prediction', 's') for entry in history[-20:]]
            
            x_hist = range(len(confidences))
            colors_hist = ['green' if pred == 's' else 'red' for pred in predictions]
            
            bars = ax4.bar(x_hist, confidences, color=colors_hist, alpha=0.7)
            ax4.axhline(y=SAFE_BET_THRESHOLD, color='black', linestyle='--', 
                       label=f'Threshold ({SAFE_BET_THRESHOLD:.0%})')
            ax4.set_title('Recent Prediction Confidences')
            ax4.set_ylabel('Confidence')
            ax4.set_xlabel('Prediction #')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No prediction history\navailable yet', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
            ax4.set_title('Prediction History')
        
        plt.tight_layout()
        
        # Save chart to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Send chart
        await query.message.reply_photo(
            photo=buf,
            caption="📊 **Visual Prediction Analysis**\n\nThis chart shows recent patterns, distributions, and prediction confidence trends."
        )
        
    except Exception as e:
        logger.error(f"Error in show_prediction_chart: {e}")
        await query.edit_message_text("❌ Error generating chart.")

async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show recent prediction history"""
    try:
        history = load_json_file(HISTORY_FILE, [])
        
        if not history:
            await update.message.reply_text("📝 No prediction history available yet.")
            return
        
        # Show last 10 predictions
        recent_history = history[-10:]
        
        history_text = "📝 **RECENT PREDICTION HISTORY**\n\n"
        
        for i, entry in enumerate(reversed(recent_history), 1):
            timestamp = entry.get('timestamp', 'Unknown')
            prediction = entry.get('prediction', 'Unknown').upper()
            confidence = entry.get('confidence', 0)
            bet_amount = entry.get('bet_amount', 0)
            
            # Format timestamp
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime('%m/%d %H:%M')
            except:
                time_str = 'Unknown'
            
            history_text += (
                f"**#{len(recent_history)-i+1}** - {time_str}\n"
                f"🎯 Prediction: {prediction} ({confidence:.1%})\n"
                f"💰 Bet Amount: ${bet_amount:.2f}\n\n"
            )
        
        history_text += f"📊 Total Predictions: {len(history)}"
        
        await update.message.reply_text(history_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in history_command: {e}")
        await update.message.reply_text("❌ Error retrieving history.")

async def simulate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Backtest strategy on historical data"""
    try:
        labels = load_json_file(LABELS_FILE)
        
        if len(labels) < 100:
            await update.message.reply_text(
                f"⚠️ Need at least 100 historical results for simulation.\n"
                f"Current data points: {len(labels)}"
            )
            return
        
        await update.message.reply_text("🔬 Running simulation... This may take a moment.")
        
        # Simulation parameters
        sim_window = min(50, len(labels) - 50)  # Test on last 50 predictions
        initial_bankroll = 1000
        bankroll = initial_bankroll
        
        results = []
        correct_predictions = 0
        total_predictions = 0
        bets_made = []
        
        # Run simulation
        for i in range(len(labels) - sim_window, len(labels)):
            if i < 20:  # Need minimum data for prediction
                continue
                
            # Use historical data up to this point
            historical_labels = labels[:i]
            actual_result = labels[i]
            
            # Make prediction using ensemble
            pred, confidence, _ = ensemble_prediction(historical_labels)
            
            if pred and confidence >= 0.6:  # Only bet if reasonably confident
                bet_size = kelly_bet_size(confidence, bankroll)
                
                if bet_size >= MIN_BET_SIZE:  # Only make bet if above minimum
                    total_predictions += 1
                    
                    if pred == actual_result:
                        correct_predictions += 1
                        profit = bet_size
                        bankroll += profit
                        result_symbol = "✅"
                    else:
                        profit = -bet_size
                        bankroll -= bet_size
                        result_symbol = "❌"
                    
                    bets_made.append({
                        'prediction': pred,
                        'actual': actual_result,
                        'confidence': confidence,
                        'bet_size': bet_size,
                        'profit': profit,
                        'bankroll': bankroll,
                        'correct': pred == actual_result
                    })
        
        # Calculate metrics
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        total_profit = bankroll - initial_bankroll
        roi = (total_profit / initial_bankroll * 100) if initial_bankroll > 0 else 0
        
        # Win streak analysis
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for bet in bets_made:
            if bet['correct']:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        # Build results message
        simulation_text = (
            f"🔬 **SIMULATION RESULTS**\n\n"
            f"📊 **Performance Metrics:**\n"
            f"• Total Predictions: {total_predictions}\n"
            f"• Correct Predictions: {correct_predictions}\n"
            f"• Accuracy: {accuracy:.1f}%\n\n"
            f"💰 **Financial Results:**\n"
            f"• Initial Bankroll: ${initial_bankroll:.2f}\n"
            f"• Final Bankroll: ${bankroll:.2f}\n"
            f"• Net P&L: ${total_profit:+.2f}\n"
            f"• ROI: {roi:+.1f}%\n\n"
            f"📈 **Risk Metrics:**\n"
            f"• Max Win Streak: {max_win_streak}\n"
            f"• Max Loss Streak: {max_loss_streak}\n"
            f"• Average Bet Size: ${sum(bet['bet_size'] for bet in bets_made)/len(bets_made):.2f if bets_made else 0:.2f}\n\n"
            f"📋 **Last 5 Simulated Bets:**\n"
        )
        
        # Show last 5 bets
        for bet in bets_made[-5:]:
            result_symbol = "✅" if bet['correct'] else "❌"
            simulation_text += (
                f"{result_symbol} {bet['prediction'].upper()} "
                f"({bet['confidence']:.1%}) "
                f"${bet['profit']:+.2f} → ${bet['bankroll']:.2f}\n"
            )
        
        await update.message.reply_text(simulation_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error in simulate_command: {e}")
        await update.message.reply_text("❌ Error running simulation.")

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show and modify bot settings"""
    user_id = str(update.effective_user.id)
    
    if user_id not in ADMIN_IDS:
        await update.message.reply_text("⚠️ This command is only available to administrators.")
        return
    
    settings_text = (
        f"⚙️ **CURRENT SETTINGS**\n\n"
        f"🎯 **Prediction Parameters:**\n"
        f"• Confidence Threshold: {SAFE_BET_THRESHOLD:.0%}\n"
        f"• Max Loss Streak: {MAX_LOSS_STREAK}\n\n"
        f"💰 **Betting Parameters:**\n"
        f"• Initial Bankroll: ${INITIAL_BANKROLL}\n"
        f"• Min Bet Size: ${MIN_BET_SIZE}\n"
        f"• Max Bet %: {MAX_BET_PERCENTAGE:.1%}\n\n"
        f"🤖 **Model Weights:**\n"
        f"• Markov Order 1: {MODEL_WEIGHTS['markov_1']:.0%}\n"
        f"• Markov Order 2: {MODEL_WEIGHTS['markov_2']:.0%}\n"
        f"• Markov Order 3: {MODEL_WEIGHTS['markov_3']:.0%}\n"
        f"• Frequency: {MODEL_WEIGHTS['frequency']:.0%}\n"
        f"• Pattern: {MODEL_WEIGHTS['pattern']:.0%}\n"
        f"• Anti-Streak: {MODEL_WEIGHTS['anti_streak']:.0%}\n\n"
        f"📊 **Data Status:**\n"
        f"• Historical Data Points: {len(load_json_file(LABELS_FILE))}\n"
        f"• Prediction History: {len(load_json_file(HISTORY_FILE, []))}"
    )
    
    await update.message.reply_text(settings_text, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show detailed help information"""
    help_text = (
        f"🤖 **LOTTERY PREDICTION BOT HELP**\n\n"
        f"📋 **Available Commands:**\n\n"
        f"🎯 `/predict` - Get AI prediction\n"
        f"   • Shows prediction with confidence level\n"
        f"   • Includes risk analysis and betting recommendation\n"
        f"   • Interactive feedback buttons\n\n"
        f"📊 `/stats` - Performance statistics\n"
        f"   • Overall accuracy and performance metrics\n"
        f"   • Visual charts and graphs\n"
        f"   • Bankroll and P&L tracking\n\n"
        f"📝 `/history` - Recent predictions\n"
        f"   • Last 10 predictions with timestamps\n"
        f"   • Confidence levels and bet amounts\n\n"
        f"🔬 `/simulate` - Strategy backtesting\n"
        f"   • Tests strategy on historical data\n"
        f"   • Shows accuracy and profitability\n"
        f"   • Risk metrics and performance analysis\n\n"
        f"⚙️ `/settings` - Bot configuration (Admin only)\n"
        f"   • View current parameters\n"
        f"   • Model weights and thresholds\n\n"
        f"🆘 `/help` - This help message\n\n"
        f"🧠 **How It Works:**\n"
        f"• Uses multiple AI models (Markov chains, pattern detection, frequency analysis)\n"
        f"• Combines predictions using weighted ensemble\n"
        f"• Applies Kelly Criterion for bet sizing\n"
        f"• Includes comprehensive risk management\n\n"
        f"⚠️ **Important Notes:**\n"
        f"• This is for educational purposes only\n"
        f"• Past performance doesn't guarantee future results\n"
        f"• Lottery numbers are fundamentally random\n"
        f"• Never bet more than you can afford to lose\n\n"
        f"💡 **Tips for Best Results:**\n"
        f"• Wait for high confidence predictions (>70%)\n"
        f"• Follow the risk warnings\n"
        f"• Use suggested bet sizes\n"
        f"• Provide feedback to improve accuracy"
    )
    
    await update.message.reply_text(help_text, parse_mode='Markdown')

# ==================== UTILITY FUNCTIONS ====================

def save_prediction_history(prediction, confidence, model_details, bet_amount):
    """Save prediction to history file"""
    try:
        history = load_json_file(HISTORY_FILE, [])
        
        entry = {
            'timestamp': get_timestamp(),
            'prediction': prediction,
            'confidence': confidence,
            'bet_amount': bet_amount,
            'model_details': model_details
        }
        
        history.append(entry)
        
        # Keep only last 100 predictions
        if len(history) > 100:
            history = history[-100:]
        
        save_json_file(HISTORY_FILE, history)
        
    except Exception as e:
        logger.error(f"Error saving prediction history: {e}")

# ==================== DATA FETCHING ====================

async def fetch_lottery_results(context: ContextTypes.DEFAULT_TYPE, initial_call=False):
    """Fetch new lottery results from API and send prediction"""
    try:
        if initial_call:
            logger.info("Initial API call on startup...")
        else:
            logger.info("Fetching lottery results...")
        
        response = requests.post(API_URL, json=API_PAYLOAD, timeout=15)
        response.raise_for_status()
        
        data = response.json().get("data", {}).get("list", [])
        
        if not data:
            logger.warning("No data received from API")
            return
        
        # Extract latest result
        latest_result = data[0]
        new_number = latest_result.get("number", "")
        
        if not new_number:
            logger.warning("No number in latest result")
            return
        
        # Get last digit
        new_digit = int(str(new_number)[-1])
        new_label = digit_to_label(new_digit)
        
        # Load existing data
        digits = load_json_file(DIGITS_FILE)
        
        # Check if this is actually new data
        if digits and len(digits) > 0 and digits[0] == new_digit:
            logger.info("No new lottery data")
            return
        
        # Add new data
        digits.insert(0, new_digit)
        labels = [digit_to_label(d) for d in digits]
        
        # Keep reasonable history size
        if len(digits) > 1000:
            digits = digits[:1000]
            labels = labels[:1000]
        
        # Save updated data
        save_json_file(DIGITS_FILE, digits)
        save_json_file(LABELS_FILE, labels)
        
        # Build basic notification text
        notification_text = (
            f"🔄 **NEW RESULT AVAILABLE**\n\n"
            f"🎲 Number: {new_digit}\n"
            f"📊 Category: {new_label.upper()} ({'Small (0-4)' if new_label == 's' else 'Big (5-9)'})\n"
            f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}\n\n"
            f"📈 Total Data Points: {len(digits)}\n"
        )
        
        # Only try to make prediction if we have enough data
        if len(labels) >= 10:  # Minimum data points for prediction
            # Generate prediction for next round
            final_pred, final_confidence, model_details = ensemble_prediction(labels)
            
            if final_pred:  # Only proceed if we got a valid prediction
                accuracy_data = load_accuracy()
                threshold = get_dynamic_threshold(accuracy_data)
                risks = is_risky_conditions(labels, accuracy_data)
                suggested_bet = kelly_bet_size(final_confidence, accuracy_data["bankroll"])
                
                # Risk level assessment
                if final_confidence >= 0.8 and not risks:
                    risk_level = "🟢 LOW RISK"
                    risk_emoji = "✅"
                elif final_confidence >= 0.65 and len(risks) <= 1:
                    risk_level = "🟡 MEDIUM RISK"
                    risk_emoji = "⚠️"
                else:
                    risk_level = "🔴 HIGH RISK"
                    risk_emoji = "❌"
                
                # Add prediction to notification
                notification_text += (
                    f"\n🔮 **NEXT PREDICTION**\n"
                    f"🎯 Prediction: {final_pred.upper()} ({'Small (0-4)' if final_pred == 's' else 'Big (5-9)'})\n"
                    f"📊 Confidence: {final_confidence:.1%}\n"
                    f"🎯 Risk Level: {risk_level}\n"
                    f"💰 Suggested bet: ${suggested_bet:.2f}\n"
                )
                
                # Add risk warnings if any
                if risks:
                    notification_text += f"\n⚠️ **Risk Warnings:**\n"
                    for risk in risks:
                        notification_text += f"• {risk}\n"
                
                # Create interactive buttons
                keyboard = [
                    [
                        InlineKeyboardButton("✅ Correct", callback_data=f"feedback_correct_{final_pred}_{suggested_bet}"),
                        InlineKeyboardButton("❌ Wrong", callback_data=f"feedback_wrong_{final_pred}_{suggested_bet}")
                    ],
                    [
                        InlineKeyboardButton("📊 Model Details", callback_data="show_models"),
                        InlineKeyboardButton("📈 Visual Chart", callback_data="show_chart")
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                # Save prediction to history
                save_prediction_history(final_pred, final_confidence, model_details, suggested_bet)
            else:
                notification_text += "\n⚠️ Not enough data to make a prediction yet."
                reply_markup = None
        else:
            notification_text += f"\n📊 Collecting more data... ({len(labels)}/10 needed for predictions)"
            reply_markup = None
        
        # Send notification
        await context.bot.send_message(
            chat_id=CHAT_ID,
            text=notification_text,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
        
        logger.info(f"New lottery result processed: {new_digit} ({new_label})")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error: {e}")
    except Exception as e:
        logger.error(f"Error in fetch_lottery_results: {e}") 
# ==================== MAIN BOT SETUP ====================

def main():
    """Main bot initialization and startup"""
    logger.info("Starting Enhanced Lottery Prediction Bot...")
    
    # Validate configuration
    if not BOT_TOKEN or BOT_TOKEN == "YOUR_BOT_TOKEN":
        logger.error("Bot token not configured!")
        return
    
    if not CHAT_ID or CHAT_ID == "YOUR_CHAT_ID":
        logger.error("Chat ID not configured!")
        return
    
    try:
        # Create application
        app = ApplicationBuilder().token(BOT_TOKEN).build()
        
        # Add command handlers
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("predict", predict_command))
        app.add_handler(CommandHandler("stats", stats_command))
        app.add_handler(CommandHandler("history", history_command))
        app.add_handler(CommandHandler("simulate", simulate_command))
        app.add_handler(CommandHandler("settings", settings_command))
        app.add_handler(CommandHandler("help", help_command))
        
        # Add callback handler for buttons
        app.add_handler(CallbackQueryHandler(button_callback))
        
        # Add periodic data fetching
        job_queue = app.job_queue
        
        # Make immediate API call on startup
        job_queue.run_once(
            lambda ctx: asyncio.create_task(fetch_lottery_results(ctx, initial_call=True)), 
            0.1  # Run after 100ms to ensure bot is fully initialized
        )
        
        # Set up repeating job every 30 seconds
        job_queue.run_repeating(
            fetch_lottery_results, 
            interval=30.0,  # Check every 30 seconds
            first=30.0      # First run 30 seconds after initial call
        )
        
        logger.info("Bot initialized successfully")
        logger.info(f"Monitoring chat ID: {CHAT_ID}")
        logger.info(f"Admin IDs: {ADMIN_IDS}")
        
        # Start the bot
        print("🤖 Enhanced Lottery Prediction Bot is running...")
        print("Press Ctrl+C to stop the bot")
        
        app.run_polling(drop_pending_updates=True)
        
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        print(f"❌ Error starting bot: {e}")

if __name__ == "__main__":
    main()