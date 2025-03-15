from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
import pandas as pd
import numpy as np
import json
import math

from tools.api import get_insider_trades, get_company_news


##### Sentiment Agent #####
def sentiment_agent(state: AgentState):
    """Analyzes market sentiment and generates trading signals for multiple tickers."""
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")

    # Initialize sentiment analysis for each ticker
    sentiment_analysis = {}

    for ticker in tickers:
        progress.update_status("sentiment_agent", ticker, "Fetching insider trades")

        # Get the insider trades
        insider_trades = get_insider_trades(
            ticker=ticker,
            end_date=end_date,
            limit=100,
        )

        progress.update_status("sentiment_agent", ticker, "Analyzing trading patterns")

        # Get the signals from the insider trades
        bullish_insider_num = 0
        bearish_insider_num = 0
        bullish_insider_score = 0.0
        bearish_insider_score = 0.0

        for trade in insider_trades:
            if trade.transaction_type == "D":
                bullish_insider_num += 1
                bullish_insider_score += trade.transaction_value
            else:
                bearish_insider_num += 1
                bearish_insider_score += trade.transaction_value

        overall_insider_score = bullish_insider_score - bearish_insider_score
        overall_insider_num = bullish_insider_num + bearish_insider_num
        overall_insider_confident = max(bullish_insider_score, bearish_insider_score)/(bullish_insider_score + bearish_insider_score) if (bullish_insider_score + bearish_insider_score) > 0 else 0.0
        bullish_insider_confident = bullish_insider_score / (bullish_insider_score + bearish_insider_score) if (bullish_insider_score + bearish_insider_score) > 0 else 0.0
        bearish_insider_confident = bearish_insider_score / (bullish_insider_score + bearish_insider_score) if (bullish_insider_score + bearish_insider_score) > 0 else 0.0
        # Calculate the overall sentiment
        if overall_insider_score > 0:
            overall_insider_sentiment = "bullish"
        elif overall_insider_score < 0:
            overall_insider_sentiment = "bearish"
        else:
            overall_insider_sentiment = "neutral"
        # Calculate the overall confidence

        # Calculate the overall sentiment
        #trade.transaction_shares = -trade.transaction_value
        #transaction_shares = pd.Series([t.transaction_shares for t in insider_trades]).dropna()
        #insider_signals = np.where(transaction_shares < 0, "bearish", "bullish").tolist()

        progress.update_status("sentiment_agent", ticker, "Fetching company news")

        # Get the company news
        company_news = get_company_news(ticker, end_date, limit=100)

        # Get the sentiment from the company news
        # 计算整体情感得分
        overall_sentiment = "Neutral"
        bullish_sentiment_num = 0
        bearish_sentiment_num = 0
        neutral_sentiment_num = 0

        bullish_sentiment_score = 0.0
        bearish_sentiment_score = 0.0
        neutral_sentiment_score = 0.0

        total_weighted_score = 0.0
        total_relevance = 0.0
        feed_num = len(company_news)

        for news in company_news:
            if 'Bullish' in news.sentiment:
                bullish_sentiment_num += news.relevance_score
                bullish_sentiment_score += news.sentiment_score * news.relevance_score
            elif 'Bearish' in news.sentiment:
                bearish_sentiment_num += news.relevance_score
                bearish_sentiment_score += news.sentiment_score * news.relevance_score
            else:
                neutral_sentiment_num += news.relevance_score
                neutral_sentiment_score += news.sentiment_score * news.relevance_score
            
            total_weighted_score += news.sentiment_score * news.relevance_score
            total_relevance += news.relevance_score
        
        avg_bullish_score = 0.0
        avg_bearish_score = 0.0
        avg_neutral_score = 0.0
        avg_weighted_score = 0.0
        if feed_num > 0:
            avg_weighted_score = total_weighted_score / total_relevance if total_relevance > 0 else 0
            avg_bullish_score = bullish_sentiment_score / bullish_sentiment_num if bullish_sentiment_num > 0 else 0
            avg_bearish_score = bearish_sentiment_score / bearish_sentiment_num if bearish_sentiment_num > 0 else 0
            avg_neutral_score = neutral_sentiment_score / neutral_sentiment_num if neutral_sentiment_num > 0 else 0
            if avg_weighted_score >= 0.35:
                overall_sentiment = "Bullish"
            elif avg_weighted_score >= 0.15:
                overall_sentiment = "Somewhat_Bullish"
            elif avg_weighted_score <= -0.35:
                overall_sentiment = "Bearish"
            elif avg_weighted_score <= -0.15:
                overall_sentiment = "Somewhat_Bearish"
        overall_sentiment_confidence = math.fabs(avg_weighted_score) #max(avg_bullish_score, -avg_bearish_score) / (avg_neutral_score+ avg_bullish_score + avg_bearish_score) if (avg_neutral_score+avg_bullish_score + avg_bearish_score) > 0 else 0.0
        #sentiment = pd.Series([n.sentiment for n in company_news]).dropna()
        #news_signals = np.where(sentiment == "negative", "bearish", 
        #                      np.where(sentiment == "positive", "bullish", "neutral")).tolist()
        # 
        #progress.update_status("sentiment_agent", ticker, "Combining signals")
        # Combine signals from both sources with weights
        insider_weight = 0.3
        news_weight = 0.7
        
        # Calculate weighted signal counts
        bullish_signals = (
            bullish_insider_confident * insider_weight +
            avg_bullish_score * news_weight
        )
        bearish_signals = (
            bearish_insider_confident * insider_weight +
            (-avg_bearish_score) * news_weight
        )

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level based on the weighted proportion

        confidence = (insider_weight * overall_insider_confident + news_weight * overall_sentiment_confidence)*100
        #confidence = 0  # Default confidence when there are no signals
        #if total_weighted_signals > 0:
        #    confidence = round(max(bullish_signals, bearish_signals) / total_weighted_signals, 2) * 100
        reasoning = f"Weighted Bullish signals: {bullish_signals:.1f}, Weighted Bearish signals: {bearish_signals:.1f}"

        sentiment_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status("sentiment_agent", ticker, "Done")

    # Create the sentiment message
    message = HumanMessage(
        content=json.dumps(sentiment_analysis),
        name="sentiment_agent",
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(sentiment_analysis, "Sentiment Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["sentiment_agent"] = sentiment_analysis

    return {
        "messages": [message],
        "data": data,
    }
