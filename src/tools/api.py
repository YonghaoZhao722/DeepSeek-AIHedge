import os
import pandas as pd
import requests
import time

from data.cache import get_cache
from data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    AlphaVantagePriceResponse,
    AlphaVantageInsiderTradeResponse,
    AlphaVantageInsiderTrade,
    AlphaVantageNewsResponse,
    AlphaVantageNews,
)

# Global cache instance
_cache = get_cache()


def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """获取价格数据"""
    financial_data = get_financial_data(ticker, start_date, end_date, ["prices"])
    return financial_data.prices if financial_data and financial_data.prices else []


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """获取财务指标"""
    financial_data = get_financial_data(ticker, end_date=end_date, data_types=["fundamentals", "overview"])
    if not financial_data:
        return []
    
    metrics = calculate_financial_metrics(financial_data, ticker)
    return [metrics] if metrics else []


def get_market_cap(ticker: str, end_date: str) -> float | None:
    """获取市值数据"""
    financial_data = get_financial_data(ticker, end_date=end_date, data_types=["overview"])
    if not financial_data or not financial_data.overview:
        return None
    return float(financial_data.overview.get("MarketCapitalization", 0))


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """获取具体财务项目数据"""
    financial_data = get_financial_data(ticker, end_date=end_date, data_types=["fundamentals"])
    if not financial_data:
        return []

    # 映射财务项目到Alpha Vantage的对应项
    av_mapping = {
        "capital_expenditure": "capitalExpenditures",
        "depreciation_and_amortization": "depreciationAndAmortization",
        "net_income": "netIncome",
        "outstanding_shares": "commonStockSharesOutstanding",
        "total_assets": "totalAssets",
        "total_liabilities": "totalLiabilities",
        "free_cash_flow": "operatingCashflow",
        "working_capital": "totalCurrentAssets"
    }

    reports = []
    if financial_data.cash_flow and "annualReports" in financial_data.cash_flow:
        for i in range(min(limit, len(financial_data.cash_flow["annualReports"]))):
            report_date = financial_data.cash_flow["annualReports"][i]["fiscalDateEnding"]
            if report_date > end_date:
                continue

            report = {
                "report_period": report_date,
                "period": period,
                "currency": "USD",
                "ticker": ticker
            }

            for item in line_items:
                if item not in av_mapping:
                    continue

                av_item = av_mapping[item]
                value = None

                # 在不同报表中查找数据
                for data_source in [
                    financial_data.cash_flow["annualReports"][i],
                    financial_data.balance_sheet["annualReports"][i] if financial_data.balance_sheet else {},
                    financial_data.income_statement["annualReports"][i] if financial_data.income_statement else {}
                ]:
                    if av_item in data_source:
                        try:
                            value = float(data_source[av_item])
                            break
                        except (ValueError, TypeError):
                            continue

                # 特殊处理计算项
                if item == "free_cash_flow" and value is not None:
                    capex = float(financial_data.cash_flow["annualReports"][i].get("capitalExpenditures", 0))
                    value = value - capex
                elif item == "working_capital" and value is not None and financial_data.balance_sheet:
                    current_liabilities = float(financial_data.balance_sheet["annualReports"][i].get("totalCurrentLiabilities", 0))
                    value = value - current_liabilities

                report[item] = value

            reports.append(LineItem(**report))

    return reports


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    """获取内部交易数据"""
    # 检查缓存
    if cached_data := _cache.get_insider_trades(ticker):
        # 过滤缓存数据
        filtered_data = [InsiderTrade(**trade) for trade in cached_data 
                        if (start_date is None or (trade.get("transaction_date") or trade["filing_date"]) >= start_date)
                        and (trade.get("transaction_date") or trade["filing_date"]) <= end_date]
        filtered_data.sort(key=lambda x: x.transaction_date or x.filing_date, reverse=True)
        if filtered_data:
            return filtered_data

    # 如果缓存中没有数据，从Alpha Vantage获取
    if not (api_key := os.environ.get("ALPHA_VANTAGE_API_KEY")):
        raise Exception("Alpha Vantage API key not found")

    try:
        # 添加延迟以避免API限制
        time.sleep(12)  # Alpha Vantage免费版API限制为每分钟5次调用

        url = f"https://www.alphavantage.co/query?function=INSIDER_TRANSACTIONS&symbol={ticker}&apikey={api_key}"
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error fetching insider trades for {ticker}: {response.status_code}")
            return []

        data = response.json()
        
        # 检查API限制
        if "Information" in data and "rate limit" in data["Information"].lower():
            print(f"API rate limit reached for {ticker} insider trades")
            return []

        # 解析响应数据
        av_response = AlphaVantageInsiderTradeResponse(**data)
        
        # 转换为统一格式
        insider_trades = []
        for transaction in av_response.insider_transactions:
            try:
                av_trade = AlphaVantageInsiderTrade(**transaction)
                # 过滤日期范围
                trade_date = av_trade.transaction_date or av_trade.filing_date
                if (start_date is None or trade_date >= start_date) and trade_date <= end_date:
                    insider_trades.append(av_trade.to_insider_trade())
            except Exception as e:
                print(f"Error processing insider trade: {str(e)}")
                continue

        # 按日期排序并限制数量
        insider_trades.sort(key=lambda x: x.transaction_date or x.filing_date, reverse=True)
        insider_trades = insider_trades[:limit]

        if insider_trades:
            # 缓存结果
            _cache.set_insider_trades(ticker, [trade.model_dump() for trade in insider_trades])

        return insider_trades

    except Exception as e:
        print(f"Error fetching insider trades for {ticker}: {str(e)}")
        return []


def analyze_news_sentiment_batch(ticker: str, news_items: list[tuple[str, str | None]]) -> list[str]:
    """批量使用DeepSeek模型分析新闻情感
    
    Args:
        ticker: 股票代码
        news_items: 新闻列表，每项为(标题, 摘要)的元组，摘要可为None
    
    Returns:
        list[str]: 情感分析结果列表，每项为'positive'、'negative'或'neutral'
    """
    from llm.models import get_model, ModelProvider
    from langchain_core.messages import HumanMessage
    
    # 构建批量提示词
    prompt = f"""分析以下多条新闻对股票 {ticker} 的影响。对每条新闻，请判断其影响是积极(positive)、消极(negative)还是中性(neutral)。

请按顺序分析以下新闻，每条新闻只返回一个词：'positive'、'negative' 或 'neutral'。
每条新闻的结果占一行。

新闻列表：
"""
    
    for i, (title, summary) in enumerate(news_items, 1):
        prompt += f"\n{i}. 标题: {title}"
        if summary:
            prompt += f"\n   摘要: {summary}"
        prompt += "\n"

    try:
        # 获取DeepSeek模型实例
        model = get_model("deepseek-ai/DeepSeek-V3", ModelProvider.SILICONFLOW)
        if not model:
            print("Error: Failed to initialize DeepSeek model")
            return ['neutral'] * len(news_items)
            
        # 调用模型获取情感分析结果
        response = model.invoke([HumanMessage(content=prompt)])
        
        # 解析响应
        sentiments = []
        for line in response.content.strip().split('\n'):
            line = line.strip().lower()
            if 'positive' in line:
                sentiments.append('positive')
            elif 'negative' in line:
                sentiments.append('negative')
            else:
                sentiments.append('neutral')
        
        # 确保返回结果数量与输入新闻数量相同
        if len(sentiments) < len(news_items):
            sentiments.extend(['neutral'] * (len(news_items) - len(sentiments)))
        elif len(sentiments) > len(news_items):
            sentiments = sentiments[:len(news_items)]
            
        return sentiments
        
    except Exception as e:
        print(f"Error analyzing sentiment batch: {str(e)}")
        return ['neutral'] * len(news_items)


def analyze_news_sentiment(ticker: str, news_title: str, news_summary: str | None = None) -> str:
    """使用DeepSeek模型分析单条新闻情感"""
    return analyze_news_sentiment_batch(ticker, [(news_title, news_summary)])[0]


def get_company_news(ticker: str, end_date: str, limit: int = 5) -> list[CompanyNews]:
    """获取公司新闻"""
    # 检查缓存
    if cached_data := _cache.get_company_news(ticker):
        # 过滤缓存数据
        filtered_data = [CompanyNews(**news) for news in cached_data 
                        if (start_date is None or news["date"] >= start_date)
                        and news["date"] <= end_date]
        filtered_data.sort(key=lambda x: x.date, reverse=True)
        if filtered_data:
            return filtered_data

    # 如果缓存中没有数据，从Alpha Vantage获取
    if not (api_key := os.environ.get("ALPHA_VANTAGE_API_KEY")):
        raise Exception("Alpha Vantage API key not found")

    try:
        # 添加延迟以避免API限制
        time.sleep(12)  # Alpha Vantage免费版API限制为每分钟5次调用

        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}"
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error fetching news for {ticker}: {response.status_code}")
            return []

        data = response.json()
        
        # 检查API限制
        if "Information" in data and "rate limit" in data["Information"].lower():
            print(f"API rate limit reached for {ticker} news")
            return []

        # 解析响应数据
        av_response = AlphaVantageNewsResponse(**data)
        
        # 收集所有符合日期范围的新闻
        valid_news_items = []
        news_data = []
        for news_item in av_response.items:
            try:
                av_news = AlphaVantageNews(**news_item)
                news_date = av_news.time_published.split("T")[0]  # 提取日期部分
                
                # 过滤日期范围
                if (start_date is None or news_date >= start_date) and news_date <= end_date:
                    valid_news_items.append((av_news.title, av_news.summary))
                    news_data.append(av_news)
            except Exception as e:
                print(f"Error processing news item: {str(e)}")
                continue

        # 批量分析情感
        if valid_news_items:
            sentiments = analyze_news_sentiment_batch(ticker, valid_news_items)
            
            # 创建CompanyNews对象
            # Parse and format news items
            company_news = []
            for av_news, sentiment in zip(news_data, sentiments):
                # Add null check and field validation
                if av_news.time_published:  # Use time_published field as publication date
                    news = CompanyNews(
                        ticker=ticker,
                        title=av_news.title,
                        author=", ".join(av_news.authors) if av_news.authors else "Unknown",
                        source=av_news.source or av_news.source_domain or "Unknown",
                        published=av_news.time_published,  # Map time_published to published
                        url=av_news.url,
                        sentiment=sentiment
                    )
                    company_news.append(news)
    
            # 按日期排序并限制数量
            company_news.sort(key=lambda x: x.date, reverse=True)
            company_news = company_news[:limit]

            if company_news:
                # 缓存结果
                _cache.set_company_news(ticker, [news.model_dump() for news in company_news])

            return company_news

        return []

    except Exception as e:
        print(f"Error fetching news for {ticker}: {str(e)}")
        return []


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)


class FinancialData:
    def __init__(self):
        self.income_statement = None
        self.balance_sheet = None
        self.cash_flow = None
        self.overview = None
        self.prices = None

# 添加一个新函数来检测 API 限制错误
def check_api_limit_error(response_data):
    """检查响应中是否包含 API 限制错误信息"""
    if isinstance(response_data, dict) and "Information" in response_data:
        info = response_data["Information"]
        if "rate limit" in info.lower() or "API key" in info.lower():
            return info
    return None

def get_financial_data(
    ticker: str,
    start_date: str = None,
    end_date: str = None,
    data_types: list[str] = ["all"],
    api_errors: list = None
) -> FinancialData:
    """统一获取财务数据的函数
    
    Args:
        ticker: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        data_types: 需要获取的数据类型，可选：["prices", "fundamentals", "overview", "all"]
        api_errors: 用于收集 API 错误的列表
    """
    if api_errors is None:
        api_errors = []
        
    if not (api_key := os.environ.get("ALPHA_VANTAGE_API_KEY")):
        raise Exception("Alpha Vantage API key not found")

    # 检查缓存
    cached_data = _cache.get_financial_data(ticker)
    if cached_data:
        result = FinancialData()
        result.income_statement = cached_data.get("income_statement")
        result.balance_sheet = cached_data.get("balance_sheet")
        result.cash_flow = cached_data.get("cash_flow")
        result.overview = cached_data.get("overview")
        result.prices = [Price(**p) for p in cached_data.get("prices", [])]
        
        # 检查是否有所需的数据类型
        has_required_data = True
        if "prices" in data_types and not result.prices:
            has_required_data = False
        if "fundamentals" in data_types and not (result.income_statement and result.balance_sheet and result.cash_flow):
            has_required_data = False
        if "overview" in data_types and not result.overview:
            has_required_data = False
            
        if has_required_data:
            return result

    result = FinancialData()
    try:
        # 获取价格数据
        if "all" in data_types or "prices" in data_types:
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                
                # 检查 API 限制
                if error_msg := check_api_limit_error(data):
                    api_errors.append(f"价格数据 ({ticker}): {error_msg}")
                    
                av_response = AlphaVantagePriceResponse(**data)
                result.prices = [
                    Price(**{**dict(price_data), "time": date})
                    for date, price_data in av_response.time_series.items()
                    if (not start_date or date >= start_date) and (not end_date or date <= end_date)
                ]
                result.prices.sort(key=lambda x: x.time)

        # 获取基本面数据
        if "all" in data_types or "fundamentals" in data_types:
            # 获取现金流量表
            response = requests.get(f"https://www.alphavantage.co/query?function=CASH_FLOW&symbol={ticker}&apikey={api_key}")
            if response.status_code == 200:
                data = response.json()
                # 检查 API 限制
                if error_msg := check_api_limit_error(data):
                    api_errors.append(f"现金流量表 ({ticker}): {error_msg}")
                result.cash_flow = data

            # 获取资产负债表
            response = requests.get(f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={api_key}")
            if response.status_code == 200:
                data = response.json()
                # 检查 API 限制
                if error_msg := check_api_limit_error(data):
                    api_errors.append(f"资产负债表 ({ticker}): {error_msg}")
                result.balance_sheet = data

            # 获取利润表
            response = requests.get(f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={api_key}")
            if response.status_code == 200:
                data = response.json()
                # 检查 API 限制
                if error_msg := check_api_limit_error(data):
                    api_errors.append(f"利润表 ({ticker}): {error_msg}")
                result.income_statement = data

        # 获取公司概览数据
        if "all" in data_types or "overview" in data_types:
            response = requests.get(f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}")
            if response.status_code == 200:
                data = response.json()
                # 检查 API 限制
                if error_msg := check_api_limit_error(data):
                    api_errors.append(f"公司概览 ({ticker}): {error_msg}")
                result.overview = data

        # 缓存结果
        cache_data = {
            "income_statement": result.income_statement,
            "balance_sheet": result.balance_sheet,
            "cash_flow": result.cash_flow,
            "overview": result.overview,
            "prices": [p.model_dump() for p in result.prices] if result.prices else []
        }
        _cache.set_financial_data(ticker, cache_data)
        return result

    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return result

def calculate_financial_metrics(financial_data: FinancialData, ticker: str) -> FinancialMetrics:
    """根据原始财务数据计算财务指标"""
    try:
        overview = financial_data.overview
        if not overview:
            print(f"No overview data available for {ticker}")
            return None

        # 获取最新的财务报表数据
        income_statement = financial_data.income_statement.get("annualReports", [{}])[0] if financial_data.income_statement else {}
        balance_sheet = financial_data.balance_sheet.get("annualReports", [{}])[0] if financial_data.balance_sheet else {}
        cash_flow = financial_data.cash_flow.get("annualReports", [{}])[0] if financial_data.cash_flow else {}

        # 打印调试信息
        print(f"Processing financial data for {ticker}")
        print(f"Overview data: {overview}")
        print(f"Income statement: {income_statement}")
        print(f"Balance sheet: {balance_sheet}")
        print(f"Cash flow: {cash_flow}")

        # 计算企业价值 (Enterprise Value)
        market_cap = float(overview.get("MarketCapitalization", 0))
        total_debt = float(balance_sheet.get("totalLongTermDebt", 0)) + float(balance_sheet.get("shortTermDebt", 0))
        cash_and_equiv = float(balance_sheet.get("cashAndCashEquivalentsAtCarryingValue", 0))
        enterprise_value = market_cap + total_debt - cash_and_equiv

        # 计算收入和自由现金流
        revenue = float(income_statement.get("totalRevenue", 0))
        operating_cashflow = float(cash_flow.get("operatingCashflow", 0))
        capital_expenditure = float(cash_flow.get("capitalExpenditures", 0))
        free_cash_flow = operating_cashflow - capital_expenditure

        # 计算其他比率
        price_to_sales = market_cap / revenue if revenue else 0
        free_cash_flow_yield = (free_cash_flow / market_cap * 100) if market_cap else 0

        # 计算增长率
        if len(financial_data.income_statement.get("annualReports", [])) > 1:
            prev_revenue = float(financial_data.income_statement["annualReports"][1].get("totalRevenue", 0))
            revenue_growth = ((revenue - prev_revenue) / prev_revenue * 100) if prev_revenue else 0
        else:
            revenue_growth = float(overview.get("QuarterlyRevenueGrowthYOY", 0)) * 100

        # 获取最新季度日期，如果没有则使用当前日期
        latest_quarter = overview.get("LatestQuarter", "") or income_statement.get("fiscalDateEnding", "") or balance_sheet.get("fiscalDateEnding", "")
        if not latest_quarter:
            from datetime import datetime
            latest_quarter = datetime.now().strftime("%Y-%m-%d")

        return FinancialMetrics(
            ticker=ticker,  # 使用传入的ticker而不是从overview中获取
            calendar_date=latest_quarter,
            report_period=latest_quarter,
            period="ttm",
            currency="USD",
            market_cap=market_cap,
            enterprise_value=enterprise_value,
            price_to_earnings_ratio=float(overview.get("PERatio", 0)),
            price_to_book_ratio=float(overview.get("PriceToBookRatio", 0)),
            price_to_sales_ratio=price_to_sales,
            enterprise_value_to_revenue_ratio=enterprise_value / revenue if revenue else 0,
            enterprise_value_to_ebitda_ratio=float(overview.get("EVToEBITDA", 0)),
            free_cash_flow_yield=free_cash_flow_yield,
            peg_ratio=float(overview.get("PEGRatio", 0)),
            gross_margin=float(overview.get("GrossProfitTTM", 0)) / revenue * 100 if revenue else 0,
            operating_margin=float(overview.get("OperatingMarginTTM", 0)),
            net_margin=float(overview.get("ProfitMargin", 0)) * 100,
            return_on_equity=float(overview.get("ReturnOnEquityTTM", 0)) * 100,
            return_on_assets=float(overview.get("ReturnOnAssetsTTM", 0)) * 100,
            return_on_invested_capital=float(overview.get("ReturnOnInvestedCapital", 0)) * 100,
            asset_turnover=revenue / float(balance_sheet.get("totalAssets", 1)),
            inventory_turnover=float(overview.get("InventoryTurnover", 0)),
            receivables_turnover=revenue / float(balance_sheet.get("currentNetReceivables", 1)),
            days_sales_outstanding=365 / (revenue / float(balance_sheet.get("currentNetReceivables", 1))),
            operating_cycle=365 / float(overview.get("InventoryTurnover", 365)),
            working_capital_turnover=revenue / (float(balance_sheet.get("totalCurrentAssets", 0)) - float(balance_sheet.get("totalCurrentLiabilities", 0))),
            current_ratio=float(balance_sheet.get("totalCurrentAssets", 0)) / float(balance_sheet.get("totalCurrentLiabilities", 1)),
            quick_ratio=(float(balance_sheet.get("totalCurrentAssets", 0)) - float(balance_sheet.get("inventory", 0))) / float(balance_sheet.get("totalCurrentLiabilities", 1)),
            cash_ratio=cash_and_equiv / float(balance_sheet.get("totalCurrentLiabilities", 1)),
            operating_cash_flow_ratio=operating_cashflow / float(balance_sheet.get("totalCurrentLiabilities", 1)),
            debt_to_equity=float(overview.get("DebtToEquityRatio", 0)),
            debt_to_assets=total_debt / float(balance_sheet.get("totalAssets", 1)),
            interest_coverage=float(income_statement.get("ebit", 0)) / float(income_statement.get("interestExpense", 1)),
            revenue_growth=revenue_growth,
            earnings_growth=float(overview.get("QuarterlyEarningsGrowthYOY", 0)) * 100,
            book_value_growth=0.0,  # 需要历史数据计算
            earnings_per_share_growth=float(overview.get("EPS", 0)),
            free_cash_flow_growth=0.0,  # 需要历史数据计算
            operating_income_growth=0.0,  # 需要历史数据计算
            ebitda_growth=0.0,  # 需要历史数据计算
            payout_ratio=float(overview.get("PayoutRatio", 0)) * 100,
            earnings_per_share=float(overview.get("EPS", 0)),
            book_value_per_share=float(overview.get("BookValue", 0)),
            free_cash_flow_per_share=free_cash_flow / float(overview.get("SharesOutstanding", 1))
        )
    except Exception as e:
        print(f"Error calculating financial metrics for {ticker}: {str(e)}")
        print(f"Overview data: {overview}")
        print(f"Income statement: {income_statement}")
        print(f"Balance sheet: {balance_sheet}")
        print(f"Cash flow: {cash_flow}")
        return None
