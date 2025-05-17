import os
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from flask_sqlalchemy import SQLAlchemy
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Replace with something secure
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stocks.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

################################################################################
#                                 DATABASE MODELS
################################################################################
class Stock(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    person_name  = db.Column(db.String(50), nullable=False)   # ← must exist
    ticker       = db.Column(db.String(20), nullable=False)
    quantity     = db.Column(db.Integer, nullable=False)


class StockAlert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), nullable=False)
    threshold_up = db.Column(db.Float, nullable=True)
    threshold_down = db.Column(db.Float, nullable=True)
    alert_triggered = db.Column(db.Boolean, default=False)
    triggered_on = db.Column(db.DateTime, nullable=True)

with app.app_context():
    db.create_all()

################################################################################
#                             PER-TICKER LSTM MODELS
################################################################################

LSTM_MODELS = {
    "DRREDDY.NS": "models/lstm_DRREDDY.NS.h5",
    "HEROMOTOCO.NS": "models/lstm_HEROMOTOCO.NS.h5",
    "BPCL.NS": "models/lstm_BPCL.NS.h5",
    "BAJFINANCE.NS": "models/lstm_BAJFINANCE.NS.h5",
    "ONGC.NS": "models/lstm_ONGC.NS.h5",
    "UPL.NS": "models/lstm_UPL.NS.h5",
    "SBIN.NS": "models/lstm_SBIN.NS.h5",
    "LT.NS": "models/lstm_LT.NS.h5",
    "SBILIFE.NS": "models/lstm_SBILIFE.NS.h5",
    "HINDUNILVR.NS": "models/lstm_HINDUNILVR.NS.h5",
    "TATAMOTORS.NS": "models/lstm_TATAMOTORS.NS.h5",
    "NESTLEIND.NS": "models/lstm_NESTLEIND.NS.h5",
    "ICICIBANK.NS": "models/lstm_ICICIBANK.NS.h5",
    "TATACONSUM.NS": "models/lstm_TATACONSUM.NS.h5",
    "ASIANPAINT.NS": "models/lstm_ASIANPAINT.NS.h5",
    "M&M.NS": "models/lstm_M&M.NS.h5",
    "BAJAJFINSV.NS": "models/lstm_BAJAJFINSV.NS.h5",
    "INFY.NS": "models/lstm_INFY.NS.h5",
    "TATASTEEL.NS": "models/lstm_TATASTEEL.NS.h5",
    "ITC.NS": "models/lstm_ITC.NS.h5",
    "BHARTIARTL.NS": "models/lstm_BHARTIARTL.NS.h5",
    "WIPRO.NS": "models/lstm_WIPRO.NS.h5",
    "SUNPHARMA.NS": "models/lstm_SUNPHARMA.NS.h5",
    "HDFCBANK.NS": "models/lstm_HDFCBANK.NS.h5",
    "TCS.NS": "models/lstm_TCS.NS.h5",
    "HINDALCO.NS": "models/lstm_HINDALCO.NS.h5",
    "BRITANNIA.NS": "models/lstm_BRITANNIA.NS.h5",
    "CIPLA.NS": "models/lstm_CIPLA.NS.h5",
    "DIVISLAB.NS": "models/lstm_DIVISLAB.NS.h5",
    "COALINDIA.NS": "models/lstm_COALINDIA.NS.h5",
    "MARUTI.NS": "models/lstm_MARUTI.NS.h5",
    "KOTAKBANK.NS": "models/lstm_KOTAKBANK.NS.h5",
    "GRASIM.NS": "models/lstm_GRASIM.NS.h5",
    "POWERGRID.NS": "models/lstm_POWERGRID.NS.h5",
    "ULTRACEMCO.NS": "models/lstm_ULTRACEMCO.NS.h5",
    "HCLTECH.NS": "models/lstm_HCLTECH.NS.h5",
    "RELIANCE.NS": "models/lstm_RELIANCE.NS.h5",
    "AXISBANK.NS": "models/lstm_AXISBANK.NS.h5",
    "NTPC.NS": "models/lstm_NTPC.NS.h5"
}
MODELS_CACHE = {}

def get_lstm_model_for_ticker(ticker):
    """Loads & caches the LSTM model for the given ticker."""
    if ticker not in LSTM_MODELS:
        raise ValueError(f"No LSTM model path found for {ticker}")
    if ticker not in MODELS_CACHE:
        path = LSTM_MODELS[ticker]
        model = tf.keras.models.load_model(path)
        MODELS_CACHE[ticker] = model
    return MODELS_CACHE[ticker]

################################################################################
#                           HELPER FUNCTIONS
################################################################################

def fetch_stock_data(ticker, period='1mo'):
    try:
        data = yf.download(ticker, period=period)
        if data.empty:
            raise ValueError("No data returned")
        return data
    except Exception as e:
        print(f"Failed to fetch {ticker}: {e}")
        return pd.DataFrame()

def get_stock_data(tickers, period='1mo'):
    info_map = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            df = tk.history(period=period)
            if df.empty:
                df = tk.history(period="5d")
            if df.empty:
                raise ValueError("No data returned at all")
            latest = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2] if len(df) > 1 else latest
            pct_chg = round(((latest - prev_close) / prev_close)*100, 2) if prev_close else 0
            info_map[t] = {
                'latest_price': latest,
                'price_change': pct_chg,
                'sector': tk.info.get('sector','N/A'),
                'high_52w': tk.info.get('fiftyTwoWeekHigh','N/A'),
                'low_52w': tk.info.get('fiftyTwoWeekLow','N/A')
            }
        except Exception as e:
            print(f"Error for {t}: {e}")
            info_map[t] = {
                'latest_price': 0,
                'price_change': 0,
                'sector': 'N/A',
                'high_52w': 'N/A',
                'low_52w': 'N/A'
            }
    return info_map

def calculate_portfolio_performance(stocks, start_date="2023-01-01", end_date=None):
    """Plot a time-series chart of the portfolio’s cumulative returns."""
    if not stocks:
        return ""
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    tickers = [s['ticker'] for s in stocks]
    df = yf.download(tickers, start=start_date, end=end_date)['Close']
    if df.empty or len(df) < 2:
        return ""

    w = np.array([s['quantity'] for s in stocks], dtype=float)
    w /= w.sum()
    daily_returns = df.pct_change().dropna()
    portfolio_cum = (daily_returns @ w).cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_cum.index,
        y=portfolio_cum,
        mode='lines',
        name='Portfolio Cumulative Returns'
    ))
    fig.update_layout(title="Portfolio Performance",
                      xaxis_title="Date",
                      yaxis_title="Cumulative Returns")
    return fig.to_html()

def create_pie_chart(tickers, weights, title):
    df = pd.DataFrame({"Stocks": tickers, "Weights": weights})
    fig = px.pie(df, names="Stocks", values="Weights", title=title)
    return fig.to_html()

def get_price_change(stocks):
    changes = {}
    for s in stocks:
        t = s['ticker']
        df = fetch_stock_data(t, period='1mo')
        if not df.empty and len(df) > 1:
            start_p = df['Close'].iloc[0]
            end_p = df['Close'].iloc[-1]
            changes[t] = round(((end_p - start_p)/start_p)*100, 2)
        else:
            changes[t] = "No data"
    return changes

def calculate_macd(df, short=12, long=26, signal=9):
    df['EMA12'] = df['Close'].ewm(span=short, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=long, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    return df

def analyze_macd_sentiment(df):
    m = df['MACD'].iloc[-1]
    s = df['Signal'].iloc[-1]
    if m > s:
        return "Bullish Sentiment"
    elif m < s:
        return "Bearish Sentiment"
    return "Neutral"

def get_nifty_returns(start_date="2020-01-01", end_date=None):
    if not end_date:
        end_date = datetime.today().strftime('%Y-%m-%d')
    n = yf.Ticker("^NSEI")
    df = n.history(start=start_date, end=end_date)
    df['Returns'] = df['Close'].pct_change()
    return df['Returns'].dropna().to_numpy()

################################################################################
#                  LSTM PREDICTION / FORECAST FOR EACH TICKER
################################################################################

def predict_stock_prices(_unused):
    """
    Predict a short-horizon return for each ticker in session to be used in optimization.
    """
    preds = []
    st = session.get('stocks', [])
    for s in st:
        t = s['ticker']
        try:
            data = yf.download(t, period="1mo")['Close']
            if data.empty:
                preds.append(0)
                continue
            window = min(30, len(data))
            arr = data[-window:].values.astype('float32').reshape(1, window, 1)
            model = get_lstm_model_for_ticker(t)
            out = model.predict(arr)
            preds.append(float(out.mean()))
        except Exception as e:
            print(f"Prediction error for {t}: {e}")
            preds.append(0)
    return preds

def forecast_future_prices(ticker, days=15, window=30):
    df = yf.download(ticker, period="3mo")['Close']
    if df.empty or len(df)<window:
        return [0]*days
    model = get_lstm_model_for_ticker(ticker)
    recent = df[-window:].values.astype('float32')
    out = []
    for _ in range(days):
        x = recent.reshape(1, window, 1)
        p = model.predict(x)
        nxt = float(p[0][0])
        out.append(nxt)
        recent = np.append(recent[1:], nxt)
    return out

################################################################################
#                PORTFOLIO OPTIMIZATION (SHARPE / SORTINO)
################################################################################

def calculate_portfolio_metrics(weights, returns, nifty_returns, rf=0.01):
    annual_ret = np.dot(weights, returns)*252
    cov_annual = np.cov(returns.T)*252
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    negs = np.where(returns<0, returns, 0)
    std_down = np.sqrt(np.dot(weights.T, np.dot(np.cov(negs.T)*252, weights)))
    var_market = np.var(nifty_returns)
    var_port = np.var(np.dot(returns, weights))
    beta = var_port/var_market if var_market>0 else 1
    return annual_ret, std_dev, std_down, beta

# ──────────── Risk-adjusted ratios ────────────
def negative_sharpe(w, r, rf=0.01):
    """Negative Sharpe ratio (annualised)."""
    ex_ret = np.dot(w, r) * 252
    vol    = np.sqrt(np.dot(w.T, np.dot(np.cov(r.T) * 252, w)))
    return -(ex_ret - rf) / vol if vol else 0.0


def negative_sortino(w, r, rf=0.01):
    """
    Proper negative Sortino ratio (annualised).

    • r  : vector of *asset* daily returns (same shape as w)
           – we use the clipped downside part to get downside deviation
    • rf : annual risk-free rate or MAR
    """
    # annualised excess return
    ex_ret = np.dot(w, r) * 252

    # downside deviation: keep only returns < rf/252
    downside = np.where(r < rf / 252, r - rf / 252, 0.0)
    dd = np.sqrt(np.dot(w.T, np.dot(np.cov(downside.T) * 252, w)))

    return -(ex_ret - rf) / dd if dd else 0.0


# ─────────────────── Optimise portfolio weights ───────────────────
def optimize_portfolio(stocks, metric="sortino", rf=0.01):
    """
    Mean-variance style optimiser:
        maximise (w·μ  – rf) / σ
    where σ uses a full covariance matrix from historical data
    and μ comes from your per-ticker LSTM models.
    """
    if not stocks:
        raise ValueError("stocks list is empty")

    tickers = [s["ticker"]      for s in stocks]
    qty     = np.array([s["quantity"] for s in stocks], dtype=float)

    # ----- 1) expected returns (μ) from the LSTM models ------------------
    μ = []
    for t in tickers:
        try:
            df_close = yf.download(t, period="1mo")["Close"]
            wnd      = min(30, len(df_close))
            x        = df_close[-wnd:].values.astype("float32").reshape(1, wnd, 1)
            μ.append(float(get_lstm_model_for_ticker(t).predict(x, verbose=0)))
        except Exception as e:            # fallback: last month's CAGR
            print(f"[WARN] LSTM for {t}: {e}")
            if len(df_close) > 1:
                μ.append( (df_close.iloc[-1] / df_close.iloc[0]) - 1 )
            else:
                μ.append(0.0)
    μ = np.asarray(μ)                     # shape (n,)

    # ----- 2) risk matrix Σ from 1-Y daily returns -----------------------
    hist = yf.download(tickers, period="1y")["Close"].pct_change().dropna()
    if hist.empty:
        raise RuntimeError("Couldn't fetch historical data for risk estimate")
    R = hist.to_numpy().T                # shape (n_assets, n_days)
    Σ = np.cov(R) * 252                  # annualise

    # ----- 3) current weights -------------------------------------------
    info   = get_stock_data(tickers)
    prices = np.array([info[t]["latest_price"] for t in tickers], dtype=float)
    w_cur  = (prices * qty) / (prices * qty).sum()

    # ----- 4) objective --------------------------------------------------
    def sharpe(w):
        port_ret = w @ μ * 252
        port_vol = np.sqrt(w @ Σ @ w)
        return -(port_ret - rf) / port_vol if port_vol else 0.0

    def sortino(w):
        downside = np.where(R < rf/252, R - rf/252, 0.0)
        Σ_down   = np.cov(downside) * 252
        dd       = np.sqrt(w @ Σ_down @ w)
        port_ret = w @ μ * 252
        return -(port_ret - rf) / dd if dd else 0.0

    base_obj = sharpe if metric == "sharpe" else sortino
    α        = 50.0                      # penalty for straying from current weights

    def objective(w):
        return base_obj(w) + α * np.sum((w - w_cur)**2)

    bounds = [(0, 1)] * len(tickers)
    cons   = {"type": "eq", "fun": lambda w: w.sum() - 1}
    res    = minimize(objective, w_cur, bounds=bounds, constraints=cons,
                      method="SLSQP")

    w_opt = res.x / res.x.sum()          # renormalise exactly
    return w_opt


################################################################################
#                         RISK ANALYSIS & DIVERSIFICATION
################################################################################

def risk_analysis(stocks):
    if not stocks:
        return "", []
    tlist = [s['ticker'] for s in stocks]
    df = yf.download(tlist, period='6mo')['Close']
    if df.empty:
        return "", []
    ret = df.pct_change().dropna()
    stats = []
    for t in tlist:
        avg_r = ret[t].mean()*252
        vol = ret[t].std()*np.sqrt(252)
        stats.append([avg_r, vol])

    stats = np.array(stats)
    if len(stats)<2:
        clusters = [0]*len(stats)
    else:
        km = KMeans(n_clusters=2, random_state=42)
        clusters = km.fit_predict(stats)
    dres = pd.DataFrame({
        "Ticker": tlist,
        "AvgReturn": stats[:,0],
        "Volatility": stats[:,1],
        "RiskCluster": clusters
    })
    fig = px.bar(dres, x="Ticker", y="Volatility", color="RiskCluster",
                 title="Risk Analysis: Volatility Clusters")
    return fig.to_html(full_html=False), dres.to_dict(orient='records')

def recommend_diversified_stocks(watchlist, top_n=5):
    df = yf.download(watchlist, period="6mo")["Close"]
    if df.empty:
        return []
    results = []
    for t in watchlist:
        try:
            r = df[t].pct_change().dropna()
            if len(r)<2:
                continue
            vol = r.std()*np.sqrt(252)
            future = forecast_future_prices(t, days=15)
            avg_f = np.mean(future)
            score = avg_f/vol if vol>0 else avg_f
            results.append({
                "Ticker": t,
                "Volatility": vol,
                "AvgForecast": avg_f,
                "Score": score
            })
        except:
            pass
    if not results:
        return []
    out = pd.DataFrame(results)
    out.sort_values("Score", ascending=False, inplace=True)
    return out.head(top_n).to_dict(orient='records')

################################################################################
#                     NEW: EFFICIENT FRONTIER (MPT INSPIRED)
################################################################################

def compute_efficient_frontier(tickers, num_portfolios=2000):
    """
    Example Markowitz approach for the tickers in the session portfolio.
    We sample random weight vectors, compute returns & risk, and store them.
    Then we plot the frontier in a Plotly chart.
    """
    df = yf.download(tickers, period='1y')['Close']
    if df.empty or df.shape[0]<2:
        return "", "No data to compute MPT."
    returns_df = df.pct_change().dropna()
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252

    results = []
    for _ in range(num_portfolios):
        w = np.random.random(len(tickers))
        w /= np.sum(w)
        # portfolio return
        port_ret = np.dot(w, mean_returns)
        # portfolio risk
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        # Sharpe (assume RF=1%)
        sharpe = (port_ret - 0.01)/port_vol if port_vol>0 else 0
        results.append({
            "Return": port_ret,
            "Volatility": port_vol,
            "Sharpe": sharpe
        })
    df_res = pd.DataFrame(results)
    # find max Sharpe
    max_sharpe = df_res.iloc[df_res['Sharpe'].idxmax()]

    fig = px.scatter(df_res, x="Volatility", y="Return", color="Sharpe",
                     title="Efficient Frontier (Random Portfolios)")
    fig.add_scatter(x=[max_sharpe['Volatility']], y=[max_sharpe['Return']],
                    mode="markers", marker=dict(color="red", size=10),
                    name="Max Sharpe")
    return fig.to_html(full_html=False), ""

################################################################################
#                           FLASK ROUTES
################################################################################

@app.route('/', methods=['GET','POST'])
def home():
    if request.method=='POST':
        user_type = request.form.get('user_type','')
        if user_type=='beginner':
            return redirect(url_for('beginner_tutorial'))
        elif user_type=='intermediate':
            return redirect(url_for('form'))
    return render_template('index.html')

@app.route('/beginner')
def beginner_tutorial():
    return render_template('beginner.html')

@app.route('/form', methods=['GET','POST'])
def form():
    """Displays a form for adding stocks to DB. Actual DB insertion is in /submit"""
    return render_template('intermediate.html', stocks=Stock.query.all())

@app.route('/delete_stock/<int:stock_id>')
def delete_stock(stock_id):
    st = Stock.query.get(stock_id)
    if st:
        db.session.delete(st)
        db.session.commit()
    return redirect(url_for('form'))

import os, pandas as pd

# ──────────── Saved-CSV optimiser ────────────
import os, pandas as pd
@app.route("/opt_saved/<filename>")
def opt_saved(filename):
    # -- sanitize & load the CSV exactly as before --------------------------
    from werkzeug.utils import secure_filename
    filename  = secure_filename(filename)
    csv_path  = os.path.join(app.root_path, "portfolios", filename)

    if not os.path.isfile(csv_path):
        flash("CSV not found.");  return redirect(url_for("view_portfolio"))

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        flash("Failed to read CSV.");  return redirect(url_for("view_portfolio"))

    df.columns = [c.lower() for c in df.columns]
    if not {"ticker", "quantity"}.issubset(df.columns):
        flash("CSV needs 'ticker' and 'quantity' columns.")
        return redirect(url_for("view_portfolio"))

    try:
        df["quantity"] = df["quantity"].astype(int)
    except ValueError:
        flash("'quantity' must be integers.");  return redirect(url_for("view_portfolio"))

    # -- load into session --------------------------------------------------
    session.clear()                                 # drop any previous portfolio
    session["stocks"] = df[["ticker", "quantity"]].to_dict("records")
    flash(f"Loaded {filename}.")

    # -- run the optimiser *now* -------------------------------------------
    #    (call the same function Flask uses for /opt)
    return opt()          # ⬅ renders results right away



# ────────────────────────────────────────────────────────────────
#  Submit portfolio (add & optional optimise)
# ────────────────────────────────────────────────────────────────
@app.route("/submit", methods=["POST"])
def submit():
    # 1. grab form fields
    person   = request.form.get("person_name")          # one name
    tickers  = request.form.getlist("stock_name")       # list of tickers
    qtys_raw = request.form.getlist("quantity")         # strings → cast below
    qtys_int = list(map(int, qtys_raw))                 # ensure numeric

    # 2. write to DB
    for t, q in zip(tickers, qtys_int):
        db.session.add(Stock(person_name=person,
                             ticker=t,
                             quantity=q))
    db.session.commit()

    # 3. save CSV
    import os, pandas as pd, datetime as dt

    csv_dir = os.path.join(app.root_path, "portfolios")
    os.makedirs(csv_dir, exist_ok=True)

    df = pd.DataFrame({"Ticker": tickers, "Quantity": qtys_int})
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename  = f"portfolio_{person}_{timestamp}.csv".replace(" ", "_")
    df.to_csv(os.path.join(csv_dir, filename), index=False)

    # 4. update session (lower-case keys)
    session["stocks"] = df.rename(columns=str.lower).to_dict("records")

    # 5. redirect once
    if "optimize_button" in request.form:
        return redirect(url_for("opt"))
    return redirect(url_for("portfolio"))

@app.route("/portfolio")
def portfolio():
    # ── 1. pull session ───────────────────────────────────────
    st = session.get("stocks", [])
    if not st:
        flash("No stocks found in session. Please add some.")
        return redirect(url_for("form"))

    # ── 2. latest market data ─────────────────────────────────
    tickers = [x["ticker"] for x in st]
    info    = get_stock_data(tickers)          # helper you already have

    # ── 3. build table rows & total value ─────────────────────
    total_val  = 0.0
    table_rows = []

    for x in st:
        t = x["ticker"]
        q = int(x["quantity"])                 # ← ensure numeric
        p = info[t]["latest_price"]            # numpy.float64 or float

        val = p * q
        total_val += val

        table_rows.append({
            "name":         t,
            "quantity":     q,
            "price":        f"₹{p:.2f}",
            "total_value":  f"₹{val:.2f}",
            "sector":       info[t].get("sector", "N/A"),
            "high_52w":     f"₹{info[t].get('high_52w', 'N/A')}",
            "low_52w":      f"₹{info[t].get('low_52w', 'N/A')}",
            "price_change": info[t].get("price_change", 0),
        })

    # ── 4. performance chart ─────────────────────────────────
    chart_html = calculate_portfolio_performance(st)

    # ── 5. render ────────────────────────────────────────────
    return render_template(
        "index1.html",
        stocks=table_rows,
        chart=chart_html,
        sum=f"₹{total_val:,.2f}"
    )

# ────────────────────────────────────────────────────────────────
#  Optimise portfolio (Sharpe & Sortino) + forecast
@app.route("/opt", methods=["POST"])
def opt():
    # 0. Grab portfolio from session
    st = session.get("stocks", [])
    if not st:
        flash("No stocks to optimize.")
        return redirect(url_for("form"))

    # 1. Prepare tickers, prices, quantities
    tickers = [x["ticker"] for x in st]
    info    = get_stock_data(tickers)

    # Latest prices as floats
    prices = np.array([float(info[t]["latest_price"]) for t in tickers])

    # Quantities as ints
    qty = np.array([int(x["quantity"]) for x in st], dtype=float)

    # 2. Compute current portfolio value
    curr_val = float((prices * qty).sum())
    if curr_val <= 0:
        flash("Portfolio value is zero—can't optimize.")
        return redirect(url_for("portfolio"))

    # 4. Run optimisations
    w_sharpe  = optimize_portfolio(st, metric="sharpe")
    w_sortino = optimize_portfolio(st, metric="sortino")

    # 5. Convert weights → integer share counts
    shr_sharpe  = np.floor((w_sharpe  * curr_val) / prices).astype(int)
    shr_sortino = np.floor((w_sortino * curr_val) / prices).astype(int)

    # 6. Compute new portfolio values
    val_sharpe  = float((prices * shr_sharpe).sum())
    val_sortino = float((prices * shr_sortino).sum())

    # 7. Build pie charts
    current_weights = (prices * qty) / curr_val
    pie_current  = create_pie_chart(tickers, current_weights,  "Current Portfolio")
    pie_sharpe   = create_pie_chart(tickers, w_sharpe,          "Optimised (Sharpe)")
    pie_sortino  = create_pie_chart(tickers, w_sortino,         "Optimised (Sortino)")

    # 8. Risk analysis
    risk_html, _ = risk_analysis(st)

    # 9. 15-day forecasts
    days = 15
    f_cur, f_sh, f_so = [], [], []
    for i, t in enumerate(tickers):
        try:
            base_fc = np.array(forecast_future_prices(t, days))
        except:
            base_fc = np.zeros(days)
        f_cur.append(base_fc * qty[i])
        f_sh.append(base_fc * shr_sharpe[i])
        f_so.append(base_fc * shr_sortino[i])

    f_cur = np.sum(f_cur, axis=0)
    f_sh  = np.sum(f_sh,  axis=0)
    f_so  = np.sum(f_so,  axis=0)

    all_vals = np.concatenate([f_cur, f_sh, f_so])
    min_val, max_val = all_vals.min(), all_vals.max()

    dt_rng = [datetime.today() + timedelta(days=i+1) for i in range(days)]
    figf   = go.Figure()
    figf.add_trace(go.Scatter(x=dt_rng, y=f_cur, mode="lines+markers", name="Current"))
    figf.add_trace(go.Scatter(x=dt_rng, y=f_sh,  mode="lines+markers", name="Sharpe-Opt"))
    figf.add_trace(go.Scatter(x=dt_rng, y=f_so,  mode="lines+markers", name="Sortino-Opt"))
    figf.update_layout(
        title="15-Day LSTM Forecast",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (₹)",
        yaxis=dict(range=[min_val * 0.95, max_val * 1.05])
    )
    forecast_graph = figf.to_html(full_html=False)

    # 10. Top 3 performers today
    top_returns = []
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="1d")
            if not hist.empty:
                o, c = hist["Open"].iloc[-1], hist["Close"].iloc[-1]
                top_returns.append((t, ((c - o) / o) * 100))
        except:
            continue
    top_returns = sorted(top_returns, key=lambda x: x[1], reverse=True)[:3]

    # 11. Pack table data
    stock_data = list(zip(tickers, qty, shr_sharpe, shr_sortino, prices))

    # 12. Render results
    return render_template(
        "results.html",
        current_value=curr_val,
        optimized_value_sharpe=val_sharpe,
        optimized_value_sortino=val_sortino,
        current_pie=pie_current,
        sharpe_pie=pie_sharpe,
        sortino_pie=pie_sortino,
        risk_chart=risk_html,
        forecast_graph=forecast_graph,
        top_returns=top_returns,
        stock_data=stock_data,
        llm_insights=""
    )



@app.route('/chat', methods=['GET'])
def chat():
    return render_template('chat.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    msg = request.form.get('user_message','')
    st = session.get('stocks', [])
    reply = ""

    if "portfolio" in msg.lower():
        if st:
            chg = get_price_change(st)
            lines = [f"{s['ticker']} x {s['quantity']}, monthly change: {chg[s['ticker']]}" for s in st]
            reply = "Portfolio Overview: " + ", ".join(lines)
        else:
            reply = "No portfolio found. Please add stocks."
    elif "stock sentiment" in msg.lower():
        if st:
            sentiments = []
            for s in st:
                t = s['ticker']
                df = fetch_stock_data(t, '1mo')
                if df.empty:
                    sentiments.append(f"{t}: No Data")
                else:
                    df = calculate_macd(df)
                    snt = analyze_macd_sentiment(df)
                    sentiments.append(f"{t}: {snt}")
            reply = "Stock Sentiments: " + ", ".join(sentiments)
        else:
            reply = "No portfolio found."
    elif "current returns" in msg.lower():
        if st:
            ret_list = []
            for s in st:
                t = s['ticker']
                dat = fetch_stock_data(t, '1mo')
                if len(dat) > 1:
                    pc = ((dat['Close'].iloc[-1] - dat['Close'].iloc[0]) / dat['Close'].iloc[0]) * 100
                    ret_list.append(f"{t}: {pc:.2f}%")
                else:
                    ret_list.append(f"{t}: no data")
            reply = "Current returns: " + ", ".join(ret_list)
        else:
            reply = "No portfolio found."
    elif "diversify" in msg.lower():
        reply = "Check out /diversify for recommendations!"
    else:
        reply = ("I can assist with portfolio overview, stock sentiment, "
                 "current returns, diversification, etc. Please try one of those commands.")

    return jsonify({"response": reply})

@app.route('/diversify')
def diversify():
    broad_watchlist = [
        "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
        "HINDUNILVR.NS","KOTAKBANK.NS","LT.NS","SBIN.NS","ITC.NS",
        "BAJFINANCE.NS","AXISBANK.NS","BHARTIARTL.NS","MARUTI.NS",
        "HCLTECH.NS","WIPRO.NS","NESTLEIND.NS","TITAN.NS","ULTRACEMCO.NS"
    ]
    recs = recommend_diversified_stocks(broad_watchlist, top_n=5)
    return render_template('diversify.html', recommendations=recs)

###########################
#   NEW: MPT FRONTIER    #
###########################
@app.route('/markowitz')
def markowitz():
    """
    Demonstrates advanced MPT approach (random portfolio sampling).
    We'll consider the user's session portfolio tickers.
    """
    st = session.get('stocks', [])
    if not st:
        flash("No stocks in session, cannot compute MPT Frontier.")
        return redirect(url_for('portfolio'))

    tickers = [s['ticker'] for s in st]
    chart, msg = compute_efficient_frontier(tickers)
    if msg:
        flash(msg)
        return redirect(url_for('portfolio'))

    return render_template('mpt.html', frontier_chart=chart)

@app.route('/alerts_form', methods=['GET','POST'])
def alerts_form():
    if request.method=='POST':
        tk = request.form.get('ticker')
        up = request.form.get('threshold_up')
        down = request.form.get('threshold_down')
        if not tk:
            flash("Please provide a ticker.")
            return redirect(url_for('alerts_form'))
        upv = float(up) if up else None
        dnv = float(down) if down else None
        alert = StockAlert(ticker=tk, threshold_up=upv, threshold_down=dnv)
        db.session.add(alert)
        db.session.commit()
        flash(f"Alert set for {tk}")
        return redirect(url_for('alerts_form'))
    all_alerts = StockAlert.query.all()
    return render_template('alerts.html', alerts=all_alerts)

@app.route('/delete_alert/<int:alert_id>')
def delete_alert(alert_id):
    al = StockAlert.query.get(alert_id)
    if al:
        db.session.delete(al)
        db.session.commit()
        flash("Alert deleted.")
    return redirect(url_for('alerts_form'))

@app.route('/my_alerts')
def my_alerts():
    triggered = StockAlert.query.filter_by(alert_triggered=True).all()
    return render_template('my_alerts.html', alerts=triggered)

@app.route("/view_portfolio")
def view_portfolio():
    all_stocks = Stock.query.all()

    # absolute path → works no matter where you start Flask
    csv_dir = os.path.join(app.root_path, "portfolios")
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

    return render_template(
        "view_portfolio.html",
        portfolios=all_stocks,
        saved_files=csv_files          # <-- send list to template
    )


###########################
#   APSCHEDULER LOGIC     #
###########################

scheduler = BackgroundScheduler()

def check_alerts():
    with app.app_context():
        pending = StockAlert.query.filter_by(alert_triggered=False).all()
        if not pending:
            return
        grouped = {}
        for al in pending:
            grouped.setdefault(al.ticker, []).append(al)
        for tk, al_list in grouped.items():
            info = get_stock_data([tk])
            px = info[tk]['latest_price']
            if px <= 0:
                continue
            for alr in al_list:
                trig = False
                if alr.threshold_up and px > alr.threshold_up:
                    trig = True
                if alr.threshold_down and px < alr.threshold_down:
                    trig = True
                if trig:
                    alr.alert_triggered = True
                    alr.triggered_on = datetime.now()
        db.session.commit()

scheduler.add_job(func=check_alerts, trigger='interval', seconds=60)
scheduler.start()

if __name__ == "__main__":
    app.run(debug=True, port=8000)
