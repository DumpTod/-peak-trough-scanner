import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.signal import argrelextrema
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
import warnings
import time
warnings.filterwarnings('ignore')

NSE_FNO_STOCKS = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","SBIN","BHARTIARTL",
    "KOTAKBANK","ITC","LT","AXISBANK","BAJFINANCE","MARUTI","TITAN","SUNPHARMA",
    "TMCV","TMPV","NTPC","POWERGRID","HCLTECH","ASIANPAINT","ULTRACEMCO","WIPRO",
    "NESTLEIND","TECHM","ONGC","TATASTEEL","JSWSTEEL","COALINDIA","GRASIM",
    "BAJAJFINSV","ADANIENT","ADANIPORTS","DIVISLAB","DRREDDY","CIPLA","APOLLOHOSP",
    "EICHERMOT","HEROMOTOCO","BPCL","TATACONSUM","BRITANNIA","HINDALCO","INDUSINDBK",
    "SBILIFE","HDFCLIFE","BAJAJ-AUTO","DABUR","PIDILITIND","HAVELLS","SIEMENS",
    "GODREJCP","DLF","BANKBARODA","PNB","IDFCFIRSTB","FEDERALBNK","INDHOTEL",
    "TRENT","ETERNAL","JIOFIN","CANBK","RECLTD","PFC","NHPC","IOC","GAIL","BEL",
    "HAL","BHEL","IRCTC","IRFC","IDEA","SAIL","NMDC","NATIONALUM","HINDPETRO",
    "PETRONET","MUTHOOTFIN","M&MFIN","LTF","MANAPPURAM","ABCAPITAL","LICHSGFIN",
    "CANFINHOME","CHOLAFIN","SHRIRAMFIN","MOTHERSON","BOSCHLTD","MRF","PAGEIND",
    "AUROPHARMA","BIOCON","LUPIN","TORNTPHARM","ALKEM","LAURUSLABS","METROPOLIS",
    "LALPATHLAB","MAXHEALTH","FORTIS","COFORGE","MPHASIS","LTTS","PERSISTENT",
    "OFSS","NAUKRI","POLICYBZR","PAYTM","DMART","TATAPOWER","ADANIGREEN",
    "ADANIENSOL","TORNTPOWER","CUMMINSIND","ABB","CROMPTON","VOLTAS","BLUESTARCO",
    "DIXON","DEEPAKNTR","ATUL","PIIND","UPL","COROMANDEL","GNFC","CHAMBLFERT",
    "RAIN","TATACOMM","ABREL","BALRAMCHIN","ESCORTS","ASHOKLEY","EXIDEIND",
    "ARE&M","BALKRISIND","APOLLOTYRE","MGL","IGL","GUJGASLTD","CONCOR",
    "GMRAIRPORT","IRCON","RVNL","NBCC","OBEROIRLTY","GODREJPROP","PRESTIGE",
    "BRIGADE","PHOENIXLTD","SUNTV","PVRINOX","NETWORK18","ZEEL",
    "UBL","COLPAL","MARICO","EMAMILTD","BATAINDIA","RELAXO","WHIRLPOOL",
    "RAJESHEXPO","TATAELXSI","KPITTECH","SONACOMS","CLEAN","SJVN","SUZLON",
    "TATACHEM","VEDL","BANDHANBNK","AUBANK","RBLBANK","CUB","DALBHARAT",
    "RAMCOCEM","JKCEMENT","AMBUJACEM","ACC","SHREECEM","ASTRAL","SUPREMEIND",
    "POLYCAB","KEI","LINDEINDIA","SYNGENE","AFFLE","ROUTE","HAPPSTMNDS",
    "NYKAA","CARTRADE","STARHEALTH","SBICARD","ICICIPRULI","ICICIGI","HDFCAMC",
    "NAVINFLUOR","FLUOROCHEM","GRINDWELL","CAMS","CDSL","BSE","MCX","ANGELONE",
    "MOTILALOFS","MFSL"
]

NSE_NON_FNO_STOCKS = [
    "YESBANK","TRIDENT","SOUTHBANK","CENTRALBK","MAHABANK","UNIONBANK","IOB",
    "UCOBANK","PSB","IBREALEST","RPOWER","JPPOWER","JPASSOCIAT","SUZLON",
    "GTLINFRA","INTELLECT","TANLA","AGARIND","ORIENTELEC","VTL","GPPL","MAHLIFE",
    "KRBL","EIHOTEL","LMW","GREAVESCOT","HEIDELBERG","ORIENTCEM","SAGCEM",
    "NILKAMAL","TDPOWERSYS","TIINDIA","GALAXYSURF","FINEORG","VINDHYATEL",
    "DBCORP","JAGRAN","NAVNETEDUL","EPL","JMFINANCIL","EDELWEISS","KFINTECH",
    "CMSINFO","NIACL","GICRE","SOLARINDS","THERMAX","ELGIEQUIP","TIMKEN",
    "SCHAEFFLER","SKFINDIA","ENDURANCE","FIVESTAR","CREDITACC","MASFIN","EQUITASBNK",
    "UJJIVANSFB","ESAFSFB","KAJARIACER","CENTURYPLY","GREENPLY","GREENPANEL",
    "APLAPOLLO","RATNAMANI","JKLAKSHMI","KPRMILL","GARFIBRES","WELSPUNLIV",
    "RAYMOND","ARVIND","PGHL","GLAXO","PFIZER","ABBOTINDIA","SANOFI","AJANTPHARM",
    "IPCALAB","GLENMARK","GRANULES","NATCOPHARM","SUDARSCHEM","AAVAS","HOMEFIRST",
    "CANFINHOME","HUDCO","IREDA","IIFL","POONAWALLA","KAYNES","DATAPATTNS","LATENTVIEW",
    "MASTEK","NEWGEN","BSOFT","ZENSARTECH","CYIENT","ECLERX","CESC",
    "JSWENERGY","KPIL","RITES","ENGINERSIN","HEG","GRAPHITE",
    "WABAG","VGUARD","BASF","BAYERCROP","FDC","CAMPUS","METROBRAND","GOKEX",
    "SWANCORP","PCBL","TTKPRESTIG","SAFARI","VMART","SHOPERSTOP",
    "DEVYANI","JUBLFOOD","JYOTHYLAB","HINDWAREAP","OLECTRA","EASEMYTRIP","IXIGO",
    "RATEGAIN","MEDPLUS","RAINBOW","NH","YATHARTH","MEDANTA","MANKIND","PGHH",
    "GILLETTE","3MINDIA","HONAUT","CARBORUNIV","SUPRAJIT","SUMICHEM","BECTORFOOD",
    "HATSUN","HERITGFOOD","DODLA","PARAGMILK","MAHSEAMLES","WELCORP","JINDALSAW",
    "MIDHANI","COCHINSHIP","MAZDOCK","GRSE",
    "BEML","TITAGARH","JWL","DHANUKA",
    "BAYERCROP","RALLIS","GODREJAGRO","FACT","RCF","GSFC","SPIC",
    "CENTENKA","RBA","LTF","TMCV","TMPV","ABREL"
]
FNO_SET = set(NSE_FNO_STOCKS)

def get_universe():
    all_syms = list(set(NSE_FNO_STOCKS + NSE_NON_FNO_STOCKS))
    return [{'symbol':s,'ticker':s+'.NS','is_fno':s in FNO_SET} for s in all_syms]

def fetch_data(ticker, period='6mo', interval='1d'):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty or len(df) < 50: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df[['Open','High','Low','Close','Volume']].dropna()
    except: return None

def fetch_weekly(ticker):
    try:
        df = yf.download(ticker, period='1y', interval='1wk', progress=False, auto_adjust=True)
        if df.empty or len(df) < 20: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df[['Open','High','Low','Close','Volume']].dropna()
    except: return None

def find_peaks_troughs(df, order=5):
    return argrelextrema(df['High'].values, np.greater_equal, order=order)[0], argrelextrema(df['Low'].values, np.less_equal, order=order)[0]

def compute_zscore(df, lookback=20):
    rm = df['Close'].rolling(lookback).mean()
    rs = df['Close'].rolling(lookback).std().replace(0, np.nan)
    return (df['Close'] - rm) / rs

def classify_trend(df, peaks, troughs):
    if len(peaks) < 2 or len(troughs) < 2: return 'sideways'
    lp = df['High'].iloc[peaks[-2:]].values
    lt = df['Low'].iloc[troughs[-2:]].values
    if lp[1] > lp[0] and lt[1] > lt[0]: return 'uptrend'
    if lp[1] < lp[0] and lt[1] < lt[0]: return 'downtrend'
    return 'sideways'

def wick_ratios(df):
    rng = (df['High'] - df['Low']).replace(0, np.nan)
    bt = np.maximum(df['Open'], df['Close'])
    bb = np.minimum(df['Open'], df['Close'])
    return (df['High'] - bt) / rng, (bb - df['Low']) / rng

def detect_bull_grab(df, lb=20, th=0.4):
    if len(df) < lb + 1: return False, 0.0
    _, lwr = wick_ratios(df)
    lw = lwr.iloc[-1]
    if pd.isna(lw) or lw < th: return False, float(lw) if not pd.isna(lw) else 0.0
    rl = df['Low'].iloc[-lb-1:-1].min()
    if df['Low'].iloc[-1] <= rl and df['Close'].iloc[-1] > rl: return True, float(lw)
    return False, float(lw)

def detect_bear_grab(df, lb=20, th=0.4):
    if len(df) < lb + 1: return False, 0.0
    uwr, _ = wick_ratios(df)
    uw = uwr.iloc[-1]
    if pd.isna(uw) or uw < th: return False, float(uw) if not pd.isna(uw) else 0.0
    rh = df['High'].iloc[-lb-1:-1].max()
    if df['High'].iloc[-1] >= rh and df['Close'].iloc[-1] < rh: return True, float(uw)
    return False, float(uw)

def vol_spike(df, lb=20, mult=1.5):
    if len(df) < lb + 1: return False, 0.0
    avg = df['Volume'].iloc[-lb-1:-1].mean()
    if avg == 0: return False, 0.0
    ratio = df['Volume'].iloc[-1] / avg
    return ratio >= mult, float(ratio)

def compute_rsi(series, period=14):
    d = series.diff()
    g = d.where(d > 0, 0.0).rolling(period, min_periods=period).mean()
    l = (-d.where(d < 0, 0.0)).rolling(period, min_periods=period).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def exhaustion_score(df, sig_type, lb=20):
    sc = {}
    z = compute_zscore(df, lb)
    cz = z.iloc[-1]
    if sig_type == 'long':
        sc['zscore_component'] = round(min(25, max(0, abs(cz)*10)) if not pd.isna(cz) and cz < 0 else 0, 2)
    else:
        sc['zscore_component'] = round(min(25, max(0, abs(cz)*10)) if not pd.isna(cz) and cz > 0 else 0, 2)
    uwr, lwr = wick_ratios(df)
    wv = (lwr.iloc[-1] if sig_type == 'long' else uwr.iloc[-1])
    wv = wv if not pd.isna(wv) else 0
    sc['wick_component'] = round(min(25, max(0, wv * 35)), 2)
    if len(df) >= lb:
        vp = stats.percentileofscore(df['Volume'].iloc[-lb:].values, df['Volume'].iloc[-1])
        sc['volume_component'] = round(min(25, (vp/100)*25), 2)
    else:
        sc['volume_component'] = 0
    rsi = compute_rsi(df['Close'])
    cr = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    if sig_type == 'long':
        sc['momentum_component'] = round(min(25, max(0, (50-cr)*0.7)), 2)
    else:
        sc['momentum_component'] = round(min(25, max(0, (cr-50)*0.7)), 2)
    sc['total'] = round(min(100, sum(sc.values())), 2)
    sc['rsi'] = round(float(cr), 2)
    sc['zscore'] = round(float(cz), 4) if not pd.isna(cz) else 0
    return sc

def reversal_probability(exh, trend_align, lg_det, mtf_conf):
    coeff = {'int': -2.5, 'z': 0.8, 'w': 1.2, 'v': 0.6, 'm': 0.9, 's': 1.0}
    zn = exh.get('zscore_component', 0) / 25.0
    wn = exh.get('wick_component', 0) / 25.0
    vn = exh.get('volume_component', 0) / 25.0
    mn = exh.get('momentum_component', 0) / 25.0
    ss = (0.5 if lg_det else 0) + (0.3 if mtf_conf else 0) + (0.2 if trend_align else 0)
    logit = coeff['int'] + coeff['z']*zn + coeff['w']*wn + coeff['v']*vn + coeff['m']*mn + coeff['s']*ss
    return round(1.0 / (1.0 + np.exp(-np.clip(logit, -500, 500))), 4)

def mtf_check(wdf, sig_type):
    if wdf is None or len(wdf) < 20: return False, 'no_data'
    wr = compute_rsi(wdf['Close'])
    cwr = wr.iloc[-1] if not pd.isna(wr.iloc[-1]) else 50
    wma = wdf['Close'].rolling(20).mean().iloc[-1]
    above = wdf['Close'].iloc[-1] > wma
    if sig_type == 'long':
        if above or cwr < 40: return True, 'confirmed'
        if cwr < 50: return True, 'weak'
        return False, 'against'
    else:
        if not above or cwr > 60: return True, 'confirmed'
        if cwr > 50: return True, 'weak'
        return False, 'against'

def backtest_stock(df, sig_type, hold=7, z_th=1.5, w_th=0.4):
    if len(df) < 60: return None
    trades = []
    zs = compute_zscore(df, 20)
    _, lwr = wick_ratios(df)
    uwr, _ = wick_ratios(df)
    for i in range(30, len(df) - hold):
        z = zs.iloc[i]
        if pd.isna(z): continue
        if sig_type == 'long':
            lw = lwr.iloc[i] if not pd.isna(lwr.iloc[i]) else 0
            if z < -z_th and lw > w_th:
                en = float(df['Close'].iloc[i])
                ex = float(df['Close'].iloc[i+hold])
                trades.append({'pnl': ((ex-en)/en)*100})
        else:
            uw = uwr.iloc[i] if not pd.isna(uwr.iloc[i]) else 0
            if z > z_th and uw > w_th:
                en = float(df['Close'].iloc[i])
                ex = float(df['Close'].iloc[i+hold])
                trades.append({'pnl': ((en-ex)/en)*100})
    return trades if trades else None

def bt_stats(trades):
    if not trades: return {'total_trades':0,'win_rate':0,'avg_pnl':0,'avg_win':0,'avg_loss':0,'expectancy':0,'sharpe':0,'max_drawdown':0,'profit_factor':0,'ci_lower':0,'ci_upper':0,'edge_valid':False}
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    tot = len(pnls)
    wr = len(wins)/tot if tot else 0
    aw = np.mean(wins) if wins else 0
    al = abs(np.mean(losses)) if losses else 0
    exp = (wr*aw) - ((1-wr)*al)
    sh = (np.mean(pnls)/np.std(pnls))*np.sqrt(52) if np.std(pnls) > 0 else 0
    cum = np.cumsum(pnls)
    rm = np.maximum.accumulate(cum)
    md = abs(np.min(cum-rm)) if len(cum) else 0
    gp = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 1
    pf = gp/gl if gl > 0 else 0
    if len(pnls) >= 5:
        bm = [np.mean(np.random.choice(pnls, len(pnls), True)) for _ in range(1000)]
        ci_l, ci_u = np.percentile(bm, 2.5), np.percentile(bm, 97.5)
    else:
        ci_l, ci_u = -999, 999
    return {'total_trades':tot,'win_rate':round(wr*100,2),'avg_pnl':round(np.mean(pnls),4),'avg_win':round(aw,4),'avg_loss':round(al,4),'expectancy':round(exp,4),'sharpe':round(sh,4),'max_drawdown':round(md,4),'profit_factor':round(pf,4),'ci_lower':round(ci_l,4),'ci_upper':round(ci_u,4),'edge_valid':bool(ci_l > 0)}

def compute_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close'].shift(1)
    tr = pd.concat([h-l, abs(h-c), abs(l-c)], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def trade_params(df, sig_type, peaks, troughs):
    cp = float(df['Close'].iloc[-1])
    atr = compute_atr(df)
    ca = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else cp*0.02
    if sig_type == 'long':
        sl = min(float(df['Low'].iloc[troughs[-1]]) - ca*0.5, cp - ca*1.5) if len(troughs) else cp - ca*2
        risk = cp - sl
        tgt = max(float(df['High'].iloc[peaks[-1]]), cp + risk*2) if len(peaks) else cp + risk*2
    else:
        sl = max(float(df['High'].iloc[peaks[-1]]) + ca*0.5, cp + ca*1.5) if len(peaks) else cp + ca*2
        risk = sl - cp
        tgt = min(float(df['Low'].iloc[troughs[-1]]), cp - risk*2) if len(troughs) else cp - risk*2
    risk = abs(cp - sl)
    reward = abs(tgt - cp)
    return {'entry':round(cp,2),'stop_loss':round(sl,2),'target':round(tgt,2),'risk_reward':round(reward/risk,2) if risk>0 else 0,'atr':round(ca,2)}

def grade_signal(exh_total, prob, bt_st, rr, mtf_conf, lg_det):
    pts = min(30, prob*40) + min(25, exh_total*0.25) + min(20, rr*8)
    if bt_st and bt_st.get('edge_valid'): pts += 15
    elif bt_st and bt_st.get('win_rate', 0) > 55: pts += 8
    if mtf_conf: pts += 5
    if lg_det: pts += 5
    pts = round(pts, 1)
    if pts >= 80: return 'A+', pts
    if pts >= 65: return 'A', pts
    if pts >= 50: return 'B+', pts
    if pts >= 35: return 'B', pts
    return 'C', pts

def sanitize(obj):
    if isinstance(obj, dict): return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list): return [sanitize(i) for i in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, pd.Timestamp): return obj.isoformat()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)): return 0
    return obj

def scan_stock(stock):
    sym, ticker, is_fno = stock['symbol'], stock['ticker'], stock['is_fno']
    try:
        df = fetch_data(ticker)
        if df is None: return None
        cp = float(df['Close'].iloc[-1])
        if cp < 100: return None
        peaks, troughs = find_peaks_troughs(df)
        trend = classify_trend(df, peaks, troughs)
        zs = compute_zscore(df, 20)
        cz = float(zs.iloc[-1]) if not pd.isna(zs.iloc[-1]) else 0
        sig_type = None
        if cz <= -1.2: sig_type = 'long'
        elif cz >= 1.2: sig_type = 'short'
        if not sig_type: return None
        if sig_type == 'long':
            lg, wr = detect_bull_grab(df)
        else:
            lg, wr = detect_bear_grab(df)
        vs, vr = vol_spike(df)
        exh = exhaustion_score(df, sig_type)
        if exh['total'] < 30: return None
        wdf = fetch_weekly(ticker)
        mtf, mtf_st = mtf_check(wdf, sig_type)
        ta = (sig_type=='long' and trend in ['uptrend','sideways']) or (sig_type=='short' and trend in ['downtrend','sideways'])
        prob = reversal_probability(exh, ta, lg, mtf)
        if prob < 0.35: return None
        bt = backtest_stock(df, sig_type)
        bts = bt_stats(bt)
        tp = trade_params(df, sig_type, peaks, troughs)
        if tp['risk_reward'] < 1.0: return None
        gr, gs = grade_signal(exh['total'], prob, bts, tp['risk_reward'], mtf, lg)
        return sanitize({
            'symbol':sym,'is_fno':is_fno,'signal_type':sig_type,'current_price':cp,
            'grade':gr,'grade_score':gs,'trend':trend,'zscore':round(cz,4),
            'num_peaks':len(peaks),'num_troughs':len(troughs),
            'liquidity_grab':lg,'wick_ratio':round(wr,4),'volume_spike':vs,'volume_ratio':round(vr,2),
            'exhaustion_score':exh['total'],
            'exhaustion_components':{'zscore_component':exh['zscore_component'],'wick_component':exh['wick_component'],'volume_component':exh['volume_component'],'momentum_component':exh['momentum_component']},
            'rsi':exh['rsi'],'probability':prob,'confidence_pct':round(prob*100,1),
            'entry':tp['entry'],'stop_loss':tp['stop_loss'],'target':tp['target'],
            'risk_reward':tp['risk_reward'],'atr':tp['atr'],
            'mtf_confirmed':mtf,'mtf_status':mtf_st,
            'backtest':bts,'scan_time':datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except:
        return None

def run_scan(max_stocks=None):
    universe = get_universe()
    if max_stocks: universe = universe[:max_stocks]
    start = time.time()
    results = {'long':[], 'short':[]}
    for stock in universe:
        sig = scan_stock(stock)
        if sig:
            results[sig['signal_type']].append(sig)
    results['long'].sort(key=lambda x: x['grade_score'], reverse=True)
    results['short'].sort(key=lambda x: x['grade_score'], reverse=True)
    elapsed = round(time.time() - start, 2)
    st = {'total_scanned':len(universe),'signals_found':len(results['long'])+len(results['short']),'long_signals':len(results['long']),'short_signals':len(results['short']),'scan_time':elapsed,'timestamp':datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    return results, st

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

cached_results = None
scan_in_progress = False

@app.route('/')
def home():
    return jsonify({'scanner':'Peak & Trough Liquidity Scanner','version':'1.0','endpoints':{'/api/scan':'Run scan','/api/scan?max=50':'Quick scan','/api/results':'Cached results','/api/health':'Health check'}})

@app.route('/api/health')
def health():
    return jsonify({'status':'healthy','timestamp':datetime.now().isoformat()})

@app.route('/api/scan')
def api_scan():
    global cached_results, scan_in_progress
    if scan_in_progress:
        return jsonify({'status':'busy','message':'Scan in progress'}), 429
    mx = request.args.get('max', default=None, type=int)
    scan_in_progress = True
    try:
        results, stats = run_scan(mx)
        cached_results = sanitize({'status':'success','scan_stats':stats,'signals':results})
        return jsonify(cached_results)
    except Exception as e:
        return jsonify({'status':'error','message':str(e)}), 500
    finally:
        scan_in_progress = False

@app.route('/api/results')
def api_results():
    if cached_results is None:
        return jsonify({'status':'no_data','message':'No scan run yet'}), 404
    return jsonify(cached_results)

application = app
@app.route('/api/track', methods=['POST'])
def track_signals():
    """
    Accepts a list of signals with entry/SL/target prices.
    Returns current price data for each symbol to check if entry was met,
    and how far price has moved relative to entry/SL/target.
    """
    try:
        data = request.get_json()
        if not data or 'signals' not in data:
            return jsonify({'error': 'Missing signals array'}), 400

        signals = data['signals']
        results = []

        for sig in signals:
            symbol = sig.get('symbol', '')
            yahoo_ticker = sig.get('yahoo_ticker', '')
            entry = sig.get('entry', 0)
            sl = sig.get('sl', 0)
            target = sig.get('target', 0)
            direction = sig.get('direction', '').upper()
            signal_date = sig.get('signal_date', '')
            signal_id = sig.get('_id', '')
            days_pending = sig.get('days_pending', 0)

            if not yahoo_ticker:
                yahoo_ticker = symbol + '.NS'

            try:
                # Fetch last 5 trading days of data
                df = yf.download(yahoo_ticker, period='5d', interval='1d', auto_adjust=True, progress=False)

                if df is None or len(df) == 0:
                    results.append({
                        '_id': signal_id,
                        'symbol': symbol,
                        'status': 'error',
                        'message': 'No data available'
                    })
                    continue

                # Flatten MultiIndex if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

                current_close = float(df['Close'].iloc[-1])
                current_high = float(df['High'].iloc[-1])
                current_low = float(df['Low'].iloc[-1])

                # Get data since signal date to check entry trigger
                # We look at all available days (up to 5)
                highs = df['High'].tolist()
                lows = df['Low'].tolist()
                closes = df['Close'].tolist()
                dates = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in df.index.tolist()]

                # Determine how many trading days of data we have after signal
                trading_days_after = len(df)

                # Check if entry was triggered in any of the available days
                entry_triggered = False
                trigger_date = ''

                for i in range(len(df)):
                    day_low = float(lows[i])
                    day_high = float(highs[i])

                    if direction in ['LONG', 'BUY']:
                        # For buy: price must have gone down to entry or below
                        if day_low <= entry:
                            entry_triggered = True
                            trigger_date = dates[i]
                            break
                    elif direction in ['SHORT', 'SELL']:
                        # For sell: price must have gone up to entry or above
                        if day_high >= entry:
                            entry_triggered = True
                            trigger_date = dates[i]
                            break

                # Check SL and Target hit
                sl_hit = False
                target_hit = False
                sl_hit_date = ''
                target_hit_date = ''

                if entry_triggered:
                    # Check from trigger point onwards
                    trigger_idx = dates.index(trigger_date)
                    for i in range(trigger_idx, len(df)):
                        day_low = float(lows[i])
                        day_high = float(highs[i])

                        if direction in ['LONG', 'BUY']:
                            if day_low <= sl and not sl_hit:
                                sl_hit = True
                                sl_hit_date = dates[i]
                            if day_high >= target and not target_hit:
                                target_hit = True
                                target_hit_date = dates[i]
                        elif direction in ['SHORT', 'SELL']:
                            if day_high >= sl and not sl_hit:
                                sl_hit = True
                                sl_hit_date = dates[i]
                            if day_low <= target and not target_hit:
                                target_hit = True
                                target_hit_date = dates[i]

                # Calculate P&L
                pnl_price = 0
                pnl_pct = 0

                if entry_triggered and entry > 0:
                    if direction in ['LONG', 'BUY']:
                        pnl_price = round(current_close - entry, 2)
                        pnl_pct = round((current_close - entry) / entry * 100, 2)
                    elif direction in ['SHORT', 'SELL']:
                        pnl_price = round(entry - current_close, 2)
                        pnl_pct = round((entry - current_close) / entry * 100, 2)

                # Determine final status
                status = 'pending'
                outcome = 'pending'
                exit_price = ''

                if entry_triggered:
                    if target_hit and sl_hit:
                        # Both hit — check which came first
                        if dates.index(target_hit_date) <= dates.index(sl_hit_date):
                            status = 'target_hit'
                            outcome = 'target_hit'
                            exit_price = target
                        else:
                            status = 'stop_hit'
                            outcome = 'stop_hit'
                            exit_price = sl
                    elif target_hit:
                        status = 'target_hit'
                        outcome = 'target_hit'
                        exit_price = target
                    elif sl_hit:
                        status = 'stop_hit'
                        outcome = 'stop_hit'
                        exit_price = sl
                    else:
                        status = 'open'
                        outcome = 'pending'
                        exit_price = current_close
                else:
                    # Entry not triggered
                    new_days_pending = days_pending + 1
                    if new_days_pending >= 2:
                        status = 'expired'
                        outcome = 'expired'
                    else:
                        status = 'pending'
                        outcome = 'pending'

                # Calculate distances
                sl_distance = 0
                target_distance = 0
                if entry_triggered and entry > 0:
                    if direction in ['LONG', 'BUY']:
                        sl_distance = round(current_close - sl, 2)
                        target_distance = round(target - current_close, 2)
                    else:
                        sl_distance = round(sl - current_close, 2)
                        target_distance = round(current_close - target, 2)

                result = {
                    '_id': signal_id,
                    'symbol': symbol,
                    'status': status,
                    'outcome': outcome,
                    'entry_triggered': entry_triggered,
                    'trigger_date': trigger_date,
                    'current_close': current_close,
                    'current_high': current_high,
                    'current_low': current_low,
                    'pnl_price': pnl_price,
                    'pnl_pct': pnl_pct,
                    'sl_hit': sl_hit,
                    'target_hit': target_hit,
                    'sl_hit_date': sl_hit_date,
                    'target_hit_date': target_hit_date,
                    'sl_distance': sl_distance,
                    'target_distance': target_distance,
                    'exit_price': float(exit_price) if exit_price != '' else '',
                    'days_pending': days_pending + 1 if not entry_triggered else days_pending,
                    'trading_days_checked': trading_days_after,
                    'last_checked': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
                }

                results.append(sanitize(result))

            except Exception as e:
                results.append({
                    '_id': signal_id,
                    'symbol': symbol,
                    'status': 'error',
                    'message': str(e)
                })

        return jsonify({
            'status': 'success',
            'tracked': len(results),
            'results': results,
            'timestamp': pd.Timestamp.now().isoformat()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
