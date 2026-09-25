from __future__ import annotations

import email.utils
import json
import math
import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests

SYMBOL = os.getenv("SYMBOL", "BTCUSDT").upper()
FAPI = "https://fapi.binance.com"
EAPI = "https://eapi.binance.com"
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/selected_model.joblib"))
SCAN_SECONDS = max(10, int(os.getenv("SCAN_SECONDS", "15")))
KLINE_REFRESH_SECONDS = max(30, int(os.getenv("KLINE_REFRESH_SECONDS", "60")))
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", "5.0"))
MAX_LATE_R = float(os.getenv("MAX_LATE_R", "0.55"))
STRUCTURE_MIN = float(os.getenv("STRUCTURE_MIN", "12.0"))
ATR_MIN_PCT = float(os.getenv("ATR_MIN_PCT", "0.15"))
ATR_MAX_PCT = float(os.getenv("ATR_MAX_PCT", "4.00"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "10"))
NEWS_BLOCK_MINUTES = int(os.getenv("NEWS_BLOCK_MINUTES", "60"))
DERIV_REFRESH_SECONDS = max(30, int(os.getenv("DERIV_REFRESH_SECONDS", "60")))
OPTIONS_REFRESH_SECONDS = max(300, int(os.getenv("OPTIONS_REFRESH_SECONDS", "600")))
NEWS_REFRESH_SECONDS = max(300, int(os.getenv("NEWS_REFRESH_SECONDS", "600")))
EXTREME_FUNDING = float(os.getenv("EXTREME_FUNDING", "0.0015"))
EXTREME_LS = float(os.getenv("EXTREME_LONG_SHORT", "2.50"))
MIN_DERIV_CONFIRM = int(os.getenv("MIN_DERIV_CONFIRM", "2"))

TFS = {"1w":"1w", "1d":"1d", "4h":"4h", "1h":"1h", "30m":"30m", "15m":"15m", "5m":"5m"}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def get(base: str, path: str, params: dict | None = None):
    r = requests.get(base + path, params=params or {}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_klines(interval: str, limit: int = 400) -> pd.DataFrame:
    rows = get(FAPI, "/fapi/v1/klines", {"symbol": SYMBOL, "interval": interval, "limit": limit})
    cols = ["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_base","taker_quote","ignore"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open","high","low","close","volume","quote_volume","trades","taker_base","taker_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["timestamp","open","high","low","close","volume","quote_volume","trades","taker_base","taker_quote"]]


def fetch_weekly_compatible(limit_daily: int = 1000) -> pd.DataFrame:
    daily = fetch_klines("1d", limit_daily)
    x = daily.set_index("timestamp")
    out = x.resample("W-SUN", label="right", closed="right").agg(
        open=("open","first"),
        high=("high","max"),
        low=("low","min"),
        close=("close","last"),
        volume=("volume","sum"),
        quote_volume=("quote_volume","sum"),
        trades=("trades","sum"),
        taker_base=("taker_base","sum"),
        taker_quote=("taker_quote","sum"),
    )
    return out.dropna(subset=["open","high","low","close"]).reset_index()


def pack(df: pd.DataFrame) -> dict[str, Any]:
    if len(df) < 80:
        return {}
    last = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    c = df.close
    e20s = c.ewm(span=20, adjust=False).mean(); e50s = c.ewm(span=50, adjust=False).mean()
    tr = pd.concat([(df.high-df.low),(df.high-df.close.shift()).abs(),(df.low-df.close.shift()).abs()],axis=1).max(axis=1)
    atr_abs_s = tr.rolling(14, min_periods=14).mean(); atr_pct_s = atr_abs_s / c
    d = c.diff(); gain=d.clip(lower=0).rolling(14).mean(); loss=(-d.clip(upper=0)).rolling(14).mean()
    rsi_s=(100-100/(1+(gain/loss.replace(0,np.nan)))).fillna(50)
    vmean=df.volume.rolling(30).mean(); vstd=df.volume.rolling(30).std()
    prev_h=df.high.rolling(20).max().shift(1); prev_l=df.low.rolling(20).min().shift(1)
    close=float(last.close); vol=float(last.volume)
    taker=float(2*last.taker_base/max(vol,1e-12)-1); qdelta=float(2*last.taker_quote/max(float(last.quote_volume),1e-12)-1)
    atr_pct=float(atr_pct_s.iloc[-2]);
    vol_reg=float(atr_pct_s.rolling(60,min_periods=20).rank(pct=True).iloc[-2]) if len(df)>=20 else 0.5
    return {
        "timestamp": str(last.timestamp), "open":float(last.open),"high":float(last.high),"low":float(last.low),"close":close,
        "volume":vol,"ema20":float(e20s.iloc[-2]),"ema50":float(e50s.iloc[-2]),"ema20_gap":close/max(float(e20s.iloc[-2]),1e-12)-1,
        "ema50_gap":close/max(float(e50s.iloc[-2]),1e-12)-1,"ema_slope":float(e20s.pct_change(5).iloc[-2]),
        "ret1":float(c.pct_change().iloc[-2]),"ret4":float(c.pct_change(4).iloc[-2]),"ret12":float(c.pct_change(12).iloc[-2]),
        "volume_z":float((vol-float(vmean.iloc[-2]))/max(float(vstd.iloc[-2]),1e-12)),
        "atr_abs":float(atr_abs_s.iloc[-2]),"atr":atr_pct,"rsi":float(rsi_s.iloc[-2]),
        "range20":float((df.high.rolling(20).max().iloc[-2]-df.low.rolling(20).min().iloc[-2])/max(close,1e-12)),
        "taker_delta":taker,"quote_delta":qdelta,
        "body":float((last.close-last.open)/max(close,1e-12)),
        "upper_wick":float((last.high-max(last.open,last.close))/max(close,1e-12)),
        "lower_wick":float((min(last.open,last.close)-last.low)/max(close,1e-12)),
        "breakout_up":float(close>float(prev_h.iloc[-2])),"breakout_down":float(close<float(prev_l.iloc[-2])),"vol_regime":vol_reg,
        "recent_low_20":float(df.low.rolling(20).min().iloc[-2]),"recent_high_20":float(df.high.rolling(20).max().iloc[-2])
    }


def pivots(df: pd.DataFrame, p: int = 3):
    highs=[]; lows=[]; h=df.high.reset_index(drop=True); l=df.low.reset_index(drop=True); end=len(df)-2
    for i in range(p, end-p+1):
        if h.iloc[i]==h.iloc[i-p:i+p+1].max(): highs.append((i,float(h.iloc[i])))
        if l.iloc[i]==l.iloc[i-p:i+p+1].min(): lows.append((i,float(l.iloc[i])))
    return highs,lows


def structure_for(df: pd.DataFrame) -> dict[str,Any]:
    z=pack(df); highs,lows=pivots(df)
    r={"direction":0,"score":0.0,"bos":0,"hh":False,"hl":False,"lh":False,"ll":False,"last_swing_high":None,"last_swing_low":None,"support":None,"resistance":None}
    if len(highs)>=2:
        h1,h2=highs[-2][1],highs[-1][1]; r["hh"]=h2>h1; r["lh"]=h2<h1; r["last_swing_high"]=h2
    if len(lows)>=2:
        l1,l2=lows[-2][1],lows[-1][1]; r["hl"]=l2>l1; r["ll"]=l2<l1; r["last_swing_low"]=l2
    close=z["close"]
    if r["last_swing_high"] is not None and close>r["last_swing_high"]: r["bos"]=1
    elif r["last_swing_low"] is not None and close<r["last_swing_low"]: r["bos"]=-1
    trend=1 if r["hh"] and r["hl"] else -1 if r["lh"] and r["ll"] else 0
    ema=1 if z["ema20"]>z["ema50"] and z["ema_slope"]>0 else -1 if z["ema20"]<z["ema50"] and z["ema_slope"]<0 else 0
    rsi=1 if z["rsi"]>=52 else -1 if z["rsi"]<=48 else 0
    r["score"]=4*trend+2.5*ema+1.5*rsi+2*r["bos"]
    r["direction"]=1 if r["score"]>=4 else -1 if r["score"]<=-4 else 0
    above=[v for _,v in highs[-12:] if v>close]; below=[v for _,v in lows[-12:] if v<close]
    r["resistance"]=min(above) if above else (r["last_swing_high"] or close)
    r["support"]=max(below) if below else (r["last_swing_low"] or close)
    return r


def fetch_timeframes():
    dfs={}; packs={}
    with ThreadPoolExecutor(max_workers=len(TFS)) as ex:
        futs={}
        for k,v in TFS.items():
            futs[ex.submit(fetch_weekly_compatible,1000) if k=="1w" else ex.submit(fetch_klines,v,400)] = k
        for f in as_completed(futs):
            k=futs[f]; dfs[k]=f.result(); packs[k]=pack(dfs[k])
    structs={k:structure_for(dfs[k]) for k in ["4h","1h","30m","15m"]}
    return dfs,packs,structs


def model_row(tf: dict[str,dict], artifact: dict) -> dict[str,float]:
    weekly = bool(artifact.get("weekly", False))
    order = ["1w","1d","4h","1h","15m","5m"] if weekly else ["1d","4h","1h","15m","5m"]
    row={}
    for name in order:
        z=tf[name]
        fields={"ret1":z.get("ret1"),"ret4":z.get("ret4"),"ret12":z.get("ret12"),
                "ema20":z.get("ema20_gap"),"ema50":z.get("ema50_gap"),"ema_slope":z.get("ema_slope"),
                "atr":z.get("atr"),"rsi":z.get("rsi"),"volz":z.get("volume_z"),
                "range20":z.get("range20"),"taker_delta":z.get("taker_delta"),
                "quote_delta":z.get("quote_delta"),"body":z.get("body"),
                "upper_wick":z.get("upper_wick"),"lower_wick":z.get("lower_wick"),
                "breakout_up":z.get("breakout_up"),"breakout_down":z.get("breakout_down")}
        for k,v in fields.items():
            row[f"{name}_{k}"]=0.0 if v is None or not np.isfinite(v) else float(v)
    daily=["1d","4h","1h","15m","5m"]
    row["trend_votes"]=float(sum(np.sign(row.get(f"{x}_ema20",0)) for x in daily))
    row["momentum_votes"]=float(sum(np.sign(row.get(f"{x}_ret4",0)) for x in daily))
    row["flow_score"]=row.get("15m_taker_delta",0)+row.get("5m_taker_delta",0)
    row["vol_regime"]=float(tf["5m"].get("vol_regime",0.5) or 0.5)

    # Historical training uses the 5m candle timestamp after right-labeled resampling.
    # Live Binance klines use candle open time, so use the completed 5m candle's close timestamp.
    ts=pd.Timestamp(tf["5m"]["timestamp"])
    ts=ts + pd.Timedelta(minutes=5)
    row["hour"]=float(ts.hour)
    row["dow"]=float(ts.dayofweek)
    row["hour_sin"]=math.sin(2*math.pi*ts.hour/24)
    row["hour_cos"]=math.cos(2*math.pi*ts.hour/24)
    row["dow_sin"]=math.sin(2*math.pi*ts.weekday()/7)
    row["dow_cos"]=math.cos(2*math.pi*ts.weekday()/7)

    trend=row["trend_votes"]; mom=row["momentum_votes"]; flow=row["flow_score"]
    if artifact.get("strategy") == "trend_alignment":
        score=trend + 0.5*mom
    elif artifact.get("strategy") == "trend_flow":
        score=trend + 0.5*mom + 2.0*flow
    elif artifact.get("strategy") == "breakout_flow":
        bu=row.get("5m_breakout_up",0.0); bd=row.get("5m_breakout_down",0.0)
        score=trend + 0.5*mom + 2.0*flow + 2.0*bu - 2.0*bd
    else:
        score=trend + 0.5*mom

    row["side"]=1.0 if score>=3.0 else -1.0 if score<=-3.0 else 0.0
    row["primary_score"]=float(score)
    strategies=artifact.get("config",{}).get("strategies",("trend_alignment","trend_flow","breakout_flow"))
    row["strategy_id"]=float(list(strategies).index(artifact["strategy"]))
    row["stop_atr"]=float(artifact["barrier"]["stop_atr"])
    row["horizon_hours"]=float(artifact["barrier"]["horizon_minutes"])/60.0
    return row


def predict(artifact, tf):
    row=model_row(tf,artifact)
    x=pd.DataFrame([{c:row.get(c,0.0) for c in artifact["features"]}]).replace([np.inf,-np.inf],np.nan).fillna(0.0)
    raw=float(artifact["model"].predict_proba(x)[0,1])
    cal=artifact.get("calibrator")
    prob=float(cal.predict([raw])[0]) if cal is not None else raw
    return int(row["side"]),prob,float(row["primary_score"])

def micro():
    with ThreadPoolExecutor(max_workers=4) as ex:
        f1=ex.submit(get,FAPI,"/fapi/v1/ticker/bookTicker",{"symbol":SYMBOL}); f2=ex.submit(get,FAPI,"/fapi/v1/depth",{"symbol":SYMBOL,"limit":50}); f3=ex.submit(get,FAPI,"/fapi/v1/premiumIndex",{"symbol":SYMBOL}); f4=ex.submit(get,FAPI,"/fapi/v1/symbolAdlRisk",{"symbol":SYMBOL})
        ticker,depth,prem,adl=f1.result(),f2.result(),f3.result(),f4.result()
    bid=float(ticker.get("bidPrice",0)); ask=float(ticker.get("askPrice",0)); mid=(bid+ask)/2 if bid and ask else float(get(FAPI,"/fapi/v1/ticker/price",{"symbol":SYMBOL}).get("price",0))
    bids=[(float(x[0]),float(x[1])) for x in depth.get("bids",[])[:20]]; asks=[(float(x[0]),float(x[1])) for x in depth.get("asks",[])[:20]]
    bqty=sum(q for _,q in bids); aqty=sum(q for _,q in asks)
    return {"bid":bid,"ask":ask,"mid":mid,"ob20":bqty/max(aqty,1e-12),"spread_bps":((ask-bid)/max(mid,1e-12)*10000 if bid and ask else None),"mark":float(prem.get("markPrice",mid)),"index":float(prem.get("indexPrice",mid)),"funding":float(prem.get("lastFundingRate",0) or 0),"adl":str(adl.get("adlRisk","unknown"))}


def derivatives():
    out={"oi":None,"oi_change_pct":None,"taker_ratio":None,"long_short":None,"basis_rate":None,"errors":[]}
    def safe(label, fn):
        try:return fn()
        except Exception: out["errors"].append(label); return None

    oi=safe("open_interest",lambda:get(FAPI,"/fapi/v1/openInterest",{"symbol":SYMBOL}))
    out["oi"]=float(oi["openInterest"]) if oi else None

    hist=safe("oi_hist",lambda:get(FAPI,"/futures/data/openInterestHist",{"symbol":SYMBOL,"period":"5m","limit":6})) or []
    hist=sorted(hist,key=lambda x:int(x.get("timestamp",0)))
    if len(hist)>=2:
        old=float(hist[0]["sumOpenInterestValue"]); new=float(hist[-1]["sumOpenInterestValue"])
        out["oi_change_pct"]=((new/old)-1)*100 if old else None

    ts=safe("taker_ratio",lambda:get(FAPI,"/futures/data/takerlongshortRatio",{"symbol":SYMBOL,"period":"5m","limit":6})) or []
    ts=sorted(ts,key=lambda x:int(x.get("timestamp",0)))
    if ts:
        out["taker_ratio"]=float(ts[-1].get("buySellRatio",1))

    ls=safe("global_ls",lambda:get(FAPI,"/futures/data/globalLongShortAccountRatio",{"symbol":SYMBOL,"period":"5m","limit":6})) or []
    ls=sorted(ls,key=lambda x:int(x.get("timestamp",0)))
    if ls:
        out["long_short"]=float(ls[-1].get("longShortRatio",1))

    bs=safe("basis",lambda:get(FAPI,"/futures/data/basis",{"pair":SYMBOL,"contractType":"PERPETUAL","period":"5m","limit":6})) or []
    bs=sorted(bs,key=lambda x:int(x.get("timestamp",0)))
    if bs:
        out["basis_rate"]=float(bs[-1].get("basisRate",0))
    return out


def options_snapshot(spot:float):
    try:
        info=get(EAPI,"/eapi/v1/exchangeInfo")
        syms=info.get("optionSymbols",[])
        if not syms: return {"available":False,"reason":"no optionSymbols"}
        now=int(time.time()*1000)
        expiries=sorted({int(s.get("expiryDate",0)) for s in syms if s.get("underlying")==SYMBOL and int(s.get("expiryDate",0))>now})[:4]
        totals=[]; near=[]
        for exp in expiries:
            exp_code=datetime.fromtimestamp(exp/1000, tz=timezone.utc).strftime("%y%m%d")
            try:
                oi=get(EAPI,"/eapi/v1/openInterest",{"underlyingAsset":SYMBOL.replace("USDT",""),"expiration":exp_code})
            except Exception:
                continue
            call=put=0.0
            for r in oi if isinstance(oi,list) else []:
                sym=str(r.get("symbol","")); val=float(r.get("sumOpenInterestUsd",0) or 0)
                if sym.endswith("-C"): call+=val
                elif sym.endswith("-P"): put+=val
            totals.append({"expiry":exp_code,"call_oi_usd":call,"put_oi_usd":put,"put_call":put/max(call,1e-12)})
            candidates=[s for s in syms if s.get("underlying")==SYMBOL and int(s.get("expiryDate",0))==exp and abs(float(s.get("strikePrice",0))-spot)/max(spot,1) <= 0.03]
            candidates=sorted(candidates,key=lambda s:abs(float(s.get("strikePrice",0))-spot))[:2]
            for c in candidates:
                try:
                    m=get(EAPI,"/eapi/v1/mark",{"symbol":c["symbol"]})
                    near.append({"symbol":c["symbol"],"side":c.get("side"),"strike":float(c.get("strikePrice",0)),"markIV":float(m[0].get("markIV",0) if isinstance(m,list) else m.get("markIV",0))})
                except Exception: pass
        return {"available":True,"expiry_totals":totals,"near_atm":near}
    except Exception as e:
        return {"available":False,"reason":f"options_error:{type(e).__name__}"}


NEWS_URL_TEMPLATE="https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
HIGH_IMPACT_TERMS=("SEC","FOMC","Federal Reserve","CPI","inflation","jobs report","payroll","hack","exploit","bankruptcy","ETF approval","ETF rejection","war","sanctions","emergency","liquidation")


def news_snapshot():
    try:
        import urllib.parse
        asset = SYMBOL.replace("USDT", "")
        query = urllib.parse.quote(f"{asset} crypto {SYMBOL}")
        news_url = NEWS_URL_TEMPLATE.format(query=query)
        r=requests.get(news_url,timeout=REQUEST_TIMEOUT,headers={"User-Agent":"Mozilla/5.0 BTC-Alpha/7.0"}); r.raise_for_status(); root=ET.fromstring(r.text); now=now_utc(); items=[]
        for it in root.findall("./channel/item")[:15]:
            title=(it.findtext("title") or "").strip(); pub=(it.findtext("pubDate") or "").strip(); link=(it.findtext("link") or "").strip();
            try: dt=email.utils.parsedate_to_datetime(pub).astimezone(timezone.utc)
            except Exception: dt=None
            if dt: items.append({"title":title,"published":dt.isoformat(),"age_min":max(0,(now-dt).total_seconds()/60),"link":link})
        high=[x for x in items if x["age_min"]<=NEWS_BLOCK_MINUTES and any(t.lower() in x["title"].lower() for t in HIGH_IMPACT_TERMS)]
        return {"available":True,"items":items,"high_impact_recent":high}
    except Exception as e:
        return {"available":False,"reason":f"news_error:{type(e).__name__}","high_impact_recent":[]}


def live_derivative_gate(side:int, d:dict, tf:dict, micro_:dict):
    if side==0: return False, []
    reasons=[]; votes=0
    tr=float(d.get("taker_ratio") or 1)
    if side==1 and tr>1.05: votes+=1; reasons.append("taker flow supports BUY")
    elif side==-1 and tr<0.95: votes+=1; reasons.append("taker flow supports SELL")
    oi=float(d.get("oi_change_pct") or 0)
    price_move=float(tf["15m"].get("ret4") or 0)
    # OI/price quadrant: price and OI moving together is not sufficient, but strong opposition is a veto.
    if side==1 and price_move>0 and oi>-0.75: votes+=1; reasons.append("price/OI not contradictory")
    elif side==-1 and price_move<0 and oi>-0.75: votes+=1; reasons.append("price/OI not contradictory")
    elif side==1 and price_move<0 and oi>1.0: return False, reasons+["BUY veto: falling price with rising OI"]
    elif side==-1 and price_move>0 and oi>1.0: return False, reasons+["SELL veto: rising price with rising OI"]
    funding=float(micro_.get("funding") or 0); ls=float(d.get("long_short") or 1)
    if side==1:
        if funding>EXTREME_FUNDING and ls>EXTREME_LS: return False,reasons+["BUY veto: extreme positive funding + crowded longs"]
        if funding<=EXTREME_FUNDING: votes+=1; reasons.append("funding not excessively crowded")
    else:
        if funding<-EXTREME_FUNDING and ls<1/EXTREME_LS: return False,reasons+["SELL veto: extreme negative funding + crowded shorts"]
        if funding>=-EXTREME_FUNDING: votes+=1; reasons.append("funding not excessively crowded")
    required = (d.get("oi_change_pct") is not None and d.get("taker_ratio") is not None and d.get("long_short") is not None and d.get("basis_rate") is not None)
    if not required:
        return False, reasons + ["derivatives data incomplete"]
    return votes >= MIN_DERIV_CONFIRM, reasons


def confluence_score(tf, structs, side, micro_):
    s=0.0; reasons=[]
    for k,w in [("4h",5),("1h",4.5),("30m",3.5),("15m",3.5)]:
        dd=structs[k]["direction"]
        if dd==side: s+=w; reasons.append(f"{k} structure aligned")
        elif dd==-side: s-=w; reasons.append(f"{k} structure opposed")
    for k,w in [("15m",2.0),("5m",1.0)]:
        mom=np.sign(tf[k].get("ret4",0))+np.sign(tf[k].get("ema_slope",0)); flow=np.sign(tf[k].get("taker_delta",0))
        if mom*side>0: s+=w*0.65
        if flow*side>0: s+=w*0.35
    if (side==1 and micro_["ob20"]>=1.10) or (side==-1 and micro_["ob20"]<=0.90): s+=1.5; reasons.append("order book supports direction")
    return s,reasons


def levels(tf, structs, side, entry, stop_atr=1.2, target_r=2.0):
    z=tf["15m"]; atr=float(z["atr_abs"]); base=stop_atr*atr
    slow=float(structs["15m"].get("last_swing_low") or z["low"]); shigh=float(structs["15m"].get("last_swing_high") or z["high"])
    if side==1:
        stop=min(entry-base,slow-0.10*atr); risk=entry-stop; tp2=entry+target_r*risk; room=min(float(structs["1h"]["resistance"]),float(structs["4h"]["resistance"]))-entry
    else:
        stop=max(entry+base,shigh+0.10*atr); risk=stop-entry; tp2=entry-target_r*risk; room=entry-max(float(structs["1h"]["support"]),float(structs["4h"]["support"]))
    return stop,risk,tp2,room


def setup_key(symbol: str, side: int, candle: str) -> str:
    # A signal is tied to the completed 15m model candle. This prevents the same
    # unchanged setup from being sent repeatedly every 15 seconds, without any
    # time-based cooldown. A new model candle can generate a new signal immediately.
    return f"{symbol}|{side}|{candle}"


def tg(msg):
    token=os.environ["TELEGRAM_BOT_TOKEN"]; chat=os.environ["TELEGRAM_CHAT_ID"]
    r=requests.post(f"https://api.telegram.org/bot{token}/sendMessage",json={"chat_id":chat,"text":msg,"disable_web_page_preview":True},timeout=15); r.raise_for_status()

def fmt(x): return f"{x:.2f}"


def main():
    if not MODEL_PATH.exists(): raise SystemExit(f"Missing {MODEL_PATH}. Run the bootstrap workflow first.")
    artifact=joblib.load(MODEL_PATH)
    if not artifact.get("hard_requirements_passed"): raise SystemExit("Model artifact failed hard requirements.")
    hold=artifact.get("selection",{}).get("holdout",{}); oos=float(artifact.get("selection",{}).get("pooled_oos_win_rate") or 0); holdwr=float(hold.get("win_rate") or 0)
    if oos<0.70 or holdwr<0.70 or not hold.get("passed"): raise SystemExit("Historical >70% deployment gate not satisfied.")
    gate=artifact.get("signal_gate",{}); threshold=float(gate.get("threshold",0.80)); margin=float(gate.get("margin",0.05)); barrier=artifact.get("barrier",{"stop_atr":1.2,"target_r":2.0,"horizon_minutes":60})
    stop_atr=float(barrier.get("stop_atr",1.2)); target_r=float(barrier.get("target_r",2.0))
    print(f"[START] {SYMBOL} | historical OOS={oos:.3%} holdout={holdwr:.3%} | model p>={threshold:.2f} | MTF 4H/1H/30M/15M + derivatives/options/news | unlimited alerts | no time cooldown",flush=True)
    try:
        tg(f"📡 {SYMBOL} Alpha Ultra V7 Telegram connection successful")
        tg(f"✅ {SYMBOL} Alpha Ultra V7 Deep monitor started\nHistorical OOS: {oos:.1%} | holdout: {holdwr:.1%}\nMTF + structure + derivatives + options + news\nPrimary target: {target_r:.1f}R | unlimited alerts | no time cooldown")
    except Exception as e: print(f"[TELEGRAM] startup failed: {e}",flush=True)
    cached=None; last_candle=None; next_refresh=0; deriv_cache={}; deriv_next=0; opt_cache={}; opt_next=0; news_cache={}; news_next=0; ob_hist=[]; sent_keys=set(); alert_count=0
    while True:
        t0=time.time()
        try:
            micro_=micro()
            if time.time()>=next_refresh or cached is None:
                dfs,tf,structs=fetch_timeframes(); cached=(dfs,tf,structs); next_refresh=time.time()+KLINE_REFRESH_SECONDS
            dfs,tf,structs=cached; now=now_utc(); side,p,ps=predict(artifact,tf); candle=str(tf["15m"]["timestamp"])
            if candle!=last_candle:
                last_candle=candle; print(f"[15M] close={candle} model={'BUY' if side==1 else 'SELL' if side==-1 else 'WAIT'} p={p:.3f} score={ps:.1f}",flush=True)
            ob_hist=(ob_hist+[float(micro_["ob20"])])[-5:]
            ob_ok=(side==1 and sum(x>=1.10 for x in ob_hist)>=2) or (side==-1 and sum(x<=0.90 for x in ob_hist)>=2)
            spread_ok=micro_["spread_bps"] is not None and micro_["spread_bps"]<=MAX_SPREAD_BPS
            prob_ok=side!=0 and p>=threshold and (p-0.5)>=margin
            ht_ok=side!=0 and structs["4h"]["direction"]==side and structs["1h"]["direction"]==side
            setup_ok=side!=0 and structs["30m"]["direction"]==side and structs["15m"]["direction"] in (side,0)
            flow_ok=(side==1 and tf["15m"]["taker_delta"]>=0.05) or (side==-1 and tf["15m"]["taker_delta"]<=-0.05)
            atr_pct=float(tf["15m"]["atr"])*100; vol_ok=ATR_MIN_PCT<=atr_pct<=ATR_MAX_PCT
            conf,reasons=confluence_score(tf,structs,side,micro_) if side else (0,[])
            entry=float(micro_["ask"] if side==1 else micro_["bid"] if side==-1 else micro_["mid"]); stop,risk,tp2,room=levels(tf,structs,side,entry,stop_atr,target_r) if side else (0,0,0,0)
            risk_pct=risk/max(entry,1e-12)*100 if entry else 999; late_r=abs(entry-float(tf["15m"]["close"]))/max(risk,1e-12); room_ok=room>=target_r*risk; risk_ok=0.10<=risk_pct<=3.50; late_ok=late_r<=MAX_LATE_R
            if time.time() >= deriv_next or not deriv_cache:
                deriv_cache = derivatives(); deriv_next = time.time() + DERIV_REFRESH_SECONDS
            deriv = deriv_cache; deriv_ok,dr=live_derivative_gate(side,deriv,tf,micro_)
            if time.time() >= opt_next or not opt_cache:
                opt_cache = options_snapshot(micro_["mid"]); opt_next = time.time() + OPTIONS_REFRESH_SECONDS
            opt = opt_cache
            if time.time() >= news_next or not news_cache:
                news_cache = news_snapshot(); news_next = time.time() + NEWS_REFRESH_SECONDS
            news = news_cache; news_ok=bool(news.get("available")) and len(news.get("high_impact_recent",[]))==0
            # Options are context: a failed options request is not itself a trade veto, but a contradictory live options skew can be.
            options_ok=True; opt_context="unavailable"
            totals=opt.get("expiry_totals",[])
            if totals:
                pc=float(totals[0].get("put_call",1)); opt_context=f"near put/call OI={pc:.2f}"
                if side==1 and pc<0.35: options_ok=False
                if side==-1 and pc>3.0: options_ok=False
            all_ok=(prob_ok and ht_ok and setup_ok and flow_ok and vol_ok and conf>=STRUCTURE_MIN and spread_ok and ob_ok and room_ok and risk_ok and late_ok and deriv_ok and news_ok and options_ok)
            print(f"[SCAN] {'BUY' if side==1 else 'SELL' if side==-1 else 'WAIT'} p={p:.3f} conf={conf:.1f} 4H={structs['4h']['direction']} 1H={structs['1h']['direction']} 30M={structs['30m']['direction']} 15M={structs['15m']['direction']} ATR={atr_pct:.2f}% OB={micro_['ob20']:.2f} spread={micro_['spread_bps']} room={room/max(risk,1e-12):.2f}R OIΔ={deriv.get('oi_change_pct')} taker={deriv.get('taker_ratio')} funding={micro_.get('funding')} -> {'ALERT' if all_ok else 'WAIT'}",flush=True)
            signal_key=setup_key(SYMBOL,side,candle) if side else ""
            if all_ok and signal_key not in sent_keys:
                tp25=entry+(2.5*risk if side==1 else -2.5*risk)
                msg=(f"{'🟢' if side==1 else '🔴'} {SYMBOL} ALPHA ULTRA V7 — {'BUY' if side==1 else 'SELL'}\n\n"
                     f"Entry: {fmt(entry)}\nStop-loss: {fmt(stop)}\nTP1: {fmt(entry+(risk if side==1 else -risk))}\nTP2 (2R): {fmt(tp2)}\nTP3 (2.5R): {fmt(tp25)}\nR:R: {target_r:.2f}:1 primary\n\n"
                     f"Historical OOS win rate: {oos:.1%}\nHistorical holdout win rate: {holdwr:.1%}\nLive model probability: {p:.3f}\nMTF confluence: {conf:.1f}\n"
                     f"4H/1H/30M/15M: {structs['4h']['direction']}/{structs['1h']['direction']}/{structs['30m']['direction']}/{structs['15m']['direction']}\n"
                     f"15M ATR: {atr_pct:.2f}% | taker delta: {tf['15m']['taker_delta']:.3f}\nOrder book: {micro_['ob20']:.2f} | spread: {micro_['spread_bps']:.2f} bps\n"
                     f"OI change: {deriv.get('oi_change_pct')}% | taker buy/sell: {deriv.get('taker_ratio')} | funding: {micro_.get('funding')}\n"
                     f"Global long/short: {deriv.get('long_short')} | basis rate: {deriv.get('basis_rate')} | ADL risk: {micro_.get('adl')}\n{opt_context}\n"
                     f"Room to HTF obstacle: {room/risk:.2f}R | late move: {late_r:.2f}R\n\n"
                     f"Why passed: {'; '.join((reasons+dr)[:8])}\n\nManual execution only. Historical win rate is not a guarantee of future performance.")
                tg(msg); sent_keys.add(signal_key); alert_count += 1; print(f"[ALERT] sent {alert_count} (new 15M setup)",flush=True)
        except Exception as e: print(f"[ERROR] {type(e).__name__}: {e} -> WAIT",flush=True)
        time.sleep(max(0.1,SCAN_SECONDS-(time.time()-t0)))


if __name__=='__main__': main()
