import time, json, warnings, pathlib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA
D=str(pathlib.Path(__file__).resolve().parent.parent / "data") + "/"
REFIT_STRIDE, REFIT_BURNIN = 20, 10

def causal(series, order=(1,1,0)):   # verbatim logic of run_dataset_causal.py
    L=len(series); fitted=np.full(L,np.nan)
    if L<=REFIT_BURNIN+1: return pd.Series(series).rolling(5,min_periods=1).mean().values
    b=REFIT_BURNIN; cur=ARIMA(series[:b],order=order).fit(); fitted[:b]=cur.predict(start=0,end=b-1); last=b-1
    for t in range(b,L):
        if (t-last)>=REFIT_STRIDE:
            cur=ARIMA(series[:t+1],order=order).fit(); fitted[t]=cur.predict(start=t,end=t)[0]; last=t
        else:
            cur=cur.append([series[t]],refit=False); p=cur.predict(start=t,end=t)
            fitted[t]=p.iloc[0] if hasattr(p,"iloc") else p[0]
    return fitted
def fullseries(series, order=(1,1,0)):
    return ARIMA(series,order=order).fit().predict(start=0,end=len(series)-1)
def ma(series): return pd.Series(series).rolling(10,min_periods=1).mean().values
def ema(series):
    e=np.zeros_like(series); e[0]=series[0]
    for t in range(1,len(series)): e[t]=0.3*series[t]+0.7*e[t-1]
    return e

res={}
for fd in ["FD001","FD002"]:
    df=pd.read_parquet(D+f"train_{fd}.parquet")
    sc=[c for c in df.columns if c.startswith("s_") and df[c].astype(float).std()>1e-4]
    rng=np.random.RandomState(0); units=rng.choice(sorted(df.unit_number.unique()),size=12,replace=False)
    T={"ma":0.0,"ema":0.0,"fullseries_arima":0.0,"causal_arima":0.0}; n=0; cyc=0
    for u in units:
        g=df[df.unit_number==u].sort_values("time_cycles")
        for c in sc:
            x=g[c].astype(float).values; x=(x-x.mean())/(x.std()+1e-9); n+=1; cyc+=len(x)
            for name,f in (("ma",ma),("ema",ema),("fullseries_arima",fullseries),("causal_arima",causal)):
                t0=time.process_time(); f(x); T[name]+=time.process_time()-t0
    res[fd]={"series":n,"cycles":cyc,**{k+"_ms_per_series":1000*v/n for k,v in T.items()},
             "ratio_causal_over_ma":T["causal_arima"]/T["ma"],"ratio_causal_over_fullseries":T["causal_arima"]/T["fullseries_arima"]}
    print(fd,json.dumps(res[fd],indent=1))
json.dump(res,open("timing_result.json","w"),indent=1)
