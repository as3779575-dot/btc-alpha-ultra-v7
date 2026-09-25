from __future__ import annotations
import argparse, os, zipfile, urllib.request, urllib.error
from datetime import date

def months(start,end):
    y,m=start.year,start.month
    while (y,m)<=(end.year,end.month):
        yield y,m
        m+=1
        if m==13:y,m=y+1,1

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--market',choices=['futures','spot'],default='futures'); ap.add_argument('--symbol',default='BTCUSDT'); ap.add_argument('--start',default='2020-01-01'); ap.add_argument('--end',default=date.today().isoformat()); ap.add_argument('--out',default='data')
    a=ap.parse_args(); s=date.fromisoformat(a.start); e=date.fromisoformat(a.end); base='https://data.binance.vision/data/{m}/monthly/klines/{s}/1m/{s}-1m-{y:04d}-{mo:02d}.zip'
    market='futures/um' if a.market=='futures' else 'spot'
    folder=os.path.join(a.out,a.symbol); os.makedirs(folder,exist_ok=True)
    for y,mo in months(s,e):
        u=base.format(m=market,s=a.symbol,y=y,mo=mo); z=os.path.join(folder,os.path.basename(u))
        if os.path.exists(z): continue
        print('GET',u)
        try:
            urllib.request.urlretrieve(u,z)
        except urllib.error.HTTPError as ex:
            if ex.code == 404 and (y,m) == (e.year,e.month):
                print('SKIP incomplete/unpublished month:', y, m)
                continue
            raise
        try:
            with zipfile.ZipFile(z) as zz: zz.extractall(folder)
            os.remove(z)
        except zipfile.BadZipFile:
            if os.path.exists(z): os.remove(z)
            raise
if __name__=='__main__': main()
