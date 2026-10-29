"""Expand Delhi contextual pollutants; preserve PM2.5 target and existing benchmark."""
from download_data import RAW,BASE,HEADERS,get
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import pandas as pd
import io,json,hashlib

def fetch(task):
    parameter,start=task
    folder=RAW/'expanded'/parameter; folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{start}.csv'
    if path.exists(): return path
    end=min(start+dt.timedelta(days=6),dt.date(2024,12,31))
    response=get(BASE+'/v1/measurements',headers=HEADERS,params=dict(city='Delhi',source='cpcb_caaqm',parameter=parameter,start=str(start),end=str(end),agg='hourly',format='csv'))
    frame=pd.read_csv(io.BytesIO(response.content))
    if len(frame)>=10000 or response.headers.get('X-Truncated','').lower()=='true': raise ValueError('Truncated response')
    if not {'station_id','period_start','mean'}.issubset(frame): raise ValueError('Invalid schema')
    path.write_bytes(response.content)
    print(parameter,start,len(frame),flush=True)
    return path

def main():
    tasks=[(p,dt.date(2024,1,1)+dt.timedelta(days=i)) for p in ('PM10','NO2') for i in range(0,366,7)]
    with ThreadPoolExecutor(max_workers=3) as pool: list(pool.map(fetch,tasks))
    counts=[]
    for p in ('PM10','NO2'):
        frame=pd.concat([pd.read_csv(f) for f in sorted((RAW/'expanded'/p).glob('*.csv'))],ignore_index=True)
        assert not frame.duplicated(['station_id','period_start']).any()
        target=RAW/f'delhi_{p.lower()}_hourly_2024.csv'; frame.to_csv(target,index=False)
        counts.append(dict(parameter=p,rows=len(frame),stations=frame.station_id.nunique(),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),path=str(target.relative_to(RAW.parent.parent)).replace('\\','/')))
    (RAW.parent/'expansion.json').write_text(json.dumps(dict(source='India Air Quality Database, XKDR Forum / CPCB CAAQM',license='CC BY 4.0',url=BASE,role='Observed contextual pollutant exploration; not additional PM2.5 labels or low-cost sensors',files=counts),indent=2))

if __name__=='__main__': main()
