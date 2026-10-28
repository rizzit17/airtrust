"""Download redistributable observations; fail rather than invent data."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import io
import json
import time
import zipfile
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'datasets' / 'raw'
BASE = 'https://airquality.xkdr.org'
# Public demonstration key explicitly published by the dataset owner.
HEADERS = {'Authorization': 'Bearer aqi_demo_wbf92Qx21zX-Wa_Tg8Dx1nXe'}

def get(url, **kwargs):
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=120, **kwargs)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 3: raise
            time.sleep(2 ** attempt)

def download_week(start):
    end = min(start + dt.timedelta(days=6), dt.date(2024, 12, 31))
    target = RAW / 'delhi_hourly' / f'{start}.csv'
    if target.exists(): return target
    params = dict(city='Delhi', source='cpcb_caaqm', parameter='PM2.5', start=str(start), end=str(end), agg='hourly', format='csv')
    r = get(BASE + '/v1/measurements', params=params, headers=HEADERS)
    df = pd.read_csv(io.BytesIO(r.content))
    if len(df) >= 10000: raise ValueError('Potential API truncation; reduce request window')
    if not {'station_id', 'period_start', 'mean'}.issubset(df): raise ValueError('Unexpected source schema')
    target.write_bytes(r.content)
    print(f'Downloaded {start}: {len(df)} rows', flush=True)
    return target

def main():
    (RAW / 'delhi_hourly').mkdir(parents=True, exist_ok=True)
    r = get(BASE + '/v1/stations', params={'city':'Delhi','source':'cpcb_caaqm'}, headers=HEADERS)
    obj = r.json()
    assert not obj['meta']['truncated']
    (RAW / 'delhi_stations.json').write_bytes(r.content)
    get(BASE+'/v1/meta',headers=HEADERS) # verify source availability
    (RAW/'xkdr_source_readme.md').write_bytes(get('https://raw.githubusercontent.com/xKDR/Air-Quality-Database/main/README.md').content)
    dates = [dt.date(2024,1,1)+dt.timedelta(days=i) for i in range(0,366,7)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths=list(pool.map(download_week,dates))
    df=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    df.to_csv(RAW/'delhi_pm25_hourly_2024.csv',index=False)
    target=RAW/'beijing_multi_site_uci.zip'
    if not target.exists():
        target.write_bytes(get('https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip').content)
    with zipfile.ZipFile(target) as z:
        for name in z.namelist():
            if name.endswith('.zip'):
                (RAW/'beijing_station_data.zip').write_bytes(z.read(name))
    sources=[dict(name='India Air Quality Database, XKDR Forum',provider='CPCB CAAQM',url=BASE,documentation='https://github.com/xKDR/Air-Quality-Database',license='CC BY 4.0',role='Delhi reference observations and station coordinates; basis for simulated low-cost clients',period='2024-01-01 through 2024-12-31; IST'),dict(name='Beijing Multi-Site Air Quality',provider='Song Chen; UCI Machine Learning Repository',url='https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data',doi='10.24432/C5RK5G',license='CC BY 4.0',role='Independent second-city benchmark; reference monitors, not low-cost sensors',period='2013-03-01 through 2017-02-28; local time')]
    files=[]
    for p in RAW.rglob('*'):
        if p.is_file(): files.append(dict(path=str(p.relative_to(ROOT)).replace('\\','/'),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    manifest=dict(downloaded_utc=dt.datetime.now(dt.timezone.utc).isoformat(),sources=sources,files=files,delhi_rows=len(df))
    (ROOT/'datasets'/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('DONE',len(df),'Delhi records',flush=True)

if __name__=='__main__': main()
