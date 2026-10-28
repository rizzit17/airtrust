import sys,json,hashlib,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'datasets'/'manifest.json').read_text())
count=0
for record in manifest['files']+manifest.get('processed_files',[]):
    p=ROOT/record['path']
    assert p.is_file(),f'Missing file: {p}'
    assert p.stat().st_size==record['bytes'],f'Size mismatch: {p}'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256'],f'Checksum mismatch: {p}'
    count+=1
for p in (ROOT/'submission').glob('*.zip'):
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None,f'Corrupt ZIP: {p}'
        assert all(not n.startswith('/') and '..' not in Path(n).parts for n in z.namelist())
    print('Verified archive:',p.name)
print(f'Verified {count} source and derived dataset files.')
