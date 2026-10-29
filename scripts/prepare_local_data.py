"""Restore the bundled public sources without requiring an API account."""
from pathlib import Path
import zipfile,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
bundle=ROOT/'datasets'/'bundled'/'source_data.zip'
index=json.loads((bundle.parent/'manifest.json').read_text())
if hashlib.sha256(bundle.read_bytes()).hexdigest()!=index['sha256']: raise ValueError('Source bundle checksum mismatch')
with zipfile.ZipFile(bundle) as archive:
    for info in archive.infolist():
        dest=(ROOT/'datasets'/info.filename).resolve()
        if not dest.is_relative_to((ROOT/'datasets'/'raw').resolve()): raise ValueError('Unexpected archive path')
    archive.extractall(ROOT/'datasets')
print('Restored verified public source data. Run start.bat or python -m uvicorn airtrust.server:app')
