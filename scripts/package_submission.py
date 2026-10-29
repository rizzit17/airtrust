"""Package current artifacts without rerunning models or regenerating datasets."""
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'submission';out.mkdir(exist_ok=True)
for name,folders in [('datasets_used.zip',['datasets']),('airtrust_project.zip',['airtrust','scripts','web','tests','docs','results','.github'])]:
    with zipfile.ZipFile(out/name,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for folder in folders:
            for p in (ROOT/folder).rglob('*'):
                if not p.is_file() or '__pycache__' in p.parts:continue
                if name=='datasets_used.zip' and 'bundled' in p.parts:continue # same raw sources already included
                archive.write(p,Path('airtrust')/p.relative_to(ROOT))
        if name=='airtrust_project.zip':
            for file in ['README.md','requirements.txt','requirements-lock.txt','start.bat','reproduce.bat','pytest.ini','LICENSE','.gitignore','datasets/README.md','datasets/manifest.json','datasets/expansion.json']:
                archive.write(ROOT/file,Path('airtrust')/file)
    print(name,(out/name).stat().st_size,flush=True)
