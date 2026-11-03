import pathlib,json
b=pathlib.Path(r"D:\GPT\gpt\graduation_project\linear_probe-and-gradient\outputs")
fs=[p for p in b.rglob('*') if p.is_file()]
from collections import Counter
c=Counter(p.suffix.lower() for p in fs)
print('dirs',sum(p.is_dir() for p in b.iterdir()),'files',len(fs),'bytes',sum(p.stat().st_size for p in fs),'GB',sum(p.stat().st_size for p in fs)/1024**3)
print(dict(c))
PY
