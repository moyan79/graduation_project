import json,pathlib
b=pathlib.Path(r"D:\GPT\gpt\graduation_project\linear_probe-and-gradient\outputs")
for d in sorted(p for p in b.iterdir() if p.is_dir()):
    f=d/'rank_report.json'
    print(d.name, end=': ')
    if f.exists():
        x=json.loads(f.read_text(encoding='utf-8'))
        print(list(x.keys()), 'dim=',x.get('dim'),'n=',x.get('n'),'feat_shape=',x.get('feat_shape'))
    else: print('NO RANK')
PY
