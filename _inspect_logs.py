import pathlib, re
base=pathlib.Path(r"D:\GPT\gpt\graduation_project\linear_probe-and-gradient\outputs")
patterns=re.compile(r"(\[diag\]|RANDOM INIT|random init|pretrained|not found|No pretrained|Loading|load_state|state_dict|missing keys|unexpected keys|warn|WARNING|using random|fallback|fall back)",re.I)
for d in sorted(p for p in base.iterdir() if p.is_dir()):
    f=d/'train.log'
    print('\n===',d.name,'===')
    if not f.exists():
        print('NO LOG'); continue
    lines=f.read_text(encoding='utf-8',errors='replace').splitlines()
    hits=[x for x in lines if patterns.search(x)]
    seen={}
    for x in hits:
        key=re.sub(r'^\S*\s+\S+\s+\d+\s+','',x)
        key=re.sub(r'\d+','N',key)
        seen.setdefault(key,x)
    for x in list(seen.values())[:80]: print(x)
