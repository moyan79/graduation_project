import pathlib,re
b=pathlib.Path(r"D:\GPT\gpt\graduation_project\linear_probe-and-gradient\outputs")
terms=re.compile(r"(RANDOM INIT|random init|pretrained|not found|No pretrained|Downloading:|lazy import|Loading '|Loading weights|load_state_dict|initial|fallback|fall back|use random|missing keys|unexpected keys|No checkpoint|weights)",re.I)
for d in sorted(p for p in b.iterdir() if p.is_dir()):
    f=d/'train.log'; print('\n===',d.name,'===')
    if not f.exists(): print('NO LOG'); continue
    lines=f.read_text(encoding='utf-8',errors='replace').splitlines()
    out=[]
    for x in lines:
        if terms.search(x) and 'it/s' not in x and len(x)<600:
            y=re.sub(r'^.*?\[rank\d+\]:','',x)
            y=re.sub(r'^\S*\s+\S+\s+\d+\s+\S+\]','',y)
            if y not in out: out.append(y)
    for x in out[:35]: print(x)
