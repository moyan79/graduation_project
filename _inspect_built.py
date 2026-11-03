import pathlib,re
b=pathlib.Path(r"D:\GPT\gpt\graduation_project\linear_probe-and-gradient\outputs")
for d in sorted(p for p in b.iterdir() if p.is_dir()):
 f=d/'train.log'; print('\n===',d.name,'===')
 if not f.exists(): print('NO LOG'); continue
 t=f.read_text(encoding='utf-8',errors='replace').splitlines()
 for x in t:
  if '[diag] model=' in x or '[diag] built ' in x or '[diag] dumped features_epoch0' in x:
   print(x.replace('[rank0]: ','').replace('[rank1]: ','').replace('[rank2]: ','').replace('[rank3]: ',''))
