import json, pathlib
base=pathlib.Path(r"D:\GPT\gpt\graduation_project\linear_probe-and-gradient\outputs")
for name in ['diag_ddp_resnet','diag_ddp_mesorch','diag_ddp_ffdn']:
    print('\n===',name,'===')
    p=base/name
    for fn in ['rank_report.json','gradient_report.json']:
        f=p/fn
        print('\n--',fn,'exists',f.exists())
        if f.exists():
            d=json.loads(f.read_text(encoding='utf-8'))
            print(json.dumps(d,ensure_ascii=False,indent=2)[:12000])
    for fn in ['epoch_loss.csv','train.log','grad_log.csv']:
        f=p/fn
        print('\n--',fn,'exists',f.exists())
        if f.exists():
            txt=f.read_text(encoding='utf-8',errors='replace')
            print(txt[:5000])
