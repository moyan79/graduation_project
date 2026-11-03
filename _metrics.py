import json, pathlib, csv, math
base=pathlib.Path(r"D:\GPT\gpt\graduation_project\linear_probe-and-gradient\outputs")
for d in sorted(p for p in base.iterdir() if p.is_dir()):
    print('\n===',d.name,'===')
    f=d/'rank_report.json'
    if f.exists():
        r=json.loads(f.read_text(encoding='utf-8'))
        print('RANK', {k:r.get(k) for k in ['n','dim','feat_shape','cumvar_top20','cumvar_top16','effective_rank','kappa_s0_s19','kappa_s0_s15','s0','s19']})
    f=d/'gradient_report.json'
    if f.exists():
        g=json.loads(f.read_text(encoding='utf-8'))
        print('GRAD', {k:g.get(k) for k in ['steps','mean_conflict_ratio','frac_steps_majority_conflict','mean_cos_overall']})
        print('PAIR_COS', g.get('per_pair_mean_cos'))
        print('PAIR_NEG', g.get('per_pair_neg_frac'))
    f=d/'epoch_loss.csv'
    if f.exists():
        text=f.read_text(encoding='utf-8',errors='replace').strip()
        print('LOSS', text.replace('\n',' | '))
