import json,pathlib
p=pathlib.Path(r"D:\GPT\gpt\graduation_project\linear_probe-and-gradient\_results.json")
for x in json.loads(p.read_text(encoding='utf-8')):
 r=x.get('rank'); r1=x.get('traj_first'); r2=x.get('traj_last')
 def met(z):
  if not z:return None
  return z.get('cumvar_top20',z.get('cumvar_top16')),z.get('effective_rank')
 print(x['model'], 'traj',x['traj_count'],met(r1),met(r2),'loss%',round(x['loss_change_pct'],2) if x['loss_change_pct'] is not None else None)
 print('  pair', (x.get('gradient') or {}).get('per_pair_mean_cos'))
PY
