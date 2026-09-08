#!/usr/bin/env python3
"""Exact, read-only reproduction check for retained MTSD attribute CIs."""
from __future__ import annotations
import csv, json, random, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(Path(__file__).resolve().parent))
import evidence_lib as ev

METRICS=ROOT/'Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics'
BOOT=ROOT/'Documents/Final-Reports/Statistical-Uncertainty/20260908-105836/bootstrap_results.csv'
OUT=ROOT/'Documents/Final-Reports/MTSD-Attribute-Bootstrap-Audit/20260908-final'

def pct(v,p): return round(ev.percentile(sorted(v),p),4)
def values(matrix, n_bootstrap=2000, seed=42):
    """Use the same random draw sequence that bootstrap_ci starts per metric."""
    n=sum(map(sum,matrix)); k=len(matrix); rng=random.Random(seed)
    stats={'accuracy':[],'macro_f1':[], **{f'f1[{i}]':[] for i in range(k)}}
    def score(c):
        total=sum(map(sum,c)); acc=sum(c[i][i] for i in range(k))/total
        f=[]
        for i in range(k):
            tp=c[i][i]; fp=sum(c[j][i] for j in range(k) if j!=i); fn=sum(c[i][j] for j in range(k) if j!=i)
            f.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        return acc,sum(f)/k,f
    samples=[(i,j) for i,row in enumerate(matrix) for j,count in enumerate(row) for _ in range(count)]
    point=score(matrix)
    for _ in range(n_bootstrap):
        c=[[0]*k for _ in range(k)]
        for _ in range(n):
            i,j=samples[rng.randrange(n)]; c[i][j]+=1
        a,m,f=score(c); stats['accuracy'].append(a);stats['macro_f1'].append(m)
        for i,x in enumerate(f):stats[f'f1[{i}]'].append(x)
    out={}
    point_map={'accuracy':point[0],'macro_f1':point[1],**{f'f1[{i}]':x for i,x in enumerate(point[2])}}
    for name,v in stats.items(): out[name]=(round(point_map[name],4),pct(v,.025),pct(v,.975))
    return out

def main():
    with BOOT.open(encoding='utf-8-sig',newline='') as f: rows=[r for r in csv.DictReader(f) if r['workstream']=='MTSD-attributes' and r['analysis']=='ci']
    stored={(r['model'],r['metric']):(r['point'],r['ci_low'],r['ci_high']) for r in rows}; checks=[]
    for p in sorted(METRICS.glob('*/test_metrics.json')):
        if p.parent.name.endswith('-smoke'):continue
        d=json.loads(p.read_text()); variant=p.parent.name
        for head,h in d['attributes'].items():
            got=values(h['confusion_matrix'])
            targets={'accuracy':'accuracy','macro_f1':'macro_f1'}
            for i,name in enumerate(h['classes']):
                if int(h['support'][name])>=20:targets[f'f1[{i}]']=f'f1[{name}]'
            for local,metric in targets.items():
                expected=tuple(str(x) for x in got[local]); actual=stored.get((variant,f'{head}/{metric}'))
                checks.append({'variant':variant,'head':head,'metric':metric,'point_reproduced':expected[0],'ci_low_reproduced':expected[1],'ci_high_reproduced':expected[2],'point_stored':actual[0] if actual else '', 'ci_low_stored':actual[1] if actual else '', 'ci_high_stored':actual[2] if actual else '', 'status':'pass' if actual==expected else 'fail'})
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'exact_ci_reproduction.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(checks[0]));w.writeheader();w.writerows(checks)
    failures=[x for x in checks if x['status']=='fail']
    (OUT/'exact_ci_reproduction.json').write_text(json.dumps({'n_checked':len(checks),'n_passed':len(checks)-len(failures),'n_failed':len(failures),'settings':{'n_bootstrap':2000,'seed':42,'interval':'percentile 95%','unit':'test crop'},'failures':failures},indent=2)+'\n')
    print(f'checked={len(checks)} passed={len(checks)-len(failures)} failed={len(failures)}')
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
