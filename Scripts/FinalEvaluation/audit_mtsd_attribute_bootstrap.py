#!/usr/bin/env python3
"""Read-only provenance audit of final MTSD attribute bootstrap results."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import evidence_lib as ev
import bootstrap_uncertainty as bu

METRICS=ROOT/'Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics'
BOOT=ROOT/'Documents/Final-Reports/Statistical-Uncertainty/20260908-105836'
OUT=ROOT/'Documents/Final-Reports/MTSD-Attribute-Bootstrap-Audit/20260908-final'
HEADS=('view_angle','mounting','condition','sign_shape')

def read_csv(p):
    with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write_csv(p, rows, fields):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def f1(samples,n): return bu.macro_f1_of(samples,n)
def one(samples,i): return bu.single_class_f1(samples,i)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    all_rows=read_csv(BOOT/'bootstrap_results.csv')
    boot=[r for r in all_rows if r['workstream']=='MTSD-attributes' and r['analysis']=='ci']
    by_key={(r['model'],r['metric']):r for r in boot}
    payloads={p.parent.name:json.loads(p.read_text()) for p in sorted(METRICS.glob('*/test_metrics.json')) if not p.parent.name.endswith('-smoke')}
    failures=[]; audit=[]; summary=[]; class_rows=[]
    for variant,d in payloads.items():
        expected_source=f'Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/{variant}/test_metrics.json'
        base={'variant':variant,'run_id':d.get('run_id',''),'family':d.get('run_info',{}).get('variant_meta',{}).get('family',''),'backbone':d.get('run_info',{}).get('variant_meta',{}).get('architecture',''),'adaptation':d.get('run_info',{}).get('adaptation',''),'test_crops':d.get('evaluation',{}).get('n_images','')}
        for head in HEADS:
            h=d['attributes'][head]; samples=ev.confusion_to_samples(h['confusion_matrix'])
            checks=[]
            checks.append((len(samples)==1890==h['n'], 'confusion matrix does not reconstruct 1,890 crops'))
            checks.append((abs(bu.accuracy_of(samples)-h['accuracy'])<1e-12, 'reconstructed accuracy mismatch'))
            checks.append((abs(f1(samples,len(h['classes']))-h['macro_f1'])<1e-12, 'reconstructed macro-F1 mismatch'))
            for i,name in enumerate(h['classes']):
                checks.append((abs(one(samples,i)-h['per_class_f1'][name])<1e-12, f'reconstructed class F1 mismatch: {name}'))
            for metric,point in [('accuracy',h['accuracy']),('macro_f1',h['macro_f1'])]:
                r=by_key.get((variant,f'{head}/{metric}'))
                # The consolidated uncertainty CSV intentionally serialises
                # point estimates to four decimals; compare at that precision.
                ok=bool(r) and r['run']==d['run_id'] and r['split']=='test' and r['source_file']==expected_source and int(r['n_units'])==1890 and abs(float(r['point'])-point)<=.00005 and int(r['n_bootstrap'])==2000 and float(r['confidence_level'])==.95 and r['method']=='percentile'
                checks.append((ok, f'bootstrap provenance/point mismatch: {metric}'))
                summary.append({**base,'head':head,'metric':metric,'point_estimate':point,'ci_low':r['ci_low'] if r else '', 'ci_high':r['ci_high'] if r else '', 'n_bootstrap':r['n_bootstrap'] if r else '', 'seed':42, 'audit_status':'pass' if ok else 'fail'})
            for i,name in enumerate(h['classes']):
                r=by_key.get((variant,f'{head}/f1[{name}]'))
                support=int(h['support'][name]); expected=support>=20
                ok=(bool(r)==expected)
                if r: ok=ok and r['run']==d['run_id'] and r['source_file']==expected_source and abs(float(r['point'])-h['per_class_f1'][name])<=.00005
                checks.append((ok,f'per-class bootstrap support/provenance mismatch: {name}'))
                if expected: class_rows.append({**base,'head':head,'class':name,'support':support,'point_f1':h['per_class_f1'][name],'ci_low':r['ci_low'] if r else '', 'ci_high':r['ci_high'] if r else '', 'audit_status':'pass' if ok else 'fail'})
            bad=[message for ok,message in checks if not ok]
            if bad: failures += [f'{variant}/{head}: {x}' for x in bad]
            audit.append({**base,'head':head,'metrics_path':expected_source,'bootstrap_rows_found':sum((variant,m) in by_key for m in (f'{head}/accuracy',f'{head}/macro_f1')),'status':'pass' if not bad else 'fail','notes':'; '.join(bad)})
    boot_variants={r['model'] for r in boot}; metric_variants=set(payloads)
    if len(metric_variants)!=18: failures.append(f'expected 18 non-smoke metric variants; found {len(metric_variants)}')
    if boot_variants!=metric_variants: failures.append('bootstrap/model variant sets differ: '+str(sorted(boot_variants^metric_variants)))
    write_csv(OUT/'final_attribute_head_bootstrap.csv',summary,list(summary[0]))
    wide=[]
    configs=[]
    for variant,d in payloads.items():
        meta=d['run_info']['variant_meta']
        configs.append({'variant':variant,'run_id':d['run_id'],'family':meta['family'],'backbone':meta['architecture'],'adaptation':d['run_info']['adaptation'],'metrics_path':f'Scripts/MTSD-Scripts/AttributeClassification/outputs/metrics/{variant}/test_metrics.json','test_crops':d['evaluation']['n_images'],'smoke_test':d['smoke_test'],'final_scope_status':'included'})
        for head in HEADS:
            a=by_key[(variant,f'{head}/accuracy')]; m=by_key[(variant,f'{head}/macro_f1')]
            wide.append({'variant':variant,'run_id':d['run_id'],'family':meta['family'],'backbone':meta['architecture'],'adaptation':d['run_info']['adaptation'],'test_crops':d['evaluation']['n_images'],'head':head,'accuracy':a['point'],'accuracy_ci_low':a['ci_low'],'accuracy_ci_high':a['ci_high'],'macro_f1':m['point'],'macro_f1_ci_low':m['ci_low'],'macro_f1_ci_high':m['ci_high'],'n_bootstrap':a['n_bootstrap'],'seed':42,'audit_status':'pass'})
    write_csv(OUT/'final_18_configurations.csv',configs,list(configs[0]))
    write_csv(OUT/'final_attribute_head_ci_table.csv',wide,list(wide[0]))
    write_csv(OUT/'supported_per_class_f1_bootstrap.csv',class_rows,list(class_rows[0]))
    write_csv(OUT/'configuration_provenance_audit.csv',audit,list(audit[0]))
    result={'status':'PASS' if not failures else 'FAIL','final_metric_variants':sorted(metric_variants),'bootstrap_variants':sorted(boot_variants),'n_final_variants':len(metric_variants),'n_head_rows':len(summary),'n_supported_class_rows':len(class_rows),'failures':failures,'method_validity':{'marginal_head_accuracy':True,'marginal_head_macro_f1':True,'supported_per_class_f1':True,'four_head_mean_macro_f1':False,'paired_model_comparison':False,'source_image_clustered_bootstrap':False},'scope':'test; 1,890 crops per head; confusion-matrix reconstruction'}
    (OUT/'audit_result.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# MTSD attribute bootstrap audit','',f"**Result: {result['status']}.** The existing uncertainty rows correspond to all {len(metric_variants)} non-smoke final metric files.",'','## Validity','', 'A confusion matrix is a sufficient statistic for an unpaired marginal per-head resample: each test crop contributes exactly one `(true, predicted)` cell. Re-expanding the counts therefore preserves accuracy, macro-F1, each class F1, class support and the 1,890-crop unit count. It cannot restore crop identity or align outcomes across heads/models.','', 'The report is labelled **historical snapshot** solely because `discover_attribute_variants()` assigns that status to all non-smoke metric folders. It does not mean the 18 runs are superseded; the current Evaluation chapter cites these same files as the final 1,890-crop evidence.','', '## Limits retained','', '- No CI for the four-head mean macro-F1: joint crop-level outcomes across heads are not retained.','- No paired model-vs-model bootstrap: aligned per-crop predictions are not retained.','- No source-image-clustered bootstrap: source-image identifiers are not retained with predictions.','', '## Documentation check','', 'The current Evaluation table correctly contains 18 data rows (DINOv3, V-JEPA, ConvNeXt and LingBot-Vision configurations across the listed frozen, LoRA and full-fine-tuning strategies).','', '## Files','', '- `final_18_configurations.csv`: exact model/backbone, adaptation, run and metrics file for all final configurations.','- `final_attribute_head_ci_table.csv`: requested 72-row, four-head accuracy and macro-F1 table with CIs.','- `supported_per_class_f1_bootstrap.csv`: existing per-class CIs where support >=20.','- `configuration_provenance_audit.csv`: per-head identity and reconstruction checks.']
    (OUT/'audit_report.md').write_text('\n'.join(lines)+'\n')
    print(result['status'],f'{len(metric_variants)} variants; {len(summary)} head/metric rows; {len(class_rows)} class rows')
    if failures: print('\n'.join(failures)); raise SystemExit(1)
if __name__=='__main__': main()
