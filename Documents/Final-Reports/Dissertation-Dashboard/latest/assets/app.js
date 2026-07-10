
function normalise(t){return (t||'').toLowerCase();}
function applyFilters(){
  const q=normalise(document.getElementById('search').value);
  const status=document.getElementById('f-status').value;
  const dataset=normalise(document.getElementById('f-dataset').value);
  const model=normalise(document.getElementById('f-model').value);
  document.querySelectorAll('[data-searchable]').forEach(el=>{
    const text=normalise(el.textContent);
    const okQ=!q||text.includes(q);
    const okS=!status||el.dataset.status===status;
    const okD=!dataset||normalise(el.dataset.dataset).includes(dataset);
    const okM=!model||normalise(el.dataset.model).includes(model);
    el.classList.toggle('hidden',!(okQ&&okS&&okD&&okM));
  });
}
window.addEventListener('DOMContentLoaded',()=>{
  ['search','f-status','f-dataset','f-model'].forEach(id=>{
    const el=document.getElementById(id);
    if(el){el.addEventListener('input',applyFilters);el.addEventListener('change',applyFilters);}
  });
});
