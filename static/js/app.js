async function predictText(){
  const text = document.getElementById('text-input').value.trim();
  const out = document.getElementById('text-output');
  out.innerHTML = "Predicting...";
  const res = await fetch('/predict_text',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
  const data = await res.json();
  renderResult(data, out);
}
function renderResult(data, el){
  if(data.error){ el.innerHTML = `<span class="badge badge-danger">Error</span> ${data.error}`; return; }
  const {label, probabilities} = data;
  let rows = Object.entries(probabilities).sort((a,b)=>b[1]-a[1]).map(([k,v])=>{
    const pct = (v*100).toFixed(1);
    return `<div><div class="row" style="justify-content:space-between"><strong>${k}</strong><span>${pct}%</span></div><div class="prog"><div style="width:${pct}%"></div></div></div>`
  }).join("");
  el.innerHTML = `<div class="badge badge-ok">Top: ${label}</div>${rows}`;
}
async function predictAudioFromUpload(){
  const fileInput = document.getElementById('audio-file');
  if(!fileInput.files.length){ alert("Choose a .wav file first"); return; }
  const form = new FormData(); form.append('file', fileInput.files[0]);
  const out = document.getElementById('audio-output'); out.innerHTML = "Predicting...";
  const res = await fetch('/predict_audio',{method:'POST',body:form});
  const data = await res.json(); renderResult(data, out);
}
async function loadSamples(){
  const res = await fetch('/api/samples'); const data = await res.json();
  const list = document.getElementById('sample-list'); list.innerHTML = "";
  data.samples.forEach(s=>{
    const div = document.createElement('div'); div.className = 'sample-item';
    div.innerHTML = `<div style="flex:1"><div style="font-weight:600">${s.label} — <span class="kbd">${s.file}</span></div><audio controls src="${s.url}"></audio></div><button class="btn btn-primary" onclick="predictSample('${s.file}')">Predict</button>`;
    list.appendChild(div);
  });
}
async function predictSample(file){
  const out = document.getElementById('audio-output'); out.innerHTML="Predicting sample...";
  const res = await fetch('/predict_audio',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sample_path:file})});
  const data = await res.json(); renderResult(data, out);
}
window.addEventListener('DOMContentLoaded', loadSamples);
