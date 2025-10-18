// elements
import './style.css';

const fileInput = document.getElementById('audioFile');
const uploadBtn = document.getElementById('uploadBtn');
const statusEl = document.getElementById('status');
const dialogueEl = document.getElementById('dialogue');
const pieImg = document.getElementById('pieImg');
const pieCanvas = document.getElementById('pieCanvas');
const legendEl = document.getElementById('legend');
const copyBtn = document.getElementById('copyBtn');
const dlBtn = document.getElementById('dlBtn');
const progressWrap = document.getElementById('progressWrap');
const progressBar = document.getElementById('progressBar');
const progressMsg = document.getElementById('progressMsg');

let lastMinutesText = '';
let lastDialogue = [];

function setStatus(s){ if(statusEl) statusEl.textContent = s; }

uploadBtn && uploadBtn.addEventListener('click', async () => {
  const f = fileInput && fileInput.files ? fileInput.files[0] : null;
  if(!f){ alert('Please select an audio file'); return; }

  // Smart worker base: change this if your worker is at another address
  const WORKER_BASE = 'http://127.0.0.1:9000';

  // If the page is being served from the same origin as WORKER_BASE, use relative paths.
  // Otherwise we will call the explicit worker address.
  const pageOrigin = window.location.protocol + '//' + window.location.host;
  const usingWorkerHost = (pageOrigin !== WORKER_BASE);
  const baseUrl = usingWorkerHost ? WORKER_BASE : '';

  try {
    setStatus('Uploading...');
    showProgressBar();
    updateProgressBar(2, 'Uploading file');

    const fd = new FormData();
    fd.append('file', f, f.name);

    const uploadUrl = `${baseUrl}/process`;
    console.log('Uploading to:', uploadUrl);
    const resp = await fetch(uploadUrl, { method:'POST', body: fd });

    if(!resp.ok){
      const txt = await resp.text().catch(()=>resp.statusText);
      hideProgressBar();
      setStatus('error');
      throw new Error(`Upload failed ${resp.status}: ${txt}`);
    }

    const j = await resp.json();
    const jobId = j.job_id;
    setStatus('Processing (background)...');
    updateProgressBar(6, 'Queued');

    // poll for status
    let finished = false;
    while(!finished){
      await new Promise(r=>setTimeout(r,700));
      try{
        const statusUrl = `${baseUrl}/status/${jobId}`;
        console.log('Polling status:', statusUrl);
        const sresp = await fetch(statusUrl);
        if(!sresp.ok){ console.warn('status fetch failed', sresp.status); continue; }
        const sdata = await sresp.json();
        updateProgressBar(sdata.progress || 0, sdata.message || `${sdata.status}`);
        if(sdata.status === 'done' || (sdata.progress || 0) >= 100) finished = true;
        if(sdata.status === 'error'){ hideProgressBar(); setStatus('error'); throw new Error('Processing error: ' + (sdata.error || sdata.message)); }
      }catch(err){
        console.warn('poll error', err);
      }
    }

    updateProgressBar(99, 'fetching result');
    const resultUrl = `${baseUrl}/result/${jobId}`;
    console.log('Fetching result:', resultUrl);
    const rresp = await fetch(resultUrl);
    if(!rresp.ok){ hideProgressBar(); setStatus('error'); throw new Error('Could not fetch result: ' + rresp.status); }
    const result = await rresp.json();
    hideProgressBar();
    setStatus('Processing complete');
    handleProcessResponse(result);

  } catch(err){
    console.error(err);
    alert('Error: ' + err.message);
    setStatus('error');
    hideProgressBar();
  }
});

function showProgressBar(){ if (progressWrap) progressWrap.style.display = 'block'; }
function hideProgressBar(){ if (progressWrap) progressWrap.style.display = 'none'; if (progressBar) progressBar.style.width = '0%'; if (progressMsg) progressMsg.textContent = ''; }
function updateProgressBar(pct, msg){ if (progressBar) progressBar.style.width = Math.min(100, Math.max(0, pct)) + '%'; if (progressMsg) progressMsg.textContent = msg || ''; }

function handleProcessResponse(data){
  let dialogue = null;
  try { dialogue = data.stages.analysis && data.stages.analysis.dialogue; } catch(e){}
  if(!dialogue){ try { dialogue = data.stages.asr.parsed.dialogue; } catch(e){} }
  if(!dialogue){
    const asrSegs = (data.stages && data.stages.asr && data.stages.asr.parsed && data.stages.asr.parsed.segments) || [];
    dialogue = (asrSegs || []).map(s=>({start: Number(s.start||0), end: Number(s.end||0), text: s.text || '', speaker: s.speaker || 'unknown'}));
  }
  lastDialogue = dialogue;
  renderDialogue(dialogue);

  const durations = data.stages && data.stages.analysis && data.stages.analysis.durations;
  const pie_png = data.stages && data.stages.analysis && data.stages.analysis.pie_png;
  if(pie_png && pieImg){
    // server should return a URL like "/results/<name>" or an absolute http URL
    if(typeof pie_png === 'string' && (pie_png.startsWith('/') || pie_png.startsWith('http'))) pieImg.src = pie_png;
    else pieImg.src = '/results/' + (String(pie_png).split('/').pop());
    pieImg.style.display = 'block'; if(pieCanvas) pieCanvas.style.display = 'none'; if(legendEl) legendEl.innerHTML = '';
    if(durations) renderLegend(durations);
    // hide image if it fails to load
    pieImg.onerror = () => {
      console.warn('Pie image failed to load:', pieImg.src);
      pieImg.style.display = 'none';
      if(pieCanvas) pieCanvas.style.display = 'block';
      if(durations) drawPieFromDurations(durations);
    };
  } else if(durations){
    if(pieImg) pieImg.style.display = 'none';
    if(pieCanvas) pieCanvas.style.display = 'block';
    drawPieFromDurations(durations); renderLegend(durations);
  } else {
    if(pieImg) pieImg.style.display = 'none';
    if(pieCanvas) pieCanvas.style.display = 'none';
    if(legendEl) legendEl.innerHTML = '<div class="muted">No duration data</div>';
  }

  lastMinutesText = generateMinutesText(dialogue, data);
}

/*
  renderDialogue: show speaker name only once per contiguous block of same speaker
  This preserves each ASR segment as a row but leaves the speaker cell empty for subsequent
  segments in a contiguous run by the same person.
*/
function renderDialogue(dialogue){
  if(!dialogue || !dialogueEl){ dialogueEl && (dialogueEl.innerHTML = '<div class="muted">No dialogue segments found.</div>'); return; }
  dialogueEl.innerHTML = '';

  // ensure sorted by start time (safe-guard)
  dialogue.sort((a,b)=> (Number(a.start||0) - Number(b.start||0)));

  let lastSpeaker = null;
  dialogue.forEach(seg => {
    const row = document.createElement('div');
    row.className = 'dialogue-row';

    const left = document.createElement('div');
    left.className = 'speaker';

    // Normalise speaker name (trim & title-case small strings)
    const rawSpeaker = (seg.speaker || 'unknown').toString().trim();
    const speaker = normalizeSpeakerName(rawSpeaker);

    // show speaker only when it changes from previous contiguous speaker
    left.textContent = (speaker && speaker !== lastSpeaker) ? speaker : '';

    const right = document.createElement('div');
    right.className = 'utterance';
    right.textContent = (seg.text || '').trim();

    row.appendChild(left);
    row.appendChild(right);
    dialogueEl.appendChild(row);

    if (speaker) lastSpeaker = speaker;
  });
}

/* helpful normalization for names (keeps simple titles nice) */
function normalizeSpeakerName(s){
  if(!s) return '';
  s = String(s).trim();
  // common patterns like "spk_0" => "Spk 0"
  if(/^[a-z0-9_]+$/.test(s)) {
    return s.split('_').map(w => w.length ? (w[0].toUpperCase()+w.slice(1).toLowerCase()) : w).join(' ');
  }
  // all-lowercase -> Title Case
  if(/^[a-z]+(?:[ a-z]+)*$/.test(s)) {
    return s.split(/\s+/).map(w => w.length ? (w[0].toUpperCase() + w.slice(1)) : w).join(' ');
  }
  // otherwise return trimmed
  return s;
}

function renderLegend(durations){
  if(!legendEl) return;
  legendEl.innerHTML = '';
  const total = Object.values(durations).reduce((s,v)=>s+(Number(v)||0),0) || 1;
  const entries = Object.entries(durations).sort((a,b)=>b[1]-a[1]);
  entries.forEach(([k,v], i) => {
    const pct = ((Number(v)||0)/total*100).toFixed(1);
    const div = document.createElement('div'); div.className = 'legend-row';
    div.innerHTML = `<span class="color" style="background:${pickColor(i)}"></span><strong>${k}</strong> <span class="muted">${(Number(v)||0).toFixed(1)}s • ${pct}%</span>`;
    legendEl.appendChild(div);
  });
}

function drawPieFromDurations(durations){
  if(!pieCanvas) return;
  const ctx = pieCanvas.getContext('2d');
  ctx.clearRect(0,0,pieCanvas.width,pieCanvas.height);
  const entries = Object.entries(durations).filter(([k,v])=> (Number(v) || 0) > 0);
  if(entries.length === 0){
    ctx.fillStyle = '#ccc';
    ctx.fillText('No data', 20, 120);
    return;
  }
  const total = entries.reduce((s,[,v])=>s+Number(v),0);
  let start = -0.5 * Math.PI;
  const cx = pieCanvas.width/2, cy = pieCanvas.height/2, r = Math.min(cx,cy)-8;
  entries.forEach(([k,v], idx) => {
    const angle = (Number(v)/total) * Math.PI * 2;
    const end = start + angle;
    ctx.beginPath(); ctx.moveTo(cx,cy); ctx.arc(cx,cy,r,start,end); ctx.closePath();
    ctx.fillStyle = pickColor(idx); ctx.fill();
    start = end;
  });
  ctx.beginPath(); ctx.fillStyle = '#fff'; ctx.arc(cx,cy,r*0.5,0,Math.PI*2); ctx.fill();
}

function pickColor(i){ const colors = ['#4f46e5','#06b6d4','#f97316','#10b981','#ef4444','#8b5cf6','#f59e0b','#14b8a6']; return colors[i % colors.length]; }

function generateMinutesText(dialogue, rawResp){
  let out = '1) Dialogue\n\n';
  (dialogue || []).forEach(d => {
    const s = (d.start||0).toFixed(2);
    const e = (d.end||0).toFixed(2);
    const sp = normalizeSpeakerName(d.speaker || 'unknown');
    out += `${s}–${e} — [${sp}] ${d.text}\n`;
  });
  out += '\n2) Summary\n\nAuto-generated (see dialogue above).\n\n3) Durations\n\n';
  const durations = rawResp && rawResp.stages && rawResp.stages.analysis && rawResp.stages.analysis.durations;
  if(durations){
    Object.entries(durations).forEach(([k,v]) => { out += `${k}: ${(Number(v)||0).toFixed(1)}s\n`; });
  } else { out += 'No duration data.\n'; }
  out += `\nRaw response note: ${rawResp.note || ''}\n`;
  return out;
}

copyBtn && copyBtn.addEventListener('click', ()=> {
  if(!lastMinutesText) { alert('No minutes to copy'); return; }
  navigator.clipboard.writeText(lastMinutesText).then(()=>alert('Copied minutes to clipboard')).catch(()=>alert('Copy failed'));
});

dlBtn && dlBtn.addEventListener('click', ()=> {
  if(!lastMinutesText) { alert('No minutes to download'); return; }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([lastMinutesText], {type:'text/plain'}));
  a.download = 'meeting_minutes.txt';
  document.body.appendChild(a); a.click(); a.remove();
});