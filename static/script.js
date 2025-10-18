// ---------- Complete script.js (drop-in) ----------

// Elements
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

/* ---------- Configuration ---------- */
const MAX_PREVIEW_CHARS = 240; // number of chars before "Read more"
const SPEAKER_COLORS = ['#4f46e5','#06b6d4','#f97316','#10b981','#ef4444','#8b5cf6','#f59e0b','#14b8a6'];
const speakerColorMap = new Map(); // maps speaker name -> color

/* ---------- small helpers ---------- */
function setStatus(s){ if(statusEl) statusEl.textContent = s; }
function pickColor(i){ return SPEAKER_COLORS[i % SPEAKER_COLORS.length]; }
function getSpeakerColor(name){
  if(!name) return '#111827';
  if(speakerColorMap.has(name)) return speakerColorMap.get(name);
  const idx = speakerColorMap.size; // assign next available color deterministically
  const color = pickColor(idx);
  speakerColorMap.set(name, color);
  return color;
}

/* ---------- Demo sample data (names changed) ---------- */
const DEMO_SPEAKERS = [
  { speaker: 'Aditya', start: 0, end: 85, text: "Good morning, everyone — this is Aditya. Let's quickly run through the agenda items and assign owners." },
  { speaker: 'Rohan', start: 86, end: 136.5, text: "Thanks Aditya. I can take the action on the API integration and follow up by Friday." },
  { speaker: 'Shourya', start: 137, end: 183, text: "I'm seeing a potential blocker in the deployment pipeline; I'll raise a ticket and coordinate with infra." },
  { speaker: 'Isha', start: 184, end: 223, text: "Isha here — I can help with QA and set up the test matrix. Also, quick note: timings shift if we add one more dependency." }
];
const DEMO_DURATIONS = { "Aditya":85, "Rohan":50.5, "Shourya":46, "Isha":39 };

/* ---------- Demo insertion & clearing ---------- */
function insertDemoSample() {
  if (!dialogueEl || dialogueEl.dataset.demoInserted) return;
  dialogueEl.dataset.demoInserted = '1';

  // Insert demo notice above dialogue container (so it stays visible)
  const notice = document.createElement('div');
  notice.className = 'demo-notice';
  notice.textContent = '⚠️ This is a sample output. Upload your audio to see real results.';
  // place notice before dialogueEl (dialogueEl is inside #dialogueCard .scroll)
  // If dialogueEl's parent exists, insert notice just before dialogueEl
  if (dialogueEl.parentNode) {
    dialogueEl.parentNode.insertBefore(notice, dialogueEl);
  }

  renderDialogue(DEMO_SPEAKERS);
  if (pieCanvas) {
    pieCanvas.style.display = 'block';
    pieImg && (pieImg.style.display = 'none');
    drawPieFromDurations(DEMO_DURATIONS);
  }
  if (legendEl) renderLegend(DEMO_DURATIONS);
  setStatus('ready (sample)');
}

function clearDemoSample() {
  if (dialogueEl && dialogueEl.dataset.demoInserted) {
    dialogueEl.innerHTML = '';
    delete dialogueEl.dataset.demoInserted;
    const old = document.querySelector('.demo-notice');
    if (old) old.remove();
  }
  if (legendEl) legendEl.innerHTML = '';
  if (pieCanvas) {
    const ctx = pieCanvas.getContext('2d');
    ctx && ctx.clearRect(0,0,pieCanvas.width,pieCanvas.height);
    pieCanvas.style.display = 'none';
  }
  if (pieImg) pieImg.style.display = 'none';
}

/* Clear demo when user selects a file (nice UX) */
fileInput && fileInput.addEventListener('change', () => { clearDemoSample(); });

/* ---------- Upload handler (clears demo immediately) ---------- */
uploadBtn && uploadBtn.addEventListener('click', async () => {
  clearDemoSample(); // clear demo immediately

  const f = fileInput && fileInput.files ? fileInput.files[0] : null;
  if(!f){ alert('Please select an audio file'); return; }

  // Base / worker URL logic kept from your original code
  const WORKER_BASE = 'http://127.0.0.1:9000';
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
    await handleProcessResponse(result);

  } catch(err){
    console.error(err);
    alert('Error: ' + err.message);
    setStatus('error');
    hideProgressBar();
  }
});

/* ---------- Progress helpers ---------- */
function showProgressBar(){ if (progressWrap) progressWrap.style.display = 'block'; }
function hideProgressBar(){ if (progressWrap) progressWrap.style.display = 'none'; if (progressBar) progressBar.style.width = '0%'; if (progressMsg) progressMsg.textContent = ''; }
let currentProgress = 0;
let targetProgress = 0;
let progressTimer = null;
function updateProgressBar(pct, msg){
  targetProgress = Math.min(100, Math.max(0, pct));
  if(progressMsg) progressMsg.textContent = msg || '';

  if(!progressTimer){
    progressTimer = setInterval(()=>{
      if(currentProgress < targetProgress){
        currentProgress += 1; // smooth step
        if(progressBar) progressBar.style.width = currentProgress + '%';
      }
      if(currentProgress >= 100){
        clearInterval(progressTimer);
        progressTimer = null;
      }
    }, 80);
  }
}

/* ---------- CSV parsing & response handling ---------- */
function parseCleanTranscriptCSV(csvText){
  const lines = csvText.split(/\r?\n/).filter(Boolean);
  if(lines.length === 0) return [];
  const header = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g,''));
  const idx = (name) => header.indexOf(name);
  const iSpeakerId = idx('speaker_id');
  const iSpeakerName = idx('speaker_name') !== -1 ? idx('speaker_name') : header.indexOf('speaker');
  const iStart = idx('start') !== -1 ? idx('start') : 0;
  const iEnd = idx('end') !== -1 ? idx('end') : (iStart+1);
  const iText = idx('text') !== -1 ? idx('text') : header.length-1;

  const out = [];
  for(let i=1;i<lines.length;i++){
    const line = lines[i];
    const row = line.match(/(".*?"|[^",\s]+)(?=\s*,|\s*$)/g) || [];
    const get = (j) => (row[j] || '').replace(/^"|"$/g,'').trim();
    const speakerName = get(iSpeakerName) || get(iSpeakerId) || 'unknown';
    const start = parseFloat(get(iStart)) || 0;
    const end = parseFloat(get(iEnd)) || start;
    const text = get(iText) || '';
    out.push({ speaker: speakerName, start: start, end: end, text: text });
  }
  return out;
}

async function handleProcessResponse(data){
  // Prefer clean transcript CSV if provided
  let dialogue = null;
  try {
    const cleanCsvPath = data.stages && data.stages.analysis && data.stages.analysis.clean_transcript_csv;
    if (cleanCsvPath) {
      const fname = String(cleanCsvPath).split('/').pop();
      const csvUrl = `/results/${fname}`;
      try {
        const txt = await fetch(csvUrl).then(r => { if(!r.ok) throw new Error('csv fetch failed'); return r.text(); });
        dialogue = parseCleanTranscriptCSV(txt);
      } catch (err) {
        console.warn('Failed to fetch/parse clean CSV, falling back:', err);
      }
    }
  } catch (e) {
    console.warn('clean csv check failed', e);
  }

  if (!dialogue) {
    try { dialogue = data.stages.analysis && data.stages.analysis.dialogue; } catch(e){}
    if (!dialogue) {
      const asrSegs = (data.stages && data.stages.asr && data.stages.asr.segments) || [];
      dialogue = asrSegs.map(s=>({
        start: Number(s.start||0),
        end: Number(s.end||0),
        text: s.text || '',
        speaker: s.speaker || 'unknown'
      }));
    }
  }

  lastDialogue = dialogue || [];
  renderDialogue(lastDialogue);

  // durations
  let durations = data.stages && data.stages.analysis && data.stages.analysis.durations;
  if (!durations) {
    try {
      const cleanCsvPath = data.stages && data.stages.analysis && data.stages.analysis.clean_transcript_csv;
      if (cleanCsvPath) {
        const stem = String(cleanCsvPath).split('/').pop().replace('_clean_transcript.csv','');
        const durUrl = `/results/${stem}_speaking_durations.json`;
        const resp = await fetch(durUrl);
        if (resp.ok) durations = await resp.json();
      }
    } catch (e) {
      console.warn('failed to fetch durations json', e);
    }
  }

  const pie_png = data.stages && data.stages.analysis && data.stages.analysis.pie_png;
  if(pie_png && pieImg){
    pieImg.src = pie_png.startsWith('/') || pie_png.startsWith('http') ? pie_png : '/results/' + (String(pie_png).split('/').pop());
    pieImg.style.display = 'block'; if(pieCanvas) pieCanvas.style.display = 'none'; if(legendEl) legendEl.innerHTML = '';
    if(durations) renderLegend(durations);
    pieImg.onerror = () => {
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

  lastMinutesText = generateMinutesText(lastDialogue, data);
}

/* ---------- Dialogue rendering (with read-more + speaker color) ---------- */
function renderDialogue(dialogue){
  if(!dialogue || !dialogueEl){ dialogueEl && (dialogueEl.innerHTML = '<div class="muted">No dialogue segments found.</div>'); return; }
  dialogueEl.innerHTML = '';

  // reset mapping so color assignments start fresh for this render (keeps order deterministic)
  speakerColorMap.clear();

  // sort by start time
  dialogue.sort((a,b)=> (Number(a.start||0) - Number(b.start||0)));

  let lastSpeaker = null;
  dialogue.forEach((seg, idx) => {
    const row = document.createElement('div');
    row.className = 'dialogue-row';

    // left column: speaker name (only when it changes)
    const left = document.createElement('div');
    left.className = 'speaker';
    const speaker = (seg.speaker || 'unknown').toString().trim();
    if (speaker && speaker !== lastSpeaker) {
      const nameSpan = document.createElement('div');
      nameSpan.className = 'speaker-name';
      const color = getSpeakerColor(speaker);
      nameSpan.textContent = speaker;
      nameSpan.style.color = color;
      nameSpan.style.fontWeight = '700';
      left.appendChild(nameSpan);
    }

    // right: meta (time pill) + bubble text (with read-more)
    const right = document.createElement('div');
    right.className = 'utterance';

    const meta = document.createElement('div');
    meta.className = 'meta';
    const timePill = document.createElement('span');
    timePill.className = 'time-pill muted';
    timePill.textContent = formatTime(Number(seg.start||0));
    meta.appendChild(timePill);

    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    const fullText = (seg.text || '').trim();

    if(fullText.length <= MAX_PREVIEW_CHARS){
      // short text: just render
      bubble.textContent = fullText;
    } else {
      // long text: create preview + expandable full node
      const preview = fullText.slice(0, MAX_PREVIEW_CHARS).replace(/\s+\S*$/, '');
      const previewNode = document.createElement('span');
      previewNode.className = 'preview-text';
      previewNode.textContent = preview + '… ';

      const fullNode = document.createElement('div');
      fullNode.className = 'full-text';
      fullNode.textContent = fullText;
      // prepare for height transition (collapsed initially)
      fullNode.style.display = 'block';
      fullNode.style.maxHeight = '0px';
      fullNode.style.overflow = 'hidden';
      fullNode.style.transition = 'max-height 320ms ease';

      const moreLink = document.createElement('a');
      moreLink.href = '#';
      moreLink.className = 'read-more';
      moreLink.textContent = 'Read more';
      moreLink.dataset.expanded = '0';
      moreLink.style.marginLeft = '8px';
      moreLink.style.cursor = 'pointer';

      // helper to collapse fully (used on collapse)
      function collapseFull() {
        // animate to 0
        fullNode.style.maxHeight = '0px';
        // after transition ends, hide fullNode and show preview
        const onEnd = (ev) => {
          if (ev.propertyName === 'max-height') {
            previewNode.style.display = 'inline';
            fullNode.removeEventListener('transitionend', onEnd);
          }
        };
        fullNode.addEventListener('transitionend', onEnd);
        moreLink.textContent = 'Read more';
        moreLink.dataset.expanded = '0';
      }

      moreLink.addEventListener('click', (ev) => {
        ev.preventDefault();
        const isExpanded = moreLink.dataset.expanded === '1';
        if(!isExpanded){
          // expand: hide preview immediately, then expand fullNode height
          previewNode.style.display = 'none';
          // must set maxHeight to scrollHeight to animate
          // temporarily set maxHeight to a large value to allow measurement, then set to scrollHeight
          fullNode.style.maxHeight = '0px';
          // force reflow to make sure the 0 is applied
          void fullNode.offsetWidth;
          const needed = fullNode.scrollHeight + 8; // +8px cushion
          fullNode.style.maxHeight = needed + 'px';
          moreLink.textContent = 'Show less';
          moreLink.dataset.expanded = '1';
          // smooth scroll bubble to center of viewport
          setTimeout(()=>{ bubble.scrollIntoView({ behavior: 'smooth', block: 'center' }); }, 60);
        } else {
          // collapse
          collapseFull();
          // scroll the row into view after collapse to keep context
          setTimeout(()=>{ row.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }, 280);
        }
      });

      // ensure that if the element's natural size changes we update maxHeight when expanded
      const resizeObserver = new ResizeObserver(() => {
        if (moreLink.dataset.expanded === '1') {
          fullNode.style.maxHeight = (fullNode.scrollHeight + 8) + 'px';
        }
      });
      resizeObserver.observe(fullNode);

      bubble.appendChild(previewNode);
      bubble.appendChild(fullNode);
      bubble.appendChild(moreLink);
    }

    right.appendChild(meta);
    right.appendChild(bubble);

    row.appendChild(left);
    row.appendChild(right);
    dialogueEl.appendChild(row);

    if (speaker) lastSpeaker = speaker;
  });
}


/* ---------- small helper to format seconds to mm:ss ---------- */
function formatTime(sec) {
  const s = Math.round(Number(sec) || 0);
  const mm = Math.floor(s / 60);
  const ss = s % 60;
  return `${String(mm).padStart(2,'0')}:${String(ss).padStart(2,'0')}`;
}

/* ---------- Legend & pie drawing ---------- */
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

/* ---------- minutes generation, copy & download ---------- */
function generateMinutesText(dialogue, rawResp){
  let out = '1) Dialogue\n\n';
  (dialogue || []).forEach(d => {
    const s = (d.start||0).toFixed(2);
    const e = (d.end||0).toFixed(2);
    const sp = d.speaker || 'unknown';
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

/* ---------- Initialize demo on page load ---------- */
document.addEventListener('DOMContentLoaded', () => {
  // show demo only when there is no existing dialogue content
  if (dialogueEl && dialogueEl.children.length === 0) {
    insertDemoSample();
  } else {
    // if dialogue exists but no pie/legend, show demo pie
    if ((!legendEl || legendEl.children.length === 0) && pieCanvas) {
      drawPieFromDurations(DEMO_DURATIONS);
      legendEl && renderLegend(DEMO_DURATIONS);
    }
  }
});
