// --------- script.js (clean, ordered, demo-preserving) ---------

/* -------------------- Globals / Elements -------------------- */
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
// Global recorder references (must be top-level!)
let mediaRecorder = null;
let recordedChunks = [];
let recordedBlob = null;
let recordedFileName = null;


window.WORKER_BASE = window.WORKER_BASE || 'http://127.0.0.1:9000';

let lastMinutesText = '';
let lastDialogue = [];

/* Speaker color palettes */
const TEXT_COLORS = [
  '#d1d0ecff', '#c9f8ebff', '#FACDEB', '#f8ca99ff',
  '#d5eef8ff', '#D8F7A3', '#fadfb9ff', '#ddc8f7ff'
];
const SPEAKER_COLORS = [
  '#2e4573ff', '#2d5f55ff', '#672147ff', '#75481cff',
  '#30492bff', '#324707ff', '#5f3833ff', '#412f51ff'
];
const speakerColorMap = new Map();

/* ---------- Small utility helpers ---------- */
function setStatus(s){ if(statusEl) statusEl.textContent = s; }
function resultFileUrl(pathOrUrl){
  if(!pathOrUrl) return null;
  try {
    const s = String(pathOrUrl);
    if(/^https?:\/\//i.test(s)) return s;
    const idx = s.lastIndexOf('/results/');
    const base = (window.WORKER_BASE && window.WORKER_BASE.replace(/\/+$/, '')) || window.location.origin;
    if(idx !== -1){
      const name = s.slice(idx + '/results/'.length);
      return base + '/results/' + name;
    }
    const parts = s.split(/[\\/]/);
    const basename = parts[parts.length - 1];
    if(!basename) return null;
    return base + '/results/' + basename;
  } catch (e){
    return null;
  }
}
function formatTime(sec) {
  const s = Math.round(Number(sec) || 0);
  const mm = Math.floor(s / 60);
  const ss = s % 60;
  return `${String(mm).padStart(2,'0')}:${String(ss).padStart(2,'0')}`;
}
function hexToRgba(hex, alpha = 1) {
  if (!hex) return `rgba(255,255,255,${alpha})`;
  const h = String(hex).replace('#','').trim();
  let r=0,g=0,b=0,a=1;
  if (h.length === 3) {
    r = parseInt(h[0]+h[0],16);
    g = parseInt(h[1]+h[1],16);
    b = parseInt(h[2]+h[2],16);
  } else if (h.length === 6) {
    r = parseInt(h.slice(0,2),16);
    g = parseInt(h.slice(2,4),16);
    b = parseInt(h.slice(4,6),16);
  } else if (h.length === 8) {
    r = parseInt(h.slice(0,2),16);
    g = parseInt(h.slice(2,4),16);
    b = parseInt(h.slice(4,6),16);
    a = parseInt(h.slice(6,8),16)/255;
  } else {
    const parts = h.split(',').map(p=>parseInt(p,10));
    if (parts.length >= 3) { r=parts[0]; g=parts[1]; b=parts[2]; }
  }
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/* ---------- Global error protection (prevents silent crash) ---------- */
window.addEventListener('error', (ev) => {
  console.error('Unhandled error', ev.error || ev.message, ev);
  try { setStatus('error'); hideProgressBar(); if(uploadBtn) uploadBtn.disabled = false; } catch(e){}
});
window.addEventListener('unhandledrejection', (ev) => {
  console.error('Unhandled promise rejection', ev.reason);
  try { setStatus('error'); hideProgressBar(); if(uploadBtn) uploadBtn.disabled = false; } catch(e){}
});

/* ---------- Demo helpers (keep demo visible until real data arrives) ---------- */
/* insertDemoSample() and clearDemoSample() mostly unchanged but removed auto-clear on file select */
const DEMO_SPEAKERS = [
  { speaker: 'Aditya', start: 0, end: 85, text: "Good morning, everyone — this is Aditya. Let's quickly run through the agenda items and assign owners." },
  { speaker: 'Rohan', start: 86, end: 136.5, text: "Thanks Aditya. I can take the action on the API integration and follow up by Friday." },
  { speaker: 'Shourya', start: 137, end: 183, text: "I'm seeing a potential blocker in the deployment pipeline; I'll raise a ticket and coordinate with the members." },
  { speaker: 'Isha', start: 184, end: 223, text: "Isha here — I can help with QA and set up the test matrix. Also, quick note: timings shift if we add one more dependency." }
];
const DEMO_DURATIONS = { "Aditya":85, "Rohan":50.5, "Shourya":46, "Isha":39 };

function insertDemoSample() {
  if (!dialogueEl || dialogueEl.dataset.demoInserted) return;
  dialogueEl.dataset.demoInserted = '1';

  const notice = document.createElement('div');
  notice.className = 'demo-notice';
  notice.textContent = '⚠️ This is a sample output. Upload your audio to see real results.';
  if (dialogueEl.parentNode) dialogueEl.parentNode.insertBefore(notice, dialogueEl);

  renderDialogue(DEMO_SPEAKERS);

  if (pieImg) {
    pieImg.dataset.demoShown = '1';
    pieImg.src = 'demo.png';
    pieImg.style.display = 'block';
    pieImg.style.width = '100%';
    pieImg.style.maxWidth = '900px';
    pieImg.style.height = 'auto';
    pieImg.style.margin = '0 auto';
    pieImg.style.boxShadow = '0 10px 30px rgba(12,22,40,0.08)';
    pieImg.style.borderRadius = '8px';
    pieImg.style.cursor = 'pointer';
    pieImg.onclick = () => window.open(pieImg.src, '_blank');
  }

  const demoAnalysis = {
    agenda: "Quick demo: project status review",
    summary: "This demo shows how the agenda, summary, and per-speaker relevance with pros, cons, and verdicts appear in the UI.",
    speakers: [
      {
        name: "Aditya",
        relevance_score: 8.1,
        pros: "Led important parts of the discussion and kept everyone on track.",
        cons: "Could summarize key outcomes more succinctly at the end.",
        final_verdict: "Outstanding — Aditya was keenly participating and made valuable contributions.",
        talk_time_seconds: 85
      },
      {
        name: "Rohan",
        relevance_score: 6.7,
        pros: "Provided updates on implementation progress and shared technical inputs.",
        cons: "Should improve clarity while explaining ongoing tasks.",
        final_verdict: "Good — Rohan participated well and supported the meeting objectives.",
        talk_time_seconds: 50.5
      },
      {
        name: "Shourya",
        relevance_score: 5.5,
        pros: "Raised a blocker and suggested alternative approaches.",
        cons: "Could engage more proactively and expand on proposed solutions.",
        final_verdict: "Needs improvement — Shourya contributed but could engage more effectively.",
        talk_time_seconds: 46
      },
      {
        name: "Isha",
        relevance_score: 7.2,
        pros: "Offered QA assistance and made useful points about test coverage.",
        cons: "Could focus more on prioritizing high-impact test areas.",
        final_verdict: "Good — Isha participated actively and added meaningful insights.",
        talk_time_seconds: 39
      }
    ]
  };

  // ensure colors assigned (fills speakerColorMap)
  demoAnalysis.speakers.forEach(s => getSpeakerColors(s.name));
  try { renderContributionAnalysis(demoAnalysis); } catch(e){ console.warn('Demo analysis failed', e); }
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
    const ctx = pieCanvas.getContext && pieCanvas.getContext('2d');
    ctx && ctx.clearRect(0,0,pieCanvas.width,pieCanvas.height);
    pieCanvas.style.display = 'none';
  }

  if (pieImg) {
    if (pieImg.dataset && pieImg.dataset.demoShown) delete pieImg.dataset.demoShown;
    pieImg.style.display = 'none';
    pieImg.src = '';
  }

  const analysisCard = document.getElementById('analysisCard');
  if (analysisCard) {
    analysisCard.style.display = 'none';
    const aAgenda = document.getElementById('analysisAgendaText');
    const aSummary = document.getElementById('analysisSummaryText');
    const aSpeakers = document.getElementById('analysisSpeakers');
    if (aAgenda) aAgenda.textContent = '—';
    if (aSummary) aSummary.textContent = '—';
    if (aSpeakers) aSpeakers.innerHTML = '';
  }
}

/* ---------- Visualization / progress helpers ---------- */
function showProgressBar(){ if (progressWrap) progressWrap.style.display = 'block'; }
function hideProgressBar(){
  if (progressWrap) progressWrap.style.display = 'none';
  if (progressBar) progressBar.style.width = '0%';
  if (progressMsg) progressMsg.textContent = '';
  currentProgress = 0;
  targetProgress = 0;
  if(progressTimer){ clearInterval(progressTimer); progressTimer = null; }
}

let currentProgress = 0;
let targetProgress = 0;
let progressTimer = null;
let simulatedStart = 0;
let simulatedDuration = 0;
let simulatedMaxPct = 0;
let simulatedRunning = false;

function startSimulatedProgress(durationMs = 95_000, maxPct = 98){
  stopSimulatedProgress();
  simulatedStart = Date.now();
  simulatedDuration = Math.max(1, durationMs);
  simulatedMaxPct = Math.min(99, Math.max(1, maxPct));
  simulatedRunning = true;
  ensureProgressTimer();
}
function stopSimulatedProgress(){
  simulatedRunning = false;
  simulatedStart = 0;
  simulatedDuration = 0;
  simulatedMaxPct = 0;
}
function getSimulatedProgress(){
  if(!simulatedRunning) return 0;
  const elapsed = Date.now() - simulatedStart;
  const frac = Math.min(1, elapsed / simulatedDuration);
  return Math.round(frac * simulatedMaxPct);
}
function updateProgressBar(serverPct, msg){
  const s = Number.isFinite(Number(serverPct)) ? Number(serverPct) : 0;
  if(progressMsg) progressMsg.textContent = msg || '';
  const sim = getSimulatedProgress();
  targetProgress = Math.min(100, Math.max(s, sim));
  ensureProgressTimer();
}
function ensureProgressTimer(){
  if(progressTimer) return;
  progressTimer = setInterval(()=> {
    const sim = getSimulatedProgress();
    targetProgress = Math.min(100, Math.max(targetProgress, sim));
    if(currentProgress < targetProgress){
      const remaining = targetProgress - currentProgress;
      const step = remaining > 10 ? Math.ceil(remaining * 0.08) : 1;
      currentProgress += step;
      if(currentProgress > targetProgress) currentProgress = targetProgress;
      if(progressBar) progressBar.style.width = currentProgress + '%';
    }
    if(currentProgress >= 100){
      clearInterval(progressTimer);
      progressTimer = null;
    }
  }, 80);
}
function waitForProgressToReach(pct = 100, timeoutMs = 15000){
  pct = Number(pct || 100);
  return new Promise((resolve) => {
    const start = Date.now();
    const check = () => {
      if(currentProgress >= pct) return resolve();
      if(Date.now() - start > timeoutMs) return resolve();
      setTimeout(check, 120);
    };
    check();
  });
}

/* ---------- CSV parsing (keeps your original parser) ---------- */
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

/* ---------- Rendering: dialogue (keeps your style + readability changes) ---------- */
function getSpeakerColors(name) {
  if (!name) name = 'unknown';
  if (speakerColorMap.has(name)) return speakerColorMap.get(name);
  const idx = speakerColorMap.size % SPEAKER_COLORS.length;
  const colors = { bg: SPEAKER_COLORS[idx], text: TEXT_COLORS[idx] };
  speakerColorMap.set(name, colors);
  return colors;
}

function renderDialogue(dialogue){
  if(!dialogue || !dialogueEl){
    dialogueEl && (dialogueEl.innerHTML = '<div class="muted">No dialogue segments found.</div>');
    return;
  }

  dialogueEl.innerHTML = '';
  speakerColorMap.clear();
  dialogue.sort((a,b)=> (Number(a.start||0) - Number(b.start||0)));

  let lastSpeaker = null;
  const fragment = document.createDocumentFragment();

  dialogue.forEach((seg) => {
    const row = document.createElement('div');
    row.className = 'dialogue-row';

    const speaker = (seg.speaker || seg.speaker_name || 'unknown').toString().trim();
    const { bg: baseColor, text: textColor } = getSpeakerColors(speaker || 'unknown');

    row.style.background = baseColor;
    row.style.padding = '12px 14px';
    row.style.borderRadius = '10px';
    row.style.marginBottom = '12px';
    row.style.display = 'flex';
    row.style.gap = '12px';
    row.style.alignItems = 'flex-start';

    const left = document.createElement('div');
    left.className = 'speaker';
    left.style.minWidth = '120px';
    left.style.flex = '0 0 140px';
    left.style.display = 'flex';
    left.style.flexDirection = 'column';

    if (speaker && speaker !== lastSpeaker) {
      const nameSpan = document.createElement('div');
      nameSpan.className = 'speaker-name';
      nameSpan.textContent = speaker;
      nameSpan.style.color = textColor;
      nameSpan.style.fontWeight = '700';
      nameSpan.style.marginBottom = '6px';
      nameSpan.style.fontSize = '20px';
      left.appendChild(nameSpan);
    }

    const right = document.createElement('div');
    right.className = 'utterance';
    right.style.flex = '1 1 auto';

    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.style.marginBottom = '8px';
    meta.style.display = 'flex';
    meta.style.alignItems = 'center';
    meta.style.gap = '10px';

    const timePill = document.createElement('span');
    timePill.className = 'time-pill';
    timePill.textContent = formatTime(Number(seg.start||0));
    timePill.style.color = textColor;
    timePill.style.fontSize = '18px';
    timePill.style.fontWeight = '700';
    timePill.style.opacity = '0.95';
    meta.appendChild(timePill);

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = (seg.text || seg.transcript || '').trim();
    bubble.style.display = 'inline-block';
    bubble.style.padding = '10px 12px';
    bubble.style.borderRadius = '8px';
    bubble.style.boxShadow = 'inset 0 0 0 1px rgba(0,0,0,0.03)';
    bubble.style.maxWidth = '100%';
    bubble.style.lineHeight = '1.45';
    bubble.style.whiteSpace = 'pre-wrap';
    bubble.style.fontSize = '17px';
    bubble.style.fontWeight = '600';
    // swap: background = textColor, text = baseColor (gives high contrast)
    bubble.style.background = textColor;
    bubble.style.color = baseColor;

    right.appendChild(meta);
    right.appendChild(bubble);

    row.appendChild(left);
    row.appendChild(right);
    fragment.appendChild(row);

    if (speaker) lastSpeaker = speaker;
  });

  const wrapper = document.createElement('div');
  wrapper.className = 'dialogue-wrapper';
  const MAX_DIALOGUE_HEIGHT = 500;
  wrapper.style.maxHeight = MAX_DIALOGUE_HEIGHT + 'px';
  wrapper.style.overflow = 'hidden';
  wrapper.style.transition = 'max-height 350ms ease';
  wrapper.appendChild(fragment);
  dialogueEl.appendChild(wrapper);

  const toggle = document.createElement('a');
  toggle.href = '#';
  toggle.className = 'read-more-global';
  toggle.style.textAlign = 'center';
  toggle.style.marginTop = '10px';
  toggle.style.display = 'block';
  toggle.style.cursor = 'pointer';
  toggle.style.userSelect = 'none';
  toggle.textContent = 'Read more';
  toggle.style.color = 'white';
  toggle.style.fontSize = '20px';

  dialogueEl.appendChild(toggle);

  requestAnimationFrame(() => {
    const contentHeight = wrapper.scrollHeight;
    if (contentHeight > MAX_DIALOGUE_HEIGHT + 4) {
      toggle.style.display = 'block';
      wrapper.style.maxHeight = MAX_DIALOGUE_HEIGHT + 'px';
      wrapper.style.overflow = 'hidden';
      toggle.textContent = 'Read more';
    } else {
      toggle.style.display = 'none';
      wrapper.style.maxHeight = 'none';
      wrapper.style.overflow = 'visible';
    }
  });

  let expanded = false;
  toggle.addEventListener('click', (e) => {
    e.preventDefault();
    if (!expanded) {
      const target = wrapper.scrollHeight;
      wrapper.style.maxHeight = target + 'px';
      const onExpandEnd = (ev) => {
        if (ev.propertyName === 'max-height') {
          wrapper.style.maxHeight = 'none';
          wrapper.style.overflow = 'visible';
          wrapper.removeEventListener('transitionend', onExpandEnd);
        }
      };
      wrapper.addEventListener('transitionend', onExpandEnd);
      toggle.textContent = 'Show less';
      toggle.style.color = 'white';
      toggle.style.fontSize = '20px';
      expanded = true;
      setTimeout(()=> wrapper.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);
    } else {
      wrapper.style.maxHeight = wrapper.scrollHeight + 'px';
      void wrapper.offsetWidth;
      wrapper.style.overflow = 'hidden';
      wrapper.style.maxHeight = MAX_DIALOGUE_HEIGHT + 'px';
      toggle.textContent = 'Read more';
      expanded = false;
      const onCollapseEnd = (ev) => {
        if (ev.propertyName === 'max-height') {
          wrapper.removeEventListener('transitionend', onCollapseEnd);
          setTimeout(()=> toggle.scrollIntoView({ behavior: 'smooth', block: 'end' }), 40);
        }
      };
      wrapper.addEventListener('transitionend', onCollapseEnd);
    }
  });
}

/* ---------- Pie image helper (unchanged, robust) ---------- */
async function showPieFromResult(data){
  if (!pieImg) return;
  try {
    let candidate = (data && data.files && (data.files.pie_png || data.files.pie_svg)) || (data && data.public_url) || null;
    if (!candidate) {
      pieImg.style.display = 'none';
      pieImg.src = '';
      return;
    }
    let url = resultFileUrl(candidate) || String(candidate);
    if (!/^https?:\/\//i.test(url)) {
      url = new URL(url, window.location.origin).toString();
    }
    let loaded = false;
    pieImg.onload = () => {
      loaded = true;
      pieImg.style.display = 'block';
      if (pieImg.dataset && pieImg.dataset.demoShown) delete pieImg.dataset.demoShown;
      pieImg.style.width = '100%';
      pieImg.style.maxWidth = '900px';
      pieImg.style.height = 'auto';
      pieImg.style.cursor = 'pointer';
    };
    pieImg.onerror = () => {
      console.warn('Failed loading pie image:', url);
      pieImg.style.display = 'none';
      pieImg.src = '';
    };
    const sep = url.includes('?') ? '&' : '?';
    pieImg.src = url + sep + '_=' + Date.now();
    pieImg.onclick = () => { const raw = pieImg.src.split('?')[0]; window.open(raw, '_blank'); };
    await new Promise(r => setTimeout(r, 300));
  } catch (e) {
    console.warn('showPieFromResult error', e);
    pieImg.style.display = 'none';
  }
}

/* ---------- Analysis fetching/rendering (robust + updated fields) ---------- */
function extractJsonFromRawOutput(raw) {
  if (!raw || typeof raw !== 'string') return null;
  const fenced = /```(?:json)?\s*([\s\S]*?)\s*```/i.exec(raw);
  const candidate = fenced ? fenced[1] : raw;
  const firstBrace = candidate.indexOf('{');
  if (firstBrace < 0) return null;
  const js = candidate.slice(firstBrace);
  try { return JSON.parse(js); } catch (e) { return null; }
}

async function fetchAnalysisFromResult(resultObj) {
  try {
    let url = null;

    if (resultObj?.files?.analysis) {
      url = resultFileUrl(resultObj.files.analysis) || String(resultObj.files.analysis);
    } else if (resultObj?.result_url) {
      url = resultFileUrl(resultObj.result_url) || String(resultObj.result_url);
    } else if (resultObj?.files?.dialogue_named) {
      const dp = resultObj.files.dialogue_named;
      const parts = String(dp).split(/[\\/]/);
      const basename = parts[parts.length - 1] || '';
      const stem = basename.replace(/_dialogue_named\.json$/, '');
      if (stem) {
        const base = (window.WORKER_BASE ? window.WORKER_BASE.replace(/\/+$/, '') : window.location.origin);
        url = `${base}/results/${stem}_analysis.json`;
      }
    }

    if (!url) return null;
    if (!/^https?:\/\//i.test(url)) url = new URL(url, window.location.origin).toString();

    const resp = await fetch(url);
    if (!resp.ok) {
      const text = await resp.text().catch(() => null);
      if (text) {
        const parsed = extractJsonFromRawOutput(text);
        if (parsed) return parsed;
      }
      return null;
    }

    const j = await resp.json().catch(async () => {
      const txt = await resp.text().catch(() => null);
      return txt ? extractJsonFromRawOutput(txt) : null;
    });

    if (!j) return null;

    if (j.raw_output && typeof j.raw_output === 'string') {
      const inner = extractJsonFromRawOutput(j.raw_output);
      if (inner) return inner;
      try { return JSON.parse(j.raw_output); } catch (_) {}
    }

    return j;
  } catch (e) {
    console.warn('fetchAnalysisFromResult failed', e);
    return null;
  }
}

function renderContributionAnalysis(analysis) {
  const card = document.getElementById('analysisCard');
  if (!card) return;
  if (!analysis) { card.style.display = 'none'; return; }
  card.style.display = 'block';

  const agendaEl = document.getElementById('analysisAgendaText');
  const summaryEl = document.getElementById('analysisSummaryText');
  const speakersWrap = document.getElementById('analysisSpeakers');

  if (agendaEl) agendaEl.textContent = analysis.agenda || '—';
  if (summaryEl) summaryEl.textContent = analysis.summary || '—';
  if (speakersWrap) speakersWrap.innerHTML = '';

  const speakers = Array.isArray(analysis.speakers) ? analysis.speakers : [];

  speakers.forEach(s => {
    const name = s.name || 'unknown';
    const score = (typeof s.relevance_score === 'number') ? s.relevance_score : null;

    // safer field selection (supports new & legacy fields)
    const pros = (s.pros ?? s.strengths ?? s.feedback ?? '');
    const cons = (s.cons ?? s.weaknesses ?? '');
    const verdict = (s.final_verdict ?? s.verdict ?? '');
    const talkTime = (typeof s.talk_time_seconds === 'number' ? s.talk_time_seconds : (s.talk_time_seconds ?? '—'));

    const colors = (typeof getSpeakerColors === 'function')
      ? (getSpeakerColors(name) || { bg: '#2b3a55', text: '#e6eefb' })
      : { bg: '#2b3a55', text: '#e6eefb' };

    // container row
    const row = document.createElement('div');
    row.className = 'speaker-row';
    const bgA = hexToRgba(colors.bg, 0.18);
    const bgB = hexToRgba(colors.bg, 0.06);
    row.style.background = `linear-gradient(180deg, ${bgA}, ${bgB})`;
    row.style.setProperty('--top-glow', hexToRgba(colors.text, 0.12));
    row.style.borderTop = `4px solid ${hexToRgba(colors.text, 0.06)}`;
    row.style.padding = '12px';
    row.style.marginBottom = '12px';
    row.style.borderRadius = '10px';
    row.style.display = 'flex';
    row.style.flexDirection = 'column';
    row.style.gap = '8px';

    // top line: name + score badge
    const top = document.createElement('div');
    top.className = 'speaker-top';
    top.style.display = 'flex';
    top.style.justifyContent = 'space-between';
    top.style.alignItems = 'center';
    top.style.gap = '12px';

    const nameEl = document.createElement('div');
    nameEl.className = 'speaker-name';
    nameEl.textContent = name;
    nameEl.style.color = colors.text;
    nameEl.style.fontWeight = '600';
    nameEl.style.fontSize = '1rem';

    const scoreBadge = document.createElement('div');
    scoreBadge.className = 'speaker-score-badge';
    scoreBadge.textContent = (score !== null) ? `${score}/10` : '—';
    scoreBadge.style.color = '#fff';
    scoreBadge.style.background = hexToRgba(colors.text, 0.12);
    scoreBadge.style.border = `1px solid ${hexToRgba(colors.text, 0.08)}`;
    scoreBadge.style.padding = '6px 10px';
    scoreBadge.style.borderRadius = '999px';
    scoreBadge.style.fontWeight = '700';
    scoreBadge.style.fontSize = '0.9rem';

    top.appendChild(nameEl);
    top.appendChild(scoreBadge);

    // relevance score section
    const relevanceSection = document.createElement('div');
    relevanceSection.className = 'relevance-section';
    relevanceSection.style.display = 'flex';
    relevanceSection.style.flexDirection = 'column';
    relevanceSection.style.gap = '6px';

    const relHeading = document.createElement('div');
    relHeading.className = 'subheading';
    relHeading.textContent = 'Relevance score:';
    relHeading.style.fontSize = '0.85rem';
    relHeading.style.fontWeight = '600';
    relHeading.style.color = hexToRgba(colors.text, 0.95);

    const barWrap = document.createElement('div');
    barWrap.className = 'speaker-bar-wrap';
    barWrap.style.width = '100%';
    barWrap.style.height = '14px';
    barWrap.style.background = hexToRgba(colors.text, 0.06);
    barWrap.style.borderRadius = '999px';
    barWrap.style.overflow = 'hidden';
    barWrap.style.border = `1px solid ${hexToRgba(colors.text, 0.04)}`;

    const bar = document.createElement('div');
    bar.className = 'speaker-bar';
    bar.setAttribute('role', 'progressbar');
    bar.setAttribute('aria-valuemin', '0');
    bar.setAttribute('aria-valuemax', '10');
    if (score !== null) bar.setAttribute('aria-valuenow', String(score));

    const barFill = document.createElement('div');
    barFill.className = 'speaker-bar-fill';
    const pct = (score !== null) ? Math.max(0, Math.min(100, Math.round((score / 10) * 100))) : 0;
    barFill.style.width = '0%';
    barFill.style.height = '100%';
    barFill.style.transition = 'width 700ms cubic-bezier(.2,.9,.2,1)';
    barFill.style.background = `linear-gradient(90deg, ${hexToRgba(colors.text, 0.95)}, ${hexToRgba(colors.bg, 0.85)})`;

    bar.appendChild(barFill);
    barWrap.appendChild(bar);
    relevanceSection.appendChild(relHeading);
    relevanceSection.appendChild(barWrap);

    // reusable builder for labeled blocks
    function makeLabeledBlock(labelText, contentText) {
      const wrap = document.createElement('div');
      wrap.className = 'label-block';
      wrap.style.display = 'flex';
      wrap.style.flexDirection = 'column';
      wrap.style.gap = '6px';

      const label = document.createElement('div');
      label.className = 'label';
      label.textContent = labelText;
      label.style.fontSize = '0.85rem';
      label.style.fontWeight = '600';
      label.style.color = hexToRgba(colors.text, 0.9);

      const content = document.createElement('div');
      content.className = 'content';
      content.style.fontSize = '0.95rem';
      content.style.color = hexToRgba(colors.text, 0.95);
      content.style.whiteSpace = 'pre-wrap';
      content.textContent = contentText || '—';

      wrap.appendChild(label);
      wrap.appendChild(content);
      return wrap;
    }

    const prosBlock = makeLabeledBlock('Pros :', pros);
    const consBlock = makeLabeledBlock('Cons :', cons);
    const verdictBlock = makeLabeledBlock('Final Verdict :', verdict);

    // talk time
    const talkEl = document.createElement('div');
    talkEl.className = 'speaker-talktime muted small';
    const timeDisplay = (typeof talkTime === 'number')
      ? `${Number(talkTime).toFixed(1)} s`
      : `${talkTime}`;
    talkEl.textContent = `Talk time: ${timeDisplay}`;
    talkEl.style.color = hexToRgba(colors.text, 0.66);
    talkEl.style.fontSize = '0.85rem';
    talkEl.style.marginTop = '4px';

    // assemble
    row.appendChild(top);
    row.appendChild(relevanceSection);
    row.appendChild(prosBlock);
    row.appendChild(consBlock);
    row.appendChild(verdictBlock);
    row.appendChild(talkEl);

    if (speakersWrap) speakersWrap.appendChild(row);

    // animate bar fill
    requestAnimationFrame(() => {
      requestAnimationFrame(() => { barFill.style.width = pct + '%'; });
    });
  });
}

async function fetchAndRenderAnalysis(resultObj) {
  try {
    const analysis = await fetchAnalysisFromResult(resultObj);
    const card = document.getElementById('analysisCard');
    if (!analysis) {
      if (card) {
        card.style.display = 'block';
        const agendaEl = document.getElementById('analysisAgendaText');
        const summaryEl = document.getElementById('analysisSummaryText');
        const speakersWrap = document.getElementById('analysisSpeakers');
        if (agendaEl) agendaEl.textContent = '—';
        if (summaryEl) summaryEl.textContent = 'Analysis not available for this meeting.';
        if (speakersWrap) speakersWrap.innerHTML =
          '<div class="muted small">No contribution analysis produced yet.</div>';
      }
      console.warn('No analysis JSON found in results.');
      return;
    }
    renderContributionAnalysis(analysis);
  } catch (err) {
    console.error('fetchAndRenderAnalysis failed:', err);
    const c = document.getElementById('analysisCard');
    if (c) c.style.display = 'none';
  }
}



/* ---------- Recorder & upload (self-contained, paste over existing recorder/upload block) ---------- */

/* --- Safe globals (do not re-declare if already present) --- */
if (typeof window.mediaRecorder === 'undefined') window.mediaRecorder = null;
if (typeof window.recordedChunks === 'undefined') window.recordedChunks = [];
if (typeof window.recordedBlob === 'undefined') window.recordedBlob = null;
if (typeof window.recordedFileName === 'undefined') window.recordedFileName = null;
if (typeof window.recTimerInterval === 'undefined') window.recTimerInterval = null;
if (typeof window.recStartTs === 'undefined') window.recStartTs = 0;

const recordBtn = document.getElementById('recordBtn');
const pauseBtn = document.getElementById('pauseBtn');
const saveRecBtn = document.getElementById('saveRecBtn');
const discardRecBtn = document.getElementById('discardRecBtn');
const recTimerEl = document.getElementById('recTimer');

/* --- small helpers (safe wrappers if your app defines these elsewhere) --- */
const _setStatus = (msg) => { try { if (typeof setStatus === 'function') return setStatus(msg); if (statusEl) statusEl.textContent = msg; } catch(e){} };
const _showProgress = () => { try { if (typeof showProgressBar === 'function') return showProgressBar(); } catch(e){} };
const _hideProgress = () => { try { if (typeof hideProgressBar === 'function') return hideProgressBar(); } catch(e){} };
const _pollAndHandleJob = async (jobId, base) => { if (typeof pollAndHandleJob === 'function') return await pollAndHandleJob(jobId, base); return null; };
const _handleProcessResponse = async (pj) => { if (typeof handleProcessResponse === 'function') return await handleProcessResponse(pj); return null; };

/* --- time formatting and timer functions --- */
function fmtTimeSec(s) {
  s = Math.max(0, Math.floor(s || 0));
  const mm = String(Math.floor(s / 60)).padStart(2, '0');
  const ss = String(s % 60).padStart(2, '0');
  return `${mm}:${ss}`;
}
function updateTimer() {
  if (!recStartTs) {
    if (recTimerEl) recTimerEl.textContent = '00:00';
    return;
  }
  const elapsed = Math.floor((Date.now() - recStartTs) / 1000);
  if (recTimerEl) recTimerEl.textContent = fmtTimeSec(elapsed);
}
function startTimer() { recStartTs = Date.now(); updateTimer(); recTimerInterval = setInterval(updateTimer, 500); }
function stopTimer() { if (recTimerInterval) clearInterval(recTimerInterval); recTimerInterval = null; recStartTs = 0; updateTimer(); }

/* --- recorder UI state --- */
function setRecorderState(state) {
  // allow external function as well, otherwise local behavior
  try {
    if (typeof window.setRecorderStateGlobal === 'function') return window.setRecorderStateGlobal(state);
  } catch(e){}

  const disable = (rec, pause, save, discard) => {
    if (recordBtn) { recordBtn.disabled = rec; }
    if (pauseBtn) { pauseBtn.disabled = pause; }
    if (saveRecBtn) { saveRecBtn.disabled = save; }
    if (discardRecBtn) { discardRecBtn.disabled = discard; }
  };
  switch (state) {
    case 'idle':
      if (recordBtn) recordBtn.textContent = '● Record';
      disable(false, true, true, true);
      if (recTimerEl) recTimerEl.textContent = '00:00';
      break;
    case 'recording':
      if (recordBtn) recordBtn.textContent = '■ Stop';
      disable(false, false, true, true);
      break;
    case 'paused':
      if (recordBtn) recordBtn.textContent = '■ Stop';
      disable(false, false, false, false);
      break;
    case 'recorded':
      if (recordBtn) recordBtn.textContent = '● Record';
      disable(false, true, false, false);
      break;
    default:
      disable(false, true, true, true);
  }
}

/* ---------- Recording control functions (defined BEFORE wiring) ---------- */

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia) {
    alert('Recording not supported in this browser.');
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    // reset
    recordedChunks = [];
    recordedBlob = null;
    recordedFileName = null;

    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (ev) => {
      if (ev.data && ev.data.size > 0) recordedChunks.push(ev.data);
    };
    mediaRecorder.onstop = async () => {
      try {
        recordedBlob = new Blob(recordedChunks, { type: recordedChunks[0]?.type || 'audio/webm' });
        const ts = new Date().toISOString().replace(/[:.]/g, '-');
        recordedFileName = `recording_${ts}.webm`;
      } catch (e) {
        console.warn('onstop assemble blob failed', e);
        recordedBlob = null;
        recordedFileName = null;
      } finally {
        stopTimer();
        setRecorderState('recorded');
        _setStatus('Recording ready');
        // optional: automatically save & process (comment out if you want manual Save)
        // await saveAndProcessRecordedAudio(recordedBlob, recordedFileName);
      }
    };
    mediaRecorder.onerror = (ev) => {
      console.error('MediaRecorder error', ev);
      _setStatus('Recording error');
      setRecorderState('idle');
      stopTimer();
      try { stream.getTracks().forEach(t=>t.stop()); } catch(e){}
      mediaRecorder = null;
    };

    mediaRecorder.start();
    startTimer();
    setRecorderState('recording');
    _setStatus('Recording...');
  } catch (err) {
    console.error('startRecording failed', err);
    alert('Could not access microphone: ' + (err.message || err));
    setRecorderState('idle');
    stopTimer();
  }
}

function togglePauseRecording() {
  if (!mediaRecorder) return;
  try {
    if (mediaRecorder.state === 'recording') {
      mediaRecorder.pause();
      stopTimer();
      setRecorderState('paused');
      _setStatus('Recording paused');
    } else if (mediaRecorder.state === 'paused') {
      mediaRecorder.resume();
      startTimer();
      setRecorderState('recording');
      _setStatus('Recording...');
    }
  } catch (e) {
    console.warn('togglePause failed', e);
  }
}

function stopRecording() {
  if (!mediaRecorder) return;
  try {
    mediaRecorder.stop();
    // release mic tracks as a fallback (onstop handler may also stop them)
    try { mediaRecorder.stream && mediaRecorder.stream.getTracks().forEach(t => t.stop()); } catch(e){}
    _setStatus('Processing recorded audio...');
  } catch (err) {
    console.warn('stopRecording failed', err);
  } finally {
    // mediaRecorder will be nulled by onstop or manual cleanup if needed
  }
}

function discardRecording() {
  try {
    // if active, stop and release tracks
    if (mediaRecorder && (mediaRecorder.state === 'recording' || mediaRecorder.state === 'paused')) {
      try { mediaRecorder.stop(); } catch(e){}
      try { mediaRecorder.stream && mediaRecorder.stream.getTracks().forEach(t=>t.stop()); } catch(e){}
      mediaRecorder = null;
    }
  } catch(e){ console.warn('discard cleanup error', e); }

  recordedChunks = [];
  recordedBlob = null;
  recordedFileName = null;
  stopTimer();
  setRecorderState('idle');
  _setStatus('Recording discarded');

  // clear file input if present
  try {
    if (fileInput) {
      const dt = new DataTransfer();
      fileInput.files = dt.files;
      fileInput.dispatchEvent(new Event('change', { bubbles: true }));
    }
  } catch(e) {}
}

/* ---------- Save recording to audio/ only (no processing) ---------- */
async function saveAndProcessRecordedAudio(blob, filename) {
  if (!blob) {
    alert('No recorded audio to save.');
    return null;
  }

  // If WORKER_BASE is set (absolute/relative base), use it; otherwise use empty string
  const base = (typeof window !== 'undefined' && window.WORKER_BASE)
    ? String(window.WORKER_BASE).replace(/\/+$/, '')
    : '';

  // Build URL relative to current origin if base is empty (use '/save_audio')
  const saveUrl = (base || '') + '/save_audio';

  try {
    _setStatus('Saving recording to server (audio/)...');
    _showProgress();

    const fd = new FormData();
    fd.append('file', blob, filename || 'recording.webm');
    // optional metadata
    fd.append('original_name', filename || '');

    // upload
    const saveResp = await fetch(saveUrl, { method: 'POST', body: fd });

    // If server returned non-2xx, try to read JSON/text for diagnostics
    if (!saveResp.ok) {
      let errBody = await saveResp.text().catch(() => saveResp.statusText || '');
      try {
        const maybeJson = JSON.parse(errBody);
        errBody = maybeJson.detail || maybeJson.message || JSON.stringify(maybeJson);
      } catch (e) { /* not JSON */ }
      console.error(`[recorder] save_audio returned ${saveResp.status}:`, errBody);
      alert('Failed to save recording to server: ' + (errBody || saveResp.statusText));
      _setStatus('Save failed');
      return null;
    }

    // Try parse JSON; if not JSON, fallback to raw text
    let saveData = null;
    const ct = (saveResp.headers.get('content-type') || '').toLowerCase();
    if (ct.includes('application/json')) {
      try {
        saveData = await saveResp.json();
      } catch (e) {
        // fallback
        const txt = await saveResp.text().catch(() => null);
        console.warn('[recorder] response had application/json but parse failed, text:', txt);
        saveData = txt;
      }
    } else {
      // try json first (some servers omit header), then fallback to text
      try {
        saveData = await saveResp.json();
      } catch (e) {
        saveData = await saveResp.text().catch(() => null);
      }
    }

    // Accept several success shapes returned by server:
    // { saved: true, filename, path, url }  OR  { ok: true, ... }  OR direct URL string
    const isObj = saveData && typeof saveData === 'object' && !Array.isArray(saveData);
    const success =
      (isObj && (saveData.saved === true || saveData.ok === true || !!saveData.filename || !!saveData.url)) ||
      (typeof saveData === 'string' && saveData.length > 0);

    if (!success) {
      console.error('[recorder] save_audio returned unexpected payload:', saveData);
      alert('Server did not confirm save. See console for details.');
      _setStatus('Save failed');
      return null;
    }

    // Extract useful info
    let fileUrl = null;
    let savedName = null;
    if (isObj) {
      fileUrl = saveData.url || saveData.saved_path || saveData.path || null;
      savedName = saveData.filename || (typeof fileUrl === 'string' ? fileUrl.split('/').pop() : null);
    } else if (typeof saveData === 'string') {
      fileUrl = saveData;
      savedName = fileUrl.split('/').pop();
    }

    if (fileUrl) {
      const userMsg = `Saved to server: ${fileUrl}`;
      _setStatus('Saved to audio folder');
      // don't spam alerts in production; useful while testing
      alert(userMsg + '\n\nYou can access it at: ' + fileUrl);
    } else {
      _setStatus('Saved (path unknown)');
      alert('Recording saved but server did not return file URL.');
    }

    return saveData;

  } catch (err) {
    console.error('save/process failed (save only):', err);
    alert('Failed to save recording: ' + (err.message || err));
    _setStatus('Save failed');
    return null;
  } finally {
    _hideProgress();
  }
}


/* ---------- Button wiring (runs AFTER functions declared) ---------- */
if (recordBtn) {
  recordBtn.addEventListener('click', async () => {
    try {
      if (!mediaRecorder || mediaRecorder.state === 'inactive') await startRecording();
      else if (mediaRecorder.state === 'recording' || mediaRecorder.state === 'paused') stopRecording();
    } catch (e) { console.warn('recordBtn handler failed', e); }
  });
}
if (pauseBtn) {
  pauseBtn.addEventListener('click', () => {
    try { togglePauseRecording(); } catch (e) { console.warn('pause handler', e); }
  });
}
if (discardRecBtn) {
  discardRecBtn.addEventListener('click', () => {
    try {
      // safe low-level cleanup if still active
      if (mediaRecorder && (mediaRecorder.state === 'recording' || mediaRecorder.state === 'paused')) {
        try { mediaRecorder.stop(); } catch (e) {}
        try { mediaRecorder.stream && mediaRecorder.stream.getTracks().forEach(t => t.stop()); } catch (e) {}
        mediaRecorder = null;
      }
    } catch (e) { console.warn('discard low-level cleanup', e); }
    try { discardRecording(); } catch (e) { console.warn('discardRecording failed', e); }
  });
}
if (saveRecBtn) {
  saveRecBtn.addEventListener('click', async () => {
    try {
      if (recordedBlob && recordedFileName) {
        await saveAndProcessRecordedAudio(recordedBlob, recordedFileName);
      } else {
        alert('No recording available to save.');
      }
    } catch (e) {
      console.error('saveRecBtn handler failed', e);
      alert('Save failed: ' + (e.message || e));
    }
  });
}

/* ---------- Choose button + dropzone (unchanged) ---------- */
const chooseBtn = document.getElementById('chooseBtn');
const dropzone = document.getElementById('dropzone');

function openFilePicker() {
  try { if (fileInput) fileInput.value = ''; } catch (e) {}
  try { if (fileInput) fileInput.click(); } catch (e) {}
}

if (chooseBtn && fileInput) {
  chooseBtn.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); openFilePicker(); });
  chooseBtn.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); openFilePicker(); } });
}

if (dropzone && fileInput) {
  dropzone.addEventListener('click', (e) => {
    if (e.target.closest && (e.target.closest('#recordBtn, #pauseBtn, #saveRecBtn, #discardRecBtn') || e.target.closest('#chooseBtn'))) {
      return;
    }
    openFilePicker();
  });
  dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openFilePicker(); }
  });
}

/* When a user selects a file via input, reflect that in UI */
if (fileInput) {
  fileInput.addEventListener('change', () => {
    recordedBlob = null;
    recordedFileName = null;

    const has = fileInput.files && fileInput.files.length > 0;
    if (uploadBtn) uploadBtn.disabled = !has;

    if (has) {
      _setStatus('File ready');
      setRecorderState('idle');
      try {
        const f = fileInput.files[0];
        const fileInfo = document.getElementById('fileInfo');
        const fileNameEl = document.getElementById('fileName');
        const filePreview = document.getElementById('filePreview');
        if (fileInfo && fileNameEl) {
          fileInfo.style.display = 'flex';
          fileNameEl.textContent = `${f.name} · ${(f.size/(1024*1024)).toFixed(2)} MB`;
        }
        if (filePreview) {
          try { filePreview.src = URL.createObjectURL(f); filePreview.style.display = 'inline-block'; } catch (err) { filePreview.style.display = 'none'; }
        }
      } catch (e) { console.warn('file input change handling failed', e); }
    } else {
      _setStatus('idle');
    }
  });
}

/* Initialize to safe state */
try { setRecorderState('idle'); } catch (e) { /* ignore */ }


/* ---------- Upload / Polling flow (robust, non-destructive) ---------- */
async function pollStatusOnce(jobId, baseUrl) {
  try {
    const statusUrl = `${baseUrl}/status/${jobId}`;
    const sresp = await fetch(statusUrl);
    if (!sresp.ok) return null;
    return await sresp.json().catch(()=>null);
  } catch (e) {
    console.warn('poll error', e);
    return null;
  }
}
async function pollAndHandleJob(jobId, baseUrl, timeoutMs = 300000) {
  const start = Date.now();
  while (true) {
    await new Promise(r=>setTimeout(r,700));
    const sdata = await pollStatusOnce(jobId, baseUrl);
    if (sdata) {
      const serverPct = Number(sdata.progress || 0);
      const serverMsg = sdata.message || `${sdata.status||''}`;
      updateProgressBar(serverPct, serverMsg);
      if (sdata.status === 'done' || serverPct >= 100) {
        updateProgressBar(100, 'Finalizing...');
        break;
      }
      if (sdata.status === 'error') {
        throw new Error('Processing error: ' + (sdata.error || sdata.message || 'unknown'));
      }
    }
    if (Date.now() - start > timeoutMs) throw new Error('Polling timeout');
  }
}

async function _getUploadFileAndName() {
  if (fileInput && fileInput.files && fileInput.files.length) {
    const f = fileInput.files[0];
    return { file: f, filename: f.name };
  }
  if (recordedBlob) {
    const name = (typeof recordedFileName === 'string' && recordedFileName) ? recordedFileName : `recording_${new Date().toISOString().replace(/[:.]/g,'-')}.webm`;
    return { file: recordedBlob, filename: name };
  }
  return { file: null, filename: null };
}

if (uploadBtn) {
  uploadBtn.addEventListener('click', async () => {
    try {
      // preserve demo while uploading/processing; only clear when we have final result to render
      const { file: f, filename } = await _getUploadFileAndName();
      if (!f) { alert('Please select or record an audio file before uploading.'); return; }

      const pageOrigin = window.location.protocol + '//' + window.location.host;
      const baseUrl = (pageOrigin !== window.WORKER_BASE) ? window.WORKER_BASE : '';

      setStatus('Uploading...');
      showProgressBar();
      startSimulatedProgress(95_000, 98);
      updateProgressBar(2, 'Uploading file');
      uploadBtn.disabled = true;

      const fd = new FormData();
      fd.append('file', f, filename || (f.name || `recording_${Date.now()}.webm`));

      const uploadUrl = `${baseUrl}/process`;
      const resp = await fetch(uploadUrl, { method:'POST', body: fd });

      if (!resp.ok) {
        const txt = await resp.text().catch(()=>resp.statusText || '');
        stopSimulatedProgress();
        hideProgressBar();
        setStatus('error');
        throw new Error(`Upload failed ${resp.status}: ${txt}`);
      }

      const j = await resp.json().catch(()=>null);
      const jobId = j && j.job_id ? j.job_id : null;
      if (!jobId) {
        stopSimulatedProgress();
        hideProgressBar();
        setStatus('error');
        throw new Error('Missing job id in upload response');
      }

      setStatus('Processing (background)...');
      updateProgressBar(6, 'Queued');

      await pollAndHandleJob(jobId, baseUrl);

      await waitForProgressToReach(100, 10000);
      stopSimulatedProgress();

      updateProgressBar(99, 'fetching result');
      const resultUrl = `${baseUrl}/result/${jobId}`;
      const rresp = await fetch(resultUrl);
      if (!rresp.ok) {
        hideProgressBar();
        setStatus('error');
        throw new Error('Could not fetch result: ' + rresp.status);
      }
      const result = await rresp.json();

      // Now we have final data — clear demo and render fresh
      clearDemoSample();

      hideProgressBar();
      setStatus('Processing complete');

      await handleProcessResponse(result);

    } catch (err) {
      console.error('Upload pipeline error', err);
      alert('Error: ' + (err.message || err));
      setStatus('error');
      stopSimulatedProgress();
      hideProgressBar();
    } finally {
      try {
        const stillHasFile = !!(fileInput && fileInput.files && fileInput.files.length) || !!recordedBlob;
        if (uploadBtn) uploadBtn.disabled = !stillHasFile;
      } catch (e){}
    }
  });
}

/* Initially set upload button disabled if no file present */
try {
  if (uploadBtn) {
    const hasInitial = !!(fileInput && fileInput.files && fileInput.files.length) || !!recordedBlob;
    uploadBtn.disabled = !hasInitial;
  }
} catch (e) {}

/* ---------- handleProcessResponse (dialogue + durations + pie + analysis) ---------- */
async function handleProcessResponse(data){
  // clear demo now happens in upload handler just before calling this; still safe to ensure
  try { if (dialogueEl && dialogueEl.dataset && dialogueEl.dataset.demoInserted) { /* already cleared */ } } catch (e){}

  // Attempt to extract dialogue (same robust approach you had)
  let dialogue = null;
  try {
    if (Array.isArray(data.dialogue)) dialogue = data.dialogue;

    if (!dialogue && data.files && data.files.dialogue_named) {
      const url = resultFileUrl(data.files.dialogue_named);
      if (url) {
        try {
          const resp = await fetch(url);
          if (resp.ok) {
            const j = await resp.json();
            if (Array.isArray(j)) dialogue = j;
            else if (Array.isArray(j.dialogue)) dialogue = j.dialogue;
            else if (Array.isArray(j.segments)) dialogue = j.segments;
            else {
              const maybeArray = Object.values(j).find(v => Array.isArray(v));
              if (maybeArray) dialogue = maybeArray;
            }
          } else console.warn('Failed to fetch dialogue file', url, resp.status);
        } catch (e) { console.warn('Error fetching dialogue_named file', e); }
      }
    }

    if (!dialogue) {
      const cleanCsv = data.stages && data.stages.analysis && data.stages.analysis.clean_transcript_csv;
      if (cleanCsv) {
        const fname = String(cleanCsv).split('/').pop();
        const csvUrl = '/results/' + fname;
        try {
          const txt = await fetch(csvUrl).then(r => { if(!r.ok) throw new Error('csv fetch failed'); return r.text(); });
          dialogue = parseCleanTranscriptCSV(txt);
        } catch (err) { console.warn('Failed to fetch/parse clean CSV, falling back:', err); }
      }
    }

    if (!dialogue) {
      try { dialogue = data.stages && data.stages.analysis && data.stages.analysis.dialogue; } catch(e){}
      if (!dialogue) {
        const asrSegs = (data.stages && data.stages.asr && data.stages.asr.segments) || [];
        if (Array.isArray(asrSegs) && asrSegs.length > 0) {
          dialogue = asrSegs.map(s=>({
            start: Number(s.start||0),
            end: Number(s.end||s.start||0),
            text: s.text || '',
            speaker: s.speaker || 'unknown'
          }));
        }
      }
    }
  } catch (e){
    console.warn('handleProcessResponse failed to extract dialogue:', e);
  }

  if (!dialogue) {
    lastDialogue = [];
    renderDialogue([]);
  } else {
    lastDialogue = dialogue;
    renderDialogue(lastDialogue);
  }

  // DURATIONS
  let durations = null;
  try {
    if (data.files && data.files.durations) {
      const durl = resultFileUrl(data.files.durations);
      if (durl) {
        const resp = await fetch(durl);
        if (resp.ok) durations = await resp.json();
      }
    }
    if (!durations && data.files && data.files.dialogue_named) {
      const dp = data.files.dialogue_named;
      const parts = String(dp).split(/[\\/]/);
      const basename = parts[parts.length - 1] || '';
      const stem = basename.replace(/_dialogue_named\.json$/,'');
      if (stem) {
        const durUrl = `/results/${stem}_speaking_durations.json`;
        try {
          const resp = await fetch(durUrl);
          if (resp.ok) durations = await resp.json();
        } catch (e){}
      }
    }
    if (!durations) {
      const maybe = data.stages && data.stages.analysis && data.stages.analysis.durations;
      if (maybe) durations = maybe;
    }
    if (!durations && Array.isArray(lastDialogue) && lastDialogue.length>0) {
      const dmap = {};
      for(const seg of lastDialogue){
        const sp = seg.speaker || seg.speaker_name || 'unknown';
        const s = Number(seg.start||0);
        const e = Number(seg.end||s);
        dmap[sp] = (dmap[sp]||0) + Math.max(0, e - s);
      }
      if (Object.values(dmap).some(v => v > 0)) durations = dmap;
    }
  } catch (e){ console.warn('failed to get durations', e); }

  // show pie
  await showPieFromResult(data);

  // fetch and render analysis (non-blocking)
  fetchAndRenderAnalysis(data).catch(e => console.warn('analysis render failed', e));
}

/* ---------- Minutes generation, copy & download ---------- */
function generateMinutesText(dialogue, info){
  let out = '1) Dialogue\n\n';
  (dialogue || []).forEach(d => {
    const s = (d.start||0).toFixed(2);
    const e = (d.end||0).toFixed(2);
    const sp = d.speaker || d.speaker_name || 'unknown';
    out += `${s}–${e} — [${sp}] ${d.text || d.transcript || ''}\n`;
  });
  out += '\n2) Summary\n\nAuto-generated (see dialogue above).\n\n3) Durations\n\n';
  const durations = (info && info.durations) || (info && info.raw && info.raw.stages && info.raw.stages.analysis && info.raw.stages.analysis.durations);
  if(durations){
    Object.entries(durations).forEach(([k,v]) => { out += `${k}: ${(Number(v)||0).toFixed(1)}s\n`; });
  } else { out += 'No duration data.\n'; }
  out += `\nRaw response note: ${(info && info.raw && info.raw.note) || ''}\n`;
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

/* ---------- Init demo on DOMContentLoaded (keeps it until replaced) ---------- */
document.addEventListener('DOMContentLoaded', () => {
  const noDialogueYet = dialogueEl && dialogueEl.children.length === 0;
  const pieAlreadyShown = pieImg && pieImg.src && !pieImg.src.includes('demo.png');
  if (noDialogueYet && !pieAlreadyShown) {
    insertDemoSample();
  } else {
    if (pieCanvas) pieCanvas.style.display = 'none';
    if (legendEl) { legendEl.style.display = 'none'; legendEl.innerHTML = ''; }
    if (pieImg && pieImg.src && pieImg.src.includes('demo.png')) pieImg.style.display = 'block';
  }
});
window.addEventListener('load', () => { document.body.classList.add('loaded'); });

/* ---------- End of script.js ---------- */
