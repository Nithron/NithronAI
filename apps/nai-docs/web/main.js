async function checkHealth() {
  const base = document.getElementById('apiBase').value;
  const badge = document.getElementById('health');
  try {
    const res = await fetch(base + '/health');
    if (!res.ok) throw new Error('not ok');
    const j = await res.json();
    badge.textContent = 'online';
    badge.style.background = '#e6ffed';
  } catch (e) {
    badge.textContent = 'offline';
    badge.style.background = '#ffe6e6';
  }
}

async function ingest() {
  const base = document.getElementById('apiBase').value;
  const filesEl = document.getElementById('files');
  const out = document.getElementById('ingestResult');
  const fd = new FormData();
  for (const f of filesEl.files) fd.append('files', f);
  out.textContent = 'Uploading...';
  try {
    const res = await fetch(base + '/ingest', { method: 'POST', body: fd });
    const j = await res.json();
    out.textContent = 'Indexed: ' + JSON.stringify(j.added);
  } catch (e) {
    out.textContent = 'Error: ' + e.message;
  }
}

async function ask() {
  const base = document.getElementById('apiBase').value;
  const q = document.getElementById('question').value;
  const ans = document.getElementById('answer');
  const cit = document.getElementById('citations');
  ans.innerHTML = 'Thinking...';
  cit.innerHTML = '';
  try {
    const res = await fetch(base + '/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q, top_k: 5 })
    });
    const j = await res.json();
    ans.innerHTML = '<pre>' + j.answer + '</pre>';
    const list = (j.citations || []).map(c => `• ${c.doc} #${c.chunk_id}  (score ${c.score.toFixed(2)})`).join('\n');
    cit.innerHTML = '<pre>' + list + '</pre>';
  } catch (e) {
    ans.textContent = 'Error: ' + e.message;
  }
}

document.getElementById('btnIngest').addEventListener('click', ingest);
document.getElementById('btnAsk').addEventListener('click', ask);
document.getElementById('apiBase').addEventListener('change', checkHealth);
checkHealth();
