/**
 * nAI Web Interface
 * Modern client-side JavaScript for the nAI document Q&A system
 */

// =============================================================================
// Configuration & State
// =============================================================================

const CONFIG = {
  apiBase: localStorage.getItem('nai_api_base') || 'http://localhost:8000',
  theme: localStorage.getItem('nai_theme') || 'dark',
};

const STATE = {
  files: [],
  documents: [],
  chatConversationId: null,
  isLoading: false,
};

// =============================================================================
// DOM Elements
// =============================================================================

const elements = {
  // Status
  statusIndicator: document.getElementById('statusIndicator'),
  
  // Tabs
  tabs: document.querySelectorAll('.tab'),
  tabPanels: document.querySelectorAll('.tab-panel'),
  
  // Ask
  questionInput: document.getElementById('questionInput'),
  askBtn: document.getElementById('askBtn'),
  useLlm: document.getElementById('useLlm'),
  topK: document.getElementById('topK'),
  resultsArea: document.getElementById('resultsArea'),
  resultMethod: document.getElementById('resultMethod'),
  answerContent: document.getElementById('answerContent'),
  citationsList: document.getElementById('citationsList'),
  
  // Ingest
  uploadZone: document.getElementById('uploadZone'),
  fileInput: document.getElementById('fileInput'),
  browseBtn: document.getElementById('browseBtn'),
  fileList: document.getElementById('fileList'),
  uploadActions: document.getElementById('uploadActions'),
  ingestBtn: document.getElementById('ingestBtn'),
  clearFilesBtn: document.getElementById('clearFilesBtn'),
  ingestResults: document.getElementById('ingestResults'),
  
  // Documents
  documentsStats: document.getElementById('documentsStats'),
  documentsList: document.getElementById('documentsList'),
  refreshDocsBtn: document.getElementById('refreshDocsBtn'),
  
  // Chat
  chatMessages: document.getElementById('chatMessages'),
  chatInput: document.getElementById('chatInput'),
  chatSendBtn: document.getElementById('chatSendBtn'),
  
  // Settings
  settingsBtn: document.getElementById('settingsBtn'),
  settingsModal: document.getElementById('settingsModal'),
  closeSettingsBtn: document.getElementById('closeSettingsBtn'),
  cancelSettingsBtn: document.getElementById('cancelSettingsBtn'),
  saveSettingsBtn: document.getElementById('saveSettingsBtn'),
  apiBaseInput: document.getElementById('apiBaseInput'),
  themeSelect: document.getElementById('themeSelect'),
  
  // Toast
  toastContainer: document.getElementById('toastContainer'),
};

// =============================================================================
// Utility Functions
// =============================================================================

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span>${message}</span>
    <button onclick="this.parentElement.remove()">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
      </svg>
    </button>
  `;
  elements.toastContainer.appendChild(toast);
  setTimeout(() => toast.remove(), 5000);
}

function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function formatDate(isoString) {
  if (!isoString) return 'Unknown';
  const date = new Date(isoString);
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

async function apiCall(endpoint, options = {}) {
  const url = `${CONFIG.apiBase}${endpoint}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  
  return response.json();
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function renderMarkdown(text) {
  // Simple markdown rendering
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`(.+?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>');
}

// =============================================================================
// Health Check
// =============================================================================

async function checkHealth() {
  const statusDot = elements.statusIndicator.querySelector('.status-dot');
  const statusText = elements.statusIndicator.querySelector('.status-text');
  
  try {
    const data = await apiCall('/health');
    statusDot.className = 'status-dot online';
    statusText.textContent = `Online (v${data.version})`;
    return true;
  } catch (e) {
    statusDot.className = 'status-dot offline';
    statusText.textContent = 'Offline';
    return false;
  }
}

// =============================================================================
// Tab Navigation
// =============================================================================

function switchTab(tabName) {
  elements.tabs.forEach(tab => {
    tab.classList.toggle('active', tab.dataset.tab === tabName);
  });
  
  elements.tabPanels.forEach(panel => {
    panel.classList.toggle('active', panel.dataset.panel === tabName);
  });
  
  // Load data when switching to certain tabs
  if (tabName === 'documents') {
    loadDocuments();
  }
}

// =============================================================================
// Ask Functionality
// =============================================================================

async function askQuestion() {
  const question = elements.questionInput.value.trim();
  if (!question) {
    showToast('Please enter a question', 'warning');
    return;
  }
  
  STATE.isLoading = true;
  elements.askBtn.disabled = true;
  elements.askBtn.innerHTML = '<span>Thinking...</span>';
  elements.resultsArea.classList.add('hidden');
  
  try {
    const data = await apiCall('/ask', {
      method: 'POST',
      body: JSON.stringify({
        question,
        top_k: parseInt(elements.topK.value),
        use_llm: elements.useLlm.checked,
      }),
    });
    
    // Display results
    elements.resultMethod.textContent = data.method;
    elements.answerContent.innerHTML = `<div class="answer-text">${renderMarkdown(escapeHtml(data.answer))}</div>`;
    
    // Display citations
    elements.citationsList.innerHTML = data.citations.map((cit, i) => `
      <div class="citation-item">
        <div class="citation-rank">${i + 1}</div>
        <div class="citation-content">
          <div class="citation-source">${escapeHtml(cit.doc_path)} (chunk #${cit.chunk_id})</div>
          <div class="citation-text">${escapeHtml(cit.text.slice(0, 300))}${cit.text.length > 300 ? '...' : ''}</div>
          <div class="citation-score">Score: ${cit.score.toFixed(3)}</div>
        </div>
      </div>
    `).join('');
    
    elements.resultsArea.classList.remove('hidden');
  } catch (e) {
    showToast(`Error: ${e.message}`, 'error');
  } finally {
    STATE.isLoading = false;
    elements.askBtn.disabled = false;
    elements.askBtn.innerHTML = '<span>Ask</span><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>';
  }
}

// =============================================================================
// Ingest Functionality
// =============================================================================

function updateFileList() {
  if (STATE.files.length === 0) {
    elements.fileList.innerHTML = '';
    elements.uploadActions.classList.add('hidden');
    return;
  }
  
  elements.fileList.innerHTML = STATE.files.map((file, i) => `
    <div class="file-item">
      <svg class="file-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
      </svg>
      <span class="file-name">${escapeHtml(file.name)}</span>
      <span class="file-size">${formatBytes(file.size)}</span>
      <button class="file-remove" data-index="${i}">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
    </div>
  `).join('');
  
  elements.uploadActions.classList.remove('hidden');
  
  // Add remove handlers
  elements.fileList.querySelectorAll('.file-remove').forEach(btn => {
    btn.addEventListener('click', () => {
      STATE.files.splice(parseInt(btn.dataset.index), 1);
      updateFileList();
    });
  });
}

function handleFiles(files) {
  STATE.files.push(...Array.from(files));
  updateFileList();
}

async function ingestFiles() {
  if (STATE.files.length === 0) {
    showToast('No files selected', 'warning');
    return;
  }
  
  elements.ingestBtn.disabled = true;
  elements.ingestBtn.innerHTML = '<span>Uploading...</span>';
  
  const formData = new FormData();
  STATE.files.forEach(file => formData.append('files', file));
  
  try {
    const response = await fetch(`${CONFIG.apiBase}/ingest`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    const data = await response.json();
    
    // Show results
    elements.ingestResults.innerHTML = `
      <div class="card">
        <h3>Ingest Results</h3>
        <p>Successfully indexed ${data.total_files} file(s) with ${data.total_chunks} chunks.</p>
        ${data.added.map(item => `
          <div class="ingest-result-item ${item.status}">
            <strong>${escapeHtml(item.filename)}</strong>
            <span>${item.chunks} chunks</span>
            <span class="status-badge">${item.status}</span>
            ${item.message ? `<small>${escapeHtml(item.message)}</small>` : ''}
          </div>
        `).join('')}
      </div>
    `;
    elements.ingestResults.classList.remove('hidden');
    
    // Clear files
    STATE.files = [];
    updateFileList();
    
    showToast(`Indexed ${data.total_files} file(s)`, 'success');
  } catch (e) {
    showToast(`Upload failed: ${e.message}`, 'error');
  } finally {
    elements.ingestBtn.disabled = false;
    elements.ingestBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>Upload & Index';
  }
}

// =============================================================================
// Documents Functionality
// =============================================================================

async function loadDocuments() {
  try {
    const data = await apiCall('/documents');
    STATE.documents = data.documents;
    
    // Stats
    const totalChunks = data.documents.reduce((sum, d) => sum + d.chunk_count, 0);
    elements.documentsStats.innerHTML = `
      <div class="stat-card">
        <div class="stat-value">${data.total}</div>
        <div class="stat-label">Documents</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${totalChunks}</div>
        <div class="stat-label">Chunks</div>
      </div>
    `;
    
    // Document list
    if (data.documents.length === 0) {
      elements.documentsList.innerHTML = `
        <div class="empty-state">
          <p>No documents indexed yet. Upload some files to get started.</p>
        </div>
      `;
      return;
    }
    
    elements.documentsList.innerHTML = data.documents.map(doc => `
      <div class="document-item">
        <div class="document-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
        </div>
        <div class="document-info">
          <div class="document-name">${escapeHtml(doc.filename)}</div>
          <div class="document-meta">
            <span>${doc.chunk_count} chunks</span>
            <span>${formatBytes(doc.total_chars)}</span>
            <span>${formatDate(doc.indexed_at)}</span>
          </div>
        </div>
        <div class="document-actions">
          <button class="btn btn-ghost btn-sm" data-action="delete" data-id="${doc.doc_id}">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
            </svg>
          </button>
        </div>
      </div>
    `).join('');
    
    // Add delete handlers
    elements.documentsList.querySelectorAll('[data-action="delete"]').forEach(btn => {
      btn.addEventListener('click', () => deleteDocument(btn.dataset.id));
    });
    
  } catch (e) {
    showToast(`Failed to load documents: ${e.message}`, 'error');
  }
}

async function deleteDocument(docId) {
  if (!confirm('Are you sure you want to delete this document?')) return;
  
  try {
    await apiCall(`/documents/${docId}`, { method: 'DELETE' });
    showToast('Document deleted', 'success');
    loadDocuments();
  } catch (e) {
    showToast(`Delete failed: ${e.message}`, 'error');
  }
}

// =============================================================================
// Chat Functionality
// =============================================================================

function addChatMessage(role, content) {
  const welcome = elements.chatMessages.querySelector('.chat-welcome');
  if (welcome) welcome.remove();
  
  const avatar = role === 'user' ? '👤' : '🤖';
  const messageEl = document.createElement('div');
  messageEl.className = `chat-message ${role}`;
  messageEl.innerHTML = `
    <div class="chat-avatar">${avatar}</div>
    <div class="chat-bubble">${renderMarkdown(escapeHtml(content))}</div>
  `;
  elements.chatMessages.appendChild(messageEl);
  elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

async function sendChatMessage() {
  const message = elements.chatInput.value.trim();
  if (!message) return;
  
  addChatMessage('user', message);
  elements.chatInput.value = '';
  elements.chatSendBtn.disabled = true;
  
  try {
    const data = await apiCall('/chat', {
      method: 'POST',
      body: JSON.stringify({
        messages: [{ role: 'user', content: message }],
        top_k: 5,
        use_context: true,
      }),
    });
    
    STATE.chatConversationId = data.conversation_id;
    addChatMessage('assistant', data.message.content);
    
  } catch (e) {
    addChatMessage('assistant', `Error: ${e.message}`);
  } finally {
    elements.chatSendBtn.disabled = false;
  }
}

// =============================================================================
// Settings
// =============================================================================

function openSettings() {
  elements.apiBaseInput.value = CONFIG.apiBase;
  elements.themeSelect.value = CONFIG.theme;
  elements.settingsModal.classList.remove('hidden');
}

function closeSettings() {
  elements.settingsModal.classList.add('hidden');
}

function saveSettings() {
  CONFIG.apiBase = elements.apiBaseInput.value.trim();
  CONFIG.theme = elements.themeSelect.value;
  
  localStorage.setItem('nai_api_base', CONFIG.apiBase);
  localStorage.setItem('nai_theme', CONFIG.theme);
  
  applyTheme();
  closeSettings();
  checkHealth();
  showToast('Settings saved', 'success');
}

function applyTheme() {
  if (CONFIG.theme === 'system') {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    document.documentElement.dataset.theme = prefersDark ? 'dark' : 'light';
  } else {
    document.documentElement.dataset.theme = CONFIG.theme;
  }
}

// =============================================================================
// Event Listeners
// =============================================================================

// Tabs
elements.tabs.forEach(tab => {
  tab.addEventListener('click', () => switchTab(tab.dataset.tab));
});

// Ask
elements.askBtn.addEventListener('click', askQuestion);
elements.questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    askQuestion();
  }
});

// Ingest
elements.browseBtn.addEventListener('click', () => elements.fileInput.click());
elements.uploadZone.addEventListener('click', (e) => {
  if (e.target === elements.uploadZone || e.target.closest('.upload-icon, .upload-title, .upload-subtitle')) {
    elements.fileInput.click();
  }
});
elements.fileInput.addEventListener('change', (e) => handleFiles(e.target.files));
elements.ingestBtn.addEventListener('click', ingestFiles);
elements.clearFilesBtn.addEventListener('click', () => {
  STATE.files = [];
  updateFileList();
});

// Drag and drop
elements.uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  elements.uploadZone.classList.add('dragover');
});
elements.uploadZone.addEventListener('dragleave', () => {
  elements.uploadZone.classList.remove('dragover');
});
elements.uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  elements.uploadZone.classList.remove('dragover');
  handleFiles(e.dataTransfer.files);
});

// Documents
elements.refreshDocsBtn.addEventListener('click', loadDocuments);

// Chat
elements.chatSendBtn.addEventListener('click', sendChatMessage);
elements.chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendChatMessage();
  }
});
elements.chatInput.addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 150) + 'px';
});

// Settings
elements.settingsBtn.addEventListener('click', openSettings);
elements.closeSettingsBtn.addEventListener('click', closeSettings);
elements.cancelSettingsBtn.addEventListener('click', closeSettings);
elements.saveSettingsBtn.addEventListener('click', saveSettings);
elements.settingsModal.querySelector('.modal-backdrop').addEventListener('click', closeSettings);

// =============================================================================
// Initialization
// =============================================================================

applyTheme();
checkHealth();
setInterval(checkHealth, 30000); // Check health every 30 seconds
