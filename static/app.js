let sessionId = generateUUID();
let messages = [];
let selectedModel = 'Qwen 2.5 0.5B';
let selectedNERModel = 'BERT Base NER';
let selectedOCREngine = 'easyocr';
let selectedOCRConfig = 'English Only';
let isGenerating = false;
let currentReader = null;
let currentAbortController = null;
let displayedConversationCount = 5;
let displayedNERCount = 5;
let displayedOCRCount = 5;

function closeMobileMenuHelper() {
    if (window.innerWidth <= 900) {
        const sidebar = document.querySelector('.sidebar');
        const backdrop = document.querySelector('.backdrop');
        const hamburgerMenu = document.querySelector('.hamburger-menu');
        
        if (sidebar) sidebar.classList.remove('open');
        if (backdrop) backdrop.classList.remove('active');
        if (hamburgerMenu) hamburgerMenu.setAttribute('aria-expanded', 'false');
    }
}

const settings = {
    temperature: 0.7,
    maxTokens: 150,
    topP: 0.9,
    topK: 50
};

function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

async function init() {
    setupEventListeners();
    setupNERExamples();
    setupOCRExamples();
    closeMobileMenuHelper();
    
    Promise.all([
        loadModels(),
        loadNERModels(),
        loadOCREngines(),
        loadOCRConfigs(),
        loadConversation(),
        loadConversationList(),
        loadNERHistory(),
        loadOCRHistory()
    ]).catch(error => {
        console.error('Error during initialization:', error);
    });
}

async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        const modelList = document.getElementById('model-list');
        modelList.innerHTML = '';
        
        Object.entries(data.models).forEach(([name, info]) => {
            const modelDiv = document.createElement('div');
            modelDiv.className = `model-option ${name === selectedModel ? 'selected' : ''}`;
            modelDiv.innerHTML = `
                <h4>${name}</h4>
                <div class="model-specs">
                    <span class="model-badge">${info.params} params</span>
                    <span class="model-badge">${info.memory}</span>
                </div>
                <div class="model-description">${info.description}</div>
            `;
            modelDiv.onclick = (evt) => selectModel(name, evt);
            modelList.appendChild(modelDiv);
        });
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

async function loadNERModels() {
    try {
        const response = await fetch('/api/ner/models');
        const data = await response.json();
        
        const modelList = document.getElementById('ner-model-list');
        modelList.innerHTML = '';
        
        Object.entries(data.models).forEach(([name, info]) => {
            const modelDiv = document.createElement('div');
            modelDiv.className = `model-option ${name === selectedNERModel ? 'selected' : ''}`;
            modelDiv.innerHTML = `
                <h4>${name}</h4>
                <div class="model-specs">
                    <span class="model-badge">${info.params} params</span>
                    <span class="model-badge">${info.memory}</span>
                </div>
                <div class="model-description">${info.description}</div>
            `;
            modelDiv.onclick = (evt) => selectNERModel(name, evt);
            modelList.appendChild(modelDiv);
        });
    } catch (error) {
        console.error('Error loading NER models:', error);
    }
}

async function loadOCREngines() {
    try {
        const engineSelector = document.getElementById('ocr-engine-selector');
        engineSelector.innerHTML = '';
        
        const engines = {
            'easyocr': {
                name: 'EasyOCR',
                description: 'Fast and accurate OCR with multilingual support'
            },
            'tesseract': {
                name: 'Tesseract',
                description: 'Fast and lightweight OCR engine'
            }
        };
        
        Object.entries(engines).forEach(([key, info]) => {
            const engineDiv = document.createElement('div');
            engineDiv.className = `model-option ${key === selectedOCREngine ? 'selected' : ''}`;
            engineDiv.innerHTML = `
                <h4>${info.name}</h4>
                <div class="model-description">${info.description}</div>
            `;
            engineDiv.onclick = (evt) => selectOCREngine(key, evt);
            engineSelector.appendChild(engineDiv);
        });
        
        updateOCRConfigVisibility();
    } catch (error) {
        console.error('Error loading OCR engines:', error);
    }
}

async function loadOCRConfigs() {
    try {
        const response = await fetch('/api/ocr/configs');
        const data = await response.json();
        
        const configList = document.getElementById('ocr-config-list');
        configList.innerHTML = '';
        
        Object.entries(data.configs).forEach(([name, info]) => {
            const configDiv = document.createElement('div');
            configDiv.className = `model-option ${name === selectedOCRConfig ? 'selected' : ''}`;
            configDiv.innerHTML = `
                <h4>${name}</h4>
                <div class="model-description">${info.description}</div>
                <div class="model-specs">
                    ${info.languages.map(lang => `<span class="model-badge">${lang}</span>`).join('')}
                </div>
            `;
            configDiv.onclick = (evt) => selectOCRConfig(name, evt);
            configList.appendChild(configDiv);
        });
    } catch (error) {
        console.error('Error loading OCR configs:', error);
    }
}

function selectModel(name, evt) {
    selectedModel = name;
    
    document.querySelectorAll('#model-list .model-option').forEach(el => {
        el.classList.remove('selected');
    });
    
    evt.target.closest('.model-option').classList.add('selected');
    
    setTimeout(() => {
        closeMobileMenuHelper();
    }, 100);
}

function selectNERModel(name, evt) {
    selectedNERModel = name;
    
    document.querySelectorAll('#ner-model-list .model-option').forEach(el => {
        el.classList.remove('selected');
    });
    
    evt.target.closest('.model-option').classList.add('selected');
    
    setTimeout(() => {
        closeMobileMenuHelper();
    }, 100);
}

function selectOCREngine(engine, evt) {
    selectedOCREngine = engine;
    
    document.querySelectorAll('#ocr-engine-selector .model-option').forEach(el => {
        el.classList.remove('selected');
    });
    
    evt.target.closest('.model-option').classList.add('selected');
    
    updateOCRConfigVisibility();
    
    setTimeout(() => {
        closeMobileMenuHelper();
    }, 300);
}

function updateOCRConfigVisibility() {
    const configSection = document.getElementById('ocr-config-section');
    if (!configSection) {
        console.warn('ocr-config-section element not found');
        return;
    }
    if (selectedOCREngine === 'easyocr') {
        configSection.style.display = 'block';
    } else {
        configSection.style.display = 'none';
    }
}

function selectOCRConfig(name, evt) {
    selectedOCRConfig = name;
    
    document.querySelectorAll('#ocr-config-list .model-option').forEach(el => {
        el.classList.remove('selected');
    });
    
    evt.target.closest('.model-option').classList.add('selected');
    
    setTimeout(() => {
        closeMobileMenuHelper();
    }, 100);
}

async function loadConversation() {
    try {
        const response = await fetch(`/api/conversations/${sessionId}`);
        const data = await response.json();
        messages = data.messages || [];
        renderMessages();
    } catch (error) {
        console.error('Error loading conversation:', error);
    }
}

async function loadConversationList() {
    try {
        const response = await fetch('/api/conversations');
        const data = await response.json();
        
        const conversationList = document.getElementById('conversation-list');
        conversationList.innerHTML = '';
        
        if (!data.conversations || data.conversations.length === 0) {
            conversationList.innerHTML = '<small style="color: #888;">No saved conversations yet</small>';
            return;
        }
        
        conversationList.innerHTML = `<small style="color: #888; display: block; margin-bottom: 15px;">Found ${data.conversations.length} saved conversation(s)</small>`;
        
        data.conversations.slice(0, displayedConversationCount).forEach(conv => {
            const item = document.createElement('div');
            item.className = 'conversation-item';
            
            const btn = document.createElement('button');
            btn.className = `conversation-btn ${conv.session_id === sessionId ? 'active' : ''}`;
            btn.innerHTML = `
                <div style="font-size: 13px; margin-bottom: 4px;">${conv.first_message || 'No messages'}</div>
                <div style="font-size: 11px; color: #888;">${conv.message_count} messages</div>
            `;
            btn.onclick = () => loadConversationById(conv.session_id);
            
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'delete-btn';
            deleteBtn.textContent = 'Delete';
            deleteBtn.onclick = () => deleteConversation(conv.session_id);
            
            item.appendChild(btn);
            item.appendChild(deleteBtn);
            conversationList.appendChild(item);
        });
        
        if (data.conversations.length > displayedConversationCount) {
            const showMoreBtn = document.createElement('button');
            showMoreBtn.className = 'btn btn-secondary show-more-btn';
            showMoreBtn.textContent = `Show More (${data.conversations.length - displayedConversationCount} remaining)`;
            showMoreBtn.onclick = () => {
                displayedConversationCount += 5;
                loadConversationList();
            };
            conversationList.appendChild(showMoreBtn);
        }
    } catch (error) {
        console.error('Error loading conversation list:', error);
    }
}

async function loadConversationById(convSessionId) {
    sessionId = convSessionId;
    await loadConversation();
    await loadConversationList();
    
    closeMobileMenuHelper();
}

async function deleteConversation(convSessionId) {
    try {
        await fetch(`/api/conversations/${convSessionId}`, {
            method: 'DELETE'
        });
        
        if (convSessionId === sessionId) {
            sessionId = generateUUID();
            messages = [];
            renderMessages();
        }
        
        await loadConversationList();
    } catch (error) {
        console.error('Error deleting conversation:', error);
    }
}

async function saveConversation() {
    try {
        await fetch('/api/conversations/save', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: sessionId,
                messages: messages
            })
        });
    } catch (error) {
        console.error('Error saving conversation:', error);
    }
}

function renderMessages() {
    const chatMessages = document.getElementById('chat-messages');
    chatMessages.innerHTML = '';
    
    if (messages.length === 0) {
        const examplePromptsHtml = `
            <div id="example-prompts" class="example-prompts">
                <div class="info-message">Welcome! Try one of these example prompts:</div>
                <div class="example-grid">
                    <button class="example-prompt" data-prompt="Classify this cargo: 20 containers of refrigerated pharmaceuticals requiring temperature control at 2-8°C. Categories: [Standard Cargo, Refrigerated Cargo, Hazardous Materials, Oversized Freight]">Classify cargo type</button>
                    <button class="example-prompt" data-prompt="Determine priority level: Shipment of automotive parts needed for assembly line restart. Customer reports production halted. Categories: [Routine, Standard, Urgent, Critical Emergency]">Classify shipment urgency</button>
                    <button class="example-prompt" data-prompt="Classify route optimization: Container ship with 5000 TEU capacity, fuel consumption 200 tons/day, scheduled ports: Singapore-Rotterdam-New York. Categories: [Direct Route, Hub-and-Spoke, Transhipment, Feeder Service]">Determine optimal routing</button>
                    <button class="example-prompt" data-prompt="Classify handling requirements: Shipment contains heavy machinery parts, individual pieces up to 15 tons, requires crane access. Categories: [Standard Handling, Heavy Lift, Breakbulk, Project Cargo]">Identify handling needs</button>
                </div>
            </div>
        `;
        chatMessages.innerHTML = examplePromptsHtml;
        
        document.querySelectorAll('.example-prompt').forEach(btn => {
            btn.addEventListener('click', () => {
                const prompt = btn.getAttribute('data-prompt');
                sendMessage(prompt);
            });
        });
    } else {
        messages.forEach(msg => {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${msg.role}`;
            
            let metricsHtml = '';
            if (msg.role === 'assistant' && msg.metrics) {
                const tokensPerSec = msg.metrics.tokens_per_sec || 0;
                let speedDisplay = 'N/A';
                
                if (tokensPerSec > 0 && tokensPerSec < 1) {
                    speedDisplay = `${(1 / tokensPerSec).toFixed(1)} sec/token`;
                } else if (tokensPerSec >= 1) {
                    speedDisplay = `${tokensPerSec.toFixed(1)} tokens/sec`;
                }
                
                let modelLoadInfo = '';
                if (msg.metrics.model_load_time) {
                    modelLoadInfo = ` | Model load: ${msg.metrics.model_load_time.toFixed(1)}s`;
                }
                
                metricsHtml = `
                    <div class="message-metrics">
                        Inference: ${msg.metrics.time.toFixed(1)}s | 
                        Tokens: ${msg.metrics.tokens} | 
                        Speed: ${speedDisplay}${modelLoadInfo}
                    </div>
                `;
            }
            
            messageDiv.innerHTML = `
                <div class="message-header">${msg.role === 'user' ? 'User' : 'Assistant'}</div>
                <div class="message-content">${escapeHtml(msg.content)}</div>
                ${metricsHtml}
            `;
            
            chatMessages.appendChild(messageDiv);
        });
    }
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function sendMessage(userMessage) {
    if (!userMessage || !userMessage.trim() || isGenerating) return;
    
    isGenerating = true;
    document.getElementById('send-btn').style.display = 'none';
    document.getElementById('stop-btn').style.display = 'inline-block';
    document.getElementById('chat-input').disabled = true;
    
    messages.push({
        role: 'user',
        content: userMessage
    });
    
    renderMessages();
    
    const chatMessages = document.getElementById('chat-messages');
    const assistantDiv = document.createElement('div');
    assistantDiv.className = 'message assistant';
    assistantDiv.innerHTML = `
        <div class="message-header">Assistant <span class="loading"></span></div>
        <div class="message-content" id="streaming-content">&nbsp;</div>
    `;
    chatMessages.appendChild(assistantDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    let fullResponse = '';
    let wasAborted = false;
    const requestStartTime = Date.now();
    let inferenceStartTime = null;
    let modelLoadTime = null;
    let loadingTimerInterval = null;
    let modelLoadStartTime = null;
    
    // Create AbortController for instant cancellation
    currentAbortController = new AbortController();
    
    try {
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model_name: selectedModel,
                messages: messages,
                temperature: settings.temperature,
                max_tokens: settings.maxTokens,
                top_p: settings.topP,
                top_k: settings.topK
            }),
            signal: currentAbortController.signal
        });
        
        currentReader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        while (true) {
            const { done, value } = await currentReader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            
            for (const line of lines) {
                const trimmedLine = line.trim();
                if (trimmedLine.startsWith('data: ')) {
                    const jsonStr = trimmedLine.substring(6).trim();
                    if (jsonStr) {
                        try {
                            const data = JSON.parse(jsonStr);
                            
                            if (data.error) {
                                document.getElementById('streaming-content').textContent = 
                                    `Error: ${data.error}`;
                                break;
                            }
                            
                            if (data.model_loading_start) {
                                // Model loading started, show loading indicator with timer
                                modelLoadStartTime = Date.now();
                                const streamingContent = document.getElementById('streaming-content');
                                
                                // Start timer that updates every 100ms
                                loadingTimerInterval = setInterval(() => {
                                    const elapsed = ((Date.now() - modelLoadStartTime) / 1000).toFixed(1);
                                    streamingContent.innerHTML = `<em style="color: #888;">Loading model... ${elapsed}s</em>`;
                                }, 100);
                            }
                            
                            if (data.model_loading_end) {
                                // Model loading completed, stop timer
                                if (loadingTimerInterval) {
                                    clearInterval(loadingTimerInterval);
                                    loadingTimerInterval = null;
                                }
                                modelLoadTime = data.load_time;
                                inferenceStartTime = Date.now();
                                // Clear the loading message
                                document.getElementById('streaming-content').innerHTML = '&nbsp;';
                            }
                            
                            if (data.text) {
                                // Start tracking inference if not already started
                                if (inferenceStartTime === null) {
                                    inferenceStartTime = Date.now();
                                }
                                fullResponse += data.text;
                                document.getElementById('streaming-content').textContent = fullResponse;
                                chatMessages.scrollTop = chatMessages.scrollHeight;
                            }
                        } catch (e) {
                            console.error('Error parsing SSE data:', e, 'Line:', jsonStr);
                        }
                    }
                }
            }
        }
        
        // Remove loading spinner
        const loadingSpinner = assistantDiv.querySelector('.loading');
        if (loadingSpinner) {
            loadingSpinner.remove();
        }
        
        // Only save if we have a response
        if (fullResponse.trim()) {
            const endTime = Date.now();
            // Use inference time only (excluding model load time)
            const inferenceTime = inferenceStartTime 
                ? (endTime - inferenceStartTime) / 1000 
                : (endTime - requestStartTime) / 1000;
            
            const tokens = fullResponse.split(/\s+/).length;
            const tokensPerSecond = tokens / inferenceTime;
            
            const metrics = {
                time: inferenceTime,
                tokens: tokens,
                tokens_per_sec: tokensPerSecond
            };
            
            // Add model load time to metrics if available
            if (modelLoadTime !== null) {
                metrics.model_load_time = modelLoadTime;
            }
            
            messages.push({
                role: 'assistant',
                content: fullResponse,
                metrics: metrics
            });
            
            await saveConversation();
            await loadConversationList();
            renderMessages();
        } else {
            // No response, remove the empty assistant div
            assistantDiv.remove();
        }
        
    } catch (error) {
        console.error('Error sending message:', error);
        wasAborted = error.name === 'AbortError';
        
        // Remove loading spinner
        const loadingSpinner = assistantDiv.querySelector('.loading');
        if (loadingSpinner) {
            loadingSpinner.remove();
        }
        
        const streamingContent = document.getElementById('streaming-content');
        if (streamingContent) {
            if (wasAborted) {
                streamingContent.textContent = 'Generation stopped by user';
                // If we have partial response, save it
                if (fullResponse.trim()) {
                    const endTime = Date.now();
                    // Use inference time only (excluding model load time)
                    const inferenceTime = inferenceStartTime 
                        ? (endTime - inferenceStartTime) / 1000 
                        : (endTime - requestStartTime) / 1000;
                    const tokens = fullResponse.split(/\s+/).length;
                    const tokensPerSecond = tokens / inferenceTime;
                    
                    const metricsAborted = {
                        time: inferenceTime,
                        tokens: tokens,
                        tokens_per_sec: tokensPerSecond
                    };
                    
                    // Add model load time if available
                    if (modelLoadTime !== null) {
                        metricsAborted.model_load_time = modelLoadTime;
                    }
                    
                    messages.push({
                        role: 'assistant',
                        content: fullResponse,
                        metrics: metricsAborted
                    });
                    
                    await saveConversation();
                    await loadConversationList();
                    renderMessages();
                } else {
                    // No partial response, just remove the div after a moment
                    setTimeout(() => assistantDiv.remove(), 1500);
                }
            } else {
                streamingContent.textContent = `Error: ${error.message}`;
            }
        }
    } finally {
        // Clean up loading timer if still running
        if (loadingTimerInterval) {
            clearInterval(loadingTimerInterval);
            loadingTimerInterval = null;
        }
        
        currentReader = null;
        currentAbortController = null;
        isGenerating = false;
        document.getElementById('send-btn').style.display = 'inline-block';
        document.getElementById('stop-btn').style.display = 'none';
        document.getElementById('chat-input').disabled = false;
        document.getElementById('chat-input').value = '';
    }
}

function stopGeneration() {
    // Abort the fetch request immediately
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }
    // Also cancel the reader if it exists
    if (currentReader) {
        currentReader.cancel();
        currentReader = null;
    }
}

function setupEventListeners() {
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage(chatInput.value);
        }
    });
    
    sendBtn.addEventListener('click', () => {
        sendMessage(chatInput.value);
    });
    
    document.getElementById('stop-btn').addEventListener('click', () => {
        stopGeneration();
    });
    
    document.getElementById('clear-chat-btn').addEventListener('click', async () => {
        messages = [];
        renderMessages();
        await saveConversation();
        await loadConversationList();
    });
    
    document.getElementById('new-conversation-btn').addEventListener('click', () => {
        sessionId = generateUUID();
        messages = [];
        renderMessages();
        loadConversationList();
        closeMobileMenuHelper();
    });
    
    const hamburgerMenu = document.querySelector('.hamburger-menu');
    const closeSidebar = document.querySelector('.close-sidebar');
    const backdrop = document.querySelector('.backdrop');
    const sidebar = document.querySelector('.sidebar');
    
    function openMobileMenu() {
        if (!sidebar || !backdrop || !hamburgerMenu) return;
        sidebar.classList.add('open');
        backdrop.classList.add('active');
        hamburgerMenu.setAttribute('aria-expanded', 'true');
    }
    
    function closeMobileMenu() {
        if (!sidebar || !backdrop || !hamburgerMenu) return;
        sidebar.classList.remove('open');
        backdrop.classList.remove('active');
        hamburgerMenu.setAttribute('aria-expanded', 'false');
    }
    
    if (hamburgerMenu) {
        hamburgerMenu.addEventListener('click', openMobileMenu);
    }
    if (closeSidebar) {
        closeSidebar.addEventListener('click', closeMobileMenu);
    }
    if (backdrop) {
        backdrop.addEventListener('click', closeMobileMenu);
    }
    
    window.addEventListener('resize', () => {
        if (window.innerWidth > 900) {
            closeMobileMenu();
        }
    });
    
    const temperatureSlider = document.getElementById('temperature-slider');
    const temperatureValue = document.getElementById('temperature-value');
    temperatureSlider.addEventListener('input', (e) => {
        settings.temperature = parseFloat(e.target.value);
        temperatureValue.textContent = settings.temperature.toFixed(1);
    });
    
    const maxTokensSlider = document.getElementById('max-tokens-slider');
    const maxTokensValue = document.getElementById('max-tokens-value');
    maxTokensSlider.addEventListener('input', (e) => {
        settings.maxTokens = parseInt(e.target.value);
        maxTokensValue.textContent = settings.maxTokens;
    });
    
    const topPSlider = document.getElementById('top-p-slider');
    const topPValue = document.getElementById('top-p-value');
    topPSlider.addEventListener('input', (e) => {
        settings.topP = parseFloat(e.target.value);
        topPValue.textContent = settings.topP.toFixed(2);
    });
    
    const topKSlider = document.getElementById('top-k-slider');
    const topKValue = document.getElementById('top-k-value');
    topKSlider.addEventListener('input', (e) => {
        settings.topK = parseInt(e.target.value);
        topKValue.textContent = settings.topK;
    });
    
    setupTabs();
    setupNER();
    setupOCR();
}

function setupTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const sidebarContents = document.querySelectorAll('.sidebar-content');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.dataset.tab;
            
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            sidebarContents.forEach(s => s.style.display = 'none');
            
            btn.classList.add('active');
            document.getElementById(`${targetTab}-tab`).classList.add('active');
            document.getElementById(`${targetTab}-sidebar`).style.display = 'block';
            
            if (targetTab === 'ner') {
                loadNERHistory();
            } else if (targetTab === 'ocr') {
                loadOCRHistory();
            }
        });
    });
}

function setupNER() {
    const submitBtn = document.getElementById('ner-submit-btn');
    const textInput = document.getElementById('ner-text-input');
    
    submitBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        if (!text) {
            alert('Please enter some text to analyze');
            return;
        }
        
        submitBtn.disabled = true;
        submitBtn.textContent = 'Extracting...';
        
        let loadingTimerInterval = null;
        let modelLoadStartTime = null;
        
        try {
            const response = await fetch('/api/ner', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text, model: selectedNERModel })
            });
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let finalData = null;
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                
                for (const line of lines) {
                    const trimmedLine = line.trim();
                    if (trimmedLine.startsWith('data: ')) {
                        const jsonStr = trimmedLine.substring(6).trim();
                        if (jsonStr) {
                            try {
                                const data = JSON.parse(jsonStr);
                                
                                if (data.error) {
                                    displayNERError(data.error);
                                    break;
                                }
                                
                                if (data.model_loading_start) {
                                    // Show loading indicator
                                    modelLoadStartTime = Date.now();
                                    submitBtn.textContent = 'Loading model...';
                                    
                                    loadingTimerInterval = setInterval(() => {
                                        const elapsed = ((Date.now() - modelLoadStartTime) / 1000).toFixed(1);
                                        submitBtn.textContent = `Loading model... ${elapsed}s`;
                                    }, 100);
                                }
                                
                                if (data.model_loading_end) {
                                    // Stop loading timer
                                    if (loadingTimerInterval) {
                                        clearInterval(loadingTimerInterval);
                                        loadingTimerInterval = null;
                                    }
                                    submitBtn.textContent = 'Extracting...';
                                }
                                
                                if (data.done) {
                                    finalData = data;
                                }
                            } catch (e) {
                                console.error('Error parsing NER SSE data:', e);
                            }
                        }
                    }
                }
            }
            
            if (finalData) {
                displayNERResults(finalData);
            }
        } catch (error) {
            console.error('NER error:', error);
            displayNERError('Network error. Please try again.');
        } finally {
            if (loadingTimerInterval) {
                clearInterval(loadingTimerInterval);
            }
            submitBtn.disabled = false;
            submitBtn.textContent = 'Extract Entities';
        }
    });
}

function displayNERError(message) {
    const resultsDiv = document.getElementById('ner-results');
    const entitiesDiv = document.getElementById('ner-entities');
    const metricsDiv = document.getElementById('ner-metrics');
    
    resultsDiv.style.display = 'block';
    entitiesDiv.innerHTML = `<div style="padding: 15px; background: #fee; border: 2px solid #c33; border-radius: 8px; color: #c33;"><strong>Error:</strong> ${message}</div>`;
    metricsDiv.innerHTML = '';
}

function displayNERResults(data) {
    const resultsDiv = document.getElementById('ner-results');
    const entitiesDiv = document.getElementById('ner-entities');
    const metricsDiv = document.getElementById('ner-metrics');
    
    resultsDiv.style.display = 'block';
    
    if (data.entities && data.entities.length > 0) {
        entitiesDiv.innerHTML = data.entities.map(entity => `
            <div class="entity-tag ${entity.label}">
                <span>${entity.text}</span>
                <span class="entity-label">${entity.label}</span>
            </div>
        `).join('');
        
        metricsDiv.innerHTML = `
            <div><strong>Processing Time:</strong> ${data.processing_time.toFixed(3)}s</div>
            <div><strong>Entities Found:</strong> ${data.entities.length}</div>
            <div><strong>Text Length:</strong> ${data.text_length} characters</div>
        `;
    } else {
        entitiesDiv.innerHTML = '<p>No entities detected in the text.</p>';
        metricsDiv.innerHTML = `
            <div><strong>Processing Time:</strong> ${data.processing_time.toFixed(3)}s</div>
        `;
    }
    
    loadNERHistory();
}

let ocrSelectedFile = null;

function setupOCR() {
    const dropZone = document.getElementById('ocr-drop-zone');
    const fileInput = document.getElementById('ocr-file-input');
    const submitBtn = document.getElementById('ocr-submit-btn');
    const previewDiv = document.getElementById('ocr-preview');
    const previewImg = document.getElementById('ocr-preview-img');
    
    dropZone.addEventListener('click', () => fileInput.click());
    
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
    
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });
    
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleFileSelect(file);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelect(file);
        }
    });
    
    function handleFileSelect(file) {
        ocrSelectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewDiv.style.display = 'block';
            dropZone.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }
    
    function clearOCRImage() {
        ocrSelectedFile = null;
        previewImg.src = '';
        previewDiv.style.display = 'none';
        dropZone.style.display = 'block';
        fileInput.value = '';
        
        const resultsDiv = document.getElementById('ocr-results');
        resultsDiv.style.display = 'none';
    }
    
    const clearBtn = document.getElementById('ocr-clear-btn');
    clearBtn.addEventListener('click', clearOCRImage);
    
    submitBtn.addEventListener('click', async () => {
        if (!ocrSelectedFile) {
            alert('Please select an image first');
            return;
        }
        
        submitBtn.disabled = true;
        const isLayout = selectedOCREngine === 'paddleocr';
        submitBtn.textContent = isLayout ? 'Analyzing...' : 'Extracting...';
        
        let loadingTimerInterval = null;
        let modelLoadStartTime = null;
        
        try {
            const formData = new FormData();
            formData.append('file', ocrSelectedFile);
            
            const endpoint = isLayout ? '/api/layout' : `/api/ocr?config=${encodeURIComponent(selectedOCRConfig)}`;
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let finalData = null;
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                
                for (const line of lines) {
                    const trimmedLine = line.trim();
                    if (trimmedLine.startsWith('data: ')) {
                        const jsonStr = trimmedLine.substring(6).trim();
                        if (jsonStr) {
                            try {
                                const data = JSON.parse(jsonStr);
                                
                                if (data.error) {
                                    displayOCRError(data.error);
                                    break;
                                }
                                
                                if (data.model_loading_start) {
                                    // Show loading indicator
                                    modelLoadStartTime = Date.now();
                                    submitBtn.textContent = 'Loading model...';
                                    
                                    loadingTimerInterval = setInterval(() => {
                                        const elapsed = ((Date.now() - modelLoadStartTime) / 1000).toFixed(1);
                                        submitBtn.textContent = `Loading model... ${elapsed}s`;
                                    }, 100);
                                }
                                
                                if (data.model_loading_end) {
                                    // Stop loading timer
                                    if (loadingTimerInterval) {
                                        clearInterval(loadingTimerInterval);
                                        loadingTimerInterval = null;
                                    }
                                    submitBtn.textContent = isLayout ? 'Analyzing...' : 'Extracting...';
                                }
                                
                                if (data.done) {
                                    finalData = data;
                                }
                            } catch (e) {
                                console.error('Error parsing OCR SSE data:', e);
                            }
                        }
                    }
                }
            }
            
            if (finalData) {
                displayOCRResults(finalData);
                await loadOCRHistory();
            }
        } catch (error) {
            console.error('OCR error:', error);
            displayOCRError('Network error. Please try again.');
        } finally {
            if (loadingTimerInterval) {
                clearInterval(loadingTimerInterval);
            }
            submitBtn.disabled = false;
            submitBtn.textContent = 'Extract Text';
        }
    });
}

function displayOCRError(message) {
    const resultsDiv = document.getElementById('ocr-results');
    const textDiv = document.getElementById('ocr-text');
    const metricsDiv = document.getElementById('ocr-metrics');
    
    resultsDiv.style.display = 'block';
    textDiv.innerHTML = `<div style="padding: 15px; background: #fee; border: 2px solid #c33; border-radius: 8px; color: #c33;"><strong>Error:</strong> ${message}</div>`;
    metricsDiv.innerHTML = '';
}

function displayOCRResults(data) {
    const resultsDiv = document.getElementById('ocr-results');
    const textDiv = document.getElementById('ocr-text');
    const metricsDiv = document.getElementById('ocr-metrics');
    
    resultsDiv.style.display = 'block';
    
    if (data.text) {
        textDiv.textContent = data.text;
        metricsDiv.innerHTML = `
            <div><strong>Processing Time:</strong> ${data.processing_time.toFixed(3)}s</div>
            <div><strong>Text Detections:</strong> ${data.num_detections}</div>
        `;
    } else {
        textDiv.textContent = 'No text detected in the image.';
        metricsDiv.innerHTML = `
            <div><strong>Processing Time:</strong> ${data.processing_time.toFixed(3)}s</div>
        `;
    }
    
}

async function loadNERHistory() {
    try {
        const response = await fetch('/api/ner/history');
        const data = await response.json();
        
        const historyList = document.getElementById('ner-history-list');
        historyList.innerHTML = '';
        
        if (!data.analyses || data.analyses.length === 0) {
            historyList.innerHTML = '<small style="color: #888;">No analyses yet</small>';
            return;
        }
        
        historyList.innerHTML = `<small style="color: #888; display: block; margin-bottom: 15px;">Found ${data.analyses.length} analysis/analyses</small>`;
        
        data.analyses.slice(0, displayedNERCount).forEach(analysis => {
            const item = document.createElement('div');
            item.className = 'conversation-item';
            
            const btn = document.createElement('button');
            btn.className = 'conversation-btn';
            btn.innerHTML = `
                <div style="font-size: 13px; margin-bottom: 4px;">${analysis.text_preview}</div>
                <div style="font-size: 11px; color: #888;">${analysis.entity_count} entities | ${analysis.model}</div>
            `;
            btn.onclick = () => loadNERAnalysis(analysis.id);
            
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'delete-btn';
            deleteBtn.textContent = 'Delete';
            deleteBtn.onclick = (e) => {
                e.stopPropagation();
                deleteNERAnalysis(analysis.id);
            };
            
            item.appendChild(btn);
            item.appendChild(deleteBtn);
            historyList.appendChild(item);
        });
        
        if (data.analyses.length > displayedNERCount) {
            const showMoreBtn = document.createElement('button');
            showMoreBtn.className = 'btn btn-secondary show-more-btn';
            showMoreBtn.textContent = `Show More (${data.analyses.length - displayedNERCount} remaining)`;
            showMoreBtn.onclick = () => {
                displayedNERCount += 5;
                loadNERHistory();
            };
            historyList.appendChild(showMoreBtn);
        }
    } catch (error) {
        console.error('Error loading NER history:', error);
    }
}

async function loadNERAnalysis(nerId) {
    try {
        const response = await fetch(`/api/ner/history/${nerId}`);
        const data = await response.json();
        
        document.getElementById('ner-text-input').value = data.text;
        selectedNERModel = data.model;
        await loadNERModels();
        
        displayNERResults(data);
        
        closeMobileMenuHelper();
    } catch (error) {
        console.error('Error loading NER analysis:', error);
    }
}

async function deleteNERAnalysis(nerId) {
    try {
        await fetch(`/api/ner/history/${nerId}`, {
            method: 'DELETE'
        });
        
        await loadNERHistory();
    } catch (error) {
        console.error('Error deleting NER analysis:', error);
    }
}

async function loadOCRHistory() {
    try {
        const [ocrResponse, layoutResponse] = await Promise.all([
            fetch('/api/ocr/history'),
            fetch('/api/layout/history')
        ]);
        
        const ocrData = await ocrResponse.json();
        const layoutData = await layoutResponse.json();
        
        const historyList = document.getElementById('ocr-history-list');
        historyList.innerHTML = '';
        
        const ocrAnalyses = ocrData.analyses || [];
        const layoutAnalyses = layoutData.analyses || [];
        const allAnalyses = [...ocrAnalyses, ...layoutAnalyses];
        
        if (allAnalyses.length === 0) {
            historyList.innerHTML = '<small style="color: #888;">No extractions yet</small>';
            return;
        }
        
        historyList.innerHTML = `<small style="color: #888; display: block; margin-bottom: 15px;">Found ${allAnalyses.length} extraction(s)</small>`;
        
        const displayedAnalyses = allAnalyses.slice(0, displayedOCRCount);
        
        displayedAnalyses.forEach(analysis => {
            const item = document.createElement('div');
            item.className = 'conversation-item';
            
            const btn = document.createElement('button');
            btn.className = 'conversation-btn';
            
            const isLayout = layoutAnalyses.includes(analysis);
            const configText = isLayout ? 'PaddleOCR' : analysis.config;
            
            btn.innerHTML = `
                <div style="font-size: 13px; margin-bottom: 4px;">${analysis.filename}</div>
                <div style="font-size: 11px; color: #888;">${analysis.num_detections} detections | ${configText}</div>
            `;
            btn.onclick = () => isLayout ? loadLayoutAnalysis(analysis.id) : loadOCRAnalysis(analysis.id);
            
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'delete-btn';
            deleteBtn.textContent = 'Delete';
            deleteBtn.onclick = (e) => {
                e.stopPropagation();
                isLayout ? deleteLayoutAnalysis(analysis.id) : deleteOCRAnalysis(analysis.id);
            };
            
            item.appendChild(btn);
            item.appendChild(deleteBtn);
            historyList.appendChild(item);
        });
        
        if (allAnalyses.length > displayedOCRCount) {
            const showMoreBtn = document.createElement('button');
            showMoreBtn.className = 'btn btn-secondary show-more-btn';
            showMoreBtn.textContent = `Show More (${allAnalyses.length - displayedOCRCount} remaining)`;
            showMoreBtn.onclick = () => {
                displayedOCRCount += 5;
                loadOCRHistory();
            };
            historyList.appendChild(showMoreBtn);
        }
    } catch (error) {
        console.error('Error loading OCR history:', error);
    }
}

async function loadOCRAnalysis(ocrId) {
    try {
        const response = await fetch(`/api/ocr/history/${ocrId}`);
        const data = await response.json();
        
        selectedOCRConfig = data.config;
        await loadOCRConfigs();
        
        displayOCRResults(data);
        
        closeMobileMenuHelper();
    } catch (error) {
        console.error('Error loading OCR analysis:', error);
    }
}

async function deleteOCRAnalysis(ocrId) {
    try {
        await fetch(`/api/ocr/history/${ocrId}`, {
            method: 'DELETE'
        });
        
        await loadOCRHistory();
    } catch (error) {
        console.error('Error deleting OCR analysis:', error);
    }
}

async function loadLayoutAnalysis(layoutId) {
    try {
        const response = await fetch(`/api/layout/history/${layoutId}`);
        const data = await response.json();
        
        selectedOCREngine = 'paddleocr';
        await loadOCREngines();
        
        displayOCRResults(data);
        
        closeMobileMenuHelper();
    } catch (error) {
        console.error('Error loading layout analysis:', error);
    }
}

async function deleteLayoutAnalysis(layoutId) {
    try {
        await fetch(`/api/layout/history/${layoutId}`, {
            method: 'DELETE'
        });
        
        await loadOCRHistory();
    } catch (error) {
        console.error('Error deleting layout analysis:', error);
    }
}

function setupNERExamples() {
    const exampleBtns = document.querySelectorAll('[data-ner-text]');
    exampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const text = btn.dataset.nerText;
            document.getElementById('ner-text-input').value = text;
            document.getElementById('ner-submit-btn').click();
        });
    });
}

async function setupOCRExamples() {
    const exampleBtns = document.querySelectorAll('[data-ocr-image]');
    exampleBtns.forEach(btn => {
        btn.addEventListener('click', async () => {
            const imageUrl = btn.dataset.ocrImage;
            try {
                const response = await fetch(imageUrl);
                const blob = await response.blob();
                const file = new File([blob], imageUrl.split('/').pop(), { type: blob.type });
                
                ocrSelectedFile = file;
                const reader = new FileReader();
                reader.onload = (e) => {
                    document.getElementById('ocr-preview-img').src = e.target.result;
                    document.getElementById('ocr-preview').style.display = 'block';
                    document.getElementById('ocr-drop-zone').style.display = 'none';
                    document.getElementById('ocr-submit-btn').click();
                };
                reader.readAsDataURL(file);
            } catch (error) {
                console.error('Error loading sample image:', error);
            }
        });
    });
}

document.addEventListener('DOMContentLoaded', init);
