let sessionId = generateUUID();
let messages = [];
let selectedModel = 'Qwen 2.5 0.5B';
let isGenerating = false;
let currentReader = null;
let currentAbortController = null;

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
    await loadModels();
    await loadConversation();
    await loadConversationList();
    setupEventListeners();
}

async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        document.getElementById('persistence-type').textContent = 
            `Using ${data.persistence_type} persistence`;
        
        const modelList = document.getElementById('model-list');
        modelList.innerHTML = '';
        
        Object.entries(data.models).forEach(([name, info]) => {
            const modelDiv = document.createElement('div');
            modelDiv.className = `model-option ${name === selectedModel ? 'selected' : ''}`;
            modelDiv.innerHTML = `
                <h4>${name}</h4>
                <div class="model-specs">${info.params} params | ${info.quantization} | ${info.memory}</div>
                <div class="model-description">${info.description}</div>
            `;
            modelDiv.onclick = (evt) => selectModel(name, evt);
            modelList.appendChild(modelDiv);
        });
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

function selectModel(name, evt) {
    selectedModel = name;
    
    document.querySelectorAll('.model-option').forEach(el => {
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
        
        data.conversations.slice(0, 10).forEach(conv => {
            const item = document.createElement('div');
            item.className = 'conversation-item';
            
            const btn = document.createElement('button');
            btn.className = `conversation-btn ${conv.session_id === sessionId ? 'active' : ''}`;
            btn.textContent = `${conv.message_count} messages`;
            btn.onclick = () => loadConversationById(conv.session_id);
            
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'delete-btn';
            deleteBtn.textContent = 'Delete';
            deleteBtn.onclick = () => deleteConversation(conv.session_id);
            
            item.appendChild(btn);
            item.appendChild(deleteBtn);
            conversationList.appendChild(item);
        });
        
        if (data.conversations.length > 10) {
            const more = document.createElement('small');
            more.style.color = '#888';
            more.textContent = `Showing 10 of ${data.conversations.length} conversations`;
            conversationList.appendChild(more);
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
                    <button class="example-prompt" data-prompt="Explain quantum computing in simple terms">Explain quantum computing in simple terms</button>
                    <button class="example-prompt" data-prompt="Write a short poem about artificial intelligence">Write a short poem about artificial intelligence</button>
                    <button class="example-prompt" data-prompt="What are the benefits of 1-bit LLMs?">What are the benefits of 1-bit LLMs?</button>
                    <button class="example-prompt" data-prompt="Tell me an interesting fact about space">Tell me an interesting fact about space</button>
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
                
                metricsHtml = `
                    <div class="message-metrics">
                        Time: ${msg.metrics.time.toFixed(1)}s | 
                        Tokens: ${msg.metrics.tokens} | 
                        Speed: ${speedDisplay}
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
    if (!userMessage.trim() || isGenerating) return;
    
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
        <div class="message-content" id="streaming-content"></div>
    `;
    chatMessages.appendChild(assistantDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    let fullResponse = '';
    let wasAborted = false;
    const startTime = Date.now();
    
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
                            
                            if (data.text) {
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
            const generationTime = (endTime - startTime) / 1000;
            
            const tokens = fullResponse.split(/\s+/).length;
            const tokensPerSecond = tokens / generationTime;
            
            const metrics = {
                time: generationTime,
                tokens: tokens,
                tokens_per_sec: tokensPerSecond
            };
            
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
                    const generationTime = (endTime - startTime) / 1000;
                    const tokens = fullResponse.split(/\s+/).length;
                    const tokensPerSecond = tokens / generationTime;
                    
                    messages.push({
                        role: 'assistant',
                        content: fullResponse,
                        metrics: {
                            time: generationTime,
                            tokens: tokens,
                            tokens_per_sec: tokensPerSecond
                        }
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
}

document.addEventListener('DOMContentLoaded', init);
