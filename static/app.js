let sessionId = generateUUID();
let messages = [];
let classificationLabels = ['positive', 'negative', 'neutral'];
let currentClassification = null;
let selectedModel = 'Qwen 2.5 7B';
let selectedNERModel = 'BERT Base';
let selectedOCRConfig = 'EasyOCR';
let isGenerating = false;
let currentReader = null;
let currentAbortController = null;
let displayedConversationCount = 5;
let availableModelsData = {};

const labelPresets = {
    sentiment: ['positive', 'negative', 'neutral'],
    intent: ['question', 'complaint', 'praise', 'request', 'information'],
    urgency: ['urgent', 'normal', 'low-priority'],
    cargo: ['standard cargo', 'refrigerated cargo', 'hazardous materials', 'oversized freight']
};

const examplePrompts = {
    sentiment: {
        info: 'Try one of these example texts for sentiment analysis:',
        examples: [
            { text: 'Major shipping alliance announces new ultra-large container vessels with 50% reduction in emissions per container. Industry leaders praise breakthrough in sustainable maritime transport.', label: 'Positive news', predicted: 'positive' },
            { text: "Port strike enters third week as dockworkers reject latest offer. Container backlog grows to record levels, threatening supply chain collapse across major retail sectors.", label: 'Negative news', predicted: 'negative' },
            { text: 'Global container shipping rates remain stable in Q3 according to latest freight index. Trans-Pacific routes show minimal fluctuation from previous quarter.', label: 'Neutral news', predicted: 'neutral' },
            { text: 'New Panama Canal expansion opens to fanfare, but analysts warn increased capacity may not offset rising fuel costs. Mixed reactions from shipping executives.', label: 'Mixed sentiment', predicted: 'neutral' }
        ]
    },
    intent: {
        info: 'Try one of these example texts for intent classification:',
        examples: [
            { text: 'How will the new IMO 2025 sulfur regulations impact freight rates for Asia-Europe routes? Shipping economists weigh in on compliance costs.', label: 'Question', predicted: 'question' },
            { text: "Cargo owners slam terminal operators for chronic delays at Los Angeles port. Shipper coalition demands immediate infrastructure improvements to prevent further disruptions.", label: 'Complaint', predicted: 'complaint' },
            { text: 'Port of Rotterdam achieves record throughput while maintaining industry-leading sustainability metrics. Officials credit advanced automation and green initiatives.', label: 'Praise', predicted: 'praise' },
            { text: 'Maritime authority seeks public comment on proposed amendments to dangerous goods handling regulations for container terminals. Submissions due by month end.', label: 'Request', predicted: 'request' }
        ]
    },
    urgency: {
        info: 'Try one of these example texts for urgency classification:',
        examples: [
            { text: 'BREAKING: Container ship experiencing engine failure in Suez Canal. Vessel blocking northbound traffic. Immediate tugboat assistance required to prevent extended closure of critical waterway.', label: 'Urgent', predicted: 'urgent' },
            { text: 'Industry conference announces 2025 dates for annual maritime logistics summit. Early bird registration opens next quarter for Singapore venue.', label: 'Low priority', predicted: 'low-priority' },
            { text: 'New container terminal at Port of Hamburg scheduled to open next spring. Facility will add 2 million TEU annual capacity to Northern European hub.', label: 'Normal', predicted: 'normal' },
            { text: 'MARITIME EMERGENCY: Typhoon Haikui forces evacuation of Shanghai port. All vessel movements suspended. Hundreds of ships seeking emergency anchorage. Coast guard on high alert.', label: 'Critical emergency', predicted: 'urgent' }
        ]
    },
    cargo: {
        info: 'Try one of these example texts for cargo type classification:',
        examples: [
            { text: 'Pharmaceutical shipment of temperature-sensitive vaccines requires uninterrupted cold chain at 2-8°C from manufacturing facility through final delivery. Advanced reefer monitoring deployed.', label: 'Refrigerated cargo', predicted: 'refrigerated cargo' },
            { text: 'Container manifest shows 500 TEU of consumer electronics and textiles loaded at Shenzhen. Standard dry containers with ambient temperature storage for trans-Pacific crossing.', label: 'Standard cargo', predicted: 'standard cargo' },
            { text: 'Vessel carries 200 tons of lithium-ion batteries classified as UN3480 Class 9 dangerous goods. Special segregation and fire suppression protocols in effect per IMDG Code.', label: 'Hazardous materials', predicted: 'hazardous materials' },
            { text: 'Breakbulk carrier loading 80-meter wind turbine blades onto reinforced flat racks. Route survey completed for overhead clearances through Panama Canal transit.', label: 'Oversized freight', predicted: 'oversized freight' }
        ]
    }
};

const hypothesisTemplates = {
    sentiment: 'The sentiment of this text is {label}.',
    intent: 'The intent of this message is {label}.',
    urgency: 'The urgency level is {label}.',
    cargo: 'This describes {label}.'
};

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

function hide(element) {
    if (typeof element === 'string') {
        element = document.getElementById(element);
    }
    if (element) {
        element.classList.add('hidden');
    }
}

function show(element) {
    if (typeof element === 'string') {
        element = document.getElementById(element);
    }
    if (element) {
        element.classList.remove('hidden');
    }
}

function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    const icon = '';
    
    toast.innerHTML = `
        <span class="toast-icon">${icon}</span>
        <span class="toast-message">${message}</span>
    `;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.add('toast-hiding');
        setTimeout(() => {
            container.removeChild(toast);
        }, 300);
    }, 3000);
}

function copyToClipboard(text, button) {
    navigator.clipboard.writeText(text).then(() => {
        const originalText = button.textContent;
        button.textContent = 'Copied!';
        button.classList.add('copied');
        showToast('Text copied to clipboard');
        setTimeout(() => {
            button.textContent = originalText;
            button.classList.remove('copied');
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
        button.textContent = 'Failed';
        showToast('Failed to copy text', 'error');
        setTimeout(() => {
            button.textContent = 'Copy';
        }, 2000);
    });
}

function updateClassificationExamples(preset = 'sentiment') {
    const infoElement = document.getElementById('classification-example-info');
    const gridElement = document.getElementById('classification-example-grid');
    
    if (!infoElement || !gridElement) {
        console.warn('Classification example elements not found');
        return;
    }
    
    const presetData = examplePrompts[preset];
    if (!presetData) {
        console.warn(`No example prompts found for preset: ${preset}`);
        return;
    }
    
    infoElement.textContent = presetData.info;
    
    gridElement.innerHTML = '';
    presetData.examples.forEach(example => {
        const button = document.createElement('button');
        button.className = 'btn btn-secondary example-prompt';
        button.setAttribute('data-classification-text', example.text);
        
        button.innerHTML = `
            <span class="expected-label">${example.label}</span>
        `;
        
        gridElement.appendChild(button);
    });
    
    setupClassificationExamples();
}

function setupClassificationExamples() {
    const exampleBtns = document.querySelectorAll('[data-classification-text]');
    exampleBtns.forEach(btn => {
        const newBtn = btn.cloneNode(true);
        btn.parentNode.replaceChild(newBtn, btn);
        
        newBtn.addEventListener('click', () => {
            const text = newBtn.dataset.classificationText;
            document.getElementById('classification-text-input').value = text;
            
            setTimeout(() => {
                document.getElementById('classify-btn').click();
            }, 100);
        });
    });
}

function updateHypothesisTemplate(preset = 'sentiment') {
    const templateInput = document.getElementById('hypothesis-template-input');
    
    if (!templateInput) {
        console.warn('Hypothesis template input not found');
        return;
    }
    
    const template = hypothesisTemplates[preset];
    if (template) {
        templateInput.value = template;
    }
}

function setupClassificationUI() {
    try {
        renderLabelChips();
        updateClassificationExamples('sentiment');
        updateHypothesisTemplate('sentiment');
        
        const presetButtons = document.querySelectorAll('.preset-btn');
        
        if (presetButtons.length === 0) {
            console.warn('No preset buttons found in DOM');
            return;
        }
        
        presetButtons.forEach((btn) => {
            btn.addEventListener('click', () => {
                const preset = btn.dataset.preset;
                if (labelPresets[preset]) {
                    classificationLabels = [...labelPresets[preset]];
                    renderLabelChips();
                    updateClassificationExamples(preset);
                    updateHypothesisTemplate(preset);
                }
            });
        });
    } catch (error) {
        console.error('Error in setupClassificationUI():', error);
    }
    
    const addLabelBtn = document.getElementById('add-label-btn');
    const newLabelInput = document.getElementById('new-label-input');
    
    addLabelBtn.addEventListener('click', () => {
        const label = newLabelInput.value.trim();
        if (label && !classificationLabels.includes(label)) {
            classificationLabels.push(label);
            renderLabelChips();
            newLabelInput.value = '';
        }
    });
    
    newLabelInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            addLabelBtn.click();
        }
    });
}

function renderLabelChips() {
    const container = document.getElementById('label-chips-container');
    if (!container) {
        console.error('Label chips container not found');
        return;
    }
    
    container.innerHTML = '';
    
    classificationLabels.forEach(label => {
        const chip = document.createElement('div');
        chip.className = 'label-chip';
        chip.innerHTML = `
            <span>${label}</span>
            <button class="remove-label" data-label="${label}">&times;</button>
        `;
        
        chip.querySelector('.remove-label').addEventListener('click', () => {
            classificationLabels = classificationLabels.filter(l => l !== label);
            renderLabelChips();
        });
        
        container.appendChild(chip);
    });
}

async function classifyText() {
    const textInput = document.getElementById('classification-text-input');
    const text = textInput.value.trim();
    
    if (!text || isGenerating || classificationLabels.length === 0) {
        if (classificationLabels.length === 0) {
            showToast('Please add at least one label', 'error');
        }
        return;
    }
    
    isGenerating = true;
    hide('classify-btn');
    show('stop-classification-btn');
    
    const resultsDiv = document.getElementById('classification-results');
    show(resultsDiv);
    resultsDiv.innerHTML = '<p>Classifying...</p>';
    
    const abstainThreshold = parseFloat(document.getElementById('abstain-threshold-slider').value);
    const useLogprobs = document.getElementById('use-logprobs-checkbox').checked;
    const hypothesisTemplate = document.getElementById('hypothesis-template-input').value;
    
    try {
        currentAbortController = new AbortController();
        
        const response = await fetch('/api/zero-shot/classify', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text,
                candidate_labels: classificationLabels,
                model: selectedModel,
                abstain_threshold: abstainThreshold,
                use_logprobs: useLogprobs,
                hypothesis_template: hypothesisTemplate
            }),
            signal: currentAbortController.signal
        });
        
        if (!response.ok) {
            throw new Error('Classification failed');
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let result = null;
        let startTime = null;
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();
            
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    
                    if (data.model_loading_start) {
                        resultsDiv.innerHTML = '<p>Loading model...</p>';
                    } else if (data.model_loading_end) {
                        resultsDiv.innerHTML = '<p>Model loaded. Classifying...</p>';
                        startTime = performance.now();
                        updateMainLoadButton(true);
                    } else if (data.result) {
                        result = data.result;
                        renderClassificationResults(result, useLogprobs);
                    } else if (data.error) {
                        resultsDiv.innerHTML = `<p class="text-error">Error: ${data.error}</p>`;
                    }
                }
            }
        }
        
        if (result && startTime) {
            const endTime = performance.now();
            const duration = ((endTime - startTime) / 1000).toFixed(2);
            
            currentClassification = {
                id: generateUUID(),
                text,
                labels: classificationLabels,
                result,
                timestamp: new Date().toISOString(),
                model: selectedModel,
                duration,
                use_logprobs: useLogprobs
            };
            
            showToast('Classification complete');
        }
        
    } catch (error) {
        if (error.name !== 'AbortError') {
            console.error('Classification error:', error);
            resultsDiv.innerHTML = `<p class="text-error">Error: ${error.message}</p>`;
            showToast('Classification failed', 'error');
        }
    } finally {
        isGenerating = false;
        show('classify-btn');
        hide('stop-classification-btn');
    }
}

function renderClassificationResults(result, useLogprobs = false) {
    const resultsDiv = document.getElementById('classification-results');
    
    let html = '<h3>Classification Results</h3>';
    
    const hasLogprobs = useLogprobs || (result.labels && result.labels.some(l => l.logprob !== null && l.logprob !== undefined));
    
    if (!hasLogprobs) {
        html += `
            <div class="top-prediction">
                <div class="prediction-header">Predicted Label</div>
                <div class="prediction-label">${result.top_label}</div>
            </div>
            <div class="info-box">
                <strong>Fast mode</strong> - Confidence scoring is disabled. Enable "Use Logprob Scoring" for detailed confidence estimates.
            </div>
        `;
    } else {
        html += `
            <div class="info-box">
                <strong>Confidence scores</strong> are computed from logprobs (model internal probabilities), not generated by the model. This provides accurate confidence estimation.
            </div>
        `;
        
        if (result.should_abstain) {
            html += `
                <div id="abstain-indicator" class="abstain-indicator">
                    <strong>Low Confidence Warning</strong>
                    <p>The model suggests abstaining from this classification. Top confidence (${(result.top_score * 100).toFixed(1)}%) is below the threshold (${(result.abstain_threshold * 100).toFixed(0)}%).</p>
                </div>
            `;
        }
        
        html += `
            <div class="top-prediction">
                <div class="prediction-header">Top Prediction</div>
                <div class="prediction-label">${result.top_label}</div>
                <div class="prediction-score">Confidence: ${(result.top_score * 100).toFixed(1)}%</div>
            </div>
        `;
        
        html += '<div class="all-predictions">';
        
        result.labels.forEach(labelResult => {
            const percentage = (labelResult.score * 100).toFixed(1);
            const logprobText = labelResult.logprob !== null && labelResult.logprob !== undefined 
                ? `<div class="prediction-logprob">logprob: ${labelResult.logprob.toFixed(3)}</div>`
                : '';
            
            html += `
                <div class="prediction-item">
                    <div class="prediction-label-row">
                        <span class="prediction-label-name">${labelResult.label}</span>
                        <span class="prediction-label-score">${percentage}%</span>
                    </div>
                    <div class="confidence-bar-container">
                        <div class="confidence-bar" style="--confidence-width: ${percentage}%">
                            ${percentage > 15 ? `<span class="confidence-bar-label">${percentage}%</span>` : ''}
                        </div>
                    </div>
                    ${logprobText}
                </div>
            `;
        });
        
        html += '</div>';
    }
    
    resultsDiv.innerHTML = html;
}

async function init() {
    try {
        setupEventListeners();
        setupNERExamples();
        await setupOCRExamples();
        setupClassificationUI();
        closeMobileMenuHelper();
        
        Promise.all([
            loadModels(),
            loadNERModels(),
            loadOCRConfigs()
        ]).catch(error => {
            console.error('Error during initialization:', error);
        });
    } catch (error) {
        console.error('Error in init():', error);
    }
}

async function loadModels() {
    try {
        const [modelsResponse, statusResponse] = await Promise.all([
            fetch('/api/models'),
            fetch('/api/models/status')
        ]);
        
        const modelsData = await modelsResponse.json();
        const statusData = await statusResponse.json();
        
        availableModelsData = modelsData.models;
        
        const modelList = document.getElementById('model-list');
        modelList.innerHTML = '';
        
        const zeroShotModels = Object.entries(modelsData.models).filter(([name, info]) => {
            const supportedTasks = info.supported_tasks || [];
            return supportedTasks.includes('zero-shot');
        });
        
        if (zeroShotModels.length === 0) {
            modelList.innerHTML = '<p class="error-message">No models available for zero-shot classification</p>';
            return;
        }
        
        const currentModelSupportsZeroShot = availableModelsData[selectedModel]?.supported_tasks?.includes('zero-shot');
        if (!currentModelSupportsZeroShot && zeroShotModels.length > 0) {
            selectedModel = zeroShotModels[0][0];
        }
        
        zeroShotModels.forEach(([name, info]) => {
            const modelDiv = document.createElement('div');
            modelDiv.className = `model-option ${name === selectedModel ? 'selected' : ''}`;
            modelDiv.dataset.modelName = name;
            
            modelDiv.innerHTML = `
                <div class="model-header-row">
                    <h4>${name}</h4>
                </div>
                <div class="model-specs">
                    <span class="model-badge">${info.params} params</span>
                    <span class="model-badge">${info.memory}</span>
                </div>
                <div class="model-description">${info.description}</div>
            `;
            
            modelDiv.onclick = (evt) => {
                selectModel(name, evt);
            };
            
            modelList.appendChild(modelDiv);
        });
        
        const modelNameDisplay = document.getElementById('selected-model-display');
        if (modelNameDisplay) {
            modelNameDisplay.textContent = selectedModel;
        }
        
        const isLoaded = statusData.status[selectedModel]?.loaded || false;
        updateMainLoadButton(isLoaded);
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

async function loadNERModels() {
    try {
        const [modelsResponse, statusResponse] = await Promise.all([
            fetch('/api/ner/models'),
            fetch('/api/ner/models/status')
        ]);
        
        const modelsData = await modelsResponse.json();
        const statusData = await statusResponse.json();
        
        const modelList = document.getElementById('ner-model-list');
        modelList.innerHTML = '';
        
        Object.entries(modelsData.models).forEach(([name, info]) => {
            const modelDiv = document.createElement('div');
            modelDiv.className = `model-option ${name === selectedNERModel ? 'selected' : ''}`;
            modelDiv.dataset.modelName = name;
            
            modelDiv.innerHTML = `
                <div class="model-header-row">
                    <h4>${name}</h4>
                </div>
                <div class="model-specs">
                    <span class="model-badge">${info.params} params</span>
                    <span class="model-badge">${info.memory}</span>
                </div>
                <div class="model-description">${info.description}</div>
            `;
            
            modelDiv.onclick = (evt) => {
                selectNERModel(name, evt);
            };
            
            modelList.appendChild(modelDiv);
        });
    } catch (error) {
        console.error('Error loading NER models:', error);
    }
}

async function loadOCRConfigs() {
    try {
        const [configsResponse, statusResponse] = await Promise.all([
            fetch('/api/ocr/configs'),
            fetch('/api/ocr/configs/status')
        ]);
        
        const configsData = await configsResponse.json();
        const statusData = await statusResponse.json();
        
        const modelList = document.getElementById('ocr-model-list');
        modelList.innerHTML = '';
        
        Object.entries(configsData.configs).forEach(([name, info]) => {
            const modelDiv = document.createElement('div');
            modelDiv.className = `model-option ${name === selectedOCRConfig ? 'selected' : ''}`;
            modelDiv.dataset.configName = name;
            
            modelDiv.innerHTML = `
                <div class="model-header-row">
                    <h4>${name}</h4>
                </div>
                <div class="model-description">${info.description}</div>
            `;
            
            modelDiv.onclick = (evt) => {
                selectOCRConfig(name, evt);
            };
            
            modelList.appendChild(modelDiv);
        });
    } catch (error) {
        console.error('Error loading OCR configs:', error);
    }
}

async function selectModel(name, evt) {
    selectedModel = name;
    
    document.querySelectorAll('#model-list .model-option').forEach(el => {
        el.classList.remove('selected');
    });
    
    evt.target.closest('.model-option').classList.add('selected');
    
    const modelNameDisplay = document.getElementById('selected-model-display');
    if (modelNameDisplay) {
        modelNameDisplay.textContent = name;
    }
    
    try {
        const statusResponse = await fetch('/api/models/status');
        const statusData = await statusResponse.json();
        const isLoaded = statusData.status[name]?.loaded || false;
        updateMainLoadButton(isLoaded);
    } catch (error) {
        console.error('Error fetching model status:', error);
        updateMainLoadButton(false);
    }
    
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

function selectOCRConfig(name, evt) {
    selectedOCRConfig = name;
    
    document.querySelectorAll('#ocr-model-list .model-option').forEach(el => {
        el.classList.remove('selected');
    });
    
    evt.target.closest('.model-option').classList.add('selected');
    
    setTimeout(() => {
        closeMobileMenuHelper();
    }, 100);
}

function updateMainLoadButton(isLoaded) {
    const mainLoadBtn = document.getElementById('main-load-model-btn');
    if (!mainLoadBtn) return;
    
    if (isLoaded) {
        mainLoadBtn.textContent = 'Model Loaded ✓';
        mainLoadBtn.disabled = true;
        mainLoadBtn.classList.add('loaded');
    } else {
        mainLoadBtn.textContent = 'Load Model';
        mainLoadBtn.disabled = false;
        mainLoadBtn.classList.remove('loaded');
    }
}

async function loadSelectedModel() {
    const mainLoadBtn = document.getElementById('main-load-model-btn');
    if (!mainLoadBtn) return;
    
    try {
        mainLoadBtn.textContent = 'Loading...';
        mainLoadBtn.disabled = true;
        
        const response = await fetch('/api/models/load', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_name: selectedModel })
        });
        
        const data = await response.json();
        
        if (data.success) {
            updateMainLoadButton(true);
            const loadTime = data.load_time ? ` (${data.load_time.toFixed(1)}s)` : '';
            showToast(`${selectedModel} loaded${loadTime}`);
        } else {
            throw new Error('Failed to load model');
        }
    } catch (error) {
        console.error('Error loading model:', error);
        mainLoadBtn.textContent = 'Load Model';
        mainLoadBtn.disabled = false;
        showToast(`Failed to load ${selectedModel}`, 'error');
    }
}

async function preloadModel(modelName) {
    const modelDiv = document.querySelector(`[data-model-name="${modelName}"]`);
    const loadBtn = modelDiv?.querySelector('.model-load-btn');
    
    if (!loadBtn) return;
    
    try {
        loadBtn.textContent = 'Loading...';
        loadBtn.disabled = true;
        loadBtn.classList.add('loading');
        
        const response = await fetch('/api/models/load', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_name: modelName })
        });
        
        const data = await response.json();
        
        if (data.success) {
            loadBtn.textContent = 'Loaded';
            loadBtn.classList.remove('loading');
            loadBtn.classList.add('loaded');
            
            const loadTime = data.load_time ? ` (${data.load_time.toFixed(1)}s)` : '';
            showToast(`${modelName} loaded${loadTime}`);
        } else {
            throw new Error('Failed to load model');
        }
    } catch (error) {
        console.error('Error preloading model:', error);
        loadBtn.textContent = 'Load';
        loadBtn.disabled = false;
        loadBtn.classList.remove('loading');
        showToast(`Failed to load ${modelName}`, 'error');
    }
}

function updateModelLoadButton(modelName, isLoaded) {
    const modelDiv = document.querySelector(`[data-model-name="${modelName}"]`);
    const loadBtn = modelDiv?.querySelector('.model-load-btn');
    
    if (!loadBtn) return;
    
    if (isLoaded) {
        loadBtn.textContent = 'Loaded';
        loadBtn.disabled = true;
        loadBtn.classList.remove('loading');
        loadBtn.classList.add('loaded');
    } else {
        loadBtn.textContent = 'Load';
        loadBtn.disabled = false;
        loadBtn.classList.remove('loading', 'loaded');
    }
}

async function preloadNERModel(modelName) {
    const modelDiv = document.querySelector(`[data-model-name="${modelName}"]`);
    const loadBtn = modelDiv?.querySelector('.model-load-btn');
    
    if (!loadBtn) return;
    
    try {
        loadBtn.textContent = 'Loading...';
        loadBtn.disabled = true;
        loadBtn.classList.add('loading');
        
        const response = await fetch('/api/ner/models/load', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_name: modelName })
        });
        
        const data = await response.json();
        
        if (data.success) {
            loadBtn.textContent = 'Loaded';
            loadBtn.classList.remove('loading');
            loadBtn.classList.add('loaded');
            
            const loadTime = data.load_time ? ` (${data.load_time.toFixed(1)}s)` : '';
            showToast(`${modelName} loaded${loadTime}`);
        } else {
            throw new Error('Failed to load NER model');
        }
    } catch (error) {
        console.error('Error preloading NER model:', error);
        loadBtn.textContent = 'Load';
        loadBtn.disabled = false;
        loadBtn.classList.remove('loading');
        showToast(`Failed to load ${modelName}`, 'error');
    }
}

async function preloadOCRConfig(configName) {
    const modelDiv = document.querySelector(`[data-config-name="${configName}"]`);
    const loadBtn = modelDiv?.querySelector('.model-load-btn');
    
    if (!loadBtn) return;
    
    try {
        loadBtn.textContent = 'Loading...';
        loadBtn.disabled = true;
        loadBtn.classList.add('loading');
        
        const response = await fetch('/api/ocr/configs/load', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ config_name: configName })
        });
        
        const data = await response.json();
        
        if (data.success) {
            loadBtn.textContent = 'Loaded';
            loadBtn.classList.remove('loading');
            loadBtn.classList.add('loaded');
            
            const loadTime = data.load_time ? ` (${data.load_time.toFixed(1)}s)` : '';
            showToast(`${configName} loaded${loadTime}`);
        } else {
            throw new Error('Failed to load OCR config');
        }
    } catch (error) {
        console.error('Error preloading OCR config:', error);
        loadBtn.textContent = 'Load';
        loadBtn.disabled = false;
        loadBtn.classList.remove('loading');
        showToast(`Failed to load ${configName}`, 'error');
    }
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
            conversationList.innerHTML = `
                <div class="empty-state">
                    <p class="empty-state">No saved conversations yet</p>
                    <small class="empty-state-hint">
                        Try selecting a model and starting a conversation!<br><br>
                        Example use cases:<br>
                        • Classify cargo shipments<br>
                        • Analyze logistics scenarios<br>
                        • Get shipping recommendations
                    </small>
                </div>
            `;
            return;
        }
        
        conversationList.innerHTML = `<small class="history-count">Found ${data.conversations.length} saved conversation(s)</small>`;
        
        data.conversations.slice(0, displayedConversationCount).forEach(conv => {
            const item = document.createElement('div');
            item.className = 'conversation-item';
            
            const btn = document.createElement('button');
            btn.className = `conversation-btn ${conv.session_id === sessionId ? 'active' : ''}`;
            btn.innerHTML = `
                <div class="history-item-preview">${conv.first_message || 'No messages'}</div>
                <div class="history-item-meta">${conv.message_count} messages</div>
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
                // Insert skeleton before the button
                const skeleton = document.createElement('div');
                skeleton.className = 'skeleton-conversation';
                skeleton.innerHTML = `
                    <div class="skeleton skeleton-conversation-title"></div>
                    <div class="skeleton skeleton-conversation-meta"></div>
                `;
                conversationList.insertBefore(skeleton, showMoreBtn);
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
    if (!chatMessages) {
        console.warn('chat-messages element not found - chat functionality not available');
        return;
    }
    
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
        messages.forEach((msg, index) => {
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
            
            const copyButtonHtml = msg.role === 'assistant' ? 
                `<button class="copy-btn" data-message-index="${index}" title="Copy to clipboard">Copy</button>` : '';
            
            messageDiv.innerHTML = `
                <div class="message-header">
                    ${msg.role === 'user' ? 'User' : 'Assistant'}
                    ${copyButtonHtml}
                </div>
                <div class="message-content">${escapeHtml(msg.content)}</div>
                ${metricsHtml}
            `;
            
            chatMessages.appendChild(messageDiv);
        });
        
        // Add click handlers for copy buttons
        chatMessages.querySelectorAll('.copy-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const index = parseInt(btn.dataset.messageIndex);
                copyToClipboard(messages[index].content, btn);
            });
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
    
    const sendBtn = document.getElementById('send-btn');
    const stopBtn = document.getElementById('stop-btn');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    
    if (!sendBtn || !stopBtn || !chatInput || !chatMessages) {
        console.warn('Chat UI elements not found - chat functionality not available');
        return;
    }
    
    isGenerating = true;
    hide(sendBtn);
    show(stopBtn);
    chatInput.disabled = true;
    
    messages.push({
        role: 'user',
        content: userMessage
    });
    
    renderMessages();
    
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
                                    streamingContent.innerHTML = `<em class="text-muted">Loading model... ${elapsed}s</em>`;
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
                streamingContent.innerHTML = `
                    <div class="text-error py-base">Error: ${error.message}</div>
                    <button class="btn btn-primary chat-retry-btn mt-base" data-retry-message="${escapeHtml(userMessage)}">Try Again</button>
                `;
                
                const retryBtn = assistantDiv.querySelector('.chat-retry-btn');
                if (retryBtn) {
                    retryBtn.addEventListener('click', () => {
                        assistantDiv.remove();
                        messages.pop();
                        sendMessage(userMessage);
                    });
                }
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
        
        const sendBtn = document.getElementById('send-btn');
        const stopBtn = document.getElementById('stop-btn');
        const chatInput = document.getElementById('chat-input');
        
        if (sendBtn) show(sendBtn);
        if (stopBtn) hide(stopBtn);
        if (chatInput) {
            chatInput.disabled = false;
            chatInput.value = '';
        }
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
    const classifyBtn = document.getElementById('classify-btn');
    const stopClassificationBtn = document.getElementById('stop-classification-btn');
    
    if (classifyBtn) {
        classifyBtn.addEventListener('click', () => {
            classifyText();
        });
    }
    
    if (stopClassificationBtn) {
        stopClassificationBtn.addEventListener('click', () => {
            stopGeneration();
        });
    }
    
    const classificationTextInput = document.getElementById('classification-text-input');
    if (classificationTextInput) {
        classificationTextInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                e.preventDefault();
                classifyText();
            }
        });
    }
    
    const mainLoadModelBtn = document.getElementById('main-load-model-btn');
    if (mainLoadModelBtn) {
        mainLoadModelBtn.addEventListener('click', async () => {
            await loadSelectedModel();
        });
    }
    
    const abstainThresholdSlider = document.getElementById('abstain-threshold-slider');
    const abstainThresholdValue = document.getElementById('abstain-threshold-value');
    if (abstainThresholdSlider && abstainThresholdValue) {
        abstainThresholdSlider.addEventListener('input', (e) => {
            abstainThresholdValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }
    
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
    if (temperatureSlider && temperatureValue) {
        temperatureSlider.addEventListener('input', (e) => {
            settings.temperature = parseFloat(e.target.value);
            temperatureValue.textContent = settings.temperature.toFixed(1);
        });
    }
    
    const maxTokensSlider = document.getElementById('max-tokens-slider');
    const maxTokensValue = document.getElementById('max-tokens-value');
    if (maxTokensSlider && maxTokensValue) {
        maxTokensSlider.addEventListener('input', (e) => {
            settings.maxTokens = parseInt(e.target.value);
            maxTokensValue.textContent = settings.maxTokens;
        });
    }
    
    const topPSlider = document.getElementById('top-p-slider');
    const topPValue = document.getElementById('top-p-value');
    if (topPSlider && topPValue) {
        topPSlider.addEventListener('input', (e) => {
            settings.topP = parseFloat(e.target.value);
            topPValue.textContent = settings.topP.toFixed(2);
        });
    }
    
    const topKSlider = document.getElementById('top-k-slider');
    const topKValue = document.getElementById('top-k-value');
    if (topKSlider && topKValue) {
        topKSlider.addEventListener('input', (e) => {
            settings.topK = parseInt(e.target.value);
            topKValue.textContent = settings.topK;
        });
    }
    
    const labelCustomizationAccordion = document.querySelector('.label-customization-accordion');
    const labelCustomizationContent = document.getElementById('label-customization-content');
    if (labelCustomizationAccordion && labelCustomizationContent) {
        labelCustomizationAccordion.addEventListener('toggle', (e) => {
            if (e.target.open) {
                labelCustomizationContent.classList.remove('hidden');
            } else {
                labelCustomizationContent.classList.add('hidden');
            }
        });
    }
    
    // NER settings sliders
    const nerConfidenceSlider = document.getElementById('ner-confidence-threshold');
    const nerConfidenceValue = document.getElementById('ner-confidence-value');
    if (nerConfidenceSlider && nerConfidenceValue) {
        nerConfidenceSlider.addEventListener('input', (e) => {
            nerConfidenceValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }
    
    // OCR settings sliders
    const ocrConfidenceSlider = document.getElementById('ocr-confidence-threshold');
    const ocrConfidenceValue = document.getElementById('ocr-confidence-value');
    if (ocrConfidenceSlider && ocrConfidenceValue) {
        ocrConfidenceSlider.addEventListener('input', (e) => {
            ocrConfidenceValue.textContent = parseFloat(e.target.value).toFixed(2);
        });
    }
    
    const ocrMinTextSizeSlider = document.getElementById('ocr-min-text-size');
    const ocrMinTextSizeValue = document.getElementById('ocr-min-text-size-value');
    if (ocrMinTextSizeSlider && ocrMinTextSizeValue) {
        ocrMinTextSizeSlider.addEventListener('input', (e) => {
            ocrMinTextSizeValue.textContent = e.target.value;
        });
    }
    
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
            sidebarContents.forEach(s => hide(s));
            
            btn.classList.add('active');
            document.getElementById(`${targetTab}-tab`).classList.add('active');
            show(`${targetTab}-sidebar`);
        });
    });
}

function setupNER() {
    const submitBtn = document.getElementById('ner-submit-btn');
    const textInput = document.getElementById('ner-text-input');
    
    // Add Enter key support
    textInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !submitBtn.disabled) {
            submitBtn.click();
        }
    });
    
    submitBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        if (!text) {
            alert('Please enter some text to analyze');
            return;
        }
        
        submitBtn.disabled = true;
        submitBtn.textContent = 'Extracting...';
        
        // Show loading skeleton
        const resultsDiv = document.getElementById('ner-results');
        show(resultsDiv);
        resultsDiv.innerHTML = `
            <div class="skeleton-results">
                <div class="skeleton skeleton-results-title"></div>
                <div class="skeleton skeleton-entity"></div>
                <div class="skeleton skeleton-entity"></div>
                <div class="skeleton skeleton-entity"></div>
                <div class="skeleton skeleton-metrics"></div>
                <div class="skeleton skeleton-metrics"></div>
            </div>
        `;
        
        // Auto-scroll to results
        setTimeout(() => {
            resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
        
        let loadingTimerInterval = null;
        let modelLoadStartTime = null;
        
        try {
            // Get settings values
            const confidenceThreshold = parseFloat(document.getElementById('ner-confidence-threshold')?.value || 0.5);
            const entityTypeCheckboxes = document.querySelectorAll('.entity-type-checkbox');
            const entityTypes = Array.from(entityTypeCheckboxes)
                .filter(cb => cb.checked)
                .map(cb => cb.value);
            
            if (entityTypes.length === 0) {
                displayNERError('Please select at least one entity type to extract.');
                submitBtn.disabled = false;
                submitBtn.textContent = 'Extract Entities';
                return;
            }
            
            console.log('Sending NER request:', { text: text.substring(0, 50) + '...', model: selectedNERModel, confidenceThreshold, entityTypes });
            
            const response = await fetch('/api/ner', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    text, 
                    model: selectedNERModel,
                    confidence_threshold: confidenceThreshold,
                    entity_types: entityTypes
                })
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                console.error('NER API error:', response.status, errorText);
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }
            
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
            let errorMessage = 'Network error. Please try again.';
            
            if (error.name === 'AbortError') {
                errorMessage = 'Request was cancelled.';
            } else if (error.message) {
                errorMessage = `Error: ${error.message}`;
            } else if (!navigator.onLine) {
                errorMessage = 'No internet connection. Please check your network.';
            }
            
            displayNERError(errorMessage);
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
    
    // Restore proper HTML structure
    resultsDiv.innerHTML = `
        <h3>Detected Entities</h3>
        <div id="ner-entities" class="entity-list"></div>
        <div id="ner-metrics" class="metrics-display"></div>
    `;
    
    const entitiesDiv = document.getElementById('ner-entities');
    const metricsDiv = document.getElementById('ner-metrics');
    
    show(resultsDiv);
    entitiesDiv.innerHTML = `
        <div class="error-box">
            <strong>Error:</strong> ${message}
        </div>
        <button id="ner-retry-btn" class="btn btn-primary mb-md">Try Again</button>
    `;
    metricsDiv.innerHTML = '';
    
    document.getElementById('ner-retry-btn').addEventListener('click', () => {
        document.getElementById('ner-submit-btn').click();
    });
}

function displayNERResults(data) {
    const resultsDiv = document.getElementById('ner-results');
    
    // Restore proper HTML structure
    resultsDiv.innerHTML = `
        <h3>Detected Entities</h3>
        <div id="ner-entities" class="entity-list"></div>
        <div id="ner-metrics" class="metrics-display"></div>
    `;
    
    const entitiesDiv = document.getElementById('ner-entities');
    const metricsDiv = document.getElementById('ner-metrics');
    
    show(resultsDiv);
    
    if (data.entities && data.entities.length > 0) {
        entitiesDiv.innerHTML = data.entities.map(entity => `
            <div class="entity-tag ${entity.label}">
                <span>${entity.text}</span>
                <span class="entity-label">${entity.label}</span>
            </div>
        `).join('');
        
        // Format entities for copying
        const entitiesText = data.entities.map(entity => 
            `${entity.text} (${entity.label})`
        ).join('\n');
        
        metricsDiv.innerHTML = `
            <div class="metrics-stats">
                <div><strong>Processing Time:</strong> ${data.processing_time.toFixed(3)}s</div>
                <div><strong>Entities Found:</strong> ${data.entities.length}</div>
                <div><strong>Text Length:</strong> ${data.text_length} characters</div>
            </div>
            <button class="copy-btn" id="ner-copy-btn">Copy Entities</button>
        `;
        
        document.getElementById('ner-copy-btn').addEventListener('click', function() {
            copyToClipboard(entitiesText, this);
        });
    } else {
        entitiesDiv.innerHTML = '<p>No entities detected in the text.</p>';
        metricsDiv.innerHTML = `
            <div><strong>Processing Time:</strong> ${data.processing_time.toFixed(3)}s</div>
        `;
    }
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
            show(previewDiv);
            hide(dropZone);
        };
        reader.readAsDataURL(file);
    }
    
    function clearOCRImage() {
        ocrSelectedFile = null;
        previewImg.src = '';
        hide(previewDiv);
        show(dropZone);
        fileInput.value = '';
        
        const resultsDiv = document.getElementById('ocr-results');
        hide(resultsDiv);
    }
    
    const clearBtn = document.getElementById('ocr-clear-btn');
    clearBtn.addEventListener('click', clearOCRImage);
    
    submitBtn.addEventListener('click', async () => {
        if (!ocrSelectedFile) {
            alert('Please select an image first');
            return;
        }
        
        submitBtn.disabled = true;
        submitBtn.textContent = 'Extracting...';
        
        // Show loading skeleton
        const resultsDiv = document.getElementById('ocr-results');
        show(resultsDiv);
        resultsDiv.innerHTML = `
            <div class="skeleton-results">
                <div class="skeleton skeleton-results-title"></div>
                <div class="skeleton skeleton-text-block"></div>
                <div class="skeleton skeleton-metrics"></div>
                <div class="skeleton skeleton-metrics"></div>
            </div>
        `;
        
        // Auto-scroll to results
        setTimeout(() => {
            resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
        
        let loadingTimerInterval = null;
        let modelLoadStartTime = null;
        
        try {
            // Get settings values
            const confidenceThreshold = parseFloat(document.getElementById('ocr-confidence-threshold')?.value || 0.5);
            const minTextSize = parseInt(document.getElementById('ocr-min-text-size')?.value || 10);
            
            const formData = new FormData();
            formData.append('file', ocrSelectedFile);
            
            const endpoint = `/api/ocr?config=${encodeURIComponent(selectedOCRConfig)}&confidence_threshold=${confidenceThreshold}&min_text_size=${minTextSize}`;
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
                                    submitBtn.textContent = 'Extracting...';
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
    
    // Restore proper HTML structure
    resultsDiv.innerHTML = `
        <h3>Extracted Text</h3>
        <div id="ocr-text" class="extracted-text"></div>
        <div id="ocr-metrics" class="metrics-display"></div>
    `;
    
    const textDiv = document.getElementById('ocr-text');
    const metricsDiv = document.getElementById('ocr-metrics');
    
    show(resultsDiv);
    textDiv.innerHTML = `
        <div class="error-box">
            <strong>Error:</strong> ${message}
        </div>
        <button id="ocr-retry-btn" class="btn btn-primary mb-md">Try Again</button>
    `;
    metricsDiv.innerHTML = '';
    
    document.getElementById('ocr-retry-btn').addEventListener('click', () => {
        document.getElementById('ocr-submit-btn').click();
    });
}

function displayOCRResults(data) {
    const resultsDiv = document.getElementById('ocr-results');
    
    // Restore proper HTML structure
    resultsDiv.innerHTML = `
        <h3>Extracted Text</h3>
        <div id="ocr-text" class="extracted-text"></div>
        <div id="ocr-metrics" class="metrics-display"></div>
    `;
    
    const textDiv = document.getElementById('ocr-text');
    const metricsDiv = document.getElementById('ocr-metrics');
    
    show(resultsDiv);
    
    if (data.text) {
        textDiv.textContent = data.text;
        
        metricsDiv.innerHTML = `
            <div class="metrics-stats">
                <div><strong>Processing Time:</strong> ${data.processing_time.toFixed(3)}s</div>
                <div><strong>Text Detections:</strong> ${data.num_detections}</div>
            </div>
            <button class="copy-btn" id="ocr-copy-btn">Copy Text</button>
        `;
        
        document.getElementById('ocr-copy-btn').addEventListener('click', function() {
            copyToClipboard(data.text, this);
        });
    } else {
        textDiv.textContent = 'No text detected in the image.';
        metricsDiv.innerHTML = `
            <div><strong>Processing Time:</strong> ${data.processing_time.toFixed(3)}s</div>
        `;
    }
    
}

function setupNERExamples() {
    const exampleBtns = document.querySelectorAll('[data-ner-text]');
    exampleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const text = btn.dataset.nerText;
            const textInput = document.getElementById('ner-text-input');
            const submitBtn = document.getElementById('ner-submit-btn');
            
            if (!textInput || !submitBtn) {
                console.error('NER input or submit button not found');
                return;
            }
            
            if (submitBtn.disabled) {
                console.warn('Submit button is disabled, waiting...');
                showToast('Please wait for the current operation to complete', 'warning');
                return;
            }
            
            textInput.value = text;
            
            setTimeout(() => {
                if (!submitBtn.disabled) {
                    submitBtn.click();
                } else {
                    console.warn('Submit button became disabled');
                    showToast('Unable to submit, please try again', 'error');
                }
            }, 100);
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
                    show('ocr-preview');
                    hide('ocr-drop-zone');
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
