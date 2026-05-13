const $ = id => document.getElementById(id);
const uploadArea = $('uploadArea');
const fileInput = $('fileInput');
const processBtn = $('processBtn');
const resetBtn = $('resetBtn');
const loading = $('loading');
const errorMsg = $('errorMsg');
const successMsg = $('successMsg');
const imagesContainer = $('imagesContainer');
const buttonsContainer = $('buttonsContainer');
const selectedFilesDiv = $('selectedFiles');
const fileList = $('fileList');
const resultsGrid = $('resultsGrid');

let selectedFiles = [];
let currentSessionId = null;
let websocket = null;
let progressInterval = null;

const showMessage = (isError, message) => {
    const [show, hide] = isError ? [errorMsg, successMsg] : [successMsg, errorMsg];
    show.textContent = message;
    show.classList.add('active');
    hide.classList.remove('active');
    
    if (!isError) {
        setTimeout(() => show.classList.remove('active'), 5000);
    }
};

const hideMessages = () => {
    errorMsg.classList.remove('active');
    successMsg.classList.remove('active');
};

const handleFileSelect = files => {
    if (!files || files.length === 0) return;

    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff', 'image/tif'];
    const validExtensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff'];
    const validFiles = [];
    
    for (const file of files) {
        const ext = file.name.toLowerCase().slice(file.name.lastIndexOf('.'));
        if (!validTypes.includes(file.type) && !validExtensions.includes(ext)) {
            showMessage(true, `❌ "${file.name}" has invalid type. Use JPG, PNG, or TIFF.`);
            continue;
        }
        if (file.size > 500 * 1024 * 1024) {
            showMessage(true, `❌ "${file.name}" is too large. Max 500MB.`);
            continue;
        }
        validFiles.push(file);
    }

    if (validFiles.length === 0) return;

    selectedFiles = validFiles;
    showMessage(false, `✅ ${validFiles.length} MRI scan(s) selected successfully!`);

    // Show file list
    fileList.innerHTML = validFiles.map(f => `<li>🩻 ${f.name}</li>`).join('');
    selectedFilesDiv.classList.remove('hidden');
    buttonsContainer.classList.remove('hidden');
    imagesContainer.classList.add('hidden');
    resultsGrid.innerHTML = '';
};

uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', e => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFileSelect(e.dataTransfer.files);
});

fileInput.addEventListener('change', e => handleFileSelect(e.target.files));

// WebSocket connection management
const connectWebSocket = (sessionId) => {
    if (websocket) {
        websocket.close();
    }
    
    currentSessionId = sessionId;
    websocket = new WebSocket(`ws://${window.location.host}/ws/progress/${sessionId}`);
    
    websocket.onopen = () => {
        console.log('WebSocket connected');
        showMessage(false, '🔗 Connected to real-time updates');
    };
    
    websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
    
    websocket.onclose = () => {
        console.log('WebSocket disconnected');
        if (progressInterval) {
            clearInterval(progressInterval);
        }
    };
    
    websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        showMessage(true, '❌ Connection lost. Please refresh the page.');
    };
};

const handleWebSocketMessage = (data) => {
    switch (data.type) {
        case 'connection_established':
            console.log('Connection established:', data.session_id);
            break;
        
        case 'progress_update':
            updateProgressDisplay(data.data);
            break;
        
        case 'task_update':
            updateTaskProgress(data.task_id, data.data);
            break;
        
        case 'task_completed':
            handleTaskCompletion(data.task_id, data.data);
            break;
        
        case 'error':
            showMessage(true, `❌ Error: ${data.message}`);
            break;
        
        case 'ping':
            // Keep connection alive
            break;
    }
};

const updateProgressDisplay = (status) => {
    const progressContainer = document.createElement('div');
    progressContainer.className = 'progress-overview';
    progressContainer.innerHTML = `
        <div class="progress-stats">
            <span class="stat-item">📊 Total: ${status.total_tasks}</span>
            <span class="stat-item">⏳ Pending: ${status.pending_tasks}</span>
            <span class="stat-item">🔄 Processing: ${status.processing_tasks}</span>
            <span class="stat-item">✅ Completed: ${status.completed_tasks}</span>
            <span class="stat-item">❌ Failed: ${status.failed_tasks}</span>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: ${(status.completed_tasks / status.total_tasks) * 100}%"></div>
        </div>
    `;
    
    // Update loading section with progress
    const existingProgress = loading.querySelector('.progress-overview');
    if (existingProgress) {
        existingProgress.remove();
    }
    loading.appendChild(progressContainer);
};

const updateTaskProgress = (taskId, taskData) => {
    const taskElement = document.querySelector(`[data-task-id="${taskId}"]`);
    if (taskElement) {
        const progressElement = taskElement.querySelector('.task-progress');
        const statusElement = taskElement.querySelector('.task-status');
        
        progressElement.value = taskData.progress;
        progressElement.textContent = `${taskData.progress}%`;
        
        if (taskData.status === 'completed') {
            statusElement.textContent = '✅ Completed';
            statusElement.className = 'task-status completed';
        } else if (taskData.status === 'failed') {
            statusElement.textContent = '❌ Failed';
            statusElement.className = 'task-status failed';
        } else if (taskData.status === 'processing') {
            statusElement.textContent = '🔄 Processing';
            statusElement.className = 'task-status processing';
        }
    }
};

const handleTaskCompletion = (taskId, result) => {
    const taskElement = document.querySelector(`[data-task-id="${taskId}"]`);
    if (taskElement) {
        const resultCard = document.createElement('div');
        resultCard.className = 'result-card';
        resultCard.innerHTML = `
            <h3 class="result-filename">🩻 ${result.filename} 
                ${result.has_tumor 
                    ? '<span class="tumor-status tumor-detected">⚠️ Tumor Detected</span>'
                    : '<span class="tumor-status tumor-not-detected">✅ No Tumor Detected</span>'
                }
            </h3>
            <div class="result-images">
                <div class="result-image-box">
                    <span class="result-label">Original MRI</span>
                    <img src="${result.original_image_url}" alt="Original MRI" />
                </div>
                <div class="result-image-box">
                    <span class="result-label">Segmentation Mask</span>
                    <img src="${result.mask_url}" alt="Segmentation Mask" />
                </div>
                ${result.overlay_url ? `
                <div class="result-image-box">
                    <span class="result-label">Overlay</span>
                    <img src="${result.overlay_url}" alt="Overlay" />
                </div>
                ` : ''}
            </div>
        `;
        
        resultsGrid.appendChild(resultCard);
    }
};

processBtn.addEventListener('click', async () => {
    if (selectedFiles.length === 0) {
        return showMessage(true, '❌ Please select MRI scans first.');
    }

    hideMessages();
    loading.classList.add('active');
    processBtn.disabled = true;
    resultsGrid.innerHTML = '';

    try {
        // Create session ID
        currentSessionId = `session_${Date.now()}`;
        
        // Connect to WebSocket
        connectWebSocket(currentSessionId);
        
        // Process files based on count
        if (selectedFiles.length === 1) {
            // Single file processing
            const file = selectedFiles[0];
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/api/v1/segment', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `Failed to process ${file.name}`);
            }

            const result = await response.json();
            handleTaskCompletion('single', result);
            
        } else {
            // Batch processing
            const formData = new FormData();
            selectedFiles.forEach(file => {
                formData.append('files', file);
            });

            const response = await fetch('/api/v1/segment-multiple', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Batch processing failed');
            }

            const result = await response.json();
            showMessage(false, `✅ Processing ${selectedFiles.length} MRI scan(s)! Use WebSocket for real-time updates.`);
            
            // Show progress tracking
            showProgressTracking();
        }

        imagesContainer.classList.remove('hidden');
        showMessage(false, `✅ Successfully processed ${selectedFiles.length} MRI scan(s)!`);
        
        setTimeout(() => {
            imagesContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);

    } catch (error) {
        showMessage(true, `❌ Error: ${error.message}`);
    } finally {
        loading.classList.remove('active');
        processBtn.disabled = false;
    }
});

const showProgressTracking = () => {
    // Create progress tracking section
    const progressSection = document.createElement('div');
    progressSection.className = 'progress-section';
    progressSection.innerHTML = `
        <h3>🔄 Processing Progress</h3>
        <div class="progress-tasks" id="progressTasks"></div>
        <div class="progress-controls">
            <button class="btn btn-secondary" onclick="refreshProgress()">
                🔄 Refresh Status
            </button>
            <button class="btn btn-secondary" onclick="cancelAllTasks()">
                ❌ Cancel All
            </button>
        </div>
    `;
    
    // Insert before results
    imagesContainer.insertBefore(progressSection, resultsGrid);
};

const refreshProgress = async () => {
    try {
        const response = await fetch('/api/v1/batch-status');
        const status = await response.json();
        
        // Update progress display
        const progressContainer = document.querySelector('.progress-overview');
        if (progressContainer) {
            progressContainer.remove();
        }
        
        updateProgressDisplay(status);
        
        // Update task list
        const tasksContainer = document.getElementById('progressTasks');
        if (tasksContainer) {
            tasksContainer.innerHTML = status.tasks.map(task => `
                <div class="task-item" data-task-id="${task.task_id}">
                    <div class="task-info">
                        <span class="task-name">${task.filename}</span>
                        <span class="task-status ${task.status}">${task.status}</span>
                    </div>
                    <div class="task-progress-bar">
                        <input type="range" class="task-progress" value="${task.progress}" 
                               max="100" readonly>
                        <span class="progress-text">${task.progress}%</span>
                    </div>
                    <div class="task-time">
                        <small>Created: ${new Date(task.created_at).toLocaleTimeString()}</small>
                    </div>
                </div>
            `).join('');
        }
        
    } catch (error) {
        showMessage(true, `❌ Error refreshing progress: ${error.message}`);
    }
};

const cancelAllTasks = async () => {
    try {
        const response = await fetch('/api/v1/batch-status');
        const status = await response.json();
        
        for (const task of status.tasks) {
            if (task.status === 'pending' || task.status === 'processing') {
                await fetch(`/api/v1/cancel-task/${task.task_id}`, {
                    method: 'POST'
                });
            }
        }
        
        showMessage(false, '✅ All pending/processing tasks cancelled');
        refreshProgress();
        
    } catch (error) {
        showMessage(true, `❌ Error cancelling tasks: ${error.message}`);
    }
};

resetBtn.addEventListener('click', () => {
    selectedFiles = [];
    fileInput.value = '';
    fileList.innerHTML = '';
    resultsGrid.innerHTML = '';
    selectedFilesDiv.classList.add('hidden');
    imagesContainer.classList.add('hidden');
    buttonsContainer.classList.add('hidden');
    
    // Remove progress tracking
    const progressSection = document.querySelector('.progress-section');
    if (progressSection) {
        progressSection.remove();
    }
    
    hideMessages();
    uploadArea.scrollIntoView({ behavior: 'smooth', block: 'center' });
    
    // Close WebSocket
    if (websocket) {
        websocket.close();
        websocket = null;
    }
});

// Auto-refresh progress for batch processing
if (selectedFiles.length > 1) {
    progressInterval = setInterval(refreshProgress, 2000);
}

// Make functions globally available
window.refreshProgress = refreshProgress;
window.cancelAllTasks = cancelAllTasks;