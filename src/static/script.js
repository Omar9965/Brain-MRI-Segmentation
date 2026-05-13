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

    if (files.length > 1) {
        showMessage(true, `❌ Please upload only one MRI scan at a time.`);
        return;
    }

    const file = files[0];
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff', 'image/tif'];
    const validExtensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff'];
    
    const ext = file.name.toLowerCase().slice(file.name.lastIndexOf('.'));
    if (!validTypes.includes(file.type) && !validExtensions.includes(ext)) {
        showMessage(true, `❌ "${file.name}" has invalid type. Use JPG, PNG, or TIFF.`);
        return;
    }
    if (file.size > 500 * 1024 * 1024) {
        showMessage(true, `❌ "${file.name}" is too large. Max 500MB.`);
        return;
    }

    selectedFiles = [file];
    showMessage(false, `✅ MRI scan selected successfully!`);

    // Show file list
    fileList.innerHTML = `<li>🩻 ${file.name}</li>`;
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

const renderResultCard = (result) => {
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
};

const handleTaskCompletion = (taskId, result) => {
    renderResultCard(result);
};

processBtn.addEventListener('click', async () => {
    if (selectedFiles.length === 0) {
        return showMessage(true, '❌ Please select an MRI scan first.');
    }

    hideMessages();
    loading.classList.add('active');
    processBtn.disabled = true;
    resultsGrid.innerHTML = '';

    try {
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

        imagesContainer.classList.remove('hidden');
        showMessage(false, `✅ Successfully processed MRI scan!`);
        
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

resetBtn.addEventListener('click', () => {
    selectedFiles = [];
    fileInput.value = '';
    fileList.innerHTML = '';
    resultsGrid.innerHTML = '';
    selectedFilesDiv.classList.add('hidden');
    imagesContainer.classList.add('hidden');
    buttonsContainer.classList.add('hidden');
    
    hideMessages();
    uploadArea.scrollIntoView({ behavior: 'smooth', block: 'center' });
});