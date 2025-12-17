const dropArea = document.getElementById('drop-area');
const fileElem = document.getElementById('fileElem');
const previewSection = document.getElementById('preview-section');
const resultSection = document.getElementById('result-section');
const imagePreview = document.getElementById('image-preview');
const loader = document.getElementById('loader');
const resultLabel = document.getElementById('result-label');
const confidenceBar = document.getElementById('confidence-bar');
const confidenceScore = document.getElementById('confidence-score');
const resetBtn = document.getElementById('reset-btn');

// Prevent default drag behaviors
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, preventDefaults, false)   
  document.body.addEventListener(eventName, preventDefaults, false)
});

function preventDefaults (e) {
  e.preventDefault()
  e.stopPropagation()
}

// Highlight drop area when item is dragged over it
['dragenter', 'dragover'].forEach(eventName => {
  dropArea.addEventListener(eventName, highlight, false)
});

['dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, unhighlight, false)
});

function highlight(e) {
  dropArea.classList.add('highlight')
}

function unhighlight(e) {
  dropArea.classList.remove('highlight')
}

// Handle dropped files
dropArea.addEventListener('drop', handleDrop, false)

function handleDrop(e) {
  const dt = e.dataTransfer
  const files = dt.files
  handleFiles(files)
}

function handleFiles(files) {
  const file = files[0];
  if (!file.type.startsWith('image/')) return;
  
  // Show image preview
  const reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onloadend = function() {
      imagePreview.src = reader.result;
      dropArea.style.display = 'none';
      previewSection.style.display = 'block';
      loader.style.display = 'block';
      
      // Upload to server
      uploadFile(file);
  }
}

function uploadFile(file) {
    const url = '/predict';
    const formData = new FormData();
    formData.append('file', file);

    fetch(url, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Mock delay for UI feeling
        setTimeout(() => {
            loader.style.display = 'none';
            showResult(data);
        }, 1500); 
    })
    .catch(error => { 
        console.error('Error:', error);
        loader.style.display = 'none';
        resultLabel.innerText = "Error analyzing image";
    });
}

function showResult(data) {
    resultSection.style.display = 'block';
    resetBtn.style.display = 'block';

    if (data.error) {
         resultLabel.innerText = "Error: " + data.error;
         return;
    }

    resultLabel.innerText = data.label;
    confidenceScore.innerText = data.confidence;
    
    // Set bar width
    let prob = data.probability;
    // Map probability to visual confidence
    let visualConf = 0;
    if (data.label.includes("Malignant")) {
        visualConf = prob * 100;
        resultLabel.style.color = 'var(--danger)';
        confidenceBar.style.background = 'var(--danger)';
    } else {
        visualConf = (1 - prob) * 100;
        resultLabel.style.color = 'var(--accent)';
        confidenceBar.style.background = 'var(--accent)';
    }
    
    confidenceBar.style.width = visualConf + '%';
}

function resetApp() {
    dropArea.style.display = 'block';
    previewSection.style.display = 'none';
    resultSection.style.display = 'none';
    resetBtn.style.display = 'none';
    fileElem.value = "";
    confidenceBar.style.width = '0%';
}

dropArea.addEventListener('click', () => {
    fileElem.click();
});
