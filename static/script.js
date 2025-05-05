// script.js
const video = document.getElementById('webcam');
const captureBtn = document.getElementById('capture-btn');
const genderOutput = document.getElementById('gender');
const ageOutput = document.getElementById('age');
const modelSelector = document.getElementById('model-selector');

// Start webcam
async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  } catch (error) {
    console.error('Error accessing webcam:', error);
  }
}

// Listen for model change
modelSelector.addEventListener('change', () => {
  const selected = modelSelector.value;
  console.log(`[Dropdown] Model changed to: ${selected}`);
});

// Capture frame from video and send to server
async function captureFrame() {
  console.log('[Capture] Button clicked');

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const context = canvas.getContext('2d');
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  const imageData = canvas.toDataURL('image/jpeg');
  const selectedModel = modelSelector.value;

  console.log(`[Capture] Sending image with model: ${selectedModel}`);

  try {
    const response = await fetch('/recognize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData, model: selectedModel })
    });

    const result = await response.json();
    console.log('[Server Response]', result);

    genderOutput.textContent = `Gender: ${result.gender}`;
    ageOutput.textContent = `Age: ${result.age}`;

    const recSection = document.getElementById('recommendations');
    recSection.classList.remove('hidden');

    setTimeout(() => {
      const yOffset = -100;
      const y = recSection.getBoundingClientRect().top + window.pageYOffset + yOffset;
      window.scrollTo({ top: y, behavior: 'smooth' });
    }, 300);

  } catch (error) {
    console.error('[Error] Sending frame to server:', error);
  }
}

captureBtn.addEventListener('click', captureFrame);
startWebcam();
