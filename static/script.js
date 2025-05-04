// script.js
const video = document.getElementById('webcam');
const captureBtn = document.getElementById('capture-btn');
const genderOutput = document.getElementById('gender');
const ageOutput = document.getElementById('age');
const modelSelector = document.getElementById('model-selector'); // <-- dropdown reference

// Start webcam
async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  } catch (error) {
    console.error('Error accessing webcam:', error);
  }
}

// Capture frame from video and send to server
async function captureFrame() {
  console.log('Capture frame button clicked');
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const context = canvas.getContext('2d');
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  const imageData = canvas.toDataURL('image/jpeg');
  const selectedModel = modelSelector.value; // <-- get selected model
  console.log('Sending image data and selected model to server:', selectedModel);

  try {
    const response = await fetch('/recognize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData, model: selectedModel }) // <-- send model too
    });

    const result = await response.json();
    console.log('Received response from server:', result);

    // Display gender and age predictions
    genderOutput.textContent = `Gender: ${result.gender}`;
    ageOutput.textContent = `Age: ${result.age}`;

    // Show and scroll to recommendations
    const recSection = document.getElementById('recommendations');
    recSection.classList.remove('hidden');

    // Give the browser a little time to render it
    setTimeout(() => {
      const yOffset = -100;
      const y = recSection.getBoundingClientRect().top + window.pageYOffset + yOffset;
      window.scrollTo({ top: y, behavior: 'smooth' });
    }, 300);

  } catch (error) {
    console.error('Error sending frame to server:', error);
  }
}

captureBtn.addEventListener('click', captureFrame);

// Start webcam on load
startWebcam();
