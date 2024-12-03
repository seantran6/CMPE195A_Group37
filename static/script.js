// script.js
const video = document.getElementById('webcam');
const captureBtn = document.getElementById('capture-btn');
const genderOutput = document.getElementById('gender');
const ageOutput = document.getElementById('age');

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
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const context = canvas.getContext('2d');
  context.drawImage(video, 0, 0, canvas.width, canvas.height);

  const imageData = canvas.toDataURL('image/jpeg');

  try {
    const response = await fetch('/recognize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData })
    });

    const result = await response.json();
    genderOutput.textContent = result.gender;
    ageOutput.textContent = result.age;
  } catch (error) {
    console.error('Error sending frame to server:', error);
  }
}

captureBtn.addEventListener('click', captureFrame);

// Start webcam on load
startWebcam();
