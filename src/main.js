import * as tf from '@tensorflow/tfjs';
import * as blazeface from '@tensorflow-models/blazeface';
import './style.css';

// Initialize components
let model = null;
let video = null;
let canvas = null;
let ctx = null;
let isDetecting = false;
let detectionCount = 0;

// DOM elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const captureBtn = document.getElementById('captureBtn');
const statusText = document.getElementById('statusText');
const detectionCountEl = document.getElementById('detectionCount');
const screenshotsContainer = document.getElementById('screenshots');

// Initialize the application
async function init() {
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    
    statusText.textContent = 'Loading face detection model...';
    
    try {
        // Load the BlazeFace model
        model = await blazeface.load();
        statusText.textContent = 'Model loaded successfully! Click "Start Camera" to begin.';
        console.log('BlazeFace model loaded successfully');
    } catch (error) {
        statusText.textContent = 'Error loading model. Please refresh the page.';
        console.error('Error loading model:', error);
    }
}

// Start camera and detection
async function startCamera() {
    if (!model) {
        statusText.textContent = 'Model not loaded yet. Please wait...';
        return;
    }
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480 } 
        });
        
        video.srcObject = stream;
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            startBtn.disabled = true;
            stopBtn.disabled = false;
            captureBtn.disabled = false;
            isDetecting = true;
            
            statusText.textContent = 'Camera started. Detecting faces...';
            detectFaces();
        };
    } catch (error) {
        statusText.textContent = 'Error accessing camera. Please check permissions.';
        console.error('Error accessing camera:', error);
    }
}

// Stop camera and detection
function stopCamera() {
    if (video.srcObject) {
        const tracks = video.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
    }
    
    isDetecting = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    startBtn.disabled = false;
    stopBtn.disabled = true;
    captureBtn.disabled = true;
    
    statusText.textContent = 'Camera stopped. Click "Start Camera" to begin again.';
    detectionCount = 0;
    updateDetectionCount();
}

// Face detection function
async function detectFaces() {
    if (!isDetecting || !model || !video.videoWidth) {
        if (isDetecting) {
            requestAnimationFrame(detectFaces);
        }
        return;
    }
    
    try {
        // Get predictions from the model
        const predictions = await model.estimateFaces(video, false);
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw bounding boxes for detected faces
        if (predictions.length > 0) {
            detectionCount = predictions.length;
            
            predictions.forEach((prediction, index) => {
                const start = prediction.topLeft;
                const end = prediction.bottomRight;
                const size = [end[0] - start[0], end[1] - start[1]];
                
                // Draw bounding box
                ctx.strokeStyle = '#00ff00';
                ctx.lineWidth = 3;
                ctx.strokeRect(start[0], start[1], size[0], size[1]);
                
                // Draw label
                ctx.fillStyle = '#00ff00';
                ctx.font = '16px Arial';
                ctx.fillText(`Face ${index + 1}`, start[0], start[1] - 10);
                
                // Draw confidence score if available
                if (prediction.probability) {
                    const confidence = (prediction.probability[0] * 100).toFixed(1);
                    ctx.fillText(`${confidence}%`, start[0], start[1] + size[1] + 20);
                }
            });
        } else {
            detectionCount = 0;
        }
        
        updateDetectionCount();
        
    } catch (error) {
        console.error('Error during face detection:', error);
    }
    
    // Continue detection loop
    if (isDetecting) {
        requestAnimationFrame(detectFaces);
    }
}

// Update detection count display
function updateDetectionCount() {
    detectionCountEl.textContent = `Faces detected: ${detectionCount}`;
}

// Capture screenshot
function captureScreenshot() {
    if (!video.videoWidth) return;
    
    // Create a temporary canvas for the screenshot
    const screenshotCanvas = document.createElement('canvas');
    const screenshotCtx = screenshotCanvas.getContext('2d');
    
    screenshotCanvas.width = video.videoWidth;
    screenshotCanvas.height = video.videoHeight;
    
    // Draw video frame
    screenshotCtx.drawImage(video, 0, 0);
    
    // Draw detection overlay
    screenshotCtx.drawImage(canvas, 0, 0);
    
    // Convert to image and display
    const imageData = screenshotCanvas.toDataURL('image/png');
    const img = document.createElement('img');
    img.src = imageData;
    img.className = 'screenshot';
    
    const timestamp = new Date().toLocaleTimeString();
    const caption = document.createElement('p');
    caption.textContent = `Screenshot taken at ${timestamp} - ${detectionCount} face(s) detected`;
    
    const screenshotDiv = document.createElement('div');
    screenshotDiv.className = 'screenshot-item';
    screenshotDiv.appendChild(img);
    screenshotDiv.appendChild(caption);
    
    screenshotsContainer.appendChild(screenshotDiv);
    
    statusText.textContent = `Screenshot captured! ${detectionCount} face(s) detected.`;
}

// Event listeners
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
captureBtn.addEventListener('click', captureScreenshot);

// Initialize the application
init();