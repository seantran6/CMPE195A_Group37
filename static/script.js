document.addEventListener('DOMContentLoaded', () => {
    // === DOM Elements ===
    const $ = id => document.getElementById(id);
    const video = $('webcam');
    const captureBtn = $('capture-btn');
    const webcamPlaceholder = $('webcam-placeholder');
    const capturedImage = $('captured-image');
    const canvas = $('canvas');
    const analyzingOverlay = $('analyzing-overlay');
    const progressBar = $('progress-bar');
    const analysisResults = $('analysis-results');
    const gender = $('gender');
    const age = $('age');
    const recommendations = $('recommendations');
    const songsGrid = $('songs-grid');
    const songsList = $('songs-list');
    const themeToggle = $('theme-toggle');
    const menuButton = document.querySelector('.menu-button');
    const mainNav = document.querySelector('.main-nav');
    const tabTriggers = document.querySelectorAll('.tab-trigger');
    const tabContents = document.querySelectorAll('.tab-content');

    // === State ===
    let stream = null;
    let isCapturing = false;
    let isAnalyzing = false;
    let progressInterval = null;

    // === Initialization ===
    initTheme();
    bindThemeToggle();
    bindMobileMenu();
    bindTabs();
    bindWebcam();
    handleResponsiveLayout();
    window.addEventListener('resize', handleResponsiveLayout);
    document.addEventListener('keydown', handleKeyboardShortcuts);
    setTimeout(improveAccessibility, 1000);

    // === Theme ===
    function initTheme() {
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const savedTheme = localStorage.getItem('theme');
        document.documentElement.classList.toggle('dark', savedTheme === 'dark' || (!savedTheme && prefersDark));
    }

    function bindThemeToggle() {
        themeToggle.addEventListener('click', () => {
            const isDark = document.documentElement.classList.toggle('dark');
            localStorage.setItem('theme', isDark ? 'dark' : 'light');
        });
    }

    // === Mobile Navigation ===
    function bindMobileMenu() {
        menuButton?.addEventListener('click', toggleMobileMenu);

        document.addEventListener('click', e => {
            if (mainNav?.style.display === 'flex' && !mainNav.contains(e.target) && !menuButton.contains(e.target)) {
                mainNav.style.display = 'none';
            }
        });

        window.addEventListener('resize', () => {
            if (window.innerWidth >= 768) {
                Object.assign(mainNav.style, {
                    display: 'flex',
                    position: 'static',
                    flexDirection: 'row',
                    padding: '0',
                    boxShadow: 'none',
                    opacity: '1',
                    transform: 'none'
                });
            }
        });
    }

    function toggleMobileMenu() {
        const visible = mainNav.style.display === 'flex';
        mainNav.style.display = visible ? 'none' : 'flex';
        if (!visible) {
            Object.assign(mainNav.style, {
                flexDirection: 'column',
                position: 'absolute',
                top: '4rem',
                right: '1rem',
                backgroundColor: 'var(--background)',
                padding: '1rem',
                borderRadius: 'var(--radius)',
                boxShadow: '0 10px 25px -5px rgba(0,0,0,0.1)',
                zIndex: '50',
                opacity: '0',
                transform: 'translateY(-10px)',
                transition: 'opacity 0.3s ease, transform 0.3s ease'
            });
            setTimeout(() => {
                mainNav.style.opacity = '1';
                mainNav.style.transform = 'translateY(0)';
            }, 10);
        }
    }

    // === Tabs ===
    function bindTabs() {
        tabTriggers.forEach(trigger => {
            trigger.addEventListener('click', () => {
                const tabName = trigger.getAttribute('data-tab');
                tabTriggers.forEach(t => t.classList.remove('active'));
                trigger.classList.add('active');
                tabContents.forEach(content => {
                    content.classList.toggle('active', content.id === `${tabName}-view`);
                });
            });
        });
    }

    // === Webcam & Capture ===
    function bindWebcam() {
        captureBtn.addEventListener('click', () => {
            isCapturing ? captureFrame() : startWebcam();
        });
    }

    async function startWebcam() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            webcamPlaceholder.style.display = 'none';
            video.style.display = 'block';
            captureBtn.textContent = 'Capture Frame';
            isCapturing = true;
        } catch (err) {
            alert('Could not access webcam.');
        }
    }

    function captureFrame() {
        if (!isCapturing) return;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg');
        capturedImage.src = imageData;
        video.style.display = 'none';
        capturedImage.style.display = 'block';
        stopWebcam();
        captureBtn.textContent = 'Try Again';
        captureBtn.addEventListener('click', resetCapture, { once: true });
        startAnalysis(imageData);
    }

    function stopWebcam() {
        stream?.getTracks().forEach(track => track.stop());
        stream = null;
        isCapturing = false;
    }

    function resetCapture() {
        capturedImage.style.display = 'none';
        webcamPlaceholder.style.display = 'block';
        analysisResults.style.display = 'none';
        recommendations.style.display = 'none';
        captureBtn.textContent = 'Start Webcam';
        isCapturing = false;
        isAnalyzing = false;
        clearInterval(progressInterval);
    }

    // === Analysis ===
    async function startAnalysis(imageData) {
        analyzingOverlay.style.display = 'flex';
        progressBar.style.width = '0%';
        isAnalyzing = true;

        simulateProgressBar();

        try {
            const res = await fetch('/recognize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });
            const data = await res.json();
            analyzingOverlay.style.display = 'none';
            showAnalysisResults(data);
        } catch (err) {
            analyzingOverlay.style.display = 'none';
            alert('Error analyzing image.');
        }
    }

    function simulateProgressBar() {
        let progress = 0;
        progressInterval = setInterval(() => {
            progress += 5;
            progressBar.style.width = `${progress}%`;
            if (progress >= 100) clearInterval(progressInterval);
        }, 100);
    }

    function showAnalysisResults({ gender: g, age: a }) {
        gender.textContent = g;
        age.textContent = a;
        analysisResults.style.display = 'block';
        analysisResults.querySelectorAll('[data-animation]').forEach(el => el.classList.add('animate-in'));
        triggerConfetti();
        fetchRecommendations(a, g);
    }

    function triggerConfetti() {
        const colors = ['#ff5bac', '#7928ca', '#0070f3'];
        for (let i = 0; i < 100; i++) {
            const confetti = document.createElement('div');
            confetti.className = 'confetti';
            confetti.style.cssText = `
                background-color: ${colors[Math.floor(Math.random() * colors.length)]};
                left: ${Math.random() * 100}vw;
                animation-duration: ${Math.random() * 3 + 2}s;
                opacity: ${Math.random()};
                transform: rotate(${Math.random() * 360}deg);
            `;
            document.body.appendChild(confetti);
            setTimeout(() => confetti.remove(), 5000);
        }
    }

    // === Recommendations ===
    async function fetchRecommendations(age, gender) {
        try {
            const res = await fetch(`/recommend_tracks?age=${age}&gender=${gender}&n=6`);
            const data = await res.json();
            renderSongs(data.tracks);
        } catch (err) {
            console.error('Error fetching recommendations:', err);
        }
    }

    function renderSongs(tracks) {
        songsGrid.innerHTML = '';
        songsList.innerHTML = '';
        recommendations.style.display = 'block';

        if (!tracks?.length) {
            songsGrid.innerHTML = '<p>No recommendations.</p>';
            songsList.innerHTML = '<p>No recommendations.</p>';
            return;
        }

        const cardTpl = $('song-card-template');
        const listTpl = $('song-list-item-template');

        tracks.forEach((track, i) => {
            setTimeout(() => {
                songsGrid.appendChild(createSongCard(track, cardTpl));
                songsList.appendChild(createSongListItem(track, listTpl));
            }, i * 100);
        });

        lucide.createIcons();
    }

    function createSongCard(track, template) {
        const card = template.content.cloneNode(true);
        const songCard = card.querySelector('.song-card');
        const img = card.querySelector('.song-image');
        const playBtn = card.querySelector('.play-button');
        const likeBtn = card.querySelector('.like-button');
        const progress = card.querySelector('.progress-indicator');

        img.src = track.image || 'https://via.placeholder.com/300?text=No+Image';
        card.querySelector('.song-title').textContent = track.title;
        card.querySelector('.song-artist').textContent = track.artist;
        playBtn.href = track.link;

        if (track.preview) {
            likeBtn.addEventListener('click', () => {
                const audio = new Audio(track.preview);
                audio.play();
                progress.style.display = 'block';
                setTimeout(() => progress.style.display = 'none', 3000);
            });
        } else {
            likeBtn.style.display = 'none';
        }

        return card;
    }

    function createSongListItem(track, template) {
        const item = template.content.cloneNode(true);
        item.querySelector('.song-title').textContent = track.title;
        item.querySelector('.song-artist').textContent = track.artist;
        item.querySelector('.play-button').href = track.link;
        return item;
    }

    // === Accessibility Enhancements ===
    function handleKeyboardShortcuts(e) {
        if (e.key === '/') {
            e.preventDefault();
            $('search')?.focus();
        }
    }

    function improveAccessibility() {
        document.querySelectorAll('button, a, input, [tabindex]').forEach(el => {
            if (!el.getAttribute('aria-label') && !el.textContent.trim()) {
                el.setAttribute('aria-label', 'button');
            }
        });
    }

    // === Responsive Layout ===
    function handleResponsiveLayout() {
        const view = window.innerWidth >= 768 ? 'grid' : 'list';
        $('grid-tab')?.classList.toggle('active', view === 'grid');
        $('list-tab')?.classList.toggle('active', view === 'list');
        $('grid-view')?.classList.toggle('active', view === 'grid');
        $('list-view')?.classList.toggle('active', view === 'list');
    }
});
