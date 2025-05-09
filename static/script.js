document.addEventListener('DOMContentLoaded', function () {
    // DOM Elements
    const video = document.getElementById('webcam');
    const captureBtn = document.getElementById('capture-btn');
    const webcamPlaceholder = document.getElementById('webcam-placeholder');
    const capturedImage = document.getElementById('captured-image');
    const canvas = document.getElementById('canvas');
    const analyzingOverlay = document.getElementById('analyzing-overlay');
    const progressBar = document.getElementById('progress-bar');
    const analysisResults = document.getElementById('analysis-results');
    const gender = document.getElementById('gender');
    const age = document.getElementById('age');
    const recommendations = document.getElementById('recommendations');
    const songsGrid = document.getElementById('songs-grid');
    const songsList = document.getElementById('songs-list');
    const tabTriggers = document.querySelectorAll('.tab-trigger');
    const tabContents = document.querySelectorAll('.tab-content');
    const themeToggle = document.getElementById('theme-toggle');
    const menuButton = document.querySelector('.menu-button');
    const mainNav = document.querySelector('.main-nav');

    // State
    let stream = null;
    let isCapturing = false;
    let isAnalyzing = false;
    let progressInterval = null;

    // Theme handling
    function initializeTheme() {
        if (localStorage.getItem('theme') === 'dark' ||
            (!localStorage.getItem('theme') && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
            document.documentElement.classList.add('dark');
        } else {
            document.documentElement.classList.remove('dark');
        }
    }

    initializeTheme();

    themeToggle.addEventListener('click', function () {
        if (document.documentElement.classList.contains('dark')) {
            document.documentElement.classList.remove('dark');
            localStorage.setItem('theme', 'light');
        } else {
            document.documentElement.classList.add('dark');
            localStorage.setItem('theme', 'dark');
        }
    });

    // Mobile menu handling
    if (menuButton && mainNav) {
        menuButton.addEventListener('click', function () {
            if (mainNav.style.display === 'flex') {
                mainNav.style.display = 'none';
                mainNav.style.opacity = '0';
                mainNav.style.transform = 'translateY(-10px)';
            } else {
                mainNav.style.display = 'flex';
                mainNav.style.flexDirection = 'column';
                mainNav.style.position = 'absolute';
                mainNav.style.top = '4rem';
                mainNav.style.right = '1rem';
                mainNav.style.backgroundColor = 'var(--background)';
                mainNav.style.padding = '1rem';
                mainNav.style.borderRadius = 'var(--radius)';
                mainNav.style.boxShadow = '0 10px 25px -5px rgba(0, 0, 0, 0.1)';
                mainNav.style.zIndex = '50';
                mainNav.style.opacity = '0';
                mainNav.style.transform = 'translateY(-10px)';
                mainNav.style.transition = 'opacity 0.3s ease, transform 0.3s ease';

                // Animate in
                setTimeout(() => {
                    mainNav.style.opacity = '1';
                    mainNav.style.transform = 'translateY(0)';
                }, 10);
            }
        });

        // Close mobile menu when clicking outside
        document.addEventListener('click', function (event) {
            if (mainNav.style.display === 'flex' &&
                !mainNav.contains(event.target) &&
                !menuButton.contains(event.target)) {
                mainNav.style.display = 'none';
            }
        });

        // Handle window resize
        window.addEventListener('resize', function () {
            if (window.innerWidth >= 768) {
                mainNav.style.display = 'flex';
                mainNav.style.position = 'static';
                mainNav.style.flexDirection = 'row';
                mainNav.style.padding = '0';
                mainNav.style.boxShadow = 'none';
                mainNav.style.opacity = '1';
                mainNav.style.transform = 'none';
            } else if (!menuButton.contains(event.target)) {
                mainNav.style.display = 'none';
            }
        });
    }

    // Tab handling
    tabTriggers.forEach(trigger => {
        trigger.addEventListener('click', () => {
            const tabName = trigger.getAttribute('data-tab');

            // Update active tab trigger
            tabTriggers.forEach(t => t.classList.remove('active'));
            trigger.classList.add('active');

            // Update active tab content
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${tabName}-view`) {
                    content.classList.add('active');
                }
            });
        });
    });

    // Webcam handling
    async function startWebcam() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            webcamPlaceholder.style.display = 'none';
            video.style.display = 'block';
            captureBtn.textContent = 'Capture Frame';
            isCapturing = true;
        } catch (err) {
            console.error('Error accessing webcam:', err);
            alert('Could not access webcam. Please make sure you have a webcam connected and have granted permission to use it.');
        }
    }

    captureBtn.addEventListener('click', function () {
        if (!isCapturing) {
            startWebcam();
        } else {
            captureFrame();
        }
    });

    async function captureFrame() {
        if (!isCapturing) return;

        // Draw the video frame to the canvas
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert the canvas to an image
        const imageData = canvas.toDataURL('image/jpeg');
        capturedImage.src = imageData;

        // Show the captured image and hide the video
        video.style.display = 'none';
        capturedImage.style.display = 'block';

        // Stop the webcam stream
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }

        // Update button
        captureBtn.textContent = 'Try Again';
        captureBtn.addEventListener('click', resetCapture, { once: true });
        isCapturing = false;

        // Start analysis
        startAnalysis(imageData);
    }

    function resetCapture() {
        // Reset UI
        capturedImage.style.display = 'none';
        webcamPlaceholder.style.display = 'block';
        analysisResults.style.display = 'none';
        recommendations.style.display = 'none';

        // Reset state
        isCapturing = false;
        isAnalyzing = false;
        if (progressInterval) {
            clearInterval(progressInterval);
            progressInterval = null;
        }

        // Reset button
        captureBtn.textContent = 'Start Webcam';
    }

    async function startAnalysis(imageData) {
        // Show analyzing overlay
        analyzingOverlay.style.display = 'flex';
        isAnalyzing = true;

        // Reset progress bar
        progressBar.style.width = '0%';

        // Simulate progress while analysis is happening
        let progress = 0;
        progressInterval = setInterval(() => {
            progress += 5;
            progressBar.style.width = `${progress}%`;

            if (progress >= 100) {
                clearInterval(progressInterval);
            }
        }, 100);

        try {
            // Send image to backend for analysis
            const response = await fetch('/recognize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageData })
            });

            const result = await response.json();

            // Hide analyzing overlay
            analyzingOverlay.style.display = 'none';
            isAnalyzing = false;

            // Update the UI with results
            gender.textContent = result.gender;
            age.textContent = result.age;

            // Show analysis results
            analysisResults.style.display = 'block';

            // Animate in the analysis items
            const animationItems = analysisResults.querySelectorAll('[data-animation]');
            animationItems.forEach(item => {
                item.classList.add('animate-in');
            });

            // Trigger confetti effect
            triggerConfetti();

            // Fetch and show recommendations
            fetchRecommendations(result.age, result.gender);

        } catch (error) {
            console.error('Error during analysis:', error);
            analyzingOverlay.style.display = 'none';
            alert('Error analyzing image. Please try again.');
        }
    }

    function triggerConfetti() {
        // Create confetti
        const colors = ['#ff5bac', '#7928ca', '#0070f3'];
        const count = 100;

        for (let i = 0; i < count; i++) {
            const confetti = document.createElement('div');
            confetti.className = 'confetti';
            confetti.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
            confetti.style.left = Math.random() * 100 + 'vw';
            confetti.style.animationDuration = (Math.random() * 3 + 2) + 's';
            confetti.style.opacity = Math.random();
            confetti.style.transform = 'rotate(' + Math.random() * 360 + 'deg)';

            document.body.appendChild(confetti);

            // Remove after animation
            setTimeout(() => {
                confetti.remove();
            }, 5000);
        }
    }

    async function fetchRecommendations(age, gender) {
        try {
            const response = await fetch(`/recommend_tracks?age=${encodeURIComponent(age)}&gender=${encodeURIComponent(gender)}&n=6`);
            const data = await response.json();

            // Display recommendations
            recommendations.style.display = 'block';

            // Populate song cards
            renderSongs(data.tracks);

        } catch (error) {
            console.error('Error fetching recommendations:', error);
        }
    }

    function renderSongs(tracks) {
        // Clear existing songs
        songsGrid.innerHTML = '';
        songsList.innerHTML = '';

        if (!tracks || !tracks.length) {
            songsGrid.innerHTML = '<p>No recommendations available.</p>';
            songsList.innerHTML = '<p>No recommendations available.</p>';
            return;
        }

        // Get templates
        const cardTemplate = document.getElementById('song-card-template');
        const listItemTemplate = document.getElementById('song-list-item-template');

        // Populate grid view
        tracks.forEach((track, index) => {
            const card = cardTemplate.content.cloneNode(true);

            // Set song data
            const image = card.querySelector('.song-image');
            image.src = track.image || track.cover || 'https://via.placeholder.com/300?text=No+Image';
            image.alt = track.name || track.title;

            card.querySelector('.song-title').textContent = track.name || track.title;
            card.querySelector('.song-artist').textContent = track.artists || track.artist;

            // Add data attribute for track ID
            const songCard = card.querySelector('.song-card');
            songCard.setAttribute('data-track-id', track.id);

            // Add event listeners
            const playButton = card.querySelector('.play-button');
            playButton.setAttribute('data-track-id', track.id);
            playButton.addEventListener('click', (e) => {
                e.stopPropagation();
                togglePlaySong(track);
            });

            const likeButton = card.querySelector('.like-button');
            likeButton.addEventListener('click', (e) => {
                e.stopPropagation();
                toggleLikeSong(likeButton);
            });

            // Add progress indicator data attribute
            const progressIndicator = card.querySelector('.progress-indicator');
            progressIndicator.setAttribute('data-track-id', track.id);

            // Make whole card clickable to open Spotify link
            songCard.addEventListener('click', () => {
                window.open(track.track_url || '#', '_blank');
            });

            // Add with delay for animation
            setTimeout(() => {
                songsGrid.appendChild(card);
            }, index * 100);
        });

        // Populate list view
        tracks.forEach((track, index) => {
            const listItem = listItemTemplate.content.cloneNode(true);

            // Set song data
            const image = listItem.querySelector('.song-list-image');
            image.src = track.image || track.cover || 'https://via.placeholder.com/300?text=No+Image';
            image.alt = track.name || track.title;

            listItem.querySelector('.song-list-title').textContent = track.name || track.title;
            listItem.querySelector('.song-list-artist').textContent = track.artists || track.artist;

            // Add data attribute for track ID
            const songListItem = listItem.querySelector('.song-list-item');
            songListItem.setAttribute('data-track-id', track.id);

            // Add event listeners
            const playButton = listItem.querySelector('.song-list-play');
            playButton.setAttribute('data-track-id', track.id);
            playButton.addEventListener('click', (e) => {
                e.stopPropagation();
                togglePlaySong(track);
            });

            // Make whole item clickable to open Spotify link
            songListItem.addEventListener('click', () => {
                window.open(track.track_url || '#', '_blank');
            });

            // Add with delay for animation
            setTimeout(() => {
                songsList.appendChild(listItem);
            }, index * 100);
        });

        // Initialize Lucide icons for the new elements
        lucide.createIcons();
    }

    function togglePlaySong(track) {
        if (track.preview_url) {
            let audio = document.getElementById(`audio-${track.id}`);

            if (!audio) {
                // Create new audio element if it doesn't exist
                audio = document.createElement('audio');
                audio.id = `audio-${track.id}`;
                audio.src = track.preview_url;
                document.body.appendChild(audio);

                audio.addEventListener('ended', () => {
                    const playButtons = document.querySelectorAll(`.play-button[data-track-id="${track.id}"]`);
                    playButtons.forEach(btn => {
                        const icon = btn.querySelector('i');
                        icon.setAttribute('data-lucide', 'play');
                        lucide.createIcons({
                            icons: {
                                play: icon
                            }
                        });
                    });

                    // Reset progress indicators
                    const progressIndicators = document.querySelectorAll(`.progress-indicator[data-track-id="${track.id}"]`);
                    progressIndicators.forEach(indicator => {
                        indicator.style.width = '0%';
                    });
                });
            }

            const isPlaying = !audio.paused;

            // Pause all other audios
            document.querySelectorAll('audio').forEach(a => {
                if (a.id !== audio.id) {
                    a.pause();

                    // Reset icons for all other tracks
                    const trackId = a.id.replace('audio-', '');
                    const otherPlayButtons = document.querySelectorAll(`.play-button[data-track-id="${trackId}"]`);
                    otherPlayButtons.forEach(btn => {
                        const icon = btn.querySelector('i');
                        icon.setAttribute('data-lucide', 'play');
                        lucide.createIcons({
                            icons: {
                                play: icon
                            }
                        });
                    });

                    // Reset progress indicators
                    const otherProgressIndicators = document.querySelectorAll(`.progress-indicator[data-track-id="${trackId}"]`);
                    otherProgressIndicators.forEach(indicator => {
                        indicator.style.width = '0%';
                    });
                }
            });

            // Toggle play/pause for this audio
            if (isPlaying) {
                audio.pause();
            } else {
                audio.play();

                // Animate progress bar
                const progressIndicators = document.querySelectorAll(`.progress-indicator[data-track-id="${track.id}"]`);
                progressIndicators.forEach(indicator => {
                    indicator.style.width = '0%';
                    indicator.style.transition = 'width 30s linear';
                    setTimeout(() => {
                        indicator.style.width = '100%';
                    }, 50);
                });
            }

            // Update all play buttons for this track
            const playButtons = document.querySelectorAll(`.play-button[data-track-id="${track.id}"]`);
            playButtons.forEach(btn => {
                const icon = btn.querySelector('i');
                icon.setAttribute('data-lucide', isPlaying ? 'play' : 'pause');
                lucide.createIcons({
                    icons: {
                        play: icon,
                        pause: icon
                    }
                });
            });
        } else {
            // No preview URL available
            window.open(track.track_url || '#', '_blank');
        }
    }

    function toggleLikeSong(button) {
        button.classList.toggle('active');
        // In a real app, you would send this to your backend
    }

    // Add track IDs to elements for easier targeting
    function addTrackDataAttributes() {
        const playButtons = document.querySelectorAll('.play-button');
        playButtons.forEach(btn => {
            const trackId = btn.closest('[data-track-id]').getAttribute('data-track-id');
            btn.setAttribute('data-track-id', trackId);
        });

        const progressIndicators = document.querySelectorAll('.progress-indicator');
        progressIndicators.forEach(indicator => {
            const trackId = indicator.closest('[data-track-id]').getAttribute('data-track-id');
            indicator.setAttribute('data-track-id', trackId);
        });
    }

    // Handle keyboard navigation
    document.addEventListener('keydown', function (e) {
        // Escape key closes mobile menu
        if (e.key === 'Escape' && mainNav && mainNav.style.display === 'flex' && window.innerWidth < 768) {
            mainNav.style.display = 'none';
        }

        // Space bar toggles play/pause when a song is focused
        if (e.key === ' ' && document.activeElement.classList.contains('play-button')) {
            e.preventDefault();
            document.activeElement.click();
        }
    });

    // Add accessibility improvements
    function improveAccessibility() {
        // Add aria-labels to buttons without text
        document.querySelectorAll('button:not([aria-label])').forEach(button => {
            if (!button.textContent.trim()) {
                const icon = button.querySelector('[data-lucide]');
                if (icon) {
                    const iconName = icon.getAttribute('data-lucide');
                    button.setAttribute('aria-label', iconName.replace(/-/g, ' '));
                }
            }
        });

        // Make sure all interactive elements are keyboard focusable
        document.querySelectorAll('.song-card, .song-list-item').forEach(item => {
            if (!item.getAttribute('tabindex')) {
                item.setAttribute('tabindex', '0');
            }
        });
    }

    // Call accessibility improvements after DOM is fully loaded
    setTimeout(improveAccessibility, 1000);

    // Add responsive behavior for the songs grid
    function handleResponsiveLayout() {
        const songsGridContainer = document.getElementById('songs-grid');
        if (songsGridContainer) {
            if (window.innerWidth < 640) {
                songsGridContainer.style.gridTemplateColumns = 'repeat(1, 1fr)';
            } else if (window.innerWidth < 1024) {
                songsGridContainer.style.gridTemplateColumns = 'repeat(2, 1fr)';
            } else {
                songsGridContainer.style.gridTemplateColumns = 'repeat(3, 1fr)';
            }
        }
    }

    // Initial call and add event listener for window resize
    handleResponsiveLayout();
    window.addEventListener('resize', handleResponsiveLayout);
});
