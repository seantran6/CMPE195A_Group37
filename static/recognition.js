// static/recognition.js
// Handles grid/list toggle and renders both views from a single data source

document.addEventListener('DOMContentLoaded', () => {
    // ── View toggle setup ─────────────────────────────────
    const gridViewBtn = document.getElementById('grid-view-btn');
    const listViewBtn = document.getElementById('list-view-btn');
    const gridView = document.getElementById('grid-view');
    const listView = document.getElementById('list-view');

    if (gridViewBtn && listViewBtn) {
        // Show grid, hide list
        gridViewBtn.addEventListener('click', () => {
            gridViewBtn.classList.add('active');
            listViewBtn.classList.remove('active');
            gridView.style.display = 'grid';
            listView.style.display = 'none';
        });

        // Show list, hide grid
        listViewBtn.addEventListener('click', () => {
            listViewBtn.classList.add('active');
            gridViewBtn.classList.remove('active');
            listView.style.display = 'flex';
            gridView.style.display = 'none';
        });
    }

    // ── Render songs into both grid and list views ─────────
    function renderSongs(songs) {
        // clear old entries
        gridView.innerHTML = '';
        listView.innerHTML = '';

        songs.forEach(song => {
            // Grid item
            const gridItem = document.createElement('div');
            gridItem.className = 'song-card gradient-border';
            gridItem.innerHTML = `
        <div class="song-image-container">
          <img src="${song.cover || '/static/placeholder.svg'}" alt="${song.title}" class="song-image">
        </div>
        <div class="song-info">
          <h3 class="song-title">${song.title}</h3>
          <p class="song-artist">${song.artist}</p>
        </div>
      `;
            gridView.appendChild(gridItem);

            // List item
            const listItem = document.createElement('div');
            listItem.className = 'song-list-item gradient-border';
            listItem.innerHTML = `
        <div class="song-list-avatar">
          <img src="${song.cover || '/static/placeholder.svg'}" alt="${song.title}" class="song-list-image">
        </div>
        <div class="song-list-info">
          <h3 class="song-list-title">${song.title}</h3>
          <p class="song-list-artist">${song.artist}</p>
        </div>
      `;
            listView.appendChild(listItem);
        });

        // update any dynamic icons
        if (window.lucide) lucide.createIcons();
    }

    // ── Expose a helper for fetching and showing tracks ─────
    window.getRecommendations = function (gender, age, count = 6) {
        fetch(`/recommend_tracks?gender=${encodeURIComponent(gender)}&age=${encodeURIComponent(age)}&n=${count}`)
            .then(res => res.json())
            .then(data => {
                if (data.tracks) {
                    renderSongs(data.tracks);
                    // ensure recommendations container is visible
                    document.getElementById('recommendations').classList.remove('hidden');
                }
            })
            .catch(err => console.error('Error fetching recommendations:', err));
    };
});
