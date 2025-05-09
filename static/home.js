document.addEventListener('DOMContentLoaded', function () {
    // Theme handling
    const themeToggle = document.getElementById('theme-toggle');

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
    const menuButton = document.querySelector('.menu-button');
    const mainNav = document.querySelector('.main-nav');

    if (menuButton && mainNav) {
        menuButton.addEventListener('click', function () {
            mainNav.style.display = mainNav.style.display === 'flex' ? 'none' : 'flex';
        });
    }

    // Animate on scroll simulation
    function animateOnScroll() {
        const elements = document.querySelectorAll('[data-aos]');

        elements.forEach(element => {
            const elementPosition = element.getBoundingClientRect().top;
            const windowHeight = window.innerHeight;

            if (elementPosition < windowHeight * 0.85) {
                element.classList.add('aos-animate');
            }
        });
    }

    // Run once on page load
    animateOnScroll();

    // Run on scroll
    window.addEventListener('scroll', animateOnScroll);

    // Add floating music notes animation
    function createFloatingNotes() {
        const container = document.querySelector('.floating-music-notes');
        if (!container) return;

        const icons = ['music', 'music-2', 'music-3', 'music-4'];
        const colors = [
            'hsl(326, 100%, 60%)', // primary
            'hsl(199, 89%, 48%)',  // secondary
            'hsl(262, 83%, 58%)'   // accent
        ];

        for (let i = 0; i < 10; i++) {
            const note = document.createElement('i');
            note.setAttribute('data-lucide', icons[Math.floor(Math.random() * icons.length)]);
            note.classList.add('note-icon', `note-random-${i}`);
            note.style.top = `${Math.random() * 100}%`;
            note.style.left = `${Math.random() * 100}%`;
            note.style.color = colors[Math.floor(Math.random() * colors.length)];
            note.style.opacity = 0.5 + Math.random() * 0.5;
            note.style.fontSize = `${0.5 + Math.random() * 1}rem`;
            note.style.animation = `float-note ${5 + Math.random() * 5}s ease-in-out infinite ${Math.random() * 5}s`;

            container.appendChild(note);
        }

        // Initialize the newly created Lucide icons
        lucide.createIcons({
            icons: {
                'music': '.note-icon',
                'music-2': '.note-icon',
                'music-3': '.note-icon',
                'music-4': '.note-icon'
            }
        });
    }

    createFloatingNotes();

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();

            const targetId = this.getAttribute('href');
            if (targetId === '#') return;

            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80, // Adjust for header height
                    behavior: 'smooth'
                });
            }
        });
    });

    // Add entrance animations for sections
    function addEntranceAnimations() {
        const sections = document.querySelectorAll('section');

        sections.forEach((section, index) => {
            section.style.opacity = '0';
            section.style.transform = 'translateY(20px)';
            section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            section.style.transitionDelay = `${index * 0.1}s`;

            setTimeout(() => {
                section.style.opacity = '1';
                section.style.transform = 'translateY(0)';
            }, 100);
        });
    }

    addEntranceAnimations();
});