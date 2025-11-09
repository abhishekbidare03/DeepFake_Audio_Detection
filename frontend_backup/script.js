// script.js - small shared interactions: nav toggle, active link, simple animations

document.addEventListener('DOMContentLoaded', () => {
    // Nav toggle (mobile)
    const navToggle = document.querySelector('.nav-toggle');
    const navMenu = document.querySelector('.nav-menu');
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', () => {
            navMenu.style.display = navMenu.style.display === 'flex' ? 'none' : 'flex';
        });
    }

    // Mark active link (by file name)
    const links = document.querySelectorAll('.nav-menu a');
    const path = window.location.pathname.split('/').pop();
    links.forEach(a => {
        const href = a.getAttribute('href');
        if (!path && href.includes('index')) a.classList.add('active');
        else if (path === href) a.classList.add('active');
    });

    // Fade-in simple on-scroll
    const observer = new IntersectionObserver(entries => {
        entries.forEach(e => {
            if (e.isIntersecting) e.target.classList.add('visible');
        });
    }, { threshold: 0.18 });

    document.querySelectorAll('.feature-card, .step, .hero-content, .audio-visualization, .analysis-panel, .contact-form').forEach(el => {
        observer.observe(el);
    });

    // Upload file chooser hookup (demo page)
    const chooseBtn = document.querySelector('.btn-upload');
    const fileInput = document.getElementById('audio-upload');
    if (chooseBtn && fileInput) {
        chooseBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => {
            const f = e.target.files[0];
            if (!f) return;
            const panelText = document.querySelector('.results-content p');
            if (panelText) panelText.textContent = `Selected: ${f.name} — (demo: simulated analysis)`;
            // Simulate progress
            const fill = document.querySelector('.progress-fill');
            if (fill) {
                fill.style.width = '0%';
                setTimeout(() => fill.style.width = '45%', 200);
                setTimeout(() => fill.style.width = '85%', 900);
                setTimeout(() => fill.style.width = '100%', 1600);
            }
        });
    }
});