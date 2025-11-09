// script.js – frontend logic for DeepFake Audio Detection system
// Handles navigation, animations, and real audio upload to FastAPI

document.addEventListener('DOMContentLoaded', () => {
    // ====== Navigation ======
    const navToggle = document.querySelector('.nav-toggle');
    const navMenu = document.querySelector('.nav-menu');
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', () => {
            navMenu.style.display = navMenu.style.display === 'flex' ? 'none' : 'flex';
        });
    }

    // Active link marker
    const links = document.querySelectorAll('.nav-menu a');
    const path = window.location.pathname.split('/').pop();
    links.forEach(a => {
        const href = a.getAttribute('href');
        if (!path && href.includes('index')) a.classList.add('active');
        else if (path === href) a.classList.add('active');
    });

    // ====== On-scroll animations ======
    const observer = new IntersectionObserver(entries => {
        entries.forEach(e => {
            if (e.isIntersecting) e.target.classList.add('visible');
        });
    }, { threshold: 0.18 });

    document.querySelectorAll('.feature-card, .step, .hero-content, .audio-visualization, .analysis-panel, .contact-form')
        .forEach(el => observer.observe(el));

    // ====== Upload + Predict ======
    const chooseBtn = document.querySelector('.btn-upload');
    const fileInput = document.getElementById('audio-upload');
    const progressFill = document.querySelector('.progress-fill');
    const resultText = document.querySelector('.results-content p');

    // Helper to show message
    function showMessage(text) {
        if (resultText) resultText.textContent = text;
        else console.log(text);
    }

    // Helper to update progress bar
    function setProgress(pct) {
        if (progressFill) progressFill.style.width = `${pct}%`;
    }

    // Upload + get prediction from FastAPI
    async function uploadAndPredict(file) {
        try {
            if (!file) {
                showMessage("Please select an audio file.");
                return;
            }

            const formData = new FormData();
            formData.append("file", file);  // Must match FastAPI param name

            showMessage(`Uploading: ${file.name} ...`);
            setProgress(10);

            // Send file to FastAPI backend
            const response = await fetch("/api/v1/audio/predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`Server error (${response.status}): ${errText}`);
            }

            const data = await response.json();

            // Response example: { prediction: "FAKE", confidence: 0.92, score: 0.92 }
            const pred = data.prediction || "Unknown";
            const conf = data.confidence ? (data.confidence * 100).toFixed(2) + "%" : "N/A";

            showMessage(`Result: ${pred} (Confidence: ${conf})`);
            setProgress(100);
        } catch (err) {
            console.error(err);
            showMessage(`Error: ${err.message}`);
            setProgress(0);
        }
    }

    // Hook upload button
    if (chooseBtn && fileInput) {
        chooseBtn.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
            setProgress(0);
            uploadAndPredict(file);
        });
    }
});
