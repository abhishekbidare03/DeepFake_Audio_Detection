document.addEventListener('DOMContentLoaded', () => {

    // ---
    // 1. Navbar Scroll Effect (Copied from index.html)
    // ---
    const navbar = document.getElementById('navbar');
    if (navbar) {
        window.addEventListener('scroll', () => {
            if (window.scrollY > 50) {
                navbar.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
            }
        });
    }

    // ---
    // 2. Bento Card Mouse Follow Effect (Copied from index.html)
    // ---
    const bentoCards = document.querySelectorAll('.bento-card');
    bentoCards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = ((e.clientX - rect.left) / rect.width) * 100;
            const y = ((e.clientY - rect.top) / rect.height) * 100;
            card.style.setProperty('--mouse-x', `${x}%`);
            card.style.setProperty('--mouse-y', `${y}%`);
        });
        // Reset on leave
        card.addEventListener('mouseleave', () => {
             card.style.setProperty('--mouse-x', `50%`);
             card.style.setProperty('--mouse-y', `50%`);
        });
    });

    // ---
    // 3. General Fade-in-on-Scroll Animation (Original)
    // ---
    const fadeInElements = document.querySelectorAll('.fade-in');

    const fadeInObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // Get delay from data-attribute
                const delay = entry.target.dataset.delay || 0;
                
                setTimeout(() => {
                    entry.target.classList.add('visible');
                }, parseInt(delay));
                
                // Stop observing once it's visible
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1 // Trigger when 10% of the element is visible
    });

    fadeInElements.forEach(el => {
        fadeInObserver.observe(el);
    });

    // ---
    // 4. Hero Pipeline Animation on Scroll (Original)
    // ---
    const pipelineNodes = document.querySelectorAll('.pipeline-node');
    
    // Define the sections that trigger each node
    const sectionToNodeMap = {
        'hero': 'pipe-node-0',
        'problem': 'pipe-node-1',
        'datasets': 'pipe-node-2',
        'model': 'pipe-node-3',
        'results': 'pipe-node-4'
    };
    
    const sections = document.querySelectorAll('#hero, #problem, #datasets, #model, #results');

    const pipelineObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const sectionId = entry.target.id;
                const nodeIdToActivate = sectionToNodeMap[sectionId];
                
                // Remove 'active' from all nodes
                pipelineNodes.forEach(node => {
                    node.classList.remove('active');
                });
                
                // Add 'active' to the corresponding node
                const activeNode = document.getElementById(nodeIdToActivate);
                if (activeNode) {
                    activeNode.classList.add('active');
                }
            }
        });
    }, {
        threshold: 0.3, 
        rootMargin: '-20% 0px -50% 0px' 
    });

    sections.forEach(sec => {
        if (sec) {
            pipelineObserver.observe(sec);
        }
    });
});