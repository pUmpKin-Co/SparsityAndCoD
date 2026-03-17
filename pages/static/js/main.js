/**
 * Sparsity & CoD - Clean Academic Theme
 * Essential JavaScript functionality
 */

document.addEventListener('DOMContentLoaded', () => {
    // Initialize Lucide icons
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }

    // Initialize scroll animations
    initScrollReveal();

    // Initialize navigation
    initNavigation();
});

/**
 * Scroll Reveal Animation
 * Simple fade-in on scroll
 */
function initScrollReveal() {
    const reveals = document.querySelectorAll('.reveal');

    if (!reveals.length) return;

    const revealObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
                revealObserver.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });

    reveals.forEach(reveal => {
        revealObserver.observe(reveal);
    });
}

/**
 * Navigation
 * Active state and smooth scrolling
 */
function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('section[id]');

    // Smooth scroll for navigation links
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href');
            const targetSection = document.querySelector(targetId);

            if (targetSection) {
                const navHeight = document.querySelector('nav').offsetHeight;
                const targetPosition = targetSection.offsetTop - navHeight - 20;

                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Update active state on scroll
    let ticking = false;

    function updateActiveNav() {
        const scrollPos = window.scrollY + 150;

        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            const sectionId = section.getAttribute('id');

            if (scrollPos >= sectionTop && scrollPos < sectionTop + sectionHeight) {
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${sectionId}`) {
                        link.classList.add('active');
                    }
                });
            }
        });

        ticking = false;
    }

    window.addEventListener('scroll', () => {
        if (!ticking) {
            requestAnimationFrame(updateActiveNav);
            ticking = true;
        }
    }, { passive: true });
}

/**
 * Copy to Clipboard
 * For BibTeX citation
 */
function copyBibtex() {
    const bibtexContent = document.getElementById('bibtex-content');
    if (!bibtexContent) return;

    const text = bibtexContent.textContent;
    const copyBtn = document.querySelector('.copy-btn');
    const copyText = document.getElementById('copy-text');

    // Use modern clipboard API with fallback
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(() => {
            showCopiedFeedback(copyBtn, copyText);
        }).catch(err => {
            console.error('Clipboard API failed:', err);
            fallbackCopy(text, copyBtn, copyText);
        });
    } else {
        fallbackCopy(text, copyBtn, copyText);
    }
}

/**
 * Fallback copy method
 */
function fallbackCopy(text, btn, textSpan) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-9999px';
    document.body.appendChild(textArea);

    try {
        textArea.select();
        const successful = document.execCommand('copy');
        if (successful) {
            showCopiedFeedback(btn, textSpan);
        }
    } catch (err) {
        console.error('Fallback copy failed:', err);
    } finally {
        document.body.removeChild(textArea);
    }
}

/**
 * Show copied feedback
 */
function showCopiedFeedback(btn, textSpan) {
    if (!btn || !textSpan) return;

    btn.classList.add('copied');
    textSpan.textContent = 'Copied!';

    setTimeout(() => {
        btn.classList.remove('copied');
        textSpan.textContent = 'Copy';
    }, 2000);
}

// Export for global access
window.copyBibtex = copyBibtex;