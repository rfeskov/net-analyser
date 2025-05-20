document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('login-form');
    const errorNotification = document.getElementById('error-notification');

    loginForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        const formData = new FormData();
        formData.append('username', document.getElementById('email').value);
        formData.append('password', document.getElementById('password').value);

        try {
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const data = await response.json();
                // Store the token
                localStorage.setItem('authToken', data.access_token);
                // Redirect to dashboard
                window.location.href = '/';
            } else {
                showError();
            }
        } catch (error) {
            console.error('Login error:', error);
            showError();
        }
    });

    function showError() {
        errorNotification.classList.remove('hidden');
        setTimeout(() => {
            errorNotification.classList.add('hidden');
        }, 3000);
    }
}); 