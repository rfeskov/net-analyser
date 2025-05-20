document.addEventListener('DOMContentLoaded', function() {
    // Check if user is authenticated
    const token = localStorage.getItem('authToken');
    if (!token) {
        window.location.href = '/login';
        return;
    }

    // Get form elements
    const currentPasswordInput = document.getElementById('current-password');
    const newPasswordInput = document.getElementById('new-password');
    const confirmPasswordInput = document.getElementById('confirm-password');
    const saveButton = document.getElementById('save-button');
    const logoutButton = document.getElementById('logout-button');
    const successNotification = document.getElementById('success-notification');
    const errorNotification = document.getElementById('error-notification');

    // Add event listeners
    if (saveButton) {
        saveButton.addEventListener('click', handleSave);
    }

    if (logoutButton) {
        logoutButton.addEventListener('click', handleLogout);
    }

    // Add input event listeners for password fields
    if (confirmPasswordInput) {
        confirmPasswordInput.addEventListener('input', function() {
            if (newPasswordInput && this.value !== newPasswordInput.value) {
                this.style.borderColor = '#EF4444'; // red
            } else {
                this.style.borderColor = '#10B981'; // green
            }
        });
    }

    // Handle save button click
    async function handleSave() {
        // Validate passwords
        if (currentPasswordInput && newPasswordInput && confirmPasswordInput) {
            if (!currentPasswordInput.value) {
                showError('Введите текущий пароль');
                return;
            }

            if (!newPasswordInput.value) {
                showError('Введите новый пароль');
                return;
            }

            if (!confirmPasswordInput.value) {
                showError('Подтвердите новый пароль');
                return;
            }

            if (newPasswordInput.value !== confirmPasswordInput.value) {
                showError('Новые пароли не совпадают');
                return;
            }

            if (newPasswordInput.value.length < 8) {
                showError('Пароль должен содержать минимум 8 символов');
                return;
            }

            try {
                const response = await fetch('/api/auth/change-password', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${token}`
                    },
                    body: JSON.stringify({
                        currentPassword: currentPasswordInput.value,
                        newPassword: newPasswordInput.value
                    })
                });

                if (response.ok) {
                    showSuccess();
                    resetForm();
                } else {
                    const data = await response.json();
                    showError(data.detail || 'Ошибка при изменении пароля');
                }
            } catch (error) {
                console.error('Error changing password:', error);
                showError('Ошибка при изменении пароля');
            }
        }
    }

    // Handle logout
    function handleLogout() {
        localStorage.removeItem('authToken');
        window.location.href = '/login';
    }

    // Show success notification
    function showSuccess() {
        if (successNotification) {
            successNotification.classList.remove('hidden');
            setTimeout(() => {
                successNotification.classList.add('hidden');
            }, 3000);
        }
    }

    // Show error notification
    function showError(message) {
        if (errorNotification) {
            const errorText = errorNotification.querySelector('span');
            if (errorText) {
                errorText.textContent = message;
            }
            errorNotification.classList.remove('hidden');
            setTimeout(() => {
                errorNotification.classList.add('hidden');
            }, 3000);
        }
    }

    // Reset form fields
    function resetForm() {
        if (currentPasswordInput) currentPasswordInput.value = '';
        if (newPasswordInput) newPasswordInput.value = '';
        if (confirmPasswordInput) confirmPasswordInput.value = '';
    }

    // Add password strength indicator
    if (newPasswordInput) {
        newPasswordInput.addEventListener('input', function() {
            const password = this.value;
            let strength = 0;

            // Check password length
            if (password.length >= 8) strength += 1;
            
            // Check for numbers
            if (/\d/.test(password)) strength += 1;
            
            // Check for special characters
            if (/[!@#$%^&*]/.test(password)) strength += 1;
            
            // Check for uppercase and lowercase
            if (/[a-z]/.test(password) && /[A-Z]/.test(password)) strength += 1;

            // Update visual feedback
            this.style.borderColor = strength >= 3 ? '#10B981' : 
                                   strength >= 2 ? '#F59E0B' : 
                                   strength >= 1 ? '#EF4444' : '#D1D5DB';
        });
    }
}); 