document.addEventListener('DOMContentLoaded', function() {
    // Get form elements
    const newPasswordInput = document.getElementById('new-password');
    const confirmPasswordInput = document.getElementById('confirm-password');
    const saveButton = document.getElementById('save-button');
    const successNotification = document.getElementById('success-notification');

    // Add event listeners
    if (saveButton) {
        saveButton.addEventListener('click', handleSave);
    }

    // Handle save button click
    function handleSave() {
        // Validate passwords
        if (newPasswordInput && confirmPasswordInput) {
            if (newPasswordInput.value !== confirmPasswordInput.value) {
                alert('Пароли не совпадают');
                return;
            }

            if (newPasswordInput.value.length < 8) {
                alert('Пароль должен содержать минимум 8 символов');
                return;
            }

            // Here you would typically make an API call to update the password
            showSuccessNotification();
            resetForm();
        }
    }

    // Show success notification
    function showSuccessNotification() {
        if (successNotification) {
            successNotification.classList.remove('hidden');
            setTimeout(() => {
                successNotification.classList.add('hidden');
            }, 3000); // Hide after 3 seconds
        }
    }

    // Reset form fields
    function resetForm() {
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