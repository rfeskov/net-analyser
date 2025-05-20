// Функция для проверки аутентификации
async function checkAuth() {
    const token = localStorage.getItem('authToken');
    if (!token) {
        // Если токена нет, перенаправляем на страницу входа
        if (window.location.pathname !== '/login') {
            window.location.href = '/login';
        }
        return false;
    }

    try {
        // Проверяем валидность токена
        const response = await fetch('/api/auth/verify', {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        });

        if (!response.ok) {
            // Если токен невалиден, удаляем его и перенаправляем на страницу входа
            localStorage.removeItem('authToken');
            if (window.location.pathname !== '/login') {
                window.location.href = '/login';
            }
            return false;
        }

        // Если мы на странице входа и токен валиден, перенаправляем на главную
        if (window.location.pathname === '/login') {
            window.location.href = '/';
        }

        return true;
    } catch (error) {
        console.error('Auth check error:', error);
        localStorage.removeItem('authToken');
        if (window.location.pathname !== '/login') {
            window.location.href = '/login';
        }
        return false;
    }
}

// Функция для выхода из системы
function logout() {
    localStorage.removeItem('authToken');
    window.location.href = '/login';
}

// Проверяем аутентификацию при загрузке страницы
document.addEventListener('DOMContentLoaded', checkAuth); 