from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional
import jwt
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import logging
import bcrypt

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# JWT settings
SECRET_KEY = "your-secret-key"  # В продакшене используйте безопасный ключ
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class User(BaseModel):
    email: str
    password: str

class PasswordChange(BaseModel):
    currentPassword: str
    newPassword: str

# User data file path
USER_DATA_FILE = Path("frontend/static/data/user_data.json")

def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def ensure_user_data_file():
    """Ensure user data file exists with default admin user"""
    try:
        if not USER_DATA_FILE.exists():
            logger.info(f"Creating user data file at {USER_DATA_FILE}")
            USER_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            # Хешируем пароль по умолчанию
            default_password = "admin123"
            hashed_password = hash_password(default_password)
            default_data = {
                "users": {
                    "admin@example.com": {
                        "password": hashed_password,
                        "role": "admin"
                    }
                }
            }
            with open(USER_DATA_FILE, "w") as f:
                json.dump(default_data, f, indent=4)
            logger.info("User data file created successfully")
        else:
            logger.info(f"User data file already exists at {USER_DATA_FILE}")
            # Проверяем и обновляем существующие пароли
            with open(USER_DATA_FILE, "r") as f:
                data = json.load(f)
                updated = False
                for email, user_data in data["users"].items():
                    # Если пароль не хеширован (не начинается с $2b$)
                    if not user_data["password"].startswith("$2b$"):
                        logger.info(f"Updating password hash for user: {email}")
                        user_data["password"] = hash_password(user_data["password"])
                        updated = True
                if updated:
                    with open(USER_DATA_FILE, "w") as f:
                        json.dump(data, f, indent=4)
                    logger.info("Updated existing passwords to hashed format")
    except Exception as e:
        logger.error(f"Error creating/updating user data file: {e}")
        raise

def get_user(email: str):
    """Get user from data file"""
    try:
        ensure_user_data_file()
        with open(USER_DATA_FILE, "r") as f:
            data = json.load(f)
            user = data["users"].get(email)
            logger.info(f"Looking up user {email}: {'Found' if user else 'Not found'}")
            return user
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        return None

def create_access_token(data: dict):
    """Create JWT token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get current user from token"""
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except jwt.JWTError:
        raise credentials_exception
    user = get_user(email=token_data.email)
    if user is None:
        raise credentials_exception
    # Add email to user data
    user["email"] = email
    return user

@router.post("/api/auth/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint"""
    logger.info(f"Login attempt for user: {form_data.username}")
    user = get_user(form_data.username)
    if not user or not verify_password(form_data.password, user["password"]):
        logger.warning(f"Failed login attempt for user: {form_data.username}")
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": form_data.username})
    logger.info(f"Successful login for user: {form_data.username}")
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/api/auth/change-password")
async def change_password(
    password_change: PasswordChange,
    current_user: dict = Depends(get_current_user)
):
    """Change password endpoint"""
    try:
        logger.info(f"Attempting to change password for user: {current_user.get('email')}")
        logger.info(f"Current user data: {current_user}")
        
        with open(USER_DATA_FILE, "r") as f:
            data = json.load(f)
            logger.info(f"Loaded user data from file: {data}")
        
        # Verify current password
        if not verify_password(password_change.currentPassword, data["users"][current_user["email"]]["password"]):
            logger.warning(f"Failed password change attempt for user: {current_user['email']} - incorrect current password")
            raise HTTPException(
                status_code=401,
                detail="Current password is incorrect"
            )
        
        # Hash and update new password
        hashed_new_password = hash_password(password_change.newPassword)
        data["users"][current_user["email"]]["password"] = hashed_new_password
        logger.info(f"Updated password for user: {current_user['email']}")
        
        with open(USER_DATA_FILE, "w") as f:
            json.dump(data, f, indent=4)
            logger.info("Successfully wrote updated user data to file")
        
        logger.info(f"Password changed successfully for user: {current_user['email']}")
        return {"message": "Password changed successfully"}
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        logger.error(f"Current user data: {current_user}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/auth/verify")
async def verify_token(current_user: dict = Depends(get_current_user)):
    """Verify token endpoint"""
    return {"message": "Token is valid", "user": current_user} 