import sqlite3
import bcrypt

DB_NAME = "users.db"


def create_user_table():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password BLOB
        )
    """)
    conn.commit()
    conn.close()


def password_policy(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not any(c.isupper() for c in password):
        return False, "Must contain uppercase letter"
    if not any(c.islower() for c in password):
        return False, "Must contain lowercase letter"
    if not any(c.isdigit() for c in password):
        return False, "Must contain number"
    if not any(c in "!@#$%^&*" for c in password):
        return False, "Must contain special character"
    return True, "Strong password"


def register_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    try:
        c.execute("INSERT INTO users VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        return True, "User registered successfully"
    except:
        return False, "Username already exists"
    finally:
        conn.close()


def login_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("SELECT password FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()

    if result and bcrypt.checkpw(password.encode(), result[0]):
        return True
    return False