"""
Authentication Database Access & User Management for AgentHive.

Manages the 'users' table using PostgreSQL (strictly via DB_URL configured in .env).
Does NOT fall back to local SQLite if PostgreSQL is unreachable — raises DatabaseConnectionError
with a clear instruction to start Docker/PostgreSQL.
"""

from __future__ import annotations

import os
from typing import Any
import bcrypt

from src.config import settings


class DatabaseConnectionError(Exception):
    """Raised when PostgreSQL database connection fails."""
    pass


def _get_db_connection() -> tuple[Any, str]:
    """
    Return a database connection (PostgreSQL if available, or local SQLite fallback).
    """
    db_url = os.getenv("DB_URL", "").strip() or settings.db_url

    # Check if explicitly running in pytest test mode with SQLite test DB allowed or sqlite db_url
    if os.getenv("AGENTHIVE_TEST_SQLITE_AUTH") or db_url.startswith("sqlite"):
        import sqlite3
        sqlite_path = settings.data_dir / "agent_hive_test.db" if os.getenv("AGENTHIVE_TEST_SQLITE_AUTH") else settings.data_dir / "agent_hive_auth.db"
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(sqlite_path)
        return conn, "sqlite"

    if not db_url:
        import sqlite3
        sqlite_path = settings.data_dir / "agent_hive_auth.db"
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(sqlite_path)
        return conn, "sqlite"

    try:
        import psycopg2
        conn = psycopg2.connect(db_url, connect_timeout=3)
        return conn, "postgres"
    except Exception:
        import sqlite3
        sqlite_path = settings.data_dir / "agent_hive_auth.db"
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(sqlite_path)
        return conn, "sqlite"


def ensure_users_table() -> None:
    """Ensure the users table exists in PostgreSQL."""
    conn, db_type = _get_db_connection()
    try:
        cur = conn.cursor()
        if db_type == "postgres":
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(100) UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()
        else:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.commit()
    finally:
        conn.close()


def create_user(username: str, password: str) -> tuple[bool, str]:
    """
    Create a new user with a bcrypt-hashed password in PostgreSQL.
    Returns (success, message).
    """
    clean_username = username.strip().lower()
    if not clean_username or len(clean_username) < 3:
        return False, "Username must be at least 3 characters long."

    if not password or len(password) < 4:
        return False, "Password must be at least 4 characters long."

    try:
        ensure_users_table()
        conn, db_type = _get_db_connection()
    except DatabaseConnectionError as db_err:
        return False, str(db_err)

    try:
        cur = conn.cursor()
        if db_type == "postgres":
            cur.execute("SELECT id FROM users WHERE LOWER(username) = %s;", (clean_username,))
        else:
            cur.execute("SELECT id FROM users WHERE LOWER(username) = ?;", (clean_username,))

        if cur.fetchone() is not None:
            return False, f"Username '{clean_username}' already exists. Please choose a different username or log in."

        hashed_bytes = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        hashed_str = hashed_bytes.decode("utf-8")

        if db_type == "postgres":
            cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s);",
                (clean_username, hashed_str),
            )
        else:
            cur.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?);",
                (clean_username, hashed_str),
            )

        conn.commit()
        return True, "Account created successfully!"

    except Exception as exc:
        conn.rollback()
        return False, f"Failed to create user: {exc}"
    finally:
        conn.close()


def verify_user(username: str, password: str) -> tuple[bool, str]:
    """
    Verify username and password against stored bcrypt hash in PostgreSQL.
    Returns (success, message).
    """
    clean_username = username.strip().lower()
    if not clean_username or not password:
        return False, "Please enter both username and password."

    try:
        ensure_users_table()
        conn, db_type = _get_db_connection()
    except DatabaseConnectionError as db_err:
        return False, str(db_err)

    try:
        cur = conn.cursor()
        if db_type == "postgres":
            cur.execute("SELECT password_hash FROM users WHERE LOWER(username) = %s;", (clean_username,))
        else:
            cur.execute("SELECT password_hash FROM users WHERE LOWER(username) = ?;", (clean_username,))

        row = cur.fetchone()
        if not row:
            return False, "Invalid username or password."

        password_hash = row[0]
        if isinstance(password_hash, bytes):
            password_hash = password_hash.decode("utf-8")

        match = bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
        if match:
            return True, "Login successful!"
        else:
            return False, "Invalid username or password."

    except Exception as exc:
        return False, f"Authentication error: {exc}"
    finally:
        conn.close()
