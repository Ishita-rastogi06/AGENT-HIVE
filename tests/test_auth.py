"""
Unit tests for AgentHive authentication system (bcrypt hashing, signup validation,
duplicate username rejection, and login verification).
"""

from __future__ import annotations

import os
import sqlite3
import pytest

from src.auth_db import create_user, verify_user, ensure_users_table


@pytest.fixture(autouse=True)
def clean_test_users(monkeypatch):
    """Ensure clean users table before each test."""
    monkeypatch.setenv("AGENTHIVE_TEST_SQLITE_AUTH", "1")
    ensure_users_table()
    yield


def test_signup_success_and_login_success(tmp_path, monkeypatch):
    """Test creating a new user and verifying login with correct credentials."""
    username = f"testuser_{os.urandom(4).hex()}"
    password = "validpassword123"

    # 1. Signup
    ok, msg = create_user(username, password)
    assert ok is True, f"Signup failed: {msg}"

    # 2. Login with correct password
    v_ok, v_msg = verify_user(username, password)
    assert v_ok is True, f"Login failed: {v_msg}"


def test_signup_rejects_duplicate_username():
    """Test that signup rejects duplicate usernames regardless of casing."""
    username = f"dupuser_{os.urandom(4).hex()}"
    password = "password123"

    # First signup
    ok1, msg1 = create_user(username, password)
    assert ok1 is True

    # Duplicate signup (same case)
    ok2, msg2 = create_user(username, password)
    assert ok2 is False
    assert "already exists" in msg2.lower()

    # Duplicate signup (different case)
    ok3, msg3 = create_user(username.upper(), password)
    assert ok3 is False
    assert "already exists" in msg3.lower()


def test_signup_validation_rules():
    """Test username and password length validation rules."""
    # Short username (< 3 chars)
    ok1, msg1 = create_user("ab", "password123")
    assert ok1 is False
    assert "at least 3 characters" in msg1

    # Short password (< 4 chars)
    ok2, msg2 = create_user("validname", "123")
    assert ok2 is False
    assert "at least 4 characters" in msg2


def test_login_rejects_wrong_password():
    """Test that login rejects incorrect passwords."""
    username = f"wrongpassuser_{os.urandom(4).hex()}"
    password = "correctpassword"

    ok, _ = create_user(username, password)
    assert ok is True

    # Attempt login with wrong password
    v_ok, v_msg = verify_user(username, "wrongpassword")
    assert v_ok is False
    assert "invalid username or password" in v_msg.lower()


def test_login_rejects_nonexistent_user():
    """Test that login fails for non-existent usernames."""
    v_ok, v_msg = verify_user("nonexistent_user_99999", "somepassword")
    assert v_ok is False
    assert "invalid username or password" in v_msg.lower()
