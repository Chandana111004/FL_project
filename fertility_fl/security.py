"""
Security Module for Fertility Risk Prediction FL Project
Covers:
  - AES-256 file encryption/decryption
  - Audit logging
  - Role-based access control (RBAC)
  - Key rotation
  - Side channel protections
  - Memory clearing
  - DP epsilon budget enforcement
  - Time-limited access tokens
"""

import os
import json
import hmac
import time
import ctypes
import random
import logging
import hashlib
import numpy as np
from datetime import datetime, timedelta, timezone
from cryptography.fernet import Fernet, MultiFernet


# ── 1. AUDIT LOGGING ──────────────────────────────────────────

def setup_audit_logger():
    os.makedirs('logs', exist_ok=True)
    logger = logging.getLogger('audit')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler('logs/audit.log')
        handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
        logger.addHandler(handler)
    return logger

audit_logger = setup_audit_logger()

def audit_log(user, action, resource, success=True):
    status = 'SUCCESS' if success else 'DENIED'
    audit_logger.info(f"USER={user} | ACTION={action} | RESOURCE={resource} | STATUS={status}")


# ── 2. ROLE-BASED ACCESS CONTROL ──────────────────────────────

ROLE_PERMISSIONS = {
    'hospital_0': ['data/processed_dp/client_0/'],
    'hospital_1': ['data/processed_dp/client_1/'],
    'hospital_2': ['data/processed_dp/client_2/'],
    'hospital_3': ['data/processed_dp/client_3/'],
    'hospital_4': ['data/processed_dp/client_4/'],
    'server_admin': ['data/processed_dp/X_test.npy', 'results/'],
    'researcher':   ['results/training_history.json'],
}

def check_access(user_role, resource_path):
    allowed_paths = ROLE_PERMISSIONS.get(user_role, [])
    for allowed in allowed_paths:
        if resource_path.startswith(allowed) or resource_path == allowed:
            audit_log(user_role, 'ACCESS_CHECK', resource_path, success=True)
            return True
    audit_log(user_role, 'ACCESS_CHECK', resource_path, success=False)
    return False

def load_data_with_rbac(user_role, file_path):
    if not check_access(user_role, file_path):
        raise PermissionError(f"Role '{user_role}' cannot access '{file_path}'")
    audit_log(user_role, 'READ', file_path)
    return np.load(file_path)


# ── 3. AES-256 ENCRYPTION ─────────────────────────────────────

KEY_FILE     = 'keys/encryption.key'
OLD_KEY_FILE = 'keys/encryption_old.key'

def generate_key(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    key = Fernet.generate_key()
    with open(path, 'wb') as f:
        f.write(key)
    print(f"✓ New key saved to {path}")
    return key

def load_fernet():
    if not os.path.exists(KEY_FILE):
        generate_key(KEY_FILE)
    with open(KEY_FILE, 'rb') as f:
        primary_key = f.read()
    keys = [Fernet(primary_key)]
    if os.path.exists(OLD_KEY_FILE):
        with open(OLD_KEY_FILE, 'rb') as f:
            keys.append(Fernet(f.read()))
    return MultiFernet(keys)

def encrypt_file(input_path, output_path=None):
    fernet = load_fernet()
    output_path = output_path or input_path + '.enc'
    with open(input_path, 'rb') as f:
        data = f.read()
    with open(output_path, 'wb') as f:
        f.write(fernet.encrypt(data))
    print(f"✓ Encrypted: {input_path} → {output_path}")

def decrypt_file(encrypted_path, output_path=None):
    fernet = load_fernet()
    with open(encrypted_path, 'rb') as f:
        encrypted_data = f.read()
    decrypted = fernet.decrypt(encrypted_data)
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(decrypted)
    return decrypted

def rotate_key():
    print("Starting key rotation...")
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, 'rb') as f:
            current = f.read()
        with open(OLD_KEY_FILE, 'wb') as f:
            f.write(current)
    generate_key(KEY_FILE)
    for root, _, files in os.walk('data/'):
        for fname in files:
            if fname.endswith('.enc'):
                path = os.path.join(root, fname)
                try:
                    data = decrypt_file(path)
                    fernet = load_fernet()
                    with open(path, 'wb') as f:
                        f.write(fernet.encrypt(data))
                    print(f"  ✓ Re-encrypted: {path}")
                except Exception as e:
                    print(f"  ✗ Failed: {path} — {e}")
    print("✓ Key rotation complete!")


# ── 4. SIDE CHANNEL PROTECTIONS ───────────────────────────────

def secure_compare(a, b):
    """Constant-time comparison — prevents timing attacks."""
    return hmac.compare_digest(
        a.encode() if isinstance(a, str) else a,
        b.encode() if isinstance(b, str) else b
    )

def add_response_noise():
    """Random delay to prevent timing-based inference."""
    time.sleep(random.uniform(0.01, 0.05))

def clear_array(arr):
    """Wipe numpy array from memory after use."""
    arr[:] = 0
    try:
        ctypes.memset(arr.ctypes.data, 0, arr.nbytes)
    except Exception:
        pass
    del arr


# ── 5. DP EPSILON BUDGET ENFORCEMENT ──────────────────────────

TARGET_EPSILON = 5.0
DELTA = 1e-5

def check_epsilon_budget(privacy_engine):
    """
    Call this after every training epoch.
    Returns (epsilon, should_stop).
    If should_stop is True, break your training loop immediately.
    """
    epsilon = privacy_engine.get_epsilon(delta=DELTA)
    should_stop = epsilon >= TARGET_EPSILON
    if should_stop:
        print(f"\n⚠ Privacy budget exhausted! ε={epsilon:.2f} ≥ {TARGET_EPSILON}")
        print("  Stopping training to protect privacy.")
        audit_log('system', 'DP_BUDGET_EXHAUSTED', f'epsilon={epsilon:.2f}')
    else:
        print(f"  Privacy: ε={epsilon:.2f}/{TARGET_EPSILON}")
    return epsilon, should_stop


# ── 6. TIME-LIMITED ACCESS TOKENS ─────────────────────────────

_TOKEN_SECRET = os.environ.get('FL_TOKEN_SECRET', 'change_this_in_production')

def generate_token(user_role, expiry_hours=1):
    expiry = (datetime.now(timezone.utc) + timedelta(hours=expiry_hours)).isoformat()
    payload = f"{user_role}:{expiry}"
    signature = hmac.new(
        _TOKEN_SECRET.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()
    audit_log(user_role, 'TOKEN_GENERATED', f'expires={expiry}')
    return f"{payload}:{signature}"

def validate_token(token):
    try:
        parts = token.rsplit(':', 1)
        if len(parts) != 2:
            return None, False
        payload, provided_sig = parts
        expected_sig = hmac.new(
            _TOKEN_SECRET.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()
        if not secure_compare(provided_sig, expected_sig):
            return None, False
        user_role, expiry_str = payload.split(':', 1)
        if datetime.now(timezone.utc) > datetime.fromisoformat(expiry_str):
            audit_log(user_role, 'TOKEN_EXPIRED', 'login_attempt')
            return user_role, False
        audit_log(user_role, 'TOKEN_VALIDATED', 'login_attempt')
        return user_role, True
    except Exception:
        return None, False


# ── 7. ENCRYPT ALL DATA (run once after data prep) ────────────

def encrypt_all_processed_data(data_dir='data/processed_dp'):
    print(f"\nEncrypting all files in {data_dir}...")
    count = 0
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith(('.npy', '.pkl')) and not fname.endswith('.enc'):
                fpath = os.path.join(root, fname)
                encrypt_file(fpath, fpath + '.enc')
                count += 1
    print(f"✓ Encrypted {count} files")


if __name__ == '__main__':
    print("Running security self-test...\n")

    print("1. Testing RBAC...")
    assert check_access('hospital_0', 'data/processed_dp/client_0/X_train.npy') == True
    assert check_access('hospital_0', 'data/processed_dp/client_1/X_train.npy') == False
    print("   ✓ RBAC works\n")

    print("2. Testing secure compare...")
    assert secure_compare('hello', 'hello') == True
    assert secure_compare('hello', 'world') == False
    print("   ✓ Constant-time comparison works\n")

    print("3. Testing tokens...")
    token = generate_token('hospital_0', expiry_hours=1)
    role, valid = validate_token(token)
    assert valid == True and role == 'hospital_0'
    expired = generate_token('hospital_0', expiry_hours=-1)
    _, valid = validate_token(expired)
    assert valid == False
    print("   ✓ Tokens work correctly\n")

    print("All tests passed! ✓")
