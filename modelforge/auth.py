from __future__ import annotations
import hashlib, hmac, os, secrets, sqlite3, time
from dataclasses import dataclass
from .store import _db

SESSION_SECONDS = 7 * 24 * 3600

def ensure_auth_schema():
    with _db() as con:
        con.executescript('''
        CREATE TABLE IF NOT EXISTS users(
          id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
          salt TEXT NOT NULL, created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions(
          token TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at REAL NOT NULL,
          expires_at REAL NOT NULL
        );
        ''')

def _hash(password:str, salt:bytes)->str:
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 240_000).hex()

def signup(email:str,password:str)->dict:
    ensure_auth_schema(); email=email.strip().lower()
    if '@' not in email or len(email)>254: raise ValueError('Enter a valid email address')
    if len(password)<8: raise ValueError('Password must be at least 8 characters')
    salt=os.urandom(16); uid=secrets.token_hex(8)
    try:
        with _db() as con:
            con.execute('INSERT INTO users VALUES(?,?,?,?,?)',(uid,email,_hash(password,salt),salt.hex(),time.time()))
    except sqlite3.IntegrityError: raise ValueError('An account with that email already exists')
    return {'id':uid,'email':email}

def login(email:str,password:str)->tuple[dict,str]:
    ensure_auth_schema(); email=email.strip().lower()
    with _db() as con: row=con.execute('SELECT * FROM users WHERE email=?',(email,)).fetchone()
    if not row: raise ValueError('Invalid email or password')
    salt=bytes.fromhex(row['salt'])
    if not hmac.compare_digest(_hash(password,salt), row['password_hash']): raise ValueError('Invalid email or password')
    token=secrets.token_urlsafe(32); now=time.time()
    with _db() as con:
        con.execute('DELETE FROM sessions WHERE expires_at<?',(now,))
        con.execute('INSERT INTO sessions VALUES(?,?,?,?)',(token,row['id'],now,now+SESSION_SECONDS))
    return {'id':row['id'],'email':row['email']}, token

def user_for_token(token:str|None)->dict|None:
    ensure_auth_schema()
    if not token: return None
    with _db() as con:
        row=con.execute('''SELECT u.id,u.email,s.expires_at FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token=?''',(token,)).fetchone()
    if not row or row['expires_at']<time.time(): return None
    return {'id':row['id'],'email':row['email']}

def logout(token:str|None):
    if token:
        with _db() as con: con.execute('DELETE FROM sessions WHERE token=?',(token,))
