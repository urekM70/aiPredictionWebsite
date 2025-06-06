import sqlite3

DATABASE = 'db.sqlite'

def get_db():
    """Vzpostavi povezavo z bazo in omogoči dostop z imeni stolpcev."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Inicializira vse tabele, če še ne obstajajo."""
    conn = get_db()
    cursor = conn.cursor()

    # --- USERS ---
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        display_name TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0,
        bio TEXT DEFAULT '',
        email TEXT DEFAULT '',
        profile_pic TEXT DEFAULT 'default.png',
        is_premium INTEGER DEFAULT 0,
        token_balance INTEGER DEFAULT 10
    )''')

    # --- BLOG POSTS ---
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS blog_posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')

    # --- COMMENTS ---
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS comments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id INTEGER NOT NULL,
        username TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(post_id) REFERENCES blog_posts(id)
    )''')

    # --- CHAT ---
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        message TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )''')

    # --- PREDICTIONS ---
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        interval TEXT NOT NULL,
        predictions TEXT NOT NULL, -- Will store JSON list
        actuals TEXT NOT NULL,     -- Will store JSON list
        timestamp DATETIME NOT NULL,
        metrics TEXT NOT NULL -- Will store JSON dict
    )''')
    # --- MARKET DATA ---
    # cursor.execute('''
    # CREATE TABLE IF NOT EXISTS market_data (
    #     id INTEGER PRIMARY KEY AUTOINCREMENT,
    #     symbol TEXT NOT NULL,
    #     timestamp INTEGER NOT NULL,
    #     open REAL NOT NULL,
    #     high REAL NOT NULL,
    #     low REAL NOT NULL,
    #     close REAL NOT NULL,
    #     volume REAL NOT NULL,
    #     UNIQUE(symbol, timestamp)
    # )''')

    conn.commit()
    conn.close()




def create_admin_user(bcrypt):
    """Ustvari admin uporabnika, če še ne obstaja."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', ('admin',))
    admin = cursor.fetchone()

    if not admin:
        hashed_password = bcrypt.generate_password_hash('admin123').decode('utf-8')
        cursor.execute('''
            INSERT INTO users (username, password, display_name, is_admin)
            VALUES (?, ?, ?, ?)
        ''', ('admin', hashed_password, 'Admin User', 1))
        conn.commit()

    conn.close()
