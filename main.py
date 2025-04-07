from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_bcrypt import Bcrypt
import sqlite3
import os
from functools import wraps

app = Flask(__name__)

# Set up secret key for session management
app.secret_key = os.urandom(24)

# Initialize Bcrypt
bcrypt = Bcrypt(app)

# Database file path
DATABASE = 'db.sqlite'


# Helper function to get the database connection
def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # Enable row access by column name
    return conn


# Initialize the database if needed
def init_db():
    conn = get_db()
    cursor = conn.cursor()

    # Create the User table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        display_name TEXT NOT NULL)''')
    conn.commit()
    conn.close()


def login_required(f):
    @wraps(f)
    def decorated_function():
        if 'username' not in session:
            flash('Please login to access this page', 'danger')
            return redirect(url_for('login'))
        return f()
    return decorated_function


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Query the database for the user
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()

        if user and bcrypt.check_password_hash(user['password'], password):
            session['username'] = username
            session['display_name'] = user['display_name']
            flash('You have been logged in successfully', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        display_name = request.form['display_name']

        # Check if user already exists
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            flash('Username already exists', 'danger')
        else:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            cursor.execute('INSERT INTO users (username, password, display_name) VALUES (?, ?, ?)',
                           (username, hashed_password, display_name))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))

        conn.close()

    return render_template('register.html')


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('display_name', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))


if __name__ == '__main__':
    # Initialize the database and create admin if necessary
    init_db()

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', ('admin',))
    admin = cursor.fetchone()

    if not admin:
        hashed_password = bcrypt.generate_password_hash('admin123').decode('utf-8')
        cursor.execute('INSERT INTO users (username, password, display_name) VALUES (?, ?, ?)',
                       ('admin', hashed_password, 'Admin User'))
        conn.commit()

    conn.close()

    app.run(debug=True)
