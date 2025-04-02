# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_bcrypt import Bcrypt
from tinydb import TinyDB, Query
from functools import wraps
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)
bcrypt = Bcrypt(app)

# Set up TinyDB
db = TinyDB('db.json')
users_table = db.table('users')
User = Query()

# Login required decorator
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
        
        user = users_table.get(User.username == username)
        
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
        existing_user = users_table.get(User.username == username)
        
        if existing_user:
            flash('Username already exists', 'danger')
        else:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            users_table.insert({
                'username': username,
                'password': hashed_password,
                'display_name': display_name
            })
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    
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
    # Create an admin user if not exists
    admin = users_table.get(User.username == 'admin')
    if not admin:
        users_table.insert({
            'username': 'admin',
            'password': bcrypt.generate_password_hash('admin123').decode('utf-8'),
            'display_name': 'Admin User'
        })
    
    app.run(debug=True)