from functools import wraps
from flask import session, redirect, url_for, flash
from .db import get_db

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please login to access this page', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            flash('Login required.', 'danger')
            return redirect(url_for('login'))

        conn = get_db()
        cur = conn.cursor()
        cur.execute('SELECT is_admin FROM users WHERE username = ?', (session['username'],))
        row = cur.fetchone()
        conn.close()

        if not row or row['is_admin'] != 1:
            flash('Admin access only.', 'danger')
            return redirect(url_for('dashboard'))

        return f(*args, **kwargs)
    return decorated
