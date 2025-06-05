from flask import render_template, request, redirect, url_for, flash, session, abort
from werkzeug.utils import secure_filename
import os
from .db import get_db
from .decorators import login_required, admin_required

cryptos = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT"]
stocks = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA"]

def setup_routes(app):
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/dashboard')
    @login_required
    def dashboard():
        conn = None
        token_balance = 0  # Default value
        is_premium = False # Default value
        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("SELECT token_balance, is_premium FROM users WHERE username = ?", (session['username'],))
            user = cursor.fetchone()
            if user:
                token_balance = user['token_balance']
                is_premium = user['is_premium'] == 1 # Convert DB integer to boolean
        except Exception as e:
            # Log error if necessary, for now use defaults
            print(f"Error fetching user details for dashboard: {e}") # Consider proper logging
        finally:
            if conn:
                conn.close()
        
        return render_template('dashboard.html', token_balance=token_balance, is_premium=is_premium)

    @app.route('/graphs')
    @login_required
    def graphs():
        return render_template('graphs.html', cryptos=cryptos, stocks=stocks)

    @app.route('/graphs/<symbol>')
    @login_required
    def graph(symbol):
        if symbol.upper() in cryptos or symbol.upper() in stocks:
            return render_template('graph.html', crypto=symbol.upper())
        abort(404)

    @app.route('/community')
    @login_required
    def community():
        return render_template('community.html')

    # --- LOGIN ---
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']

            conn = get_db()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            user = cursor.fetchone()
            conn.close()

            from app import bcrypt
            if user and bcrypt.check_password_hash(user['password'], password):
                session['username'] = username
                session['display_name'] = user['display_name']
                flash('You have been logged in successfully', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password', 'danger')

        return render_template('login.html')

    # --- LOGOUT ---
    @app.route('/logout')
    def logout():
        session.pop('username', None)
        session.pop('display_name', None)
        flash('You have been logged out', 'info')
        return redirect(url_for('login'))

    # --- REGISTER ---
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            display_name = request.form['display_name']

            conn = get_db()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            existing_user = cursor.fetchone()

            if existing_user:
                flash('Username already exists', 'danger')
            else:
                from app import bcrypt
                hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
                cursor.execute('INSERT INTO users (username, password, display_name) VALUES (?, ?, ?)',
                               (username, hashed_password, display_name))
                conn.commit()
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('login'))

            conn.close()

        return render_template('register.html')

    # --- PROFILE ---
    @app.route('/profile/<username>')
    @login_required
    def view_profile(username):
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT username, display_name, bio, email, profile_pic FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()

        if user is None:
            abort(404)

        return render_template('profile.html', user=dict(user))

    @app.route('/profile/edit', methods=['GET', 'POST'])
    @login_required
    def edit_profile():
        if request.method == 'POST':
            email = request.form.get('email', '').strip()
            bio = request.form.get('bio', '').strip()

            conn = get_db()
            cur = conn.cursor()
            cur.execute('UPDATE users SET email = ?, bio = ? WHERE username = ?', (email, bio, session['username']))
            conn.commit()
            conn.close()
            flash('Profile updated!', 'success')
            return redirect(url_for('view_profile', username=session['username']))

        conn = get_db()
        cur = conn.cursor()
        cur.execute('SELECT email, bio FROM users WHERE username = ?', (session['username'],))
        user = cur.fetchone()
        conn.close()

        return render_template('edit_profile.html', user=user)

    @app.route('/profile/upload_pic', methods=['POST'])
    @login_required
    def upload_profile_pic():
        if 'profile_pic' not in request.files:
            flash('No file part', 'danger')
            return redirect(url_for('view_profile', username=session['username']))

        file = request.files['profile_pic']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(url_for('view_profile', username=session['username']))

        if file and allowed_file(file.filename):
            filename = secure_filename(session['username'] + '_pic.' + file.filename.rsplit('.', 1)[1].lower())
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            conn = get_db()
            cur = conn.cursor()
            cur.execute('UPDATE users SET profile_pic = ? WHERE username = ?', (filename, session['username']))
            conn.commit()
            conn.close()

            flash('Profile picture updated.', 'success')

        return redirect(url_for('view_profile', username=session['username']))

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

    # --- BLOG THREAD ---
    @app.route('/blog/<int:post_id>', methods=['GET', 'POST'])
    @login_required
    def blog_post(post_id):
        conn = get_db()
        cur = conn.cursor()

        cur.execute('SELECT * FROM blog_posts WHERE id = ?', (post_id,))
        post = cur.fetchone()
        if not post:
            conn.close()
            abort(404)

        if request.method == 'POST':
            content = request.form.get('content', '').strip()
            if content:
                cur.execute('INSERT INTO comments (post_id, username, content) VALUES (?, ?, ?)',
                            (post_id, session['username'], content))
                conn.commit()
                conn.close()
                return redirect(url_for('blog_post', post_id=post_id))

        cur.execute('SELECT id, username, content, timestamp FROM comments WHERE post_id = ? ORDER BY timestamp ASC',
                    (post_id,))
        comments = [dict(row) for row in cur.fetchall()]
        conn.close()

        return render_template('blog_thread.html', post=dict(post), comments=comments)

    # --- ADMIN PANEL ---
    @app.route('/admin_panel')
    @admin_required
    def admin_panel():
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute('SELECT id, username, is_admin, is_premium, token_balance FROM users ORDER BY username')
        users = [dict(row) for row in cursor.fetchall()]

        cursor.execute('SELECT id, username, title FROM blog_posts ORDER BY timestamp DESC LIMIT 10')
        blog_posts = [dict(row) for row in cursor.fetchall()]

        conn.close()
        return render_template('admin.html', users=users, blog_posts=blog_posts)

    @app.route('/admin/toggle_premium/<int:user_id>', methods=['POST'])
    @admin_required
    def toggle_premium(user_id):
        conn = get_db()
        cursor = conn.cursor()
        # Toggle the is_premium status (0 to 1, 1 to 0)
        cursor.execute("UPDATE users SET is_premium = 1 - is_premium WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()
        flash('User premium status updated.', 'success')
        return redirect(url_for('admin_panel'))

    @app.route('/admin/add_tokens/<int:user_id>', methods=['POST'])
    @admin_required
    def add_tokens(user_id):
        try:
            tokens_to_add = int(request.form.get('tokens', 0))
            if tokens_to_add <= 0:
                flash('Please provide a positive number of tokens to add.', 'danger')
                return redirect(url_for('admin_panel'))

            conn = get_db()
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET token_balance = token_balance + ? WHERE id = ?", (tokens_to_add, user_id))
            conn.commit()
            conn.close()
            flash(f'{tokens_to_add} tokens added to user.', 'success')
        except ValueError:
            flash('Invalid number of tokens.', 'danger')
        return redirect(url_for('admin_panel'))
