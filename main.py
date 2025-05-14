from flask import Flask, render_template, request, redirect, url_for, flash, session, abort, jsonify
import requests
from flask_bcrypt import Bcrypt
from flask_caching import Cache
import sqlite3
import os
from functools import wraps
from werkzeug.utils import secure_filename

from werkzeug.exceptions import RequestEntityTooLarge

cryptos = ['bitcoin', 'ethereum', 'solana', 'cardano', 'ripple', 'dogecoin', 'polkadot', 'litecoin', 'chainlink', 'uniswap']
app = Flask(__name__)

# Set up secret key for session management
app.secret_key = "aaaaaaaaaaaaaaaaaaaaaa"

app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2 MB max


# Initialize Bcrypt
bcrypt = Bcrypt(app)

# Database file path
DATABASE = 'db.sqlite'

cache = Cache(app, config={
    'CACHE_TYPE': 'simple',       # in-memory cache
    'CACHE_DEFAULT_TIMEOUT': 300  # 5 minutes
})


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
                        display_name TEXT NOT NULL,
                        is_admin INTEGER DEFAULT 0
)''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS blog_posts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS chat_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        message TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    conn.commit()
    conn.close()

@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    flash('File is too large. Max size is 2MB.', 'danger')
    return redirect(request.referrer or url_for('index'))

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
    
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please login to access this page', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)  # Forward the arguments to the original function
    return decorated_function

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

    # Preload data
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT email, bio FROM users WHERE username = ?', (session['username'],))
    user = cur.fetchone()
    conn.close()

    return render_template('edit_profile.html', user=user)


@app.route('/api/binance/ohlcv/<symbol>')
@cache.cached(timeout=300, query_string=True)
@login_required
def binance_ohlcv(symbol):
    interval = request.args.get('interval','1d')
    limit    = int(request.args.get('limit',1000))

    url = 'https://api.binance.com/api/v3/klines'
    params = {'symbol': symbol.upper(), 'interval': interval, 'limit': limit}
    resp = requests.get(url, params=params, timeout=5)

    if resp.status_code != 200:
        return jsonify({'error': resp.json().get('msg','Binance error')}), resp.status_code

    data = resp.json()
    if not isinstance(data,list):
        return jsonify({'error':'Unexpected response format'}),400

    times   = [item[0] for item in data]
    opens   = [float(item[1]) for item in data]
    highs   = [float(item[2]) for item in data]
    lows    = [float(item[3]) for item in data]
    closes  = [float(item[4]) for item in data]
    volumes = [float(item[5]) for item in data]

    return jsonify({
        'timestamps': times,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })

@app.route('/graphs/<crypto_name>')
@login_required
def crypto_graph(crypto_name):
    print(f"Requested graph for: {crypto_name}")  # Debugging print
    if crypto_name not in cryptos:
        abort(404)
    return render_template('graph.html', crypto=crypto_name)

@app.route('/community')
@login_required
def community():
    return render_template('community.html')

@app.route('/api/chat/delete/<int:msg_id>', methods=['POST'])
@login_required
def delete_chat_message(msg_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT username FROM chat_messages WHERE id = ?', (msg_id,))
    row = cur.fetchone()

    if not row:
        return jsonify({'error': 'Message not found'}), 404

    if session['username'] != row['username'] and session['username'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403

    cur.execute('DELETE FROM chat_messages WHERE id = ?', (msg_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/api/blog/delete/<int:post_id>', methods=['POST'])
@login_required
def delete_blog_post(post_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT username FROM blog_posts WHERE id = ?', (post_id,))
    row = cur.fetchone()

    if not row:
        return jsonify({'error': 'Post not found'}), 404

    if session['username'] != row['username'] and session['username'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403

    cur.execute('DELETE FROM blog_posts WHERE id = ?', (post_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})



# --- Blog API ---
@app.route('/api/blog/new', methods=['POST'])
@login_required
def new_blog_post():
    data = request.get_json()
    title = data.get('title', '').strip()
    content = data.get('content', '').strip()
    if not title or not content:
        return jsonify({'error': 'Title and content required'}), 400

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO blog_posts (username, title, content) VALUES (?, ?, ?)',
                   (session['username'], title, content))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

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

        # Shrani v bazo
        conn = get_db()
        cur = conn.cursor()
        cur.execute('UPDATE users SET profile_pic = ? WHERE username = ?', (filename, session['username']))
        conn.commit()
        conn.close()

        flash('Profile picture updated.', 'success')

    return redirect(url_for('view_profile', username=session['username']))


@app.route('/api/blog/posts')
@login_required
def get_blog_posts():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, title, content, timestamp FROM blog_posts ORDER BY timestamp DESC LIMIT 20')
    posts = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(posts)



# --- Chat API ---
@app.route('/api/chat/send', methods=['POST'])
@login_required
def send_chat_message():
    data = request.get_json()
    message = data.get('message', '').strip()
    if not message:
        return jsonify({'error': 'Message cannot be empty'}), 400

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO chat_messages (username, message) VALUES (?, ?)',
                   (session['username'], message))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


@app.route('/api/chat/messages')
@login_required
def get_chat_messages():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, message, timestamp FROM chat_messages ORDER BY timestamp DESC LIMIT 50')
    messages = [dict(row) for row in cursor.fetchall()][::-1]
    conn.close()
    return jsonify(messages)




# Route for listing all available graphs
@app.route('/graphs')
@login_required
def graphs():
    cryptocurrencies =  cryptos
    return render_template('graphs.html', cryptos=cryptocurrencies)

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

@app.route('/admin/delete_user/<username>', methods=['POST'])
@admin_required
def delete_user(username):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE username = ?', (username,))
    conn.commit()
    conn.close()
    flash(f'User {username} deleted.', 'success')
    return redirect(url_for('admin_panel'))

@app.route('/admin/clear_chat', methods=['POST'])
@admin_required
def clear_chat():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM chat_messages')
    conn.commit()
    conn.close()
    flash('Chat cleared.', 'success')
    return redirect(url_for('admin_panel'))

@app.route('/admin/delete_blog/<int:post_id>', methods=['POST'])
@admin_required
def delete_blog(post_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM blog_posts WHERE id = ?', (post_id,))
    conn.commit()
    conn.close()
    flash(f'Blog post {post_id} deleted.', 'success')
    return redirect(url_for('admin_panel'))

@app.route('/admin_panel',methods=['GET'])
@admin_required
def admin_panel():
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('SELECT username, is_admin FROM users ORDER BY username')
    users = [dict(row) for row in cursor.fetchall()]

    cursor.execute('SELECT id, username, title FROM blog_posts ORDER BY timestamp DESC LIMIT 10')
    blog_posts = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return render_template('admin.html',users=users, blog_posts=blog_posts)

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('display_name', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))


def migrate_add_profile_fields():
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("ALTER TABLE users ADD COLUMN bio TEXT DEFAULT ''")
        cur.execute("ALTER TABLE users ADD COLUMN email TEXT DEFAULT ''")
        cur.execute("ALTER TABLE users ADD COLUMN profile_pic TEXT DEFAULT 'default.png'")
        print("Dodan bio, email in profilna slika.")
    except sqlite3.OperationalError as e:
        print("Morda stolpci že obstajajo:", e)
    conn.commit()
    conn.close()




if __name__ == '__main__':
    # Initialize the database and create admin if necessary
    init_db()

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', ('admin',))
    admin = cursor.fetchone()
    migrate_add_profile_fields()

    if not admin:
        hashed_password = bcrypt.generate_password_hash('admin123').decode('utf-8')
        cursor.execute('INSERT INTO users (username, password, display_name,is_admin) VALUES (?, ?, ?,?)',
                       ('admin', hashed_password, 'Admin User',1))
        conn.commit()

    conn.close() 

    app.run(debug=True)
