from flask import Blueprint, request, jsonify, session
from app.decorators import login_required, admin_required
from app.db import get_db
from app import cache
import requests
import yfinance as yf
from tasks.core import test_task

api_bp = Blueprint('api', __name__)

# --- Celery test task ---
@api_bp.route('/api/test-task')
def test_route():
    task = test_task.delay()
    return jsonify({'task_id': task.id})


# --- Binance OHLCV ---
@api_bp.route('/api/binance/ohlcv/<symbol>')
@login_required
def binance_ohlcv(symbol):
    interval = request.args.get('interval', '1d')
    limit = int(request.args.get('limit', 1000))

    url = 'https://api.binance.com/api/v3/klines'
    params = {'symbol': symbol.upper(), 'interval': interval, 'limit': limit}
    resp = requests.get(url, params=params, timeout=5)

    if resp.status_code != 200:
        return jsonify({'error': resp.json().get('msg', 'Binance error')}), resp.status_code

    data = resp.json()
    return jsonify({
        'timestamps': [item[0] for item in data],
        'open': [float(item[1]) for item in data],
        'high': [float(item[2]) for item in data],
        'low': [float(item[3]) for item in data],
        'close': [float(item[4]) for item in data],
        'volume': [float(item[5]) for item in data]
    })


# --- YFinance OHLCV ---
@api_bp.route('/api/stocks/ohlcv/<symbol>')
@cache.cached(timeout=300, query_string=True)
@login_required
def stock_ohlcv(symbol):
    interval = request.args.get('interval', '1d')
    limit = int(request.args.get('limit', 60))

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='max', interval=interval).dropna()
        hist = hist.tail(limit)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    data = {
        'timestamps': [int(row.name.timestamp() * 1000) for _, row in hist.iterrows()],
        'open': hist['Open'].tolist(),
        'high': hist['High'].tolist(),
        'low': hist['Low'].tolist(),
        'close': hist['Close'].tolist(),
        'volume': hist['Volume'].tolist()
    }

    return jsonify(data)


# --- Chat send ---
@api_bp.route('/api/chat/send', methods=['POST'])
@login_required
def send_chat_message():
    data = request.get_json()
    message = data.get('message', '').strip()
    if not message:
        return jsonify({'error': 'Message cannot be empty'}), 400

    conn = get_db()
    cur = conn.cursor()
    cur.execute('INSERT INTO chat_messages (username, message) VALUES (?, ?)',
                (session['username'], message))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


# --- Chat messages ---
@api_bp.route('/api/chat/messages')
@login_required
def get_chat_messages():
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT id, username, message, timestamp FROM chat_messages ORDER BY timestamp DESC LIMIT 50')
    messages = [dict(row) for row in cur.fetchall()][::-1]
    conn.close()
    return jsonify(messages)


# --- Chat delete ---
@api_bp.route('/api/chat/delete/<int:msg_id>', methods=['POST'])
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


# --- Blog new post ---
@api_bp.route('/api/blog/new', methods=['POST'])
@login_required
def new_blog_post():
    data = request.get_json()
    title = data.get('title', '').strip()
    content = data.get('content', '').strip()
    if not title or not content:
        return jsonify({'error': 'Title and content required'}), 400

    conn = get_db()
    cur = conn.cursor()
    cur.execute('INSERT INTO blog_posts (username, title, content) VALUES (?, ?, ?)',
                (session['username'], title, content))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


# --- Blog posts (preview) ---
@api_bp.route('/api/blog/posts')
@login_required
def get_blog_posts():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
    SELECT blog_posts.id, blog_posts.username, blog_posts.title, blog_posts.content, blog_posts.timestamp,
           COUNT(comments.id) AS comment_count
    FROM blog_posts
    LEFT JOIN comments ON blog_posts.id = comments.post_id
    GROUP BY blog_posts.id
    ORDER BY blog_posts.timestamp DESC
    LIMIT 20
''')
    posts = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return jsonify(posts)


# --- Blog delete ---
@api_bp.route('/api/blog/delete/<int:post_id>', methods=['POST'])
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


# --- Comment delete ---
@api_bp.route('/api/comment/delete/<int:comment_id>', methods=['POST'])
@login_required
def delete_comment(comment_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT username FROM comments WHERE id = ?', (comment_id,))
    comment = cur.fetchone()

    if not comment:
        return jsonify({'error': 'Comment not found'}), 404

    if session['username'] != comment['username'] and session['username'] != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403

    cur.execute('DELETE FROM comments WHERE id = ?', (comment_id,))
    conn.commit()
    conn.close()
    return jsonify({'success': True})


# --- Trigger Celery train_model_task ---
@api_bp.route('/api/train/<crypto>', methods=['POST'])
@login_required
def trigger_training(crypto):
    task = train_model_task.delay(crypto)
    return jsonify({'status': f'Training started for {crypto}', 'task_id': task.id})
