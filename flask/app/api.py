from flask import Blueprint, request, jsonify, session, flash
from app.decorators import login_required, admin_required
from app.db import get_db
from app import cache
import requests
import yfinance as yf
from tasks.core import test_task, train_model_task
import json
from datetime import datetime

api_bp = Blueprint('api', __name__)

# --- Celery test task ---
@api_bp.route('/api/test-task/<crypto>')
def test_route(crypto):
    pass

# --- Trigger Celery train_model_task ---
@api_bp.route('/api/train/<crypto>', methods=['POST'])
@login_required
def trigger_training(crypto):
    task = train_model_task.delay(crypto)
    return jsonify({'status': f'Training started for {crypto}', 'task_id': task.id})

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

# --- Local Market Data OHLCV ---
@api_bp.route('/api/local/ohlcv/<string:symbol>', methods=['GET'])
@login_required 
def get_local_ohlcv(symbol):
    """
    Fetches stored OHLCV data for a given symbol from the local database.
    """
    limit = request.args.get('limit', default=100, type=int)
    conn = None
    try:
        conn = get_db()
        cursor = conn.cursor()
        query = """
            SELECT symbol, timestamp, open, high, low, close, volume 
            FROM market_data 
            WHERE symbol = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        cursor.execute(query, (symbol.upper(), limit)) # Convert symbol to upper, common practice
        rows = cursor.fetchall()
        data = [dict(row) for row in rows]
        return jsonify(data)
    except Exception as e:
        # Basic error handling, can be more specific
        return jsonify({'error': str(e)}), 500
    finally:
        if conn:
            conn.close()

@api_bp.route('/api/predictions/<symbol>', methods=['GET'])
@login_required
def get_predictions(symbol):
    symbol = symbol.upper().replace("USDT", "")  # Ensure symbol is uppercase

    conn = None
    token_deducted_this_request = False # Moved initialization here
    try:
        conn = get_db()
        cursor = conn.cursor()

        # Get current user's details (is_premium, token_balance)
        cursor.execute("SELECT is_premium, token_balance FROM users WHERE username = ?", (session['username'],))
        user = cursor.fetchone()

        if not user:
            return jsonify({'error': 'User not found'}), 404

        user_is_premium = user['is_premium'] == 1
        user_token_balance = user['token_balance']

        # Authorization logic
        can_access = False
        # token_deducted_this_request = False # Original position, moved up
        if user_is_premium:
            can_access = True
        elif user_token_balance > 0:
            can_access = True
            # Deduct token
            cursor.execute("UPDATE users SET token_balance = token_balance - 1 WHERE username = ?", (session['username'],))
            conn.commit() 
            token_deducted_this_request = True
        
        if not can_access:
            return jsonify({'error': 'Upgrade required or insufficient tokens.'}), 403

        # Fetch prediction
        interval = request.args.get('interval', None)
        
        query = """
            SELECT symbol, interval, predictions, actuals, timestamp 
            FROM predictions 
            WHERE symbol = ? 
        """
        params = [symbol.upper()]

        if interval:
            query += " AND interval = ? "
            params.append(interval)
        
        query += " ORDER BY timestamp DESC LIMIT 1"
        
        cursor.execute(query, tuple(params))
        prediction_row = cursor.fetchone()

        if prediction_row:
            prediction_data = dict(prediction_row)
            prediction_data['predictions'] = json.loads(prediction_data['predictions'])
            prediction_data['actuals'] = json.loads(prediction_data['actuals'])
            return jsonify(prediction_data)
        else:
            return jsonify({'message': 'No prediction available yet.'}), 404

    except Exception as e:
        # Note: Current logic does not refund token if error occurs after deduction.
        # This is per current understanding of "deduct per API call".
        # If a refund is needed on error post-deduction, more complex transaction handling
        # or moving deduction after successful fetch would be required.
        return jsonify({'error': str(e)}), 500
    finally:
        if conn:
            conn.close()


@api_bp.route('/admin/delete_user/<username>', methods=['POST'])
@admin_required
def delete_user(username):
    conn = get_db()
    cur = conn.cursor()
    cur.execute('DELETE FROM users WHERE username = ?', (username,))
    conn.commit()
    conn.close()
    return jsonify({f'User {username} has been deleted.':'success'})

@api_bp.route('/api/chat_clear', methods=['POST'])
@admin_required
def clear_chat():
    conn = get_db()
    cur = conn.cursor()
    cur.execute('DELETE FROM chat_messages')
    conn.commit()
    conn.close()
    return jsonify({'success': True})


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



