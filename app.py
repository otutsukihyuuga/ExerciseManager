from flask import Flask, render_template, request, Response, redirect, url_for, session, flash
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from dynamic_tracker import get_frames, start_exercise, stop_exercise, set_websocket
from datetime import timedelta
from dotenv import load_dotenv
import os
from chat import init_chat
from flask_sock import Sock

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key_here')
app.permanent_session_lifetime = timedelta(days=7)

# MongoDB Atlas setup
app.config["MONGO_URI"] = os.environ.get('MONGO_URI')
mongo = PyMongo(app)
users_collection = mongo.db.users

# Flask extensions
bcrypt = Bcrypt(app)
sock = Sock(app)

# Initialize chat module
init_chat(app, users_collection)

@app.route('/')
def index():
    if 'user' in session:
        user = users_collection.find_one({'email': session['user']})
        history = user.get('history', [])
        return render_template('index.html', history=history)
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']

        if users_collection.find_one({'email': email}):
            flash('Email already exists. Please log in.', 'error')
            return redirect(url_for('login'))

        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        users_collection.insert_one({'email': email, 'password': hashed_pw, 'history': [], 'chats': []})
        flash('Registered successfully. Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip().lower()
        password = request.form['password']
        user = users_collection.find_one({'email': email})

        if user and bcrypt.check_password_hash(user['password'], password):
            session['user'] = user['email']
            return redirect(url_for('index'))
        else:
            flash('Incorrect email or password.', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/exercise', methods=['POST'])
def exercise():
    if 'user' not in session:
        return redirect(url_for('login'))
    exercise_input = request.form['exercise']
    start_exercise(exercise_input)
    return render_template('exercise.html', exercise_name=exercise_input)

@app.route('/video_feed')
def video_feed():
    if 'user' not in session:
        return redirect(url_for('login'))
    return Response(get_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/finish', methods=['POST'])
def finish():
    exercise_input = request.form['exercise']
    results = stop_exercise()

    if 'user' in session:
        users_collection.update_one(
            {'email': session['user']},
            {'$push': {
                'history': {
                    'exercise': exercise_input,
                    'count': results['count'],
                    'duration': round(results['duration'], 2),
                    'feedback': results['feedback']
                }
            }}
        )

    return render_template(
        'result.html',
        exercise_name=exercise_input.title(),
        count=results['count'],
        time_taken=round(results['duration'], 2),
        feedback=results['feedback']
    )

@sock.route('/ws')
def websocket(ws):
    """Handle WebSocket connection for exercise tracking"""
    if 'user' not in session:
        return
    
    set_websocket(ws)
    try:
        while True:
            # Keep the connection alive and wait for messages
            ws.receive()
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        set_websocket(None)  # Clear the websocket when connection ends

if __name__ == '__main__':
    app.run(debug=True)
