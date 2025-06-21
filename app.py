from flask import Flask, render_template, request, Response
from flask_sock import Sock
from dynamic_tracker import get_frames, start_exercise, stop_exercise, set_websocket

app = Flask(__name__)
sock = Sock(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(get_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/exercise', methods=['POST'])
def exercise():
    # Get user input from form
    exercise_input = request.form['exercise'].strip()

    # Run generic exercise tracker
    if start_exercise(exercise_input):
        return render_template('exercise.html', exercise_name=exercise_input.title())
    return render_template('index.html', error="Gemini Error")

@sock.route('/ws')
def websocket(ws):
    set_websocket(ws)
    while True:
        ws.receive()  # Keep the connection alive

@app.route('/finish', methods=['POST'])
def finish():
    exercise_input = request.form['exercise']
    results = stop_exercise()
    
    return render_template(
        'result.html',
        exercise_name=exercise_input,
        count=results['count'],
        time_taken=round(results['duration'], 2),
        feedback=results['feedback']
    )

if __name__ == '__main__':
    app.run(debug=True)
