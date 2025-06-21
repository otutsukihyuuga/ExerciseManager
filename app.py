from flask import Flask, render_template, request
from dynamic_tracker import run_exercise

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/exercise', methods=['POST'])
def exercise():
    # Get user input from form
    exercise_input = request.form['exercise'].strip()

    # Run generic exercise tracker
    count, time_taken, feedback = run_exercise(exercise_input)

    return render_template(
        'result.html',
        exercise_name=exercise_input.title(),  # Title case for display
        count=count,
        time_taken=round(time_taken, 2),
        feedback=feedback
    )

if __name__ == '__main__':
    app.run(debug=True)
