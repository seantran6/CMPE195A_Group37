from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Render the index.html template

@app.route('/recognition')
def recognition():
    return render_template('recognition.html')  # Render the recognition.html template

if __name__ == '__main__':
    app.run(debug=True)
