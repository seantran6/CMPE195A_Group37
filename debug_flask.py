from flask import Flask, render_template, redirect, url_for

app = Flask(__name__)

# Root route, redirects to home
@app.route('/')
def root():
    return redirect(url_for('home'))  # This redirects to /home

# Home route serving index.html
@app.route('/home')
def home():
    return render_template('index.html')  # This serves index.html for the home page

# Recognition route serving recognition.html
@app.route('/recognition')
def recognition():
    return render_template('recognition.html')  # This serves recognition.html

if __name__ == '__main__':
    app.run(debug=True)
