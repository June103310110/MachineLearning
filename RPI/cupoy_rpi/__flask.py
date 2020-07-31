from flask import Flask
app = Flask(__name__)

@app.route("/")

def hello():
    return 'hello june'

if __name__ == "__main__":
    app.run(host='10.17.4.132', port = 8080, debug = True, threaded = True)