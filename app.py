from flask import Flask
from flask import request



app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    # breakpoint()
    return "THIS IS HOME!"


@app.route("/prediction", methods=["POST"])
def prediction():
    return "PREDICTION!"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
