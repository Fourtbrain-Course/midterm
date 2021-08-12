from flask import Flask
from flask import request
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import numpy as np


class Pipeline:
    train_data = None
    validation_data = None
    test_data =None
    model = None

    def get_data(self):
        self.train_data, self.validation_data, self.test_data = tfds.load(
            name="imdb_reviews",
            split=("train[:60%]", "train[60%:]", "test"),
            as_supervised=True
        )

    def create_model(self):
        embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
        hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)

        self.model = tf.keras.Sequential()
        self.model.add(hub_layer)
        self.model.add(tf.keras.layers.Dense(16, activation="relu"))
        self.model.add(tf.keras.layers.Dense(1))

        print("Model Summary")
        print(self.model.summary())

    def compile_model(self):
        self.model.compile(optimizer="adam",
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

    def train_model(self):
        history = self.model.fit(self.train_data.shuffle(10000).batch(512),
            epochs=10,
            validation_data=self.validation_data.batch(512),
            verbose=1
        )

    def evaluate_model(self):
        results = self.model.evaluate(self.test_data.batch(512), verbose=2)
        for name, value in zip(self.model.metrics_names, results):
            print("%s: %.3f"% (name, value))

    def save_model(self):
        self.model.save("sentiment-classification-model")

    def load_model(self):
        self.model = tf.keras.models.load_model("sentiment-classification-model")

    def predict(self, review):
        prediction = self.model.predict(np.array([review]))
        result = tf.keras.activations.sigmoid(prediction)
        return result.numpy()[0][0]


pipeline = Pipeline()


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    # breakpoint()
    return "THIS IS HOME!"


@app.route("/training", methods=["GET"])
def training():
    pipeline.get_data()
    pipeline.create_model()
    pipeline.train_model()
    pipeline.evaluate_model()
    pipeline.save_model()

    return "Model trained successfully!"


@app.route("/prediction", methods=["POST"])
def prediction():
    pipeline.load_model()
    data = request.json
    review = data["review"]
    result = pipeline.predict(np.array([review]))

    return f"The prediction is: {result}"


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
