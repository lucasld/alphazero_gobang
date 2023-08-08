configurations = {
    "config 1": {
        "name": "configuration2",
        "game": {
            "width": 5,
            "height": 5,
            "number win pieces": 4,
            "images folder": "./data/visualisations/",
            "number pits": 30
        },
        "alpha zero": {
            "number iterations": 1000,
            "number self-play": 200,
            "self play deque length": 5_000,
            "max example number": 20,
            "pit win threshold": 0.55
        },
        "mcts": {
            "number search traverses": 64
        },
        "neural network": {
            "model path": "./data/models_and_weights/",
            "weight path": "./data/models_and_weights/"
        }
    }
}

# this is the configuration that will be used when running main.py
# for training
TRAINING_CONFIG = configurations["config 1"]
# for playing
PLAYING_CONFIG = configurations["config 1"]