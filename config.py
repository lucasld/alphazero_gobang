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
    },
    "config 2": {
        "name": "configuration02",
        "game": {
            "width": 5,
            "height": 5,
            "number win pieces": 4,
            "images folder": "./data/visualisations/",
            "number pits": 16
        },
        "alpha zero": {
            "number iterations": 1000,
            "number self-play": 200,
            "self play deque length": 5_000,
            "max example number": 20,
            "pit win threshold": 0.55,
            "experience path": "./data/experience/"
        },
        "mcts": {
            "number search traverses": 64
        },
        "neural network": {
            "model path": "./data/models_and_weights/",
            "weight path": "./data/models_and_weights/",
            "architecture": [("conv", 64), ("conv", 32), ("conv", 16), ("dense", 128), ("dense", 64), ("dense", 32)]
        }
    },
    "config 3": {
        "name": "configuration03",
        "game": {
            "width": 5,
            "height": 5,
            "number win pieces": 4,
            "images folder": "./data/visualisations/",
            "number pits": 16
        },
        "alpha zero": {
            "number iterations": 1000,
            "number self-play": 200,
            "self play deque length": 5_000,
            "max example number": 30,
            "pit win threshold": 0.55,
            "experience path": "./data/experience/"
        },
        "mcts": {
            "number search traverses": 64
        },
        "neural network": {
            "batch size": 64,
            "epochs": 50,
            "model path": "./data/models_and_weights/",
            "weight path": "./data/models_and_weights/",
            "architecture": [("conv", 64), ("conv", 32), ("conv", 16), ("dense", 128), ("dense", 64), ("dense", 32)]
        }
    }
}

# this is the configuration that will be used when running main.py
# for training
TRAINING_CONFIG = configurations["config 3"]
# for playing
PLAYING_CONFIG = configurations["config 3"]