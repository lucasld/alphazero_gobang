configurations = {
    "config 1": {
        "name": "configuration1",
        "game": {
            "width": 5,
            "height": 5,
            "number win pieces": 3,
            "images folder": "./src/training/training_vis",
            "number pits": 30
        },
        "alpha zero": {
            "number iterations": 100,
            "number self-play": 50,
            "self play deque length": 5000,
            "max example number": 100_000,
            "pit win threshold": 0.55
        },
        "mcts": {
            "number search traverses": 64
        }
    },
    "config 2": {
        "name": "configuration2",
        "game": {
            "width": 5,
            "height": 5,
            "number win pieces": 3,
            "images folder": "./src/training/training_vis",
            "number pits": 30
        },
        "alpha zero": {
            "number iterations": 1000,
            "number self-play": 100,
            "self play deque length": 2_000,
            "max example number": 25,
            "pit win threshold": 0.55
        },
        "mcts": {
            "number search traverses": 64
        }
    }
}

# this is the configuration that will be used when running main.py
# for training
TRAINING_CONFIG = configurations["config 2"]
# for playing
PLAYING_CONFIG = configurations["config 2"]