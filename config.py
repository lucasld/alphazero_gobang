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
    },
    "config 5": {
        "name": "configuration05",
        "game": {
            "width": 5,
            "height": 5,
            "number win pieces": 4,
            "images folder": "./data/visualisations/",
            "number pits": 16
        },
        "alpha zero": {
            "number iterations": 1000,
            "number self-play": 32,
            "self play deque length": 2000,
            "max example number": 30,
            "pit win threshold": 0.55,
            "experience path": "./data/experience/"
        },
        "mcts": {
            "number search traverses": 24
        },
        "neural network": {
            "batch size": 32,
            "epochs": 50,
            "model path": "./data/models_and_weights/",
            "weight path": "./data/models_and_weights/",
            "architecture": [("conv", 32), ("conv", 32), ("dense", 64), ("dense", 32), ("dense", 16)]
        }
    },
    "config 7": {
        "name": "configuration07",
        "game": {
            "width": 5,
            "height": 5,
            "number win pieces": 4,
            "images folder": "./data/visualisations/",
            "number pits": 16
        },
        "alpha zero": {
            "number iterations": 1000,
            "number self-play": 25,
            "self play deque length": 2000,
            "max example number": 1_000_000,
            "pit win threshold": 0.55,
            "experience path": "./data/experience/config6/"
        },
        "mcts": {
            "number search traverses": 32
        },
        "neural network": {
            "batch size": 32,
            "epochs": 20,
            "model path": "./data/models_and_weights/",
            "weight path": "./data/models_and_weights/",
            "architecture": [("conv", 32), ("conv", 32), ("dense", 64), ("dense", 32), ("dense", 16)]
        }
    },
    "config 8": {
        "name": "configuration08",
        "game": {
            "width": 5,
            "height": 5,
            "number win pieces": 4,
            "images folder": "./data/visualisations/",
            "number pits": 6
        },
        "alpha zero": {
            "number iterations": 1000,
            "number self-play": 3,
            "self play deque length": 2000,
            "max example number": 1_000_000,
            "pit win threshold": 0.55,
            "experience path": "./data/experience/"
        },
        "mcts": {
            "number search traverses": 32
        },
        "neural network": {
            "batch size": 32,
            "epochs": 2,
            "model path": "./data/models_and_weights/",
            "weight path": "./data/models_and_weights/",
            "architecture": [("conv", 32), ("conv", 32), ("dense", 64), ("dense", 32), ("dense", 16)]
        }
    },
    "config 9": {
        "name": "configuration09",
        "game": {
            "width": 5,
            "height": 5,
            "number win pieces": 4,
            "images folder": "./data/visualisations/",
            "number pits": 6,
            "experience path": "./data/experience/dataset/"
        },
        "alpha zero": {
            "number iterations": 1000,
            "number self-play": 3,
            "self play deque length": 4000,
            "max example number": 70,
            "pit win threshold": 0.55,
            "experience path": "./data/experience/dataset/"
        },
        "mcts": {
            "number search traverses": 32
        },
        "neural network": {
            "batch size": 32,
            "epochs": 8,
            "model path": "./data/models_and_weights/",
            "weight path": "./data/models_and_weights/",
            "architecture": [("conv", 16), ("conv", 16), ("dense", 8), ("dense", 32), ("dense", 16)]
        }
    },
    "config 10": {
        "name": "configuration10",
        "game": {
            "width": 5,
            "height": 5,
            "number win pieces": 4,
            "images folder": "./data/visualisations/",
            "number pits": 10,
            "experience path": "./data/experience/dataset2/"
        },
        "alpha zero": {
            "number iterations": 10_000,
            "number self-play": 20,
            "self play deque length": 10_000,
            "max example number": 30,
            "pit win threshold": 0.55,
            "experience path": "./data/experience/dataset2/"
        },
        "mcts": {
            "number search traverses": 64
        },
        "neural network": {
            "batch size": 64,
            "epochs": 8,
            "model path": "./data/models_and_weights/",
            "weight path": "./data/models_and_weights/",
            "architecture": [("conv", 16), ("conv", 16), ("dense", 8), ("dense", 32), ("dense", 16)]
        }
    },
    "config 12": {
        "name": "configuration12",
        "game": {
            "width": 5,
            "height": 5,
            "number win pieces": 4,
            "images folder": "./data/visualisations/",
            "number pits": 10,
            "experience path": "./data/experience/dataset3/"
        },
        "alpha zero": {
            "number iterations": 10_000,
            "number self-play": 5,
            "self play deque length": 10_000,
            "max example number": 30,
            "pit win threshold": 0.55,
            "experience path": "./data/experience/dataset3/"
        },
        "mcts": {
            "number search traverses": 32
        },
        "neural network": {
            "batch size": 64,
            "epochs": 8,
            "model path": "./data/models_and_weights/",
            "weight path": "./data/models_and_weights/",
            "architecture": [("conv", 64), ("conv", 32), ("conv", 16), ("dense", 32), ("dense", 16)]
        }
    },
    "config 001": {
        "name": "configuration001",
        "game": {
            "width": 5,
            "height": 5,
            "number win pieces": 4,
            "images folder": "./data/visualisations/config001/",
            "number pits": 8,
            "experience path": "./data/experience/dataset_main/"
        },
        "alpha zero": {
            "number iterations": 10_000,
            "number self-play": 20,
            "self play deque length": 10_000,
            "max example number": 30,
            "pit win threshold": 0.55,
            "experience path": "./data/experience/dataset_main/"
        },
        "mcts": {
            "number search traverses": 64
        },
        "neural network": {
            "batch size": 64,
            "epochs": 8,
            "model path": "./data/models_and_weights/config001/",
            "weight path": "./data/models_and_weights/config001/",
            "architecture": [("conv", 64), ("conv", 32), ("conv", 24), ("dense", 42), ("dense", 16)]
        }
    },
    "config 002": {
        "name": "configuration002",
        "game": {
            "width": 5,
            "height": 5,
            "number win pieces": 4,
            "images folder": "./data/visualisations/config002/",
            "number pits": 10,
            "experience path": "./data/experience/dataset_main/"
        },
        "alpha zero": {
            "number iterations": 10_000,
            "number self-play": 20,
            "self play deque length": 10_000,
            "max example number": 20,
            "pit win threshold": 0.55,
            "experience path": "./data/experience/dataset_main/"
        },
        "mcts": {
            "number search traverses": 32
        },
        "neural network": {
            "batch size": 128,
            "epochs": 4,
            "model path": "./data/models_and_weights/config002/",
            "weight path": "./data/models_and_weights/config002/",
            "architecture": [("conv", 32), ("conv", 64), ("dense", 64), ("dense", 32)]
        }
    },
    "config 003": {
        "name": "configuration003",
        "game": {
            "width": 4,
            "height": 4,
            "number win pieces": 3,
            "images folder": "./data/visualisations/config003/",
            "number pits": 100,
            "experience path": "./data/experience/dataset_3wins/"
        },
        "alpha zero": {
            "number iterations": 10_000,
            "number self-play": 180,
            "self play deque length": 10_000,
            "max example number": 15,#30,
            "pit win threshold": 0.55,
            "experience path": "./data/experience/dataset_3wins/"
        },
        "mcts": {
            "number search traverses": 16
        },
        "neural network": {
            "batch size": 32,
            "epochs": 1,
            "model path": "./data/models_and_weights/config003/",
            "weight path": "./data/models_and_weights/config003/",
            "architecture": [("conv", 24), ("conv", 32), ("dense", 32), ("dense", 16)]
        }
    }
}

# this is the configuration that will be used when running main.py
# for training
TRAINING_CONFIG = configurations["config 003"]
# for playing
PLAYING_CONFIG = configurations["config 003"]