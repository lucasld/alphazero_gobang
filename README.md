# AlphaZero for Gobang

## Project Structure

The `alphazero_gobang` project is organized into several key components to ensure modularity and maintainability. Below is an overview of the directory structure and an explanation of each component:

### Source Code (`src/`)

All core source code is contained within the `src/` directory, organized into specific modules:

#### `environment/`

#### `model/`

#### `training/`

#### `play/`

#### `utils/`

### Configuration and Entry Point

- **`main.py`**: The entry point for both training the model and playing the game. Includes command-line arguments to select different modes and options.

### How to Use

To train the model, you can run the following command from the root directory:

```bash
python main.py --train
```

To play against the trained AI:

```bash
python main.py --play
```