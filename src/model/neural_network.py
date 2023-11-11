import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.models import Model
import numpy as np
import os

# Check available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        # Set memory growth for each GPU
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPUs available.")


class NeuralNetwork:
    def __init__(self, config, load_existing_model=False):
        self.config = config
        name = config["name"]
        self.weight_path = config["neural network"]["weight path"] + name
        self.input_shape = (config["game"]["height"],
                            config["game"]["width"],
                            1)
        self.policy_shape = self.input_shape[0] * self.input_shape[1]
        self.batch_size = config["neural network"]["batch size"]
        self.epochs = config["neural network"]["epochs"]

        # load existing weights into model if path is given
        model_path = f"{config['neural network']['model path']}{name}.keras"
        if os.path.exists(model_path) and load_existing_model:
            print("Old model loaded")
            #self.model.load_weights(load_weights_path)
            self.model = tf.keras.models.load_model(model_path)
        else:
            print("Creating new model")
            # convert to float16 for memory saving
            tf.keras.backend.set_floatx('float16')
            self.model = self.create_value_policy_network(self.input_shape,
                                                          self.policy_shape)



    def create_value_policy_network(self, input_shape: tuple,
                                    policy_shape: int) -> tf.keras.Model:
        input_layer = tf.keras.Input(shape=input_shape)
        x = input_layer
        """# Convolutional layers
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        #x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

        # Flatten the output of the convolutional layers
        x = Flatten()(x)

        # Fully connected layers
        #x = Dense(256, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        x = Dense(8, activation='relu')(x)
        """
        last_layer_2d = True
        architecture = self.config["neural network"]["architecture"]
        for layer_type, number in architecture:
            if layer_type == "conv":
                x = Conv2D(number, (3, 3), activation='relu', padding='same')(x)
            elif layer_type == "dense":
                if last_layer_2d:
                    x = Flatten()(x)
                    last_layer_2d = False
                x = Dense(number, activation='relu')(x)
        # Output layers for policy and value estimates
        policy_output = Dense(policy_shape, activation='softmax', name='policy_output')(x)
        value_output = Dense(1, activation='tanh', name='value_output')(x)

        model = Model(inputs=input_layer, outputs=[policy_output, value_output])
        
        model.compile(
            optimizer='adam',
            loss={
                'policy_output': 'categorical_crossentropy',
                'value_output': 'mean_squared_error'
            },
            metrics={
                'policy_output': 'accuracy'
            }
        )
        return model


    def train_nnet(self, train_examples):
        input_boards, target_pis, target_vs = list(zip(*train_examples))
        input_boards = np.array(input_boards)
        target_pis = np.array(target_pis)
        target_vs = np.array(target_vs)
        print(input_boards.shape, target_pis.shape, target_vs.shape)
        self.model.fit(x=input_boards, y=[target_pis, target_vs],
                      batch_size=self.batch_size, epochs=self.epochs)
    

    def clone_network(self):
        print("Cloning model")
        print(self.input_shape, self.policy_shape)
        nnet = NeuralNetwork(self.config)
        nnet.model = tf.keras.models.clone_model(self.model)
        return nnet
    
    
    def save_weights(self):
        self.model.save_weights(self.weight_path)


    def load_weights(self):
        self.model.load_weights(self.weight_path)

    def save_model(self):
        self.model.save(f'{self.weight_path}.keras')
    

    def __call__(self, input):
        input = input.reshape(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        return self.model(input)