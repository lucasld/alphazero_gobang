import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import os

from src.utils import tools
from src.model.data_generator import DataGenerator

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
        """Initializes the neural network manager.

        :param config: Configuration parameters for the neural network.
        :type config: dict
        :param load_existing_model: Flag to indicate whether to load an existing model.
        :type load_existing_model: bool
        """
        self.config = config
        name = config["name"]
        self.weight_path = config["neural network"]["weight path"] + name
        self.input_shape = (config["game"]["height"],
                            config["game"]["width"],
                            3)
        self.policy_shape = self.input_shape[0] * self.input_shape[1]
        self.batch_size = config["neural network"]["batch size"]
        self.epochs = config["neural network"]["epochs"]

        # Learning rate scheduler
        """self.lr_schedule = ExponentialDecay(
            initial_learning_rate=0.05,
            decay_steps=1,
            decay_rate=0.6
        )
        # Custom callback to print the learning rate
        self.print_lr_callback = LambdaCallback(on_epoch_begin=\
            lambda epoch, logs: print(f"\nLearning Rate: {self.lr_schedule(epoch)}")
        )"""

        # Load existing weights into model if path is given
        model_path = f"{config['neural network']['model path']}{name}.keras"
        if os.path.exists(model_path) and load_existing_model:
            self.model = tf.keras.models.load_model(model_path)
            print("Old model loaded")
        else:
            print("Creating new model")
            # Convert to float16 for memory saving
            tf.keras.backend.set_floatx('float16')
            self.model = self.create_value_policy_network(self.input_shape,
                                                          self.policy_shape)
    

    def create_value_policy_network(self, input_shape: tuple,
                                    policy_shape: int) -> tf.keras.Model:
        """Creates a neural network model for value and policy estimation.

        :param input_shape: Shape of the input data.
        :type input_shape: tuple
        :param policy_shape: Shape of the policy output.
        :type policy_shape: int
        :return: Compiled TensorFlow model for value and policy estimation.
        :rtype: tf.keras.Model
        """
        # Define the inut layer
        input_layer = tf.keras.Input(shape=input_shape)
        x = input_layer
        last_layer_2d = True
        
        # Retrieve the neural network architecture from the configuration
        architecture = self.config["neural network"]["architecture"]
        
        # Build the neural network layers based on the specified architecture
        for layer_type, number in architecture:
            if layer_type == "conv":
                x = Conv2D(number, (3, 3), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = MaxPooling2D((2, 2), strides=(1,1), padding='same')(x)
            elif layer_type == "dense":
                if last_layer_2d:
                    x = Flatten()(x)
                    last_layer_2d = False
                x = Dropout(0.4)(x)
                x = Dense(number, activation='relu')(x)
        
        # Output layers for policy and value estimates
        policy_output = Dense(policy_shape, activation='softmax',
                              name='policy_output')(x)
        value_output = Dense(1, activation='tanh', name='value_output')(x)

        # Create the model with input and output layers
        model = Model(inputs=input_layer, outputs=[policy_output, value_output])

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(),#learning_rate=self.lr_schedule),
            loss={
                'policy_output': 'categorical_crossentropy',
                'value_output': 'mean_squared_error'
            },
            metrics={
                'policy_output': ['accuracy']
            }
        )
        return model


    def train_nnet(self, train_examples):
        """Trains the neural network using the provided training examples.

        :param train_examples: Training examples for training the neural network
        :type train_examples: list
        """
        # Check if network should not be trained.
        config_net = self.config['neural network']
        if 'freeze' in config_net.keys() and config_net['freeze']:
            input("!!This model is marked as frozen!!")
            return False
        # Creating dataset generator
        self.dataset_generator = DataGenerator(self.batch_size, train_examples)
        print("Number of train examples:", len(train_examples))
        
        # Train the model using the dataset generator
        history = self.model.fit(self.dataset_generator,
                                 epochs=self.epochs)#,
        #                         #callbacks=[self.print_lr_callback])
        
        # Save training history to a file
        history_path = self.config["neural network"]["weight path"] + "history.json"
        tools.add_and_plot_history(history_path, history)
    

    def clone_network(self):
        """Clones the current neural network model.

        :return: A new NeuralNetwork instance with a cloned model.
        :rtype: NeuralNetwork object
        """
        print("Cloning model")
        print(self.input_shape, self.policy_shape)
        nnet = NeuralNetwork(self.config)
        nnet.model = tf.keras.models.clone_model(self.model)
        return nnet
    
    
    def save_weights(self):
        """Saves the models weights to self.weight_path"""
        self.model.save_weights(self.weight_path)


    def load_weights(self):
        """Loads models weights from self.weight_path"""
        self.model.load_weights(self.weight_path)


    def save_model(self):
        """Saves the model to self.weight_path"""
        self.model.save(f'{self.weight_path}.keras')
    

    def __call__(self, _input):
        """Calls the neural network on the input data.

        :param _input: Input data
        :type _input: numpy.ndarray
        :return: Output of the neural network
        :rtype: tensor
        """
        # Check if GPU is available
        devices = tf.config.list_physical_devices('GPU')
        shape_re = (1, self.input_shape[0], self.input_shape[1])
        if devices:
            # Use GPU for computation
            with tf.device('/GPU:0'):
                _input = _input.reshape(shape_re)
                _input = tf.one_hot(_input, depth=3)
                output = self.model(_input)
        else:
            # If no GPU available, use CPU
            _input = _input.reshape(shape_re)
            _input = tf.one_hot(_input, depth=3)
            output = self.model(_input)
        
        return output