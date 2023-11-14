import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from keras.utils import Sequence
import numpy as np
import os
from time import time

# Check available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        # Set memory growth for each GPU
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPUs available.")

class DataGenerator(Sequence):
    def __init__(self, batch_size, train_examples):
        #self.train_examples = train_examples
        self.batch_size = batch_size
        self.add_data(train_examples)
        

    def add_data(self, data):
        input_boards, target_pis, target_vs = zip(*data)
        # generate tf.dataset from data
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (np.array(input_boards), (np.array(target_pis), np.array(target_vs)))
        )
        # left right flipping
        # create same datset to expand dataset by adding flipped samples
        new_flip_data_lr = tf.data.Dataset.from_tensor_slices(
            (np.array(input_boards), (np.array(target_pis), np.array(target_vs)))
        )
        # apply flipping to the boards
        new_flip_data_lr = new_flip_data_lr.map(self.board_flipping_lr)
        # up down flipping
        # create same datset to expand dataset by adding flipped samples
        new_flip_data_ud = tf.data.Dataset.from_tensor_slices(
            (np.array(input_boards), (np.array(target_pis), np.array(target_vs)))
        )
        # apply flipping to the boards
        new_flip_data_ud = new_flip_data_ud.map(self.board_flipping_ud)
        
        
        # add flipped data to dataset
        self.dataset = self.dataset.concatenate(new_flip_data_lr)
        self.dataset = self.dataset.concatenate(new_flip_data_ud)
        # dataset shuffling
        self.dataset = self.dataset.shuffle(buffer_size=len(self.dataset))
        # Repeat the dataset for multiple epochs
        #self.dataset = self.dataset.repeat()
    

    def board_flipping_lr(self, x, y):
        # flip left right
        x = tf.reverse(x, axis=[1])
        return x, y
    

    def board_flipping_ud(self, x, y):
        # flip up down
        x = tf.reverse(x, axis=[0])
        return x, y


    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))


    def __getitem__(self, idx):
        batch = self.dataset.batch(self.batch_size).take(1)
        # Extract features and labels from the batch
        boards, (pis, vs) = next(iter(batch))

        return boards, [pis, vs]


"""class DataGenerator(Sequence):
    def __init__(self, train_examples, batch_size):
        self.train_examples = train_examples
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.train_examples) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.train_examples[idx * self.batch_size:(idx + 1) * self.batch_size]
        input_boards, target_pis, target_vs = list(zip(*batch))
        input_boards = np.array(input_boards)
        target_pis = np.array(target_pis)
        target_vs = np.array(target_vs)
        return input_boards, [target_pis, target_vs]"""


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
        # learning rate scheduler
        self.lr_schedule = ExponentialDecay(
            initial_learning_rate=0.1,
            decay_steps=1,
            decay_rate=0.8
        )
        # custom callback to print the learning rate
        self.print_lr_callback = LambdaCallback(on_epoch_begin=\
            lambda epoch, logs: print(f"\nLearning Rate: {self.lr_schedule(epoch)}")
        )

        # load existing weights into model if path is given
        model_path = f"{config['neural network']['model path']}{name}.keras"
        print(model_path)
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
        # Create a TensorFlow session
        sess = tf.compat.v1.Session()
        # Get the list of available devices
        devices = sess.list_devices()
        # Check if a GPU is available
        gpu_available = any(device.device_type == 'GPU' for device in devices)
        if gpu_available:
            print("GPU is available and being used.")
        else:
            print("GPU is not available. TensorFlow is running on CPU.")



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
        # normalization layer
        x = tf.keras.layers.BatchNormalization()(x)
        # Output layers for policy and value estimates
        policy_output = Dense(policy_shape, activation='softmax', name='policy_output')(x)
        value_output = Dense(1, activation='tanh', name='value_output')(x)

        model = Model(inputs=input_layer, outputs=[policy_output, value_output])

        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=self.lr_schedule),
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
        # creating dataset generator
        self.dataset_generator = DataGenerator(self.batch_size, train_examples)
        print("Number of train examples:", len(train_examples))
        #self.dataset_generator.add_data(train_examples)
        #train_generator = DataGenerator(train_examples, batch_size=self.batch_size)
        self.model.fit(self.dataset_generator,
                       epochs=self.epochs,
                       callbacks=[self.print_lr_callback])
    

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
        with tf.device('/GPU:0'):
            input = input.reshape(1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
            output = self.model(input)
        return output