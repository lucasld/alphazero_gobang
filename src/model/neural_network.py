import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.models import Model
import numpy as np


class NeuralNetwork:
    def __init__(self, weight_path, input_shape=(19, 19, 1), policy_shape=361,
                 load_weights_path=False):
        self.weight_path = weight_path
        self.input_shape = input_shape
        self.policy_shape = policy_shape
        self.model = self.create_value_policy_network(input_shape,
                                                      policy_shape)
        # load existing weights into model if path is given
        if load_weights_path:
            print("Old model loaded")
            #self.model.load_weights(load_weights_path)
            tf.keras.models.load_model(f"{weight_path}.keras")


    def create_value_policy_network(self, input_shape: tuple,
                                    policy_shape: int) -> tf.keras.Model:
        input_layer = tf.keras.Input(shape=input_shape)

        # Convolutional layers
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        #x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

        # Flatten the output of the convolutional layers
        x = Flatten()(x)

        # Fully connected layers
        #x = Dense(256, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        x = Dense(8, activation='relu')(x)

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


    def train_nnet(self, train_examples, batch_size=4, epochs=50):
        input_boards, target_pis, target_vs = list(zip(*train_examples))
        input_boards = np.array(input_boards)
        target_pis = np.array(target_pis)
        target_vs = np.array(target_vs)
        print(input_boards.shape, target_pis.shape, target_vs.shape)
        self.model.fit(x=input_boards, y=[target_pis, target_vs],
                      batch_size=batch_size, epochs=epochs)
    

    def clone_network(self):
        print(self.input_shape, self.policy_shape)
        nnet = NeuralNetwork(self.weight_path, self.input_shape, self.policy_shape)
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