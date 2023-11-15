import numpy as np
from keras.utils import Sequence
import tensorflow as tf


class DataGenerator(Sequence):
    def __init__(self, batch_size, train_examples):
        """Initializes the DataGenerator.

        :param batch_size: Size of each batch
        :type batch_size: int
        :param train_examples: Training examples
        :type train_examples: list
        """
        self.batch_size = batch_size
        self.add_data(train_examples)
        

    def add_data(self, data):
        """Adds data to the dataset and performs data augmentation.

        :param data: Data to be added
        :type data: list
        """
        # Unpack data
        input_boards, target_pis, target_vs = zip(*data)
        # Generate tf.dataset from data
        self.dataset = tf.data.Dataset.from_tensor_slices(
            (np.array(input_boards), (np.array(target_pis), np.array(target_vs)))
        )

        # Left right-flipping
        new_flip_data_lr = tf.data.Dataset.from_tensor_slices(
            (np.array(input_boards), (np.array(target_pis), np.array(target_vs)))
        )
        new_flip_data_lr = new_flip_data_lr.map(self.board_flipping_lr)
        
        # Up down flipping
        new_flip_data_ud = tf.data.Dataset.from_tensor_slices(
            (np.array(input_boards), (np.array(target_pis), np.array(target_vs)))
        )
        new_flip_data_ud = new_flip_data_ud.map(self.board_flipping_ud)
        
        # Add flipped data to dataset
        self.dataset = self.dataset.concatenate(new_flip_data_lr)
        self.dataset = self.dataset.concatenate(new_flip_data_ud)
        
        # One-hot encoding for boards
        self.dataset = self.dataset.map(self.one_hot)
        
        # Dataset shuffling
        self.dataset = self.dataset.shuffle(buffer_size=len(self.dataset))
    

    def board_flipping_lr(self, x, y):
        """Flip the board left-right."""
        x = tf.reverse(x, axis=[1])
        return x, y


    def board_flipping_ud(self, x, y):
        """Flip the board up-down."""
        x = tf.reverse(x, axis=[0])
        return x, y


    def one_hot(self, x, y):
        """One-hot encode the boards."""
        x = tf.cast(x, tf.int32)
        x = tf.one_hot(x, depth=3)
        return x, y


    def __len__(self):
        """Return the number of batches in the dataset."""
        return int(np.ceil(len(self.dataset) / self.batch_size))


    def __getitem__(self, idx):
        """Returns a batch of data."""
        batch = self.dataset.batch(self.batch_size).take(1)
        # Extract features and labels from the batch
        boards, (pis, vs) = next(iter(batch))
        return boards, [pis, vs]