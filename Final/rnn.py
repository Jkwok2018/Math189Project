import tensorflow as tf
import numpy as np
import os
import time


def rnn(clustering, seq_length, k):
    """
    input: clustering, list of integers, time series of clusters

    """
    examples_per_epoch = len(clustering) - 1
    dataset = preprocess(clustering, seq_length)
     # Batch size
    BATCH_SIZE = 64
    # len(clustering)/(seq_length+1)
    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    # dataset                                               
    # print dataset
    for input_example, target_example in  dataset.take(1):
        print ('Input data: ', input_example)
        print ('Target data:', target_example)

    # Build the model
    # Length of the vocabulary in chars = k
    vocab_size = k
    # The embedding dimension
    embedding_dim = 256
    # Number of RNN units
    rnn_units = 1024
    model = build_model(
                vocab_size = vocab_size,
                embedding_dim=embedding_dim,
                rnn_units=rnn_units,
                batch_size=BATCH_SIZE)

    # Try the model
    # for input_example_batch, target_example_batch in dataset.take(1):
    #     example_batch_predictions = model(input_example_batch)
    #     print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    #     model.summary()
    #     # Train the model
    EPOCHS=10
    train_model(dataset, model, EPOCHS)
    
  
      #ValueError: Tensor's shape (4, 2, 1024) is not compatible with supplied shape [4, 1, 1024]
def predict(model, clustering):
    model.reset_states()

    prediction_L = []
    
    for i in range(len(clustering)-4):
        input = clustering[i:i+4]
        input = tf.expand_dims(input, 0)
        predictions = model(input)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        # temperature = 0.1
        # using a categorical distribution to predict the character returned by the model
        # predictions = predictions / temperature
        # print(predicted_id)
        prediction_L.append(predicted_id)
    print(len(prediction_L))
    
    return prediction_L

def preprocess(clustering, seq_length):
    # define length of each sequence
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(clustering)
    # Put clustering in sequences with length of 5
    sequences = dataset.batch(seq_length+1, drop_remainder=True)

    # TODO: testing purpose, delete later
    # for item in sequences.take(5):
    #     # print(repr(''.join(idx2char[item.numpy()])))
    #     print(item.numpy())

    # Using sliding window method
    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text
    return sequences.map(split_input_target)

    # TODO: testing purpose, delete later
    # for input_example, target_example in  dataset.take(1):
    #     print ('Input data: ', input_example)
    #     print ('Target data:', target_example)

    # for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    #     print("Step {:4d}".format(i))
    #     print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    #     print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

   
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

def train_model(dataset, model, epochs):
    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    
    # Actual training process
    history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])


# sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
# sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
# # sampled_indices                                       # print sampled_indices
# print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
# print()
# print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ]))) # prediction of untrained model

# Train the model
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def accuracy(predictions_L, actual_L):
    # actual_L = clustering[4:]
    correct = 0
    for i in range(len(actual_L)):
        if (predictions_L[i]==actual_L[i]):
            correct += 1
    return correct/len(actual_L)


