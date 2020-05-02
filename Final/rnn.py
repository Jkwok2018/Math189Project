import tensorflow as tf
import numpy as np
import os
import time


# # Read the data
# path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# # Read, then decode for py2 compat.
# text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# # length of text is the number of characters in it
# print ('Length of text: {} characters'.format(len(text)))
# # Take a look at the first 250 characters in text
# print(text[:250])
# # The unique characters in the file
# vocab = sorted(set(text))
# print ('{} unique characters'.format(len(vocab)))


# # Process the text
# # Creating a mapping from unique characters to indices
# char2idx = {u:i for i, u in enumerate(vocab)}
# idx2char = np.array(vocab)

# text_as_int = np.array([char2idx[c] for c in text])
# print('{')
# for char,_ in zip(char2idx, range(20)):
#     print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
# print('  ...\n}')
# # Show how the first 13 characters from the text are mapped to integers
# print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))



def rnn(clustering):
    """
    input: clustering, list of integers, time series of clusters

    """
    examples_per_epoch = len(clustering) - 1
    dataset = preprocess(clustering)
     # Batch size
    BATCH_SIZE = 64
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
    vocab_size = 7
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
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
        model.summary()
        # Train the model
        train_model(dataset, model, target_example_batch, example_batch_predictions)


  
def preprocess(clustering):
    # define length of each sequence
    seq_length = 4
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

def train_model(dataset, model, target_example_batch, example_batch_predictions):
    example_batch_loss  = loss(target_example_batch, example_batch_predictions)
    print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    print("scalar_loss:      ", example_batch_loss.numpy().mean())
    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    
    # Actual training process
    EPOCHS=10
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])



# sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
# sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
# # sampled_indices                                       # print sampled_indices
# print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
# print()
# print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ]))) # prediction of untrained model

# Train the model
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)




"""
word: abba
mapping: a->0 (delete)
        b->1, etc.
input matrix: [[1,0,0,0]
               [0,1,0,0]
               [0,1,0,0]
               [1,0,0,0]
              ]
our case:
cluster sequence: [0,1,2,1]
input matrix: [[1,0,0,0]
               [0,1,0,0]
               [0,0,1,0]
               [0,1,0,0]
              ]
"""