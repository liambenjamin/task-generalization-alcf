import numpy as np
import tensorflow as tf

"""
Wrapper for loading experiments
"""
def load_experiment(name, permute=False, pad=0, orientation=None, top_words=5000, max_words=500):

    assert name in ['mnist', 'fashion_mnist', 'cifar10', 'imdb', 'reuters', 'adding'], 'Dataset requested is not supported.'

    if name == 'mnist':
        (x_train, y_train), (x_test, y_test) = load_mnist(permute=permute, pad=pad, orientation=orientation)

    elif name == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = load_fashion_mnist(permute=permute, pad=pad, orientation=orientation)

    elif name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = load_cifar10(permute=permute, pad=pad, orientation=orientation)

    elif name == 'adding':
        (x_train, y_train), (x_test, y_test) = generate_adding_problem(T=750)

    elif name == 'imdb':
        (x_train, y_train), (x_test, y_test) = load_imdb(permute=permute, pad=pad, orientation=orientation, top_words=top_words, max_words=max_words)

    elif name == 'reuters':
        (x_train, y_train), (x_test, y_test) = load_reuters(permute=permute, pad=pad, orientation=orientation, top_words=top_words, max_words=max_words)

    return (x_train, y_train), (x_test, y_test)


"""
loads imdb tasks
"""
def load_imdb(permute=False, pad=0, orientation=None, top_words=5000, max_words=500):
    from tensorflow.keras.datasets import imdb

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)
    X, Y = np.concatenate((x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0)
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_words)
    Y = np.reshape(Y, (Y.shape[0], 1))

    # permute and pad
    X = permute_data(X) if permute else X
    X = X if orientation not in ['post', 'uniform'] else nlp_pad_noise(X, pad=pad, orientation=orientation)

    # split 80/20 train/test split
    index = 40000
    (x_train, y_train), (x_test, y_test) = (X[:index], Y[:index]), (X[index:], Y[index:])

    return (x_train, y_train), (x_test, y_test)


"""
loads reuters tasks
"""
def load_reuters(permute=False, pad=1000, orientation=None, top_words=5000, max_words=500):
    from tensorflow.keras.datasets import reuters
    #train: (8982,) ; test: (2246,)

    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=top_words)
    X, Y = np.concatenate((x_train, x_test), axis=0), np.concatenate((y_train, y_test), axis=0)
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_words)
    Y = np.reshape(Y, (Y.shape[0], 1))

    # permute & permute
    X = permute_data(X) if permute else X
    X = X if orientation not in ['post', 'uniform'] else nlp_pad_noise(X, pad=pad, orientation=orientation)

    # split according to standard train/test split
    index = 8982
    (x_train, y_train), (x_test, y_test) = (X[:index], Y[:index]), (X[index:], Y[index:])

    return (x_train, y_train), (x_test, y_test)


"""
load fashion mnist tasks
"""
def load_fashion_mnist(permute=False, pad=0, orientation=None):
    from tensorflow.keras.datasets import fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    n_train = x_train.shape[0]
    X = tf.concat([x_train, x_test], axis=0)
    X = tf.reshape(X, (X.shape[0],784,1))
    X = tf.cast(X, tf.float32) / 255.

    X = permute_data(X) if permute else X
    X = X if orientation not in ['post', 'uniform'] else add_noise(X, pad=pad, orientation=orientation)

    return (X[:n_train], y_train), (X[n_train:], y_test)


"""
load mnist task
"""
def load_mnist(permute=False, pad=0, orientation=None):
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    n_train = x_train.shape[0]
    X = tf.concat([x_train, x_test], axis=0)
    X = tf.reshape(X, (X.shape[0],784,1))
    X = tf.cast(X, tf.float32) / 255.

    X = permute_data(X) if permute else X
    X = X if orientation not in ['post', 'uniform'] else add_noise(X, pad=pad, orientation=orientation)

    return (X[:n_train], y_train), (X[n_train:], y_test)


"""
load cifar10 tasks
"""
def load_cifar10(permute=False, pad=0, orientation=None):
    from tensorflow.keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    n_train = x_train.shape[0]
    X = tf.concat([x_train, x_test], axis=0)
    X = tf.reshape(X, (X.shape[0],1024,3))
    X = tf.cast(X, tf.float32) / 255.

    X = permute_data(X) if permute else X
    X = X if orientation not in ['post', 'uniform'] else add_noise(X, pad=pad, orientation=orientation)

    return (X[:n_train], y_train), (X[n_train:], y_test)

"""
Generate adding problem dataset/labels
"""
def generate_adding_problem(T=750):
    N = 70000
    features = np.zeros( (N, T, 2) )
    labels = np.zeros( (N) )
    for i in range(0,N):
        rng = np.random.default_rng(i)
        features[i,:,0] = rng.uniform(0,1,T)
        # markers
        marker1 = np.int(np.random.uniform(0,np.floor(T/2)))
        marker2 = np.int(np.random.uniform(np.floor(T/2),T))
        features[i,marker1,1] = 1.0
        features[i,marker2,1] = 1.0
        # labels
        labels[i] = features[i,marker1,0] + features[i,marker2,0]
    # split train and test
    return (features[:60000], labels[:60000]), (features[60000:], labels[60000:])


def uniform_merge_two_lists(data, noise):
    """
    Given single data sample and noise, returns uniformly merged list
    """
    iterables = data, noise
    return tf.stack(list(interleave_evenly(iterables)))


def uniformly_pad_dataset(data, noise):
    """
    data: (N samples, time steps, ft_dim)
    noise: (N samples, 1000, ft_dim)

    output: (N samples, time steps + 1000, ft_dim)
    """
    data_pad = [ uniform_merge_two_lists(data[i], noise[i]) for i in range(0,len(data)) ]
    return tf.stack(data_pad)


"""
Appends noise (i.e. padding) specified by pad and orientation
"""
def add_noise(X, pad, orientation=None):
    """
    X: dataset to pad [N samples, time steps, ft dim]
    pad: [# padding steps, orientation] where orientation={'post', 'uniform'}
    """
    (N, T, ft_dim) = X.shape
    Xu = tf.random.uniform(shape=(N,pad,ft_dim), minval=0, maxval=1, seed=0)
    X_out = uniformly_pad_dataset(X, Xu) if orientation == 'uniform' else tf.concat([X, Xu], axis=1)

    return X_out


"""
Uniformly pads input with pad_step sequences of noise ~ U(0,1)
"""
def uniformly_pad(data, T, pad_step):
    # below is to create a new index list reordered to pass to tf.gather
    total_steps = T + pad_step # T + noise
    reordered_idx = list(range(total_steps))
    for i in range(len(reordered_idx)):
        if i in range(T):
            idx = reordered_idx[i]
            switch = round(idx/T*total_steps) # the position the input data should be inserted to
            if idx in range(T) and switch in range(T): # handle a special case
                reordered_idx[idx] = round(switch/T*total_steps)
                reordered_idx[switch] = idx
                reordered_idx[round(switch/T*total_steps)] = switch
            else: # switch the positions of index values
                reordered_idx[idx], reordered_idx[switch] = switch, idx
        else:
            break
    return tf.gather(data, reordered_idx, axis=1)


"""
uniformly merges data and noise
"""
def uniformly_pad2(data, noise):
    data_pad = [ x[1] for x in heapq.merge(zip(itertools.count(0,len(noise)),data), zip(itertools.count(0,len(data)),noise))]
    return data_pad

"""
Permutes dataset along axis=1 (temporal index of input)
"""
def permute_data(data):
    p = np.random.RandomState(seed=92916).permutation(data.shape[1])
    return tf.gather(data, p, axis=1)


"""
Padding for IMDB and Reuters tasks (require embedding matrices)
"""
def nlp_pad_noise(data, pad=0, orientation=None):
    N, ft_dim = data.shape
    # fixed: vocabulary size, max input length and padding
    vocab_size = 5000
    max_words = 500

    noise = np.random.RandomState(seed=123).randint(low=1, high=vocab_size+1, size=(N,pad))
    pad_data = None

    if orientation == 'post':
        pad_data = tf.concat((data, noise), axis=1)

    elif orientation == 'uniform':

        noise_split = np.split(noise, max_words, axis=1)
        ft_split = np.split(data, max_words, axis=1)

        for i in range(0, len(noise_split)):
            split_i = np.concatenate((ft_split[i], noise_split[i]), axis=1)
            pad_data = split_i if pad_data is None else np.concatenate((pad_data, split_i), axis=1)

    return pad_data


"""
Used for uniform padding -- taken from source code of 'more_itertools'
"""
def interleave_evenly(iterables, lengths=None):
    """
    Interleave multiple iterables so that their elements are evenly distributed
    throughout the output sequence.

    >>> iterables = [1, 2, 3, 4, 5], ['a', 'b']
    >>> list(interleave_evenly(iterables))
    [1, 2, 'a', 3, 4, 'b', 5]

    >>> iterables = [[1, 2, 3], [4, 5], [6, 7, 8]]
    >>> list(interleave_evenly(iterables))
    [1, 6, 4, 2, 7, 3, 8, 5]

    This function requires iterables of known length. Iterables without
    ``__len__()`` can be used by manually specifying lengths with *lengths*:

    >>> from itertools import combinations, repeat
    >>> iterables = [combinations(range(4), 2), ['a', 'b', 'c']]
    >>> lengths = [4 * (4 - 1) // 2, 3]
    >>> list(interleave_evenly(iterables, lengths=lengths))
    [(0, 1), (0, 2), 'a', (0, 3), (1, 2), 'b', (1, 3), (2, 3), 'c']

    Based on Bresenham's algorithm.
    """
    if lengths is None:
        try:
            lengths = [len(it) for it in iterables]
        except TypeError:
            raise ValueError(
                'Iterable lengths could not be determined automatically. '
                'Specify them with the lengths keyword.'
            )
    elif len(iterables) != len(lengths):
        raise ValueError('Mismatching number of iterables and lengths.')

    dims = len(lengths)

    # sort iterables by length, descending
    lengths_permute = sorted(
        range(dims), key=lambda i: lengths[i], reverse=True
    )
    lengths_desc = [lengths[i] for i in lengths_permute]
    iters_desc = [iter(iterables[i]) for i in lengths_permute]

    # the longest iterable is the primary one (Bresenham: the longest
    # distance along an axis)
    delta_primary, deltas_secondary = lengths_desc[0], lengths_desc[1:]
    iter_primary, iters_secondary = iters_desc[0], iters_desc[1:]
    errors = [delta_primary // dims] * len(deltas_secondary)

    to_yield = sum(lengths)
    while to_yield:
        yield next(iter_primary)
        to_yield -= 1
        # update errors for each secondary iterable
        errors = [e - delta for e, delta in zip(errors, deltas_secondary)]

        # those iterables for which the error is negative are yielded
        # ("diagonal step" in Bresenham)
        for i, e in enumerate(errors):
            if e < 0:
                yield next(iters_secondary[i])
                to_yield -= 1
                errors[i] += delta_primary
