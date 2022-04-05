import os
import joblib
import tensorflow as tf
from typing import Dict, Any
import hashlib
import json

def load_preprocessed_data(base_path, data_folder, data_name, labels_name, index):

    data_folder = os.path.join(base_path, data_folder)
    data = joblib.load(os.path.join(data_folder, "{}_{}".format(index, data_name))) # data
    labels = joblib.load(os.path.join(data_folder, "{}_{}".format(index, labels_name))) # labels

    return data, labels


def online_aug_generate_multi_scale_label(generator, labels, batch_size, seed, n_outputs=3):
    label_generator = generator.flow(labels, batch_size=batch_size, seed=seed)
    while True:
        multi_scale_labels = [label_generator.next()]
        for i in range(n_outputs - 1):
            next_label = tf.nn.max_pool(multi_scale_labels[-1], ksize=2, strides=2, padding="SAME")
            multi_scale_labels.append(next_label)
        yield multi_scale_labels


# From: https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html
def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()