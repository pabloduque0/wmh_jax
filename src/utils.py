import os
import joblib
import tensorflow as tf

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
