import os
import gzip
import shutil
import cv2
import numpy as np
import SimpleITK
import joblib
from shutil import copyfile
import tensorflow as tf

def get_all_image_paths():
    paths = []

    for root, dirs, files in os.walk('../'):
        for file in files:
            filepath = root + '/' + file

            if file.endswith('.gz') and file[:-3] not in files:
                with gzip.open(filepath, 'rb') as f_in:
                    with open(filepath[:-3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

            if file.startswith('brain') and file.endswith('.nii'):
                paths.append(filepath)

    return paths


def get_all_images_and_labels(path_utrech, path_singapore, path_amsterdam):
    utrech_dataset = get_images_and_labels(path_utrech)
    singapore_dataset = get_images_and_labels(path_singapore)
    amsterdam_dataset = get_images_and_labels(path_amsterdam)

    return utrech_dataset, singapore_dataset, amsterdam_dataset


def get_images_and_labels(path):
    full_dataset = []
    data_and_labels = {}

    for root, dirs, files in os.walk(path):
        for file in files:
            filepath = os.path.join(root, file)
            if file == 'wmh.nii.gz':
                if data_and_labels:
                    full_dataset.append(data_and_labels.copy())
                    data_and_labels.clear()

            if file.endswith(".nii.gz"):
                data_and_labels[file.split(".")[0]] = filepath

    full_dataset.append(data_and_labels.copy())

    return full_dataset

def get_all_sets_paths(dataset_paths, t1_key="T1_rsfl", flair_key="FLAIR", label_key="wmh", mask_key="T1_bet_mask_rsfl"):

    t1 = [row[t1_key] for row in dataset_paths]
    flair = [row[flair_key] for row in dataset_paths]
    labels = [row[label_key] for row in dataset_paths]
    common_mask = [row[mask_key] for row in dataset_paths]

    return t1, flair, labels, common_mask


def full_custom_split(datasets, indices_split):


    train_data = []
    validation_data = []
    test_data = []

    for dataset in datasets:
        single_dataset_train, single_dataset_validation, single_dataset_test = custom_split(dataset,
                                                                                            indices_split)
        train_data.append(np.concatenate(single_dataset_train))
        validation_data.append(np.concatenate(single_dataset_validation))
        test_data.append(single_dataset_test)

    train_data, validation_data, test_data = (np.expand_dims(np.concatenate(train_data), -1),
                                              np.expand_dims(np.concatenate(validation_data), -1),
                                              np.expand_dims(np.concatenate(test_data), -1))

    return train_data, validation_data, test_data


def concat_save_data(data_t1, data_flair):
    concat_data = []
    for t1, flair in zip(data_t1, data_flair):
        concat = np.concatenate([np.expand_dims(t1, axis=-1),
                                 np.expand_dims(flair, axis=-1)], axis=3)
        concat_data.append(concat)

    return concat_data


def full_custom_split_for_save(datasets, indices_split):

    validation_data = []
    test_data = []

    for dataset in datasets:
        single_dataset_train, single_dataset_validation, single_dataset_test = custom_split(dataset,
                                                                                            indices_split)
        validation_data.append(np.concatenate(single_dataset_validation))
        test_data.append(single_dataset_test)



    return validation_data, test_data


def custom_split(data, indices):

    data = np.array(data)

    validation_data = data[indices[1:]]
    test_data = data[indices[0]]
    train_indices = [idx for idx in range(len(data)) if idx not in indices]

    data = data[train_indices]

    return data, validation_data, test_data


def permute_data(data_splits):

    shuffled_splits = []
    for split_data, split_labels in data_splits:
        shuffled_data, shuffled_labels = single_permute(split_data, split_labels)
        shuffled_splits.append((shuffled_data, shuffled_labels))

    return shuffled_splits


def single_permute(data, labels):

    idx_swap = np.arange(data.shape[0])
    np.random.shuffle(idx_swap)
    data = data[idx_swap]
    labels = labels[idx_swap]

    return data, labels


def save_data_labels(dir_path, index, data, data_name, labels, labels_name):

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    joblib.dump(data, os.path.join(dir_path, "{}_{}".format(str(index), data_name)))
    joblib.dump(labels, os.path.join(dir_path, "{}_{}".format(str(index), labels_name)))



def preprocess_all_datasets(datasets, slice_shape, masks, remove_pct_top, remove_pct_bot):

    preprocessed_data = []
    for data, dataset_masks in zip(datasets, masks):
        out_data = preprocess_dataset(data, slice_shape, dataset_masks, remove_pct_top, remove_pct_bot)
        preprocessed_data.append(list(out_data))

    return preprocessed_data


def preprocess_dataset(data, slice_shape, masks, remove_pct_top, remove_pct_bot):

    data = get_all_images_np_twod(data)
    data = np.asanyarray(data) * np.asanyarray(masks)
    resized = resize_slices(data, slice_shape)
    resized = remove_top_bot_slices(resized, masks, remove_pct_top, remove_pct_bot)
    stand = standarize(resized)
    normalized = normalize_minmax(stand)
    return normalized


def preprocess_all_labels(labels_paths, masks_all, slice_shape, remove_pct_top, remove_pct_bot):

    preprocessed_labels = []
    for labels, masks in zip(labels_paths, masks_all):
        out_labels = preprocess_dataset_labels(labels, masks, slice_shape, remove_pct_top, remove_pct_bot)
        preprocessed_labels.append(list(out_labels))

    return preprocessed_labels

def preprocess_dataset_labels(label_paths, masks, slice_shape, remove_pct_top, remove_pct_bot):

    labels_imgs = get_all_images_np_twod(label_paths)
    labels_resized = resize_slices(labels_imgs, slice_shape)
    labels_resized = remove_third_label(labels_resized)
    labels_resized = remove_top_bot_slices(labels_resized, masks, remove_pct_top, remove_pct_bot)

    return labels_resized


def remove_top_bot_slices(dataset, masks, remove_pct_top, remove_pct_bot):

    output_images = []
    for image, mask in zip(dataset, masks):
        remove_bot = int(np.ceil(image.shape[0] * remove_pct_bot))
        remove_top = int(np.ceil(image.shape[0] * remove_pct_top))
        first_slice_brain, last_slice_brain = get_first_last_slice(mask)
        output_img = image[first_slice_brain + remove_bot: last_slice_brain - remove_top + 1]
        output_images.append(output_img)

    return output_images


def get_all_images_np_twod(paths_list, wrong_shape=(232, 256)):

    patient_list = []
    for path in paths_list:
        image = SimpleITK.ReadImage(path)
        np_image = SimpleITK.GetArrayFromImage(image)

        if np_image.shape[1:] == wrong_shape: #ad hoc fix for Singapore
            # only for even differences
            diff = (wrong_shape[1] - wrong_shape[0]) // 2
            np_image = np.pad(np_image, [(0, 0), (diff, diff), (0, 0)], 'constant', constant_values=0)
            np_image = np_image[..., diff:-diff]

        patient_list.append(np_image)

    return patient_list

def resize_slices(image_list, to_slice_shape):

    resized_list = []

    for image in image_list:
        resized_image = []
        for slice in image:
            slice_copy = slice.copy()

            if slice.shape[0] < to_slice_shape[0]:
                diff = to_slice_shape[0] - slice.shape[0]
                if is_odd(diff):
                    slice_copy = cv2.copyMakeBorder(slice_copy, diff//2, diff//2 + 1, 0, 0,
                                                    cv2.BORDER_CONSTANT,
                                                    value=0.0)
                else:
                    slice_copy = cv2.copyMakeBorder(slice_copy, diff // 2, diff // 2, 0, 0,
                                                    cv2.BORDER_CONSTANT,
                                                    value=0.0)

            elif slice.shape[0] > to_slice_shape[0]:
                diff = slice.shape[0] - to_slice_shape[0]
                if is_odd(diff):
                    slice_copy = slice_copy[diff // 2: -diff//2, :]
                else:
                    slice_copy = slice_copy[diff // 2: -diff // 2, :]

            if slice.shape[1] < to_slice_shape[1]:
                diff = to_slice_shape[1] - slice.shape[1]
                if is_odd(diff):
                    slice_copy = cv2.copyMakeBorder(slice_copy, 0, 0, diff // 2, diff // 2 + 1,
                                                    cv2.BORDER_CONSTANT,
                                                    value=0.0)
                else:
                    slice_copy = cv2.copyMakeBorder(slice_copy, 0, 0, diff // 2, diff // 2,
                                                    cv2.BORDER_CONSTANT,
                                                    value=0.0)
            elif slice.shape[1] > to_slice_shape[1]:
                diff = slice.shape[1] - to_slice_shape[1]
                if is_odd(diff):
                    slice_copy = slice_copy[:, diff // 2: -diff // 2]
                else:
                    slice_copy = slice_copy[:, diff // 2: -diff // 2]

            resized_image.append(slice_copy)

        resized_list.append(np.asanyarray(resized_image))

    return resized_list

def generate_multi_scale_label(labels, n_outputs=5):
    multi_scale_labels = [labels]

    for i in range(n_outputs - 1):
        next_label = tf.nn.max_pool(multi_scale_labels[-1], ksize=2, strides=2, padding="SAME")
        multi_scale_labels.append(next_label)

    print([tf.shape(label) for label in multi_scale_labels])
    return multi_scale_labels

def is_odd(number):

    return number % 2 != 0

def threed_resize(image, slice_shape):

    all_slices = []
    for index in range(image.shape[2]):
        slice = image[:, :, index]
        resized = cv2.resize(slice, (slice_shape[1], slice_shape[0]), cv2.INTER_CUBIC)
        all_slices.append(resized)

    return np.asanyarray(all_slices)

def normalize_minmax(images_list):

    normalized_list = []

    for image in images_list:
        section_max = np.max(image)
        section_min = np.min(image)
        normalized = (image - section_min) / (section_max - section_min)
        normalized_list.append(normalized)

    return normalized_list


def get_first_last_slice(mask):

    first_slice_brain = np.argmax(mask) // np.prod(mask.shape[1:])
    last_slice_brain = (np.prod(mask.shape) - np.argmax(np.flip(mask, axis=0))) // np.prod(mask.shape[1:])

    return first_slice_brain, last_slice_brain


def normalize_neg_pos_one(images_list, slice_number):

    normalized_list = []

    np_list = np.asanyarray(images_list)
    for image_idx in range(np_list.shape[0] // slice_number):
        this_section = np_list[image_idx * slice_number:(image_idx + 1) * slice_number, :, :]
        normalized = 2. * (this_section - np.min(this_section)) / np.ptp(this_section) - 1
        normalized_list.append(normalized)

    normalized_list = np.concatenate(normalized_list)
    return normalized_list



def standarize(image_list):

    standarized_imgs = []

    for image in image_list:
        stand_image = (image - np.mean(image)) / np.std(image)
        standarized_imgs.append(stand_image)

    return standarized_imgs

def remove_third_label(labels_list):

    new_labels_list = []

    for image in labels_list:
        image[image > 1.] = 0.0
        new_labels_list.append(image)

    return new_labels_list

def get_coordinates(data_list):

    coord_data_list = []

    for data_set in data_list:

        assert data_set.ndim == 4
        imgs, rows, columns, channels = data_set.shape
        grid_rows, grid_columns = np.mgrid[0:rows,0:columns]
        grid_rows = np.expand_dims(np.repeat(np.expand_dims(grid_rows, 0), imgs, axis=0), -1) / (rows - 1)
        grid_columns = np.expand_dims(np.repeat(np.expand_dims(grid_columns, 0), imgs, axis=0), -1) / (columns - 1)

        coord_data_list.append(np.concatenate([data_set, grid_rows, grid_columns], axis=-1))


    return coord_data_list


def copy_original_data(input_paths, output_path, sub_folder, global_index, raw_folder="raw_copies", filename="{}_flair_raws.pkl"):
    path = os.path.join(output_path, sub_folder, raw_folder)
    if not os.path.exists(path):
        os.mkdir(path)

    raw_niis = [SimpleITK.ReadImage(input_path) for input_path in input_paths]
    joblib.dump(raw_niis, os.path.join(path, filename.format(global_index)))

