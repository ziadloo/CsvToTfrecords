import tensorflow as tf
import csv
import os


def _create_csv_iterator(csv_file_path, skip_header):
    """Returns an iterator to read the CSV file line by line"""

    with tf.io.gfile.GFile(csv_file_path) as csv_file:
        reader = csv.reader(csv_file)
        if skip_header: # Skip the header
            next(reader)
        for row in reader:
            yield row


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""

    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    if value == "" or value is None:
        return tf.train.Feature()
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def _float_feature(value):
    """Returns a float_list from a float / double."""

    try:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))
    except ValueError as ve:
        return tf.train.Feature()


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""

    try:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))
    except ValueError as ve:
        return tf.train.Feature()


def _create_example(row, HEADER, FLOAT_FEATURES, INT_FEATURES, CATEGORICAL_FEATURES):
    """Returns a tensorflow.Example Protocol Buffer object."""

    features = {}

    for feature_index, feature_name in enumerate(HEADER):

        feature_value = row[feature_index]

        if feature_name in FLOAT_FEATURES:
            features[feature_name] = _float_feature(feature_value)

        elif feature_name in CATEGORICAL_FEATURES:
            features[feature_name] = _bytes_feature(feature_value)

        elif feature_name in INT_FEATURES:
            features[feature_name] = _int64_feature(feature_value)

    return tf.train.Example(features=tf.train.Features(feature=features))


def c2t(input_csv_file, output_tfrecord_file, config):
    """
    Creates a TFRecords file for the given input data and
    example transofmration function
    """

    filename = os.path.splitext(os.path.basename(output_tfrecord_file))[0]
    folder = os.path.dirname(output_tfrecord_file)
    output_tfrecord_file_template = os.path.join(folder, filename + "_{}.tfrecords")

    if not os.path.isdir(folder):
        os.makedirs(folder)

    filename_index = 1
    output_tfrecord_file = output_tfrecord_file_template.format(filename_index)
    writer = tf.io.TFRecordWriter(output_tfrecord_file)

    print("Creating TFRecords file at", output_tfrecord_file, "...")

    filesize = 100000000
    if "filesize" in config:
        filesize = config["filesize"]

    for i, row in enumerate(_create_csv_iterator(input_csv_file, skip_header=True)):

        if len(row) == 0:
            continue

        example = _create_example(row, config["header"], config["floats"], config["integers"], config["categoricals"])
        content = example.SerializeToString()
        writer.write(content)

        if os.stat(output_tfrecord_file).st_size > filesize:
            filename_index += 1
            writer.close()
            output_tfrecord_file = output_tfrecord_file_template.format(filename_index)
            writer = tf.io.TFRecordWriter(output_tfrecord_file)
            print("Creating TFRecords file at", output_tfrecord_file, "...")

    writer.close()
