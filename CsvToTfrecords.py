import tensorflow as tf
import csv
import os


def _create_csv_iterator(csv_file_path, skip_header: bool=True):
    """Returns an iterator to read the CSV file line by line

    Args:
        csv_file_path: str, The filesystem path to the CSV file
        skip_header: bool, Whether the CSV file has a header row
            that needs to be ignored

    Yields:
        A list of strings holding one row of data
    """

    with tf.io.gfile.GFile(csv_file_path) as csv_file:
        reader = csv.reader(csv_file)
        if skip_header: # Skip the header
            next(reader)
        for row in reader:
            yield row


def _bytes_feature(value, mapping):
    """Returns a bytes_list from a string / byte.

    Args:
        value: str, the string representation of the feature value

    Returns:
        Converts and returns the feature either as a byte list or
            as an empty feature
    """

    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    if value == "" or value is None:
        return tf.train.Feature()
    else:
        try:
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[mapping(value).encode()]))
        except Exception as e:
            return tf.train.Feature()

def _float_feature(value, mapping):
    """Returns a float_list from a float / double.

    Args:
        value: str, the string representation of the feature value

    Returns:
        Converts and returns the feature either as a float list or
            as an empty feature
    """

    try:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[float(mapping(value))]))
    except Exception as e:
        return tf.train.Feature()


def _int64_feature(value, mapping):
    """Returns an int64_list from a bool / enum / int / uint.

    Args:
        value: str, the string representation of the feature value

    Returns:
        Converts and returns the feature either as an integer list or
            as an empty feature
    """

    try:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(mapping(value))]))
    except ValueError as ve:
        return tf.train.Feature()


def _create_example(row, header, float_features, int_features, categorical_features, mappings):
    """Converts a single row of CSV into a tensorflow.Example Protocol Buffer object.

    Args:
        row: [str], A list of feature values in string format
        header: [str], A list of all the feature names in the same column order
            as they appear in the CSV file
        float_features: [str], A lsit of feature names that are of floating point
            type
        int_features: [str], A list of feature names that are of integer type
        categorical_features: [str], A list of feature names that are of string
            type and are considered categorical

    Returns:
        An instance of tf.train.Example class which represents a single sample in
            dataset
    """

    features = {}

    for feature_index, feature_name in enumerate(header):

        feature_value = row[feature_index]
        m = mappings[feature_name] if feature_name in mappings else lambda x: x

        if feature_name in float_features:
            features[feature_name] = _float_feature(feature_value, m)

        elif feature_name in int_features:
            features[feature_name] = _int64_feature(feature_value, m)

        elif feature_name in categorical_features:
            features[feature_name] = _bytes_feature(feature_value, m)

    return tf.train.Example(features=tf.train.Features(feature=features))


def c2t(input_csv_file, output_tfrecord_file, config):
    """Creates a TFRecords file for the given input data and
        example transofmration function

    Args:
        input_csv_file: str, The filesystem location of the CSV file to
            be read
        output_tfrecord_file: str, The filesystem location of the TFRecords
            file to be created
        config: A dictionary with the following entries:
            "headers": [str], A list of all features in the same order as they
                appear in the CSV columns. This list should have all the columns
                in the CSV file, even if they are not going to be used.
            "integers": [str], A list of all integer features (a subset of
                "headers") - optional
            "floats": [str], A list of all float features (a subset of
                "headers") - optional
            "categoricals": [str], A list of all categoricals (string) features
                (a subset of "headers") - optional
            "mappings": [str] -> func, A dictionary, mapping the field names into
                a function. If provided, the function will be applied to the field
                value before being cast into the designated type - optional
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

    filesize = config["filesize"] if "filesize" in config else 100000000
    floats = config["floats"] if "floats" in config else []
    integers = config["integers"] if "integers" in config else []
    categoricals = config["categoricals"] if "categoricals" in config else []
    mappings = config["mappings"] if "mappings" in config else {}

    for i, row in enumerate(_create_csv_iterator(input_csv_file, skip_header=True)):

        if len(row) == 0:
            continue

        example = _create_example(row, config["headers"], floats, integers, categoricals, mappings)
        content = example.SerializeToString()
        writer.write(content)

        if os.stat(output_tfrecord_file).st_size > filesize:
            filename_index += 1
            writer.close()
            output_tfrecord_file = output_tfrecord_file_template.format(filename_index)
            writer = tf.io.TFRecordWriter(output_tfrecord_file)
            print("Creating TFRecords file at", output_tfrecord_file, "...")

    writer.close()
