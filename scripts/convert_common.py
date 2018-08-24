import tensorflow as tf



def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _convert_to_example(image_raw, lab):

    image_format = b'jpg'
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_raw),
        'image/format': bytes_feature(image_format),
        'label': int64_feature(lab)
    }))
    return example