# Some Structure Function
import tensorflow as tf
# def conv3(layer_in, w, b, do_bn, phase, name, momentum=0.99) :
#     conv = tf.nn.conv2d(layer_in, w, strides=[1,1,1,1], padding='SAME')
#     conv = tf.nn.bias_add(conv, b)
#     if do_bn :
#         conv = tf.layers.batch_normalization(conv, axis=3, center=True, scale=True, training=phase, momentum=momentum)
#     relu = tf.nn.relu(conv, name=name)
#     tf.summary.histogram("weights", w)
#     tf.summary.histogram("bias", b)
#     tf.summary.histogram("layer", conv)
#     tf.summary.histogram("activations", relu)
#     return relu

def conv_univ(layer_in, W, B, S, N, phase, name_scope, momentum=0.99, do_bn=False, do_histogram=False) :
    with tf.name_scope(name_scope) :
        conv = layer_in
        for i in range(len(W)) :
            conv = tf.nn.conv2d(conv, W[i], strides=S[i], padding='SAME')
            conv = tf.nn.bias_add(conv, B[i])
            if do_bn :
                conv = tf.layers.batch_normalization(conv, axis=3, center=True, scale=True, training=phase, momentum=momentum)
            conv = tf.nn.relu(conv, name=N[i])
            if do_histogram :
                tf.summary.histogram("weights", W[i])
                tf.summary.histogram("bias", B[i])
                tf.summary.histogram("activations", conv)

        # pooling
        pool = tf.nn.max_pool(conv, ksize=S[-1], strides=S[-1], padding='SAME', name=N[-1])
    return pool

def fc_univ(layer_in, w, b, n, phase, name_scope, keep_prob=None, momentum=0.99, do_bn=False, do_do=False, do_histogram=False) :
    with tf.name_scope(name_scope) :
        fc = tf.nn.bias_add(tf.matmul(layer_in, w), b)
        if do_bn :
            fc = tf.layers.batch_normalization(fc, axis=1, center=True, scale=True, training=phase, momentum=momentum)
        fc = tf.nn.relu(fc, name=n)
        if do_histogram :
            tf.summary.histogram("weights", w)
            tf.summary.histogram("bias", b)
            tf.summary.histogram("activations", fc)
        if do_do :
            fc = tf.nn.dropout(fc, keep_prob)
    return fc

# def conv33pool2(layer_in, w1, b1, w2, b2, phase, keep_prob, l1_name, l2_name, p_name, name_scope, do_bn=False) :
#     with tf.name_scope(name_scope) :
#         layer1 = conv3(layer_in, w1, b1, do_bn, phase, l1_name)
#         layer2 = conv3(layer1, w2, b2, do_bn, phase, l2_name)
#         pool = tf.nn.max_pool(layer2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=p_name)
# #         pool = tf.nn.dropout(pool, keep_prob)
#     return pool

# def conv33pool4(layer_in, w1, b1, w2, b2, phase, keep_prob, l1_name, l2_name, p_name, name_scope, do_bn=False) :
#     with tf.name_scope(name_scope) :
#         layer1 = conv3(layer_in, w1, b1, do_bn, phase, l1_name)
#         layer2 = conv3(layer1, w2, b2, do_bn, phase, l2_name)
#         pool = tf.nn.max_pool(layer2, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME', name=p_name)
# #         pool = tf.nn.dropout(pool, keep_prob)
#     return pool

# def conv333pool2(layer_in, w1, b1, w2, b2, phase, name_scope) :
#     with tf.name_scope(name_scope) :
#         layer1 = conv3_bn(layer_in, w1, b1, phase)
#         layer2 = conv3_bn(layer1, w2, b2, phase)
#         layer3 = conv3_bn(layer2, w3, b3, phase)
#         pool = tf.nn.max_pool(layer3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#     return pool

# def conv333pool5(layer_in, w1, b1, w2, b2, w3, b3, phase, name_scope) :
#     with tf.name_scope(name_scope) :
#         layer1 = conv3_bn(layer_in, w1, b1, phase)
#         layer2 = conv3_bn(layer1, w2, b2, phase)
#         layer3 = conv3_bn(layer2, w3, b3, phase)
#         pool = tf.nn.max_pool(layer3, ksize=[1,5,5,1], strides=[1,5,5,1], padding='SAME')
#     return pool

# def conv3pool_half(layer_in, w, b, phase, name_scope) :
#     with tf.name_scope(name_scope) :
#         layer1 = conv3_bn(layer_in, w, b, phase)
#         pool = tf.nn.max_pool(layer1, ksize=[1,2,1,1], strides=[1,2,1,1], padding='SAME')
#     return pool