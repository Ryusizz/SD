import os
import itertools
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import glob
from scipy import ndimage

class Vars() :
    def __init__(self) :
        self.handler = {}
        
    def get(self, key) :
        return self.handler[key]
    
    def put(self, key, value) :
        self.handler[key] = value
        
        
def load_image(f) :
    return ndimage.imread(f)

def load_data_cell(V, rm_list, x_grid_std, y_grid_std) :
    # load Data
    data_dir = V.get('data_dir')
    cell_xy_file = V.get('cell_xy_file')
    cats = os.listdir(data_dir)
    for c in cats :
        if not filter(os.path.isdir, glob.glob("{}/{}".format(data_dir, c))) :
             cats.remove(c)
    for rm in rm_list :
        cats.remove(rm)
    cats = sorted(cats)
    print "Classes :"
    print cats

    cell_xy = pickle.load(open(cell_xy_file, 'rb'))
    fList = []
    cList = []
    xyList = []
    cDic = { cats[i] : i for i in range(len(cats)) }
    for c in cats :
        fCat = os.listdir("{}/{}".format(data_dir, c))
        fCat = sorted(fCat, key=str.lower)
        f_check = fCat[0].split("_")[0]
        f_arr = []
        xy_arr = []
        for f in fCat :
            tokens = f.split("_")
            if tokens[0] != f_check :
                f_check = tokens[0]
                fList.append(f_arr)
                xyList.append(xy_arr)
                cList.append(cDic[c])
                f_arr = []
                xy_arr = []
            f_name = "{}/{}/{}".format(data_dir, c, f)
            f_key = "{}/{}/{}".format("/notebooks/sd/../cell_unscale", c, f)
            f_arr.append(f_name)
            if f_key in cell_xy :
                xy = list(cell_xy[f_key])
#                 for i in range(len(x_grid_std)-1) :
#                     if x_grid_std[i] <= xy[1] < x_grid_std[i+1] :
#                         xy[1] = i
#                         break
#                 for i in range(len(y_grid_std)-1) :
#                     if y_grid_std[i] <= xy[0] < y_grid_std[i+1] :
#                         xy[0] = i
#                         break
                xy_arr.append(xy)
        fList.append(f_arr)
        cList.append(cDic[c])
        xyList.append(xy_arr)

    V.put('n_output', len(cDic))
    print "# of class : {:d}".format(len(cDic))
    print "# of images : {:d}".format(len(fList))
    print "# of cells : {:d}".format(len(list(itertools.chain.from_iterable(fList))))
    print "Loading Done"
    return [fList, cList, xyList, cDic]

def partition(V, fList, cList, xyList, cDic) :
    test_set_size = V.get('test_set_size')
    train_set_size = len(fList) - test_set_size
    V.put("train_set_size", train_set_size)
    partitions = np.random.permutation(len(fList))
    train_f_list = []
    train_l_list = []
    train_xy_list = []
    for p in partitions[:train_set_size] :
        train_f_list.extend(fList[p])
        train_l_list.extend([cList[p] for i in range(len(fList[p]))])
        train_xy_list.extend(xyList[p])
    print "# of train cells : {:d}".format(len(train_f_list))
    print "Train partition done"
    test_f_list = []
    test_labels = []
    test_xys = []
    for p in partitions[train_set_size:] :
        test_f_list.append(fList[p])
        test_labels.append(cList[p])
        test_xys.append(xyList[p])
    test_images = []
    for fs in test_f_list :
        imgs = []
        for f in fs :
            imgs.append(ndimage.imread(f))
        test_images.append(np.array(imgs))
    print "Test partition done"
    return [train_f_list, train_l_list, train_xy_list, test_images, test_labels, test_xys]

def oversampling(train_f_list, train_l_list, cDic) :
    n_max = 0
    l_max = 0
    n_list = np.zeros(len(cDic))
    for i in range(len(cDic)) :
        n = train_l_list.count(i)
        n_list[i] = n
        if n > n_max :
            n_max = n
            l_max = i

    ratio_base = 0.9
    ratio_list = (n_max / n_list * ratio_base - 1)
    new_train_f_list = train_f_list[:]
    new_train_l_list = train_l_list[:]
    for i in range(len(train_l_list)) :
        r_float, r_int = math.modf(ratio_list[train_l_list[i]])
        new_train_f_list.extend([train_f_list[i]] * int(r_int))
        new_train_l_list.extend([train_l_list[i]] * int(r_int))
    #         train_xy_list.extend([train_xy_list[i]] * int(r_int))
        nexon = np.random.random()
        if nexon < r_float :
            new_train_f_list.extend([train_f_list[i]])
            new_train_l_list.extend([train_l_list[i]])
    #             train_xy_list.extend([train_xy_list[i]])

    # permutation
    perm = np.random.permutation(len(new_train_f_list))
    new_train_f_list = [ new_train_f_list[p] for p in perm ]
    new_train_l_list = [ new_train_l_list[p] for p in perm ]
    #     new_train_xy_list = [ train_xy_list[p] for p in perm ]

    print "# of data after oversampling : {:d}".format(len(new_train_f_list))
    n_list = np.zeros(len(cDic))
    for i in range(len(cDic)) :
        n_list[i] = new_train_l_list.count(i)
    print n_list
    return new_train_f_list, new_train_l_list

def undersampling(train_f_list, train_l_list, cDic) :
    n_max = 0
    l_max = 0
    n_list = np.zeros(len(cDic))
    for i in range(len(cDic)) :
        n = train_l_list.count(i)
        n_list[i] = n
        if n > n_max :
            n_max = n
            l_max = i
    
    ratio_base = 0.6
    new_train_f_list = []
    new_train_l_list = []
    for i in range(len(train_l_list)) :
        if train_l_list[i] != l_max :
            new_train_f_list.append(train_f_list[i])
            new_train_l_list.append(train_l_list[i])
        else :
            nexon = np.random.random()
            if nexon < ratio_base :
                new_train_f_list.append(train_f_list[i])
                new_train_l_list.append(train_l_list[i])
                
    # permutation
    perm = np.random.permutation(len(new_train_f_list))
    new_train_f_list = [ new_train_f_list[p] for p in perm ]
    new_train_l_list = [ new_train_l_list[p] for p in perm ]
            
    print "# of data after undersampling : {:d}".format(len(new_train_f_list))
    n_list = np.zeros(len(cDic))
    for i in range(len(cDic)) :
        n_list[i] = new_train_l_list.count(i)
    print n_list
    return new_train_f_list, new_train_l_list

def pipeline_train(V, train_f_list, train_l_list, train_xy_list=None) :
# build pipeline
    cell_x = V.get('cell_x')
    cell_y = V.get('cell_y')
    batch_size = V.get('batch_size')
#     test_set_size = V.get('test_set_size')
#     train_set_size = V.get('train_set_size')
    n_input = cell_y*cell_x
    n_output = V.get('n_output')
    if train_xy_list != None :
        do_xy = True
    else :
        do_xy = False
    
    train_images = tf.convert_to_tensor(train_f_list, dtype=tf.string)
    train_labels = tf.convert_to_tensor(train_l_list, dtype=tf.uint8)
    if do_xy :
        train_xys = tf.convert_to_tensor(train_xy_list)

    if do_xy :
        train_input_queue = tf.train.slice_input_producer([train_images, train_labels, train_xys], shuffle=True)
    else :
        train_input_queue = tf.train.slice_input_producer([train_images, train_labels], shuffle=True)

    file_content = tf.read_file(train_input_queue[0])
    train_image = tf.image.decode_png(file_content)
    train_image.set_shape([cell_y, cell_x, 3])
    train_label = train_input_queue[1]
    if do_xy :
        train_xy = train_input_queue[2]
    
    if do_xy :
        batch_out = tf.train.batch([train_image, train_label, train_xy], batch_size = batch_size)
    else :
        batch_out = tf.train.batch([train_image, train_label], batch_size = batch_size)

    print "Batching done"
    return batch_out

def resize(V, x) :
    # preprocessing
    with tf.name_scope("Resize") :
        x_resize = tf.image.resize_images(x, [V.get('size_re'), V.get('size_re')])
    return x_resize

def zero_channel(V, x) :
    with tf.name_scope("Zero_Channel") :
        ch_mean = tf.reduce_mean(x, [1,2])
        ch_mean = tf.expand_dims(ch_mean, 1)
        ch_mean = tf.expand_dims(ch_mean, 2)
        ch_mean = tf.tile(ch_mean, [1, V.get('size_re'), V.get('size_re'), 1])
        x_std = x - ch_mean
    return x_std

def confmat_normalize(V, conf_arr, mode) :
    norm_conf = np.zeros((V.get('n_output'), V.get('n_output')))
    if mode == "row" :
        for i in range(len(conf_arr)) :
            tmp_arr = conf_arr[i,:]
            a = np.sum(tmp_arr)
            if a != 0 :
                tmp_arr = tmp_arr.astype(np.float32)/a
            norm_conf[i,:] = tmp_arr
    if mode == "col" :
        for i in range(len(conf_arr)) :
            tmp_arr = conf_arr[:,i]
            a = np.sum(tmp_arr)
            if a != 0 :
                tmp_arr = tmp_arr.astype(np.float32)/a
            norm_conf[:,i] = tmp_arr
    return norm_conf

def plotNNFilter(units):
    filters = units.shape[2]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[:,:,i], interpolation="nearest", cmap="gray")
        
def plotNNFilterRGB(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[:,:,:,i], interpolation="nearest")
        
def plotNNOutput(units):
    n_images = units.shape[0]
    plt.figure(1, figsize=(15,15))
    plt.axis('off')
    n_columns = 6
    n_rows = math.ceil(n_images / n_columns) + 1
    for i in range(n_images):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Image ' + str(i))
        plt.imshow(units[i,:,:], interpolation="nearest", cmap="gray")
        
def make_recorder(V, result_dir) :
    with tf.name_scope("Cost") :
        train_loss = tf.placeholder(tf.float32, shape=(), name="train_loss")
        test_loss = tf.placeholder(tf.float32, shape=(), name="test_loss")
        tf.summary.scalar("test_loss", test_loss)
        tf.summary.scalar("train_loss", train_loss)
    with tf.name_scope("Accuracy") :
        train_accuracy = tf.placeholder(tf.float32, shape=(), name="train_accuracy")
        test_accuracy = tf.placeholder(tf.float32, shape=(), name="test_accuracy")
        tf.summary.scalar("train_accuracy", train_accuracy)
        tf.summary.scalar("test_accuracy", test_accuracy)
        
    if not os.path.exists(result_dir) :
        os.makedirs(result_dir)
    rec_writer = open("{}/recall.txt".format(result_dir), 'w')
    prec_writer = open("{}/precision.txt".format(result_dir), 'w')
    f1_writer = open("{}/f1.txt".format(result_dir), 'w')
    conf_stack = np.zeros((V.get('n_output'), V.get('n_output'), 0))
    return train_loss, test_loss, train_accuracy, test_accuracy, rec_writer, prec_writer, f1_writer, conf_stack

def record(rec_writer, recall, prec_writer, precision, f1_writer, f1, step) :
    rec_writer.write("{:d} ".format(step))
    np.savetxt(rec_writer, recall, fmt='%0.3f')
    prec_writer.write("{:d} ".format(step))
    np.savetxt(prec_writer, precision, fmt='%0.3f')
    f1_writer.write("{:d} ".format(step))
    np.savetxt(f1_writer, f1, fmt='%0.3f')
    rec_writer.flush()
    prec_writer.flush()
    f1_writer.flush()