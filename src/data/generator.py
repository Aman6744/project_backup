import h5py
import string
import numpy as np
import tensorflow as tf

import Preprocess_image as pp
from data.tokenizer import Tokenizer

def preprocessor_helper(x, y, charset, partition, maxTextLength):
    charset = charset.numpy().decode()
    partition = partition.numpy().decode()
    maxTextLength = maxTextLength.numpy()
    y = y.numpy()
    x = x.numpy()
    tokenizer = Tokenizer(
        filters=string.printable.translate(
            str.maketrans("", "", charset)
            ), 
        charset=charset
        )
    if y.any():
        y_ = []
        for word in y:
            seq = tokenizer.texts_to_sequences(word.decode())[0]
            padded_seq = np.pad(seq, (0, maxTextLength-len(seq)))
            y_.append(padded_seq)

        y = np.array(y_)

    if partition in ["train"]:
        x = pp.augmentation(x, 
                rotation_range=5.0, 
                scale_range=0.05, 
                height_shift_range=0.025, 
                width_shift_range=0.05, 
                erode_range=5, 
                dilate_range=3)

    x = pp.normalization(x)

    if y.any():
        return x, y
    else:
        return x

def create_dataset(source_path, charset, partition, maxTextLength, batch_size=32, buf_size=1000, prefetch_size=10):
    with h5py.File(source_path, "r") as f:
        imgs = f[partition]["image"][:100]
        labels = f[partition]["label"][:100]
    
    def get_img_label(x):
        index = x.numpy()
        if partition in ["test"]:
            return imgs[index]
        else:
            return imgs[index], labels[index]

    dataset_size = len(labels)
    indexes = [i for i in range(dataset_size)]
    np.random.shuffle(indexes)

    index_ds = tf.data.Dataset.from_tensor_slices(indexes)
    if partition in ["train"]:
        ds = index_ds.map(lambda x: tf.py_function(get_img_label, [x], [tf.uint8, tf.string])).shuffle(buf_size).batch(batch_size)
        final_ds = ds.map(lambda x,y: tf.py_function(preprocessor_helper, [x,y,charset,partition,maxTextLength], [tf.float32, tf.float32]))
    elif partition in ["valid"]:
        ds = index_ds.map(lambda x: tf.py_function(get_img_label, [x], [tf.uint8, tf.string])).batch(batch_size)
        final_ds = ds.map(lambda x,y: tf.py_function(preprocessor_helper, [x,y,charset,partition,maxTextLength], [tf.float32, tf.float32]))
    else:
        ds = index_ds.map(lambda x: tf.py_function(get_img_label, [x], [tf.uint8])).batch(batch_size)
        final_ds = ds.map(lambda x: tf.py_function(preprocessor_helper, [x,False,charset,partition,maxTextLength], [tf.float32]))

    return final_ds.prefetch(prefetch_size)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2
    charset = string.printable[:84]
    tokenizer = Tokenizer(
        filters=string.printable.translate(
            str.maketrans("", "", charset)
            ), 
        charset=charset
        )
    ds = create_dataset("../data/dataset_hdf5/iam_words.hdf5", string.printable[:84], "train", 32, 2, 10, 2)
    for i, l in ds.take(1):
        print(i.shape, tokenizer.sequences_to_texts(np.swapaxes([l.numpy()], 0, 1)))
        plt.subplot(121)
        plt.imshow(pp.adjust_to_see(i[0].numpy()), cmap="gray")
        plt.subplot(122)
        plt.imshow(pp.adjust_to_see(i[1].numpy()), cmap="gray")
        plt.show()

# class Datagenerator(tf.keras.utils.Sequence):
#     def __init__(self, source_path, charset, partition, maxTextLength, batch_size=32, buf_size=0):
#         self.maxTextLength = maxTextLength
#         self.tokenizer = Tokenizer(
#                 filters=string.printable.translate(
#                     str.maketrans("", "", charset)
#                     ), 
#                 charset=charset
#                 )
#         # self.tokenizer.fit_on_texts(charset)
#         self.batch_size = batch_size
#         self.partition = partition
#         self.source_path = source_path
#         with h5py.File(self.source_path, 'r') as a:
#             dataset = a[self.partition]
#             self.size = dataset['label'].shape[0]
#         self.steps = int(np.ceil(self.size/self.batch_size))
#         self.buf_size = buf_size

#         # if self.partition in ['train'] and self.buf_size:
#         #     self.img_buf = self.dataset['image'][0:self.buf_size]
#         #     self.lab_buf = self.dataset['label'][0:self.buf_size]

#         # for p in self.partitions:
#         #     self.size[p] = self.dataset[p]['image'].shape[0]
#         #     self.steps[p] = int(np.ceil(self.size[p]/self.batch_size))
#         #     self.index[p] = 0


#     def __len__(self):
#         return self.steps
    
#     def __getitem__(self, idx):
#         # if self.partition in ['valid', 'test'] or not self.buf_size:
#         with h5py.File(self.source_path, 'r') as a:
#             dataset = a[self.partition]
#             index = idx*self.batch_size
#             until = index+self.batch_size

#             x = np.array(dataset['image'][index:until]) 
#             if self.partition in ['train']:
#                 x = pp.augmentation(x, 
#                         rotation_range=5.0, 
#                         scale_range=0.05, 
#                         height_shift_range=0.025, 
#                         width_shift_range=0.05, 
#                         erode_range=5, 
#                         dilate_range=3)
#             x = pp.normalization(x)
#             if self.partition in ['valid', 'train']:
#                 y = dataset['label'][index:until]
#                 # y = [self.tokenizer.texts_to_sequences(word.decode())[0] for word in y]
#                 # y = np.array([np.pad(np.asarray(seq), (0, self.maxTextLength-len(seq)), constant_values=(-1, self.PAD)) for seq in y])
#                 y_ = []
#                 for word in y:
#                     seq = self.tokenizer.texts_to_sequences(word.decode())[0]
#                     padded_seq = np.pad(seq, (0, self.maxTextLength-len(seq)))
#                     y_.append(padded_seq)

#                 y = np.array(y_)

#                 return (x, y)
#             return x

#         # else :
#         #     index = idx*self.batch_size + self.buf_size
#         #     until = index+self.batch_size

#         #     zipped = list(zip(self.img_buf, self.lab_buf))
#         #     np.random.shuffle(zipped)

#         #     X, Y = zip(*zipped)
#         #     X = list(X)
#         #     Y = list(Y)

#         #     x = np.array(X[:self.batch_size])
#         #     y = Y[:self.batch_size]

#         #     if until < self.size:
#         #         X[:self.batch_size] = self.dataset['image'][index:until]
#         #         Y[:self.batch_size] = self.dataset['label'][index:until]

#         #     elif index < self.size:
#         #         X = X[until-self.size:]
#         #         Y = Y[until-self.size:]
#         #         until = self.size
#         #         X[:until-index] = self.dataset['image'][index:until]
#         #         Y[:until-index] = self.dataset['label'][index:until]

#         #     else:
#         #         X = X[self.batch_size:]
#         #         Y = Y[self.batch_size:]

#         #     self.img_buf = X
#         #     self.lab_buf = Y

#         #     x = pp.augmentation(x, 
#         #             rotation_range=5.0, 
#         #             scale_range=0.05, 
#         #             height_shift_range=0.025, 
#         #             width_shift_range=0.05, 
#         #             erode_range=5, 
#         #             dilate_range=3)
#         #     x = pp.normalization(x)
#         #     # y = [self.tokenizer.texts_to_sequences(word.decode())[0] for word in y]
#         #     # y = np.array([np.pad(np.asarray(seq), (0, self.maxTextLength-len(seq)), constant_values=(-1, self.PAD)) for seq in y])
#         #     y_ = []
#         #     for word in y:
#         #         seq = self.tokenizer.texts_to_sequences(word.decode())[0]
#         #         padded_seq = np.pad(seq, (0, self.maxTextLength-len(seq)))
#         #         y_.append(padded_seq)

#         #     y = np.array(y_)

#         #     return (x, y)



#     # def on_epoch_end(self):
#     #     if self.partition in ['train'] and self.buf_size:
#     #         self.img_buf = self.dataset['image'][0:self.buf_size]
#     #         self.lab_buf = self.dataset['label'][0:self.buf_size]

