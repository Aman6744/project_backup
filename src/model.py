import os
import numpy as np
import tensorflow as tf

from glob import glob
from contextlib import redirect_stdout
from tensorflow.keras import backend as K
from tensorflow.keras import Model

from tensorflow.keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.constraints import MaxNorm
from tensorflow.python.keras import distribute

from network.layers import FullGatedConv2D, GatedConv2D
from tensorflow.keras.layers import Conv2D, Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, PReLU
from tensorflow.keras.layers import Input, Add, Activation, Lambda, MaxPooling2D, Reshape


class MyModel:

    def __init__(self,
                 input_size,
                 vocab_size,
                 greedy=False,
                 beam_width=10,
                 top_paths=1,
                 stop_tolerance=20,
                 reduce_tolerance=15):

        self.input_size = input_size
        self.vocab_size = vocab_size

        self.model = None
        self.greedy = greedy
        self.beam_width = beam_width
        self.top_paths = max(1, top_paths)

        self.stop_tolerance = stop_tolerance
        self.reduce_tolerance = reduce_tolerance

    def summary(self, output=None, target=None):

        self.model.summary()

        if target is not None:
            os.makedirs(output, exist_ok=True)

            with open(os.path.join(output, target), "w") as f:
                with redirect_stdout(f):
                    self.model.summary()

    def load_checkpoint(self, target):
        if os.path.isfile(target):
            if self.model is None:
                self.compile()

            self.model.load_weights(target)

    def get_callbacks(self, logdir, checkpoint, monitor="val_loss", verbose=0):

        callbacks = [
            CSVLogger(
                filename=os.path.join(logdir, "epochs.log"),
                separator=";",
                append=True),
            TensorBoard(
                log_dir=logdir,
                histogram_freq=10,
                profile_batch=0,
                write_graph=True,
                write_images=False,
                update_freq="epoch"),
            ModelCheckpoint(
                filepath=checkpoint,
                monitor=monitor,
                save_best_only=True,
                save_weights_only=True,
                verbose=verbose),
            EarlyStopping(
                monitor=monitor,
                min_delta=1e-8,
                patience=self.stop_tolerance,
                restore_best_weights=True,
                verbose=verbose),
            ReduceLROnPlateau(
                monitor=monitor,
                min_delta=1e-8,
                factor=0.2,
                patience=self.reduce_tolerance,
                verbose=verbose)
        ]

        return callbacks

    def compile(self, learning_rate=None, initial_step=0):

        # define inputs, outputs and optimizer of the chosen architecture
        inputs, outputs = self.architecture(self.input_size, self.vocab_size + 1)

        if learning_rate is None:
            learning_rate = CustomSchedule(d_model=self.vocab_size + 1, initial_step=initial_step)
            self.learning_schedule = True
        else:
            self.learning_schedule = False

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

        # create and compile
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=optimizer, loss=lambda y1,y2: tf.py_function(self.ctc_loss_lambda_func, [y1,y2], [tf.float32]))

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            **kwargs):

        # remove ReduceLROnPlateau (if exist) when use schedule learning rate
        if callbacks and self.learning_schedule:
            callbacks = [x for x in callbacks if not isinstance(x, ReduceLROnPlateau)]

        out = self.model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose,
                             callbacks=callbacks, validation_split=validation_split,
                             validation_data=validation_data, shuffle=shuffle,
                             class_weight=class_weight, sample_weight=sample_weight,
                             initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch,
                             validation_steps=validation_steps, validation_freq=validation_freq,
                             max_queue_size=max_queue_size, workers=workers,
                             use_multiprocessing=use_multiprocessing, **kwargs)
        return out

    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=1,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False,
                ctc_decode=True):

        if verbose == 1:
            print("Model Predict")

        out = self.model.predict(x=x, batch_size=batch_size, verbose=verbose, steps=steps,
                                 callbacks=callbacks, max_queue_size=max_queue_size,
                                 workers=workers, use_multiprocessing=use_multiprocessing)

        if not ctc_decode:
            return np.log(out.clip(min=1e-8)), []

        steps_done = 0
        if verbose == 1:
            print("CTC Decode")
            progbar = tf.keras.utils.Progbar(target=steps)

        batch_size = int(np.ceil(len(out) / steps))
        input_length = len(max(out, key=len))

        predicts, probabilities = [], []

        while steps_done < steps:
            index = steps_done * batch_size
            until = index + batch_size

            x_test = np.asarray(out[index:until])
            x_test_len = np.asarray([input_length for _ in range(len(x_test))])

            decode, log = self.ctc_decode(x_test,
            	x_test_len,
            	greedy=self.greedy,
            	beam_width=self.beam_width,
            	top_paths=self.top_paths)

            if not self.greedy:
            	probabilities.extend([np.exp(x)[0] for x in log])
            else:
            	probabilities.extend([np.exp(-x)[0] for x in log])
            decode = [[[int(p) for p in x if p != -1] for x in y] for y in decode]
            predicts.extend(np.swapaxes(decode, 0, 1))

            steps_done += 1
            if verbose == 1:
                progbar.update(steps_done)
                
        return (predicts, probabilities)

    def ctc_decode(self, y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    	input_shape = y_pred.shape
    	num_samples, num_steps = input_shape[0], input_shape[1]
    	y_pred = tf.math.log(tf.transpose(y_pred, perm=[1, 0, 2]) + K.epsilon())
    	input_length = tf.cast(input_length, tf.int32)

    	if greedy:
    		(decoded, log_prob) = tf.nn.ctc_greedy_decoder(
				inputs=y_pred, sequence_length=input_length)
    	else:
    		(decoded, log_prob) = tf.nn.ctc_beam_search_decoder(
				inputs=y_pred,
				sequence_length=input_length,
				beam_width=beam_width,
				top_paths=top_paths)
    	decoded_dense = []
    	for st in decoded:
    		# st = tf.sparse.SparseTensor(
    		# 	st.indices, st.values, (num_samples, num_steps))
    		decoded_dense.append(
    			tf.sparse.to_dense(sp_input=st, default_value=-1))
    	return (decoded_dense, log_prob)


    @staticmethod
    def ctc_loss_lambda_func(y_true, y_pred):

        if len(y_true.shape) > 2:
            y_true = tf.squeeze(y_true)

        # y_pred.shape = (batch_size, string_length, alphabet_size_1_hot_encoded)
        # output of every model is softmax
        # so sum across alphabet_size_1_hot_encoded give 1
        #               string_length give string length
        input_length = tf.math.reduce_sum(y_pred, axis=-1, keepdims=False)
        input_length = tf.math.reduce_sum(input_length, axis=-1, keepdims=True)

        # y_true strings are padded with 0
        # so sum of non-zero gives number of characters in this string
        label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype="int64")

        loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
        # average loss across all entries in the batch
        loss = tf.reduce_mean(loss)

        return loss

    def architecture(self, input_size, d_model):
    
        input_data = Input(name="input", shape=input_size)
        
        cnn = Reshape((input_size[0]//2, input_size[1]//2, input_size[2]*4))(input_data)
        cnn = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 2), padding="same", kernel_initializer="he_uniform")(cnn)
        cnn = PReLU(shared_axes=[1, 2])(cnn)
        cnn = BatchNormalization(renorm=True)(cnn)
        cnn = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(cnn)
    
        cnn = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
        cnn = PReLU(shared_axes=[1, 2])(cnn)
        cnn = BatchNormalization(renorm=True)(cnn)
        cnn = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(cnn)
    
        cnn = Conv2D(filters=40, kernel_size=(2, 4), strides=(2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
        cnn = PReLU(shared_axes=[1, 2])(cnn)
        cnn = BatchNormalization(renorm=True)(cnn)
        cnn = FullGatedConv2D(filters=40, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
        cnn = Dropout(rate=0.2)(cnn)
    
        cnn = Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
        cnn = PReLU(shared_axes=[1, 2])(cnn)
        cnn = BatchNormalization(renorm=True)(cnn)
        cnn = FullGatedConv2D(filters=48, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
        cnn = Dropout(rate=0.2)(cnn)
    
        cnn = Conv2D(filters=56, kernel_size=(2, 4), strides=(2, 4), padding="same", kernel_initializer="he_uniform")(cnn)
        cnn = PReLU(shared_axes=[1, 2])(cnn)
        cnn = BatchNormalization(renorm=True)(cnn)
        cnn = FullGatedConv2D(filters=56, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(cnn)
        cnn = Dropout(rate=0.2)(cnn)
    
        cnn = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(cnn)
        cnn = PReLU(shared_axes=[1, 2])(cnn)
        cnn = BatchNormalization(renorm=True)(cnn)
    
        shape = cnn.get_shape()
        bgru = Reshape((shape[1], shape[2] * shape[3]))(cnn)
    
        bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
        bgru = Dense(units=256)(bgru)

        bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
        bgru = Dense(units=256)(bgru)
    
        bgru = Bidirectional(GRU(units=128, return_sequences=True, dropout=0.5))(bgru)
        output_data = Dense(units=d_model, activation="softmax")(bgru)
    
        return (input_data, output_data)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, initial_step=0, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, dtype="float32")
        self.initial_step = initial_step
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step + self.initial_step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)