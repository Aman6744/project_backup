import os
import json
from tensorflow import keras
from config.config import json_file

class ModelCheckpoint(keras.callbacks.ModelCheckpoint):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def on_train_begin(self, logs=None):
		super().on_train_begin(logs=logs)

		if os.path.isfile(json_file):
			with open(json_file, "r") as f:
				initial_params = json.load(f)

			self.best = initial_params[self.monitor]
			keras.backend.set_value(self.model.optimizer.lr, np.float32(initial_params["lr"]))

	def on_epoch_end(self, epoch, logs=None):
		_current = logs.get(self.monitor)
		if self.monitor_op(_current, self.best):
			initial_params = {
			"epoch":epoch,  
			"lr":str(keras.backend.get_value(self.model.optimizer.lr)), 
			self.monitor:_current
			}
			with open(json_file, "w") as f:
				json.dump(initial_params, f)

		super().on_epoch_end(epoch=epoch, logs=logs)


class EarlyStopping(keras.callbacks.EarlyStopping):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def on_train_begin(self, logs=None):
		super().on_train_begin(logs=logs)

		if os.path.isfile(json_file):
			with open(json_file, "r") as f:
				initial_params = json.load(f)

			self.best = initial_params[self.monitor]


class ReduceLROnPlateau(keras.callbacks.ReduceLROnPlateau):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def on_train_begin(self, logs=None):
		super().on_train_begin(logs=logs)

		if os.path.isfile(json_file):
			with open(json_file, "r") as f:
				initial_params = json.load(f)

			self.best = initial_params[self.monitor]