import os
import datetime
os.environ['SM_FRAMEWORK'] = 'tf.keras'

import numpy as np
import albumentations as al
from osgeo import gdal, gdalconst
import matplotlib.pyplot as plt

import efficientnet
import tensorflow as tf
import tensorboard
from segmentation_models.models.unet import Unet
from segmentation_models.losses import bce_jaccard_loss, binary_crossentropy
from segmentation_models.metrics import iou_score


ds_path = '/home/andrew/HDD_Data/clouds/38-Clouds/'
train_count = 8400
val_percent = 0.2



def visualize(rows=None, cols=None, **images):
	"""PLot images in one row."""
	if rows is None:
		rows = 1
	if cols is None:
		cols = int(np.ceil(len(images) / rows))
	plt.figure(figsize=(16, 5))
	for i, (name, image) in enumerate(images.items()):
		plt.subplot(rows, cols, i + 1)
		plt.xticks([])
		plt.yticks([])
		plt.title(' '.join(name.split('_')).title())
		plt.imshow(image)
	plt.show()


def load_decorator(func, train):
	def wrapper(id):
		return tf.numpy_function(
			func,
			(id, train),
			(tf.float32, tf.float32) if train else tf.float32
		)
	return wrapper


def load_one(id, train):
	if train:
		real_ds_path = os.path.join(ds_path, '38-Cloud_training')
		pref = 'train_'
		lst = ['red', 'green', 'blue', 'nir', 'gt']
	else:
		real_ds_path = os.path.join(ds_path, '38-Cloud_test')
		pref = 'test_'
		lst = ['red', 'green', 'blue', 'nir']

	id = str(id, 'utf-8')
	d = []
	for sp in lst:
		band_path = os.path.join(real_ds_path, pref + sp, sp + '_' + id + '.TIF')
		band_ds = gdal.Open(band_path, gdalconst.GA_ReadOnly)
		d.append(band_ds.GetRasterBand(1).ReadAsArray())

	if train:
		x = np.stack(d[:-1], -1).astype(np.float32) / 65535
		y = np.expand_dims(d[-1].astype(np.float32) / 255, -1)

		aug = al.Compose([al.RandomRotate90(), al.HorizontalFlip(),
						  al.ShiftScaleRotate(scale_limit=0.2, rotate_limit=10, shift_limit=0.2, p=0.5, border_mode=0),
						  al.OneOf(
							  [al.MotionBlur(p=0.2), al.MedianBlur(blur_limit=3, p=0.1), al.Blur(blur_limit=3, p=0.1)],
							  p=0.5)])
		t = aug(image=x, mask=y)

		return t['image'], t['mask']

	else:
		x = np.stack(d, -1).astype(np.float32) / 65535
		return x


def create_datasets():
	dss = []
	for csv, train in [(os.path.join(ds_path, '38-Cloud_training', 'training_patches_38-Cloud.csv'), True),
					   (os.path.join(ds_path, '38-Cloud_test', 'test_patches_38-Cloud.csv'), False)]:
		ds = tf.data.experimental.CsvDataset(csv, [str()], header=True)
		ds = ds.map(load_decorator(load_one, train), 16)
		dss.append(ds)

	train_ds = dss[0].skip(int(train_count * val_percent)).prefetch(tf.data.experimental.AUTOTUNE)
	val_ds = dss[0].take(int(train_count * val_percent))
	test_ds = dss[1].prefetch(tf.data.experimental.AUTOTUNE)

	return train_ds, val_ds, test_ds


def train():
	start_time = datetime.datetime.now()
	log_dir = os.path.join('logs', start_time.strftime("%d.%m.%Y-%H:%M:%S"))
	os.makedirs(os.path.join(log_dir, 'weights'))

	strategy = tf.distribute.MirroredStrategy()
	with strategy.scope():
		train_ds, val_ds, test_ds = create_datasets()

		callbacks = [tf.keras.callbacks.TensorBoard(log_dir),
					 tf.keras.callbacks.ModelCheckpoint(os.path.join(log_dir, 'weights', '{epoch:02d}_{val_iou_score:.2f}.hdf5'),
														monitor='val_iou_score', mode='max', save_best_only=True)]

		model = Unet('efficientnetb1', (384, 384, 4), 1, encoder_weights=None,
					 layers=tf.keras.layers, utils=tf.keras.utils, models=tf.keras.models, backend=tf.keras.backend)
		model.compile('adam', bce_jaccard_loss, metrics=[iou_score])
		model.fit(x=train_ds.batch(8),
				  validation_data=val_ds.batch(8),
				  epochs=500,
				  callbacks=callbacks)


def test(model_path):
	model = tf.keras.models.load_model(model_path, compile=False)
	train_ds, val_ds, test_ds = create_datasets()

	test_pred = model.predict(test_ds.take(100).batch(8))
	test_pred = test_pred[..., 0]
	test_pred[test_pred >= 0.8] = 1
	test_pred[test_pred < 0.8] = 0

	for i, img in enumerate(test_ds):
		if np.max(img.numpy()) > 0:
			visualize(img=img[..., :-1], mask=test_pred[i])


if __name__ == '__main__':
	# test('/home/andrew/Temp/108_0.81.hdf5')
	train()
