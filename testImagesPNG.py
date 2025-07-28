#!/usr/bin/env python3

### SAFE AND REQUIRED IMPORTS
import ast
import getopt
import multiprocessing
import numpy as np
import os
import shutil
import sys
import textwrap



### CONSTANT
WRAP = shutil.get_terminal_size().columns



### PRINT TO STANDARD ERROR
def eprint(*args, **kwargs):
	print(*args, file = sys.stderr, **kwargs)

### WRAP TEXT
def eprintWrap(string, columns = WRAP):
	eprint(wrap(string, columns))

def wrap(string, columns = WRAP):
	return '\n'.join(textwrap.wrap(string, columns))



### USER SETTINGS
settings = {}
settings['auc'] = ''
settings['batch'] = 64 
settings['classWeight'] = None
settings['cpu'] = False
settings['dataTest'] = ''
settings['errorMatrix'] = ''
settings['gpu'] = '0'
settings['inputSize'] = 64
settings['jcrop'] = False
settings['model'] = ''
settings['numberMap'] = ''
settings['outputArray'] = None
settings['processors'] = multiprocessing.cpu_count()
settings['vitPreprocessing'] = False



### OTHER SETTINGS
settings['bands'] = 3
settings['means'] = [151.232678,129.847499,97.245936] ### ./tfrPNGdecode2stats.py -i Sphagnum-Sphagnum-5b5-vegetative64-train.tfr -r 396692098 -v 12 
settings['variances'] = [1159.658289,1189.655168,965.479194] ### ./tfrPNGdecode2stats.py -i Sphagnum-Sphagnum-5b5-vegetative64-train.tfr -r 396692098 -v 12 




### READ OPTIONS
arrayError = 'Number of elements in the output array (required): -a int | --array=int'
dataTestError = 'Input test data (required): -t file.tfr | --test=file.tfr'
modelError = 'Input model file (required): -m file.keras | --model=file.keras'
try:
	arguments, values = getopt.getopt(sys.argv[1:], 'a:b:ce:g:hi:jm:n:p:t:u:vw:', ['array=', 'batch=', 'cpu', 'error=', 'gpu=', 'help', 'input=', 'jcrop', 'model=', 'numberMap=', 'processors=', 'test=', 'under=', 'vit', 'weight='])
except getopt.error as err:
	eprintWrap(str(err))
	sys.exit(2)
for argument, value in arguments:
	if argument in ('-a', '--array') and int(value) > 0:
		settings['outputArray'] = int(value)
	elif argument in ('-b', '--batch') and int(value) > 0:
		settings['batch'] = int(value)
	elif argument in ('-c', '--cpu'):
		settings['cpu'] = True
	elif argument in ('-e', '--error'):
		settings['errorMatrix'] = value
	elif argument in ('-g', '--gpu') and int(value) >= 0: ### does not test if device is valid
		settings['gpu'] = value
	elif argument in ('-h', '--help'):
		eprint('')
		eprintWrap('A Python3 script to test models on images from .tfr files with TensorFlow 2.13.0.')
		eprintWrap(arrayError)
		eprintWrap(f"Batch size (optional; default = {settings['batch']}): -b int | --batch=int")
		eprintWrap(f"CPU only (optional; default = {not settings['cpu']}): -c | --cpu")
		eprintWrap('Calculate and save confusion (error) matrix (optional): -e file.npz | --error=file.npz')
		eprintWrap(f"Run on specified GPU (optional; default = {settings['gpu']}; CPU option overrides GPU settings): -g int | --gpu int")
		eprintWrap(f"Input image size (optional; default = {settings['inputSize']}): -i int | --input=int")
		eprintWrap(f"Use multicrop dataset (optional; default = {settings['jcrop']}): -j | --jcrop")
		eprintWrap(modelError)
		eprintWrap('File to remap ID numbers for better one-hot encoding (optional; header assumed: .tfr ID, new ID): -n file.tsv | --numberMap=file.tsv')
		eprintWrap(f"Processors (optional; default = {settings['processors']}): -p int | --processors=int")
		eprintWrap(dataTestError)
		eprintWrap('Calculate and save area under the values (optional): -u file.npz | --under=file.npz')
		eprintWrap("Class weight dictionary (optional): -w '{0:float,1:float,2:float...}' | --weight='{0:float,1:float,2:float...}'")
		eprint('')
		sys.exit(0)
	elif argument in ('-i', '--input') and int(value) > 0:
		settings['inputSize'] = int(value)
	elif argument in ('-j', '--jcrop'):
		settings['jcrop'] = True
	elif argument in ('-m', '--model'):
		if os.path.isfile(value) or os.path.isdir(value):
			settings['model'] = value
		else:
			eprintWrap(f"Model file '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-n', '--numberMap'):
		if os.path.isfile(value):
			settings['numberMap'] = value
		else:
			eprintWrap(f"Map file '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-p', '--processors') and int(value) > 0:
		settings['processors'] = int(value)
	elif argument in ('-t', '--test'):
		if os.path.isfile(value):
			settings['dataTest'] = value
		else:
			eprintWrap(f"Input file '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-u', '--under'):
		settings['auc'] = value
	elif argument in ('-v', '--vit'):
		settings['vitPreprocessing'] = True
	elif argument in ('-w', '--weight'):
		settings['classWeight'] = ast.literal_eval(value)



### START/END
if not settings['outputArray']:
	eprintWrap(arrayError)
	sys.exit(2)
elif not settings['dataTest']:
	eprintWrap(dataTestError)
	sys.exit(2)
elif not settings['model']:
	eprintWrap(modelError)
	sys.exit(2)
if not settings['classWeight']:
	settings['classWeight'] = {}
	for k in range(0, settings['outputArray']):
		settings['classWeight'][k] = 1.0

eprintWrap('started...')
for key, value in settings.items():
	eprintWrap(f"{key} = {value}")



### DISABLE OR SET GPU, THEN IMPORT TENSORFLOW
if settings['cpu'] == True:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
else:
	os.environ['CUDA_VISIBLE_DEVICES'] = settings['gpu']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from IterativeNormalization import IterativeNormalization
eprintWrap(f"TensorFlow GPUs = {len(tf.config.experimental.list_physical_devices('GPU'))}")
eprintWrap(f"TensorFlow {tf.version.VERSION}\n")



### INIT REMAP
settings['remap'] = True if len(settings['numberMap']) else False
if settings['remap']:
	keys = []
	values = []
	with open(settings['numberMap'], mode = 'rt', encoding = 'utf8', errors = 'replace') as file:
		for k, line in enumerate(file):
			if k > 0:
				columns = line.strip().split('\t')
				keys.append(int(columns[0]))
				values.append(int(columns[1]))
	remapper = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(tf.cast(keys, dtype = tf.dtypes.int64), tf.cast(values, dtype = tf.dtypes.int64)), default_value = 0)



### DATASET FUNCTIONS
def decodeTFR(record):
	if settings['jcrop']:
		feature = {
			'category': tf.io.FixedLenFeature([], tf.int64),
			'images': tf.io.VarLenFeature(tf.string)
		}
		record = tf.io.parse_single_example(record, feature)
		images = tf.sparse.to_dense(record['images'], default_value = '', validate_indices = True)
		image = tf.cast(tf.io.decode_png(
			channels = 3,
			contents = images[0]
		), tf.float32)
	else:
		feature = {
			'category': tf.io.FixedLenFeature([], tf.int64),
			'image': tf.io.FixedLenFeature([], tf.string)
		}
		record = tf.io.parse_single_example(record, feature)
		image = tf.cast(tf.io.decode_png(
			channels = 3,
			contents = record['image']
		), tf.float32)
	record['category'] = tf.one_hot(
		depth = settings['outputArray'],
		indices = remapper.lookup(record['category']) if settings['remap'] else record['category']
	)
	return image, record['category']

NORMALIZE = tf.keras.Sequential([
	tf.keras.layers.Normalization(
		axis = -1, 
		invert = False,
		mean = settings['means'], 
		variance = settings['variances']
	)
])
NORMALIZE.build((settings['batch'], settings['inputSize'], settings['inputSize'], settings['bands'])) ### kluge

def normalize(image, labels):
	if settings['vitPreprocessing']:
		image = NORMALIZE(image)
	return image, labels



### DATASET
testData = (
	tf.data.TFRecordDataset(settings['dataTest'])
	.map(
		decodeTFR,
		deterministic = True,
		num_parallel_calls = tf.data.AUTOTUNE
	).batch(
		batch_size = settings['batch'],
		deterministic = True,
		# drop_remainder = True,
		drop_remainder = False,
		num_parallel_calls = tf.data.AUTOTUNE
	).map(
		lambda image, labels: (normalize(image, labels)),
		deterministic = False,
		num_parallel_calls = tf.data.AUTOTUNE
	).prefetch(tf.data.AUTOTUNE)
)



### READ MODEL
model = tf.keras.models.load_model(settings['model'], compile = False, custom_objects = {'IterativeNormalization': IterativeNormalization})
model.compile(
	loss = tf.keras.losses.CategoricalCrossentropy(
		from_logits = True, 
		label_smoothing = 0.1
	),
	metrics = [
		tf.keras.metrics.CategoricalAccuracy(name = 'accuracy'),
		tf.keras.metrics.AUC(
			curve = 'PR',
			from_logits = True,
			multi_label = False, 
			name = 'auc',
			num_thresholds = 200, 
			summation_method = 'interpolation'
		),
		tf.keras.metrics.F1Score(
			average = 'macro',
			name = 'f1',
			threshold = None
		)
	],
	optimizer = tf.keras.optimizers.AdamW()
)
print(model.summary())



### TEST STATS
stats = model.evaluate(testData)
print(f"Test loss: {stats[0]:.4f}")
print(f"Test accuracy: {(stats[1]*100):.2f}%")
print(f"Test AUCPR: {(stats[2]*100):.2f}%")
print(f"Test macro F1: {(stats[3]*100):.2f}%")



### TEST ERROR MATRIX AND AUCPR
if settings['auc'] or settings['errorMatrix']:
	labels = tf.math.argmax(
		axis = -1,
		input = np.concatenate([y for x, y in testData], axis = 0),
		output_type = tf.dtypes.int64
	)
	rawPredictions = model.predict(testData)
	if settings['auc']:
		predictions = tf.nn.softmax(
			axis = -1,
			logits = rawPredictions
		)
		np.savez(settings['auc'], predictions = predictions, labels = labels.numpy())
	if settings['errorMatrix']:
		predictions = tf.math.argmax(
			axis = -1,
			input = rawPredictions,
			output_type = tf.dtypes.int64
		)
		matrix = tf.math.confusion_matrix(
			dtype = tf.int32,
			labels = labels,
			num_classes = settings['outputArray'],
			predictions = predictions,
			# weights = settings['classWeight']
#
# implement!
# 			
			weights = None
		)
		np.savez(settings['errorMatrix'], predictions = predictions.numpy(), labels = labels.numpy(), matrix = matrix.numpy())



sys.exit(0)
