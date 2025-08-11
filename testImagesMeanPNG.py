#!/usr/bin/env python3

### SAFE AND REQUIRED IMPORTS
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
settings['cpu'] = False
settings['dataTest'] = ''
settings['errorMatrix'] = ''
settings['gpu'] = '0'
settings['inputSize'] = 64
settings['model'] = ''
settings['numberMap'] = ''
settings['outputArray'] = None
settings['processors'] = multiprocessing.cpu_count()



### OTHER SETTINGS
settings['bands'] = 3



### READ OPTIONS
arrayError = 'Number of elements in the output array (required): -a int | --array=int'
dataTestError = 'Input test data (required): -t file.tfr | --test=file.tfr'
modelError = 'Input model file (required): -m file.keras | --model=file.keras'
try:
	arguments, values = getopt.getopt(sys.argv[1:], 'a:ce:g:hi:m:n:p:t:u:', ['array=', 'cpu', 'error=', 'gpu=', 'help', 'input=', 'model=', 'numberMap=', 'processors=', 'test=', 'under='])
except getopt.error as err:
	eprintWrap(str(err))
	sys.exit(2)
for argument, value in arguments:
	if argument in ('-a', '--array') and int(value) > 0:
		settings['outputArray'] = int(value)
	elif argument in ('-c', '--cpu'):
		settings['cpu'] = True
	elif argument in ('-e', '--error'):
		settings['errorMatrix'] = value
	elif argument in ('-g', '--gpu') and int(value) >= 0: ### does not test if device is valid
		settings['gpu'] = value
	elif argument in ('-h', '--help'):
		eprint('')
		eprintWrap('A Python3 script to test models on multitile images from .tfr files with TensorFlow 2.13.0.')
		eprintWrap(arrayError)
		eprintWrap(f"CPU only (optional; default = {not settings['cpu']}): -c | --cpu")
		eprintWrap('Calculate and save confusion (error) matrix (optional): -e file.npz | --error=file.npz')
		eprintWrap(f"Run on specified GPU (optional; default = {settings['gpu']}; CPU option overrides GPU settings): -g int | --gpu int")
		eprintWrap(f"Input image size (optional; default = {settings['inputSize']}): -i int | --input=int")
		eprintWrap(modelError)
		eprintWrap('File to remap ID numbers for better one-hot encoding (optional; header assumed: .tfr ID, new ID): -n file.tsv | --numberMap=file.tsv')
		eprintWrap(f"Processors (optional; default = {settings['processors']}): -p int | --processors=int")
		eprintWrap(dataTestError)
		eprintWrap('Calculate and save area under the values (optional): -u file.npz | --under=file.npz')
		eprint('')
		sys.exit(0)
	elif argument in ('-i', '--input') and int(value) > 0:
		settings['inputSize'] = int(value)
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
else:
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
from PatchEncoder import PatchEncoder
from PositionalEmbeddings import PositionalEmbeddings
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
def decodePNG(image):
	return tf.cast(tf.io.decode_png(
		channels = settings['bands'],
		contents = image
	), tf.float32)

def decodeTFR(record):
	feature = {
		'category': tf.io.FixedLenFeature([], tf.int64),
		'images': tf.io.VarLenFeature(tf.string)
	}
	record = tf.io.parse_single_example(record, feature)
	images = tf.sparse.to_dense(record['images'], default_value = '', validate_indices = True)
	decodedImages = tf.map_fn(
		back_prop = False,
		elems = images,
		fn = decodePNG,
		fn_output_signature = tf.float32,
		infer_shape = True,
		parallel_iterations = None,
		swap_memory = False
	)
	record['category'] = tf.one_hot(
		depth = settings['outputArray'],
		indices = remapper.lookup(record['category']) if settings['remap'] else record['category']
	)
	return decodedImages, record['category']



### DATASET
testData = (
	tf.data.TFRecordDataset(settings['dataTest'])
	.map(
		decodeTFR,
		deterministic = True,
		num_parallel_calls = tf.data.AUTOTUNE
	).prefetch(tf.data.AUTOTUNE)
)



### READ MODEL
model = tf.keras.models.load_model(settings['model'], compile = False, custom_objects = {'IterativeNormalization': IterativeNormalization, 'PatchEncoder': PatchEncoder, 'PositionalEmbeddings': PositionalEmbeddings})
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
acc = tf.keras.metrics.CategoricalAccuracy(name = 'accuracy')
auc = tf.keras.metrics.AUC(
	curve = 'PR',
	from_logits = True,
	multi_label = False, 
	name = 'auc',
	num_thresholds = 200, 
	summation_method = 'interpolation'
)
dataSize = sum(1 for _ in testData)
f1 = tf.keras.metrics.F1Score(
	average = 'macro',
	name = 'f1',
	threshold = None
)
labels = np.zeros((dataSize), dtype = np.int64)
predictionsAUC = np.zeros((dataSize, settings['outputArray']), dtype = np.float32)
predictionsF1 = np.zeros((dataSize), dtype = np.int64)
row = 0
for images, label in testData:
	rawPredictions = tf.math.reduce_mean(model.predict(images), axis = 0)
	labels[row] = tf.math.argmax(
		axis = -1,
		input = label,
		output_type = tf.dtypes.int64
	).numpy()
	predictionsAUC[row, :] = tf.nn.softmax(
		axis = -1,
		logits = rawPredictions
	).numpy()
	predictionsF1[row] = tf.math.argmax(
		axis = -1,
		input = rawPredictions,
		output_type = tf.dtypes.int64
	).numpy()
	acc.update_state(
		y_true = label, 
		y_pred = rawPredictions, 
		sample_weight=None
	)
	auc.update_state(
		y_true = label, 
		y_pred = rawPredictions, 
		sample_weight=None
	)
	f1.update_state(
		y_true = tf.expand_dims(label, axis = 0), 
		y_pred = tf.expand_dims(rawPredictions, axis = 0), 
		sample_weight=None
	)
	row += 1

print(f"Test Accuracy: {(float(acc.result())*100):.2f}%")
print(f"Test AUCPR: {(float(auc.result())*100):.2f}%")
print(f"Test macro F1: {(float(f1.result())*100):.2f}%")



### TEST ERROR MATRIX AND AUCPR
if settings['auc'] or settings['errorMatrix']:
	if settings['auc']:
		np.savez(settings['auc'], predictions = predictionsAUC, labels = labels)
	if settings['errorMatrix']:
		matrix = tf.math.confusion_matrix(
			dtype = tf.int32,
			labels = labels,
			num_classes = settings['outputArray'],
			predictions = predictionsF1,
			weights = None
		)
		np.savez(settings['errorMatrix'], predictions = predictionsF1, labels = labels, matrix = matrix.numpy())



sys.exit(0)
