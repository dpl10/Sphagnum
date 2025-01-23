#!/usr/bin/env python3

### SAFE AND REQUIRED IMPORTS
import ast
import datetime
import getopt
import hashlib
import inspect
import json
import multiprocessing
import numpy as np
import os
import re
import shutil
import sys
import textwrap
import time



### CONSTANTS
DIGITS = 4
EPSILON = 1e-07
VECTORS = re.compile(r'^(?:-?[0-9]+\.[0-9]+,?)+$')
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
settings['batch'] = 64
settings['classWeight'] = None
settings['cpu'] = False
settings['dataSoup'] = ''
settings['dataValidate'] = ''
settings['gpu'] = '0'
settings['inputSize'] = 64
settings['jcrop'] = False
settings['means'] = []
settings['modelDirectory'] = ''
settings['numberMap'] = ''
settings['outputArray'] = None
settings['outputDirectory'] = ''
settings['processors'] = multiprocessing.cpu_count()
settings['tile'] = 0
settings['threshold'] = 0.98
settings['variances'] = []



### OTHER SETTINGS
settings['analysisTime'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
settings['bands'] = 3
settings['normalize'] = False
settings['optimize'] = {}
settings['optimize']['name'] = 'AUCPR'
settings['optimize']['index'] = 2



### READ OPTIONS
arrayError = 'Number of elements in the output array (required): -a int | --array=int'
dataSoupError = 'Input soup data (required): -d file.tfr | --data file.tfr'
dataValidateError = 'Input validation data (required): -v file.tfr | --validate file.tfr'
modelDirectoryError = 'Input model directory (required): -m directory | --model=directory'
outputDirectoryError = 'Model output directory (required): -o directory | --output=directory'
try:
	arguments, values = getopt.getopt(sys.argv[1:], 'a:b:cd:g:hi:jM:m:n:o:p:T:t:V:v:w:', ['array=', 'batch=', 'cpu', 'data', 'gpu=', 'help', 'input=', 'jcrop', 'means=', 'model=', 'numberMap=', 'output=', 'processors=', 'tile=', 'threshold=', 'variances=', 'validate=', 'weight='])
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
	elif argument in ('-d', '--data'):
		if os.path.isfile(value):
			settings['dataSoup'] = value
		else:
			eprintWrap(f"Input data file '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-g', '--gpu') and int(value) >= 0: ### does not test if device is valid
		settings['gpu'] = value
	elif argument in ('-h', '--help'):
		eprint('')
		eprintWrap('A Python3 script to make greedy model soups (arXiv:2203.05482v1) using image data in .tfr files with TensorFlow 2.13.0.')
		eprintWrap(arrayError)
		eprintWrap(f"Batch size (optional; default = {settings['batch']}): -b int | --batch=int")
		eprintWrap(f"CPU only (optional; default = {not settings['cpu']}): -c | --cpu")
		eprintWrap(dataSoupError)
		eprintWrap(f"Run on specified GPU (optional; default = {settings['gpu']}; CPU option overrides GPU settings): -g int | --gpu=int")
		eprintWrap(f"Input image size (optional; default = {settings['inputSize']}): -i int | --input=int")
		eprintWrap(f"Use multicrop dataset (optional; default = {settings['jcrop']}): -j | --jcrop")
		eprintWrap(f"Band means for image normalization (optional): -M float,float,... | --means=float,float,...")
		eprintWrap(modelDirectoryError)
		eprintWrap('File to remap ID numbers for better one-hot encoding (optional; header assumed: .tfr ID, new ID): -n file.tsv | --numberMap=file.tsv')
		eprintWrap(outputDirectoryError)
		eprintWrap(f"Processors (optional; default = {settings['processors']}): -p int | --processors=int")
		eprintWrap(dataValidateError)
		eprintWrap(f"Metric validation percentile threshold for possible soup inclusion (optional; overrides threshold; default = {settings['tile']}): -T int | --tile=int")
		eprintWrap(f"Metric validation threshold for possible soup inclusion (optional; default = {settings['threshold']}): -t float | --threshold=float")
		eprintWrap(f"Band variances for image normalization (optional): -V float,float,... | --variances=float,float,...")
		eprintWrap("Class weight dictionary (optional): -w '{0:float,1:float,2:float...}' | --weight='{0:float,1:float,2:float...}'")
		eprint('')
		sys.exit(0)
	elif argument in ('-i', '--input') and int(value) > 0:
		settings['inputSize'] = int(value)
	elif argument in ('-j', '--jcrop'):
		settings['jcrop'] = True
	elif argument in ('-M', '--means') and re.search(VECTORS, value):
		settings['means'] = [float(x) for x in value.split(',')]
	elif argument in ('-m', '--model'):
		if os.path.isdir(value):
			settings['modelDirectory'] = value
		else:
			eprintWrap(f"Model directory '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-n', '--numberMap'):
		if os.path.isfile(value):
			settings['numberMap'] = value
		else:
			eprintWrap(f"Map file '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-o', '--output'):
		if os.path.isdir(value):
			settings['outputDirectory'] = value
		else:
			eprintWrap(f"Model output directory '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-p', '--processors') and int(value) > 0:
		settings['processors'] = int(value)
	elif argument in ('-T', '--tile') and int(value) > 0 and int(value) < 100:
		settings['tile'] = int(value)
	elif argument in ('-t', '--threshold') and float(value) > 0.0 and float(value) <= 1.0:
		settings['threshold'] = float(value)
	elif argument in ('-v', '--validate'):
		if os.path.isfile(value):
			settings['dataValidate'] = value
		else:
			eprintWrap(f"Input validation file '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-V', '--variances') and re.search(VECTORS, value):
		settings['variances'] = [float(x) for x in value.split(',')]
	elif argument in ('-w', '--weight'):
		settings['classWeight'] = ast.literal_eval(value)
if len(settings['means']) == settings['bands'] and len(settings['variances']) == settings['bands']:
	settings['normalize'] = True



### START/END
if not settings['outputArray']:
	eprintWrap(arrayError)
	sys.exit(2)
elif not settings['dataSoup']:
	eprintWrap(dataSoupError)
	sys.exit(2)
elif not settings['dataValidate']:
	eprintWrap(dataValidateError)
	sys.exit(2)
elif not settings['modelDirectory']:
	eprintWrap(modelDirectoryError)
	sys.exit(2)
elif not settings['outputDirectory']:
	eprintWrap(outputDirectoryError)
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
import tensorflow.keras.backend as K
from IterativeNormalization import IterativeNormalization
availableGPUs = len(tf.config.experimental.list_physical_devices('GPU'))
if settings['cpu'] == False and availableGPUs == 0:
	eprintWrap('No GPUs are available to TensorFlow. Rerun the script with -c for CPU processing only.')
	sys.exit(2)
eprintWrap(f"TensorFlow GPUs = {availableGPUs}")
eprintWrap(f"TensorFlow {tf.version.VERSION}\n")



### INIT METRIC
metric = None
if settings['optimize']['name'] == 'ACCURACY':
	metric = tf.keras.metrics.CategoricalAccuracy(name = 'accuracy')
elif settings['optimize']['name'] == 'AUCPR':
	metric =	tf.keras.metrics.AUC(
		curve = 'PR',
		from_logits = True,
		multi_label = False, 
		name = 'auc',
		num_thresholds = 200, 
		summation_method = 'interpolation'
	)
elif settings['optimize']['name'] == 'F1':
	metric = tf.keras.metrics.F1Score(
		average = 'macro',
		name = 'f1',
		threshold = None
	)



### INIT NORM
if settings['normalize']:
	NORMALIZE = tf.keras.Sequential([
		tf.keras.layers.Normalization(
			axis = -1, 
			invert = False,
			mean = settings['means'], 
			variance = settings['variances']
		)
	])
	NORMALIZE.build((settings['inputSize'], settings['inputSize'], settings['bands'])) ### kluge



### DATASET FUNCTION
def decodeTFR(record):
	if settings['jcrop']:
		feature = {
			'category': tf.io.FixedLenFeature([], tf.int64),
			'images': tf.io.VarLenFeature(tf.string)
		}
		record = tf.io.parse_single_example(record, feature)
		images = tf.sparse.to_dense(record['images'], default_value = '', validate_indices = True)
		image = tf.cast(tf.io.decode_png(
			channels = settings['bands'],
			contents = images[0]
		), tf.float32)
	else:
		feature = {
			'category': tf.io.FixedLenFeature([], tf.int64),
			'image': tf.io.FixedLenFeature([], tf.string)
		}
		record = tf.io.parse_single_example(record, feature)
		image = tf.cast(tf.io.decode_jpeg(
			channels = settings['bands'],
			contents = record['image']
		), tf.float32)
	if settings['normalize']:
		image = NORMALIZE(image)
	record['category'] = tf.one_hot(
		depth = settings['outputArray'],
		indices = record['category']
	)
	return image, record['category']

### DATASET
soupData = (
	tf.data.TFRecordDataset(settings['dataSoup'])
	.map(
		decodeTFR,
		deterministic = False,
		num_parallel_calls = tf.data.AUTOTUNE
	).batch(
		batch_size = settings['batch'],
		deterministic = False,
		drop_remainder = False,
		num_parallel_calls = tf.data.AUTOTUNE
	).prefetch(tf.data.AUTOTUNE)
)

validationData = (
	tf.data.TFRecordDataset(settings['dataValidate'])
	.map(
		decodeTFR,
		deterministic = False,
		num_parallel_calls = tf.data.AUTOTUNE
	).batch(
		batch_size = settings['batch'],
		deterministic = False,
		drop_remainder = False,
		num_parallel_calls = tf.data.AUTOTUNE
	).prefetch(tf.data.AUTOTUNE)
)


### MODEL FUNCTION
def getModel(path):
	model = tf.keras.models.load_model(path, compile = False, custom_objects = {'IterativeNormalization': IterativeNormalization})
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
	# print(model.summary())
	return model

### READ AND EVALUATE MODELS
model2score = {} ### path => score
print(f"MODEL\t{settings['optimize']['name']}")
for path, dirs, files in os.walk(settings['modelDirectory']):
	for filename in files:
		if filename.endswith('.keras'):
			modelFile = os.path.join(path, filename)
			model = getModel(modelFile)
			stats = model.evaluate(validationData)
			score = float(stats[settings['optimize']['index']])
			model2score[modelFile] = score
			print(f"{modelFile}\t{score:.{DIGITS}f}")
models = list(reversed(dict(sorted(model2score.items(), key = lambda item: item[1])).keys()))
end = len(models)
threshold = settings['threshold'] if settings['tile'] == 0 else np.percentile(list(model2score.values()), settings['tile'])
for k, model in enumerate(models):
	if model2score[model] < threshold:
		end = k
		break
models = models[0:end]
eprintWrap(f"End = {k}: {len(models):,} models for soup...")



### SOUP FUNCTIONS 
### based on https://github.com/Burf/ModelSoups
def uniformSoup(model, path, by_name = False):
	if not isinstance(path, list):
		path = [path]
	soups = []
	for modelPath in path:
		model.load_weights(modelPath, by_name = by_name)
		soup = [np.array(w) for w in model.weights]
		soups.append(soup)
	if 0 < len(soups):
		for w1, w2 in zip(model.weights, list(zip(*soups))):
			K.set_value(w1, np.mean(w2, axis = 0))
	return model

def greedySoup(model, path, data, metric, updateGreedy = False, compare = np.greater_equal, by_name = False, verbose = True, y_true = 'y_true'):
	if not isinstance(path, list):
		path = [path]
	score = None
	soup = []
	inputKey = [inp.name for inp in model.inputs]
	inputCount = len(inputKey)
	for modelPath in path:
		if updateGreedy:
			model.load_weights(modelPath, by_name = by_name)
			for w1, w2 in zip(model.weights, soup):
				K.set_value(w1, np.mean([w1, w2], axis = 0))
		else:
			model = uniformSoup(model, soup + [modelPath], by_name = by_name)
		iterator = iter(data)
		history = []
		step = 0
		startTime = time.time()
		while True:
			try:
				iter_data = next(iterator)
				if not isinstance(iter_data, dict):
					x = iter_data[:inputCount]
					y = list(iter_data[inputCount:])
					d_cnt = len(y[0])
				else:
					x = [iter_data[k] for k in inputKey if k in iter_data]
				step += 1
				logits = model.predict(x)
				if not isinstance(logits, list):
					logits = [logits]
				if isinstance(iter_data, dict):
					metric_key = [key for key in inspect.getfullargspec(metric).args if key != "self"]
					if len(metric_key) == 0:
						metric_key = [y_true]
					y = [iter_data[k] for k in metric_key if k in iter_data]
					d_cnt = len(y[0])
				metric_val = np.array(metric(*(y + logits)))
				if np.ndim(metric_val) == 0:
					metric_val = [float(metric_val)] * d_cnt
				history += list(metric_val)
				if verbose:
					key = metric.__name__ if hasattr(metric, '__name__') else str(metric)
					sys.stdout.write(f"\r[{os.path.basename(modelPath)}] step: {step} - time: {(time.time()-startTime):.2f}s - {key}: {np.nanmean(history):.{DIGITS}f}")
					sys.stdout.flush()
			except (tf.errors.OutOfRangeError, StopIteration):
					print('')
					break
		if 0 < len(history) and (score is None or compare(np.nanmean(history), score)):
			score = np.nanmean(history)
			if updateGreedy:
				soup = [np.array(w) for w in model.weights]
			else:
				soup += [modelPath]
	if len(soup) != 0:
		if updateGreedy:
			for w1, w2 in zip(model.weights, soup):
				K.set_value(w1, w2)
		else:
			model = uniformSoup(model, soup, by_name = by_name)
		if verbose:
			print(f"Greedy soup best score: {score:.{DIGITS}f}")
	return model

### AVERAGE MODELS
if len(models) >= 2:
	greedyModel = greedySoup(
		compare = np.greater_equal, 
		data = soupData, 
		metric = metric, 
		model = getModel(models[0]), 
		path = list(models), 
		verbose = False,
		updateGreedy = True
	)
	stats = greedyModel.evaluate(validationData)
	print(f"Greedy soup best model (of {len(models):,} models): loss = {float(stats[0]):.{DIGITS}f}; {settings['optimize']['name']} = {stats[settings['optimize']['index']]:.{DIGITS}f}")
else:
	if settings['tile'] == 0:
		eprintWrap(f"No models meeting the required threshold ({settings['threshold']:.{DIGITS}f}) were found!")
	else:
		eprintWrap(f"Not enough models were found!")
	sys.exit(0)



### OUTPUT
encoded = json.dumps(settings, ensure_ascii = False, indent = 3, sort_keys = True).encode()
hexMD5 = hashlib.md5(encoded).hexdigest()
directory = os.path.join(settings['outputDirectory'], f"{hexMD5}-best")
if not os.path.exists(directory):
	os.makedirs(directory)
greedyModel.save(os.path.join(directory, 'soup-model.keras'))
with open(os.path.join(directory, 'soup-settings.json'), 'w') as file:
	print(json.dumps(settings, ensure_ascii = False, indent = 3, sort_keys = True).encode(), file = file)



sys.exit(0)
