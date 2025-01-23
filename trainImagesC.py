#!/usr/bin/env python3

### SAFE AND REQUIRED IMPORTS
import ast
import datetime
import getopt
import hashlib
import json
import multiprocessing
import numpy as np
import os
import random
import re
import shutil
import sys
import textwrap



### CONSTANTS
EPSILON = 1e-07
I32 = 2147483647
LOSSES = ('bel', 'cbf', 'cbs', 'cbbs', 'ccm', 'ce', 'cos', 'la', 'lab', 'lw', 'lwb', 'mae', 'mse', 'nis')
OPTIMIZERS = ('a', 'aw', 'sgd')
SCHEDULERS =  ('clr', )
TRAINERS = []
for loss in LOSSES:
	for optimizer in OPTIMIZERS:
		for scheduler in SCHEDULERS:
			TRAINERS.append(f"{loss}+{scheduler}+{optimizer}")
WEIGHTS = re.compile('^\[[0-9]\.[0-9],[0-9]\.[0-9],[0-9]\.[0-9]\]$')
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
settings['batch'] = 16 
settings['cbBeta'] = 0.99
settings['cbGamma'] = 1.0
settings['ccmWeight'] = [1.0, 1.0, 1.0]
settings['classCount'] = None
settings['classWeight'] = None
settings['cpu'] = False
settings['dataTrain'] = ''
settings['dataValidate'] = []
settings['epochs'] = 256
settings['gpu'] = '0'
settings['inputSize'] = 64
settings['jcrop'] = False
settings['kappa'] = 1.0
settings['laTau'] = 0.0
settings['learningRate'] = 0.005
settings['loss'] = 'ce'
settings['model'] = ''
settings['numberMap'] = ''
settings['optimizer'] = 'aw'
settings['outputArray'] = None
settings['outputDirectory'] = ''
settings['processors'] = multiprocessing.cpu_count()
settings['randomSeed'] = 123456789
settings['remix'] = False
settings['remixTau'] = 0.0
settings['scheduler'] = 'clr'
settings['saveQuantity'] = False
settings['smoothModel'] = ''
settings['weightDecay'] = 0.0001



### OTHER SETTINGS
settings['analysisTime'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
settings['beta1'] = 0.9 ### AdamW
settings['beta2'] = 0.999 ### AdamW
settings['brightnessDelta'] = 0.4
settings['channels'] = 3
settings['clrInitial'] = 4 ### try 3
settings['clrStep'] = 4 ### try 2-8
settings['contrastRange'] = [0.6, 1.4]
settings['ema_momentum'] = 0.99 ### AdamW + SGD
settings['GaussianSigma'] = (0.1, 2.0)
settings['hue'] = 0.1
settings['leniency'] = 256
settings['perBatchRemix'] = False
settings['randomMax'] = 2**32 ### 64 is unsafe (53 is max safe)
settings['randomMin'] = 0
settings['resizeRatio'] = (0.75, 1.3333333333333333) ### default in RandomResizedCrop
settings['resizeScale'] = (0.08, 1.0) ### default in RandomResizedCrop; used (0.4, 1.0) unsuccessfully
settings['saturationRange'] = [0.8, 1.2]



### READ OPTIONS
arrayError = 'Number of elements in the output array (required): -a int | --array=int'
dataTrainError = 'Input train data (required): -t file.tfr | --train file.tfr'
dataValidateError = 'Input validation data (required): -v file.tfr,file.tfr,... | --validate file.tfr,file.tfr,...'
modelError = 'Input model file (required): -m file.keras | --model=file.keras'
outputDirectoryError = 'Model output directory (required): -o directory | --output=directory'
try:
	arguments, values = getopt.getopt(sys.argv[1:], 'a:B:b:C:cd:e:f:G:g:hi:jk:l:m:n:o:p:Qqr:s:T:t:u:v:w:xy:', ['array=', 'beta=', 'batch=', 'ccm=', 'cpu', 'decay=', 'epochs=', 'function=', 'gamma=', 'gpu=', 'help', 'input=', 'jcrop', 'kappa=', 'learning=', 'model=', 'numberMap=', 'output=', 'processors=', 'quantity', 'remix', 'random=', 'smooth=', 'Tau=', 'train=', 'tau=', 'validate=', 'weight=', 'ycount='])
except getopt.error as err:
	eprintWrap(str(err))
	sys.exit(2)
for argument, value in arguments:
	if argument in ('-a', '--array') and int(value) > 0:
		settings['outputArray'] = int(value)
	elif argument in ('-B', '--beta') and float(value) > 0.0 and float(value) < 1.0:
		settings['cbBeta'] = float(value)
	elif argument in ('-b', '--batch') and int(value) > 0:
		settings['batch'] = int(value)
	elif argument in ('-C', '--ccm') and re.search(WEIGHTS, value):
		settings['ccmWeight'] = ast.literal_eval(value)	
	elif argument in ('-c', '--cpu'):
		settings['cpu'] = True
	elif argument in ('-d', '--decay') and float(value) > 0.0 and float(value) < 1.0:
		settings['weightDecay'] = float(value)
	elif argument in ('-e', '--epochs') and int(value) > 0:
		settings['epochs'] = int(value)
	elif argument in ('-f', '--function') and value in TRAINERS:
		items = value.split('+')
		settings['loss'] = items[0]
		settings['scheduler'] = items[1]
		settings['optimizer'] = items[2]
	elif argument in ('-G', '--gamma') and float(value) > 0.0 and float(value) < 4.0:
		settings['cbGamma'] = float(value)
	elif argument in ('-g', '--gpu') and int(value) >= 0: ### does not test if device is valid
		settings['gpu'] = value
	elif argument in ('-h', '--help'):
		eprint('')
		eprintWrap('A Python3 script to train models on images from .tfr files with TensorFlow 2.13.0.')
		eprintWrap(arrayError)
		eprintWrap(f"Class balanced loss beta (optional; default = {settings['cbBeta']}): -B float | --beta=float")
		eprintWrap(f"Batch size (optional; default = {settings['batch']}): -b int | --batch=int")
		eprintWrap(f"CCM loss weights (optional; default = {settings['ccmWeight']}): -C [float,float,float] | --ccm=[float,float,float]")
		eprintWrap(f"CPU only (optional; default = {not settings['cpu']}): -c | --cpu")
		eprintWrap(f"Weight decay for AdamW/SGD optimizer (optional; default = {settings['weightDecay']}): -d float | --decay=float")
		eprintWrap(f"Number of epochs (optional; default = {settings['epochs']}): -e int | --epochs=int")
		eprintWrap(f"Loss, scheduler, and optimization function combination (optional; a = adam; aw = adamW; bel = per-batch class balanced sigmoid + logit adjusted loss; cbf = class balanced focal loss; cbs = class balanced sigmoid loss; cbbs = per-batch class balanced sigmoid loss; ccm = cos + ce + mse; ce = cross entropy; clr = cyclical learning rate; cos = cosine loss; la = logit adjusted loss; lab = per-batch logit adjusted loss; lw = logit adjusted loss with normalized inverse weights; lwb = per-batch logit adjusted loss with normalized inverse weights; mae = mean absolute error; mse = mean squared error; nis = normalized inverse weights sigmoid loss; sgd = stochastic gradient descent; default = {settings['loss']}+{settings['scheduler']}+{settings['optimizer']}): -f {'|'.join(TRAINERS)} | --function={'|'.join(TRAINERS)}")
		eprintWrap(f"Class balanced loss gamma (optional; default = {settings['cbGamma']}): -G float | --gamma=float")
		eprintWrap(f"Run on specified GPU (optional; default = {settings['gpu']}; CPU option overrides GPU settings): -g int | --gpu=int")
		eprintWrap(f"Input image size (optional; default = {settings['inputSize']}): -i int | --input=int")
		eprintWrap(f"Use multicrop dataset (optional; default = {settings['jcrop']}): -j | --jcrop")
		eprintWrap(f"Remix Kappa (optional; default = {settings['kappa']}; arXiv:2007.03943): -k float | --kappa=float")
		eprintWrap(f"Learning rate (optional; default = {settings['learningRate']}): -l float | --learning=float")
		eprintWrap(modelError)
		eprintWrap('File to remap ID numbers for better one-hot encoding (optional; header assumed: .tfr ID, new ID): -n file.tsv | --numberMap=file.tsv')
		eprintWrap(outputDirectoryError)
		eprintWrap(f"Processors (optional; default = {settings['processors']}): -p int | --processors=int")
		eprintWrap(f"Save model at the end of each epoch (optional; default = {settings['saveQuantity']}): -Q | --quantity")
		eprintWrap(f"Reweigh labels in favor of rare classes using remix (arXiv:2007.03943) using a class count dictionary (optional; default = {settings['remix']}): -q | --remix")
		eprintWrap(f"Random seed (optional; default = {settings['randomSeed']}): -r int | --random=int")
		eprintWrap(f"Improve the model with fine tuning by setting the last non-trainable layer name (optional; default = {settings['smoothModel']}): -s string | --smooth=string")
		eprintWrap(f"Logit adjusted loss tau (optional; default = {settings['laTau']}): -T float | --Tau=float")
		eprintWrap(dataTrainError)
		eprintWrap(f"Remix Tau (optional; default = {settings['remixTau']}; arXiv:2007.03943): -u float | --tau=float")
		eprintWrap(dataValidateError)
		eprintWrap("Class weight dictionary (optional): -w '{0:float,1:float,2:float...}' | --weight='{0:float,1:float,2:float...}'")
		eprintWrap("Class count dictionary (optional): -y '{0:int,1:int,2:int...}' | --ycount='{0:int,1:int,2:int...}'")
		eprint('')
		sys.exit(0)
	elif argument in ('-i', '--input') and int(value) > 0:
		settings['inputSize'] = int(value)
	elif argument in ('-j', '--jcrop'):
		settings['jcrop'] = True
	elif argument in ('-k', '--kappa') and float(value) > 0.0 and float(value) < 1.0:
		settings['kappa'] = float(value)
	elif argument in ('-l', '--learning') and float(value) > 0.0 and float(value) < 1.0:
		settings['learningRate'] = float(value)
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
	elif argument in ('-o', '--output'):
		if os.path.isdir(value):
			settings['outputDirectory'] = value
		else:
			eprintWrap(f"Model output directory '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-p', '--processors') and int(value) > 0:
		settings['processors'] = int(value)
	elif argument in ('-Q', '--quantity'):
		settings['saveQuantity'] = True
	elif argument in ('-q', '--remix'):
		settings['remix'] = True
	elif argument in ('-r', '--random') and int(value) >= settings['randomMin'] and int(value) <= settings['randomMax']:
		settings['randomSeed'] = int(value)
	elif argument in ('-s', '--smooth') and len(value):
		settings['smoothModel'] = value
	elif argument in ('-T', '--Tau') and float(value) >= 0.0 and float(value) <= 10.0:
		settings['laTau'] = float(value)
	elif argument in ('-t', '--train'):
		if os.path.isfile(value):
			settings['dataTrain'] = value
		else:
			eprintWrap(f"Input train file '{value}' does not exist!")
			sys.exit(2)
	elif argument in ('-u', '--tau') and float(value) >= 0.0 and float(value) <= 1.0:
		settings['remixTau'] = float(value)
	elif argument in ('-v', '--validate'):
		files = value.split(',')
		for file in files:
			if os.path.isfile(file):
				settings['dataValidate'].append(file)
			else:
				eprintWrap(f"Input validation file '{file}' does not exist!")
				sys.exit(2)
	elif argument in ('-w', '--weight'):
		settings['classWeight'] = ast.literal_eval(value)
	elif argument in ('-y', '--ycount'):
		settings['classCount'] = ast.literal_eval(value)



### START/END
if not settings['outputArray']:
	eprintWrap(arrayError)
	sys.exit(2)
elif not len(settings['dataValidate']):
	eprintWrap(dataValidateError)
	sys.exit(2)
elif not settings['dataTrain']:
	eprintWrap(dataTrainError)
	sys.exit(2)
elif not settings['model']:
	eprintWrap(modelError)
	sys.exit(2)
elif not settings['outputDirectory']:
	eprintWrap(outputDirectoryError)
	sys.exit(2)
if not settings['classWeight']:
	settings['classWeight'] = {}
	for k in range(0, settings['outputArray']):
		settings['classWeight'][k] = 1.0
if not settings['classCount']:
	settings['classCount'] = {}
	for k in range(0, settings['outputArray']):
		settings['classCount'][k] = 1

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
# from Bias import BiasLayer
from IterativeNormalization import IterativeNormalization
# from LayerScale import LayerScale
# from PatchEncoder import PatchEncoder
# from PositionalEmbeddings import PositionalEmbeddings
availableGPUs = len(tf.config.experimental.list_physical_devices('GPU'))
if settings['cpu'] == False and availableGPUs == 0:
	eprintWrap('No GPUs are available to TensorFlow. Rerun the script with -c for CPU processing only.')
	sys.exit(2)
eprintWrap(f"TensorFlow GPUs = {availableGPUs}")
eprintWrap(f"TensorFlow {tf.version.VERSION}\n")



### INIT
random.seed(settings['randomSeed'])
tf.random.set_seed(random.randint(settings['randomMin'], settings['randomMax']))

settings['GaussianKernel'] = settings['inputSize']//10



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



### INIT REMIX AND WEIGHTED LOSSES
if settings['remix'] or settings['loss'] in ('cbf', 'cbs', 'la', 'lw', 'nis'):
	### common
	classCounts = tf.cast(tf.constant([settings['classCount'][key] for key in sorted(settings['classCount'].keys())]), tf.int32)
	total = sum(settings['classCount'].values())
	### cbf + cbs
	effective = 1.0 - np.power(settings['cbBeta'], classCounts)
	cbLossWeights = (1.0 - settings['cbBeta']) / effective
	cbLossWeights /= np.sum(cbLossWeights)
	cbLossWeights = tf.expand_dims(tf.cast(cbLossWeights, dtype = tf.float32), 0)
	### la
	laLossWeights = tf.math.log(tf.cast(tf.constant([(settings['classCount'][key]/total)**settings['laTau'] + EPSILON for key in sorted(settings['classCount'].keys())]), tf.float32))
	### lw; normalized inverse of class frequency weights
	weights = [(1-(settings['classCount'][key]/total)) for key in sorted(settings['classCount'].keys())]
	minimum = min(weights)
	lwLossWeights = tf.cast(tf.constant([w/minimum for w in weights]), tf.float32)
	### nis
	nisLossWeights = lwLossWeights



### DATASET FUNCTIONS
def augmentApplyOne(image, label, functions, probability = 1.0): ### modified from https://github.com/hirune924/imgaug-tf
	def _applyOne(image, label, functions):
		selection = tf.random.uniform(
			dtype = tf.int32,
			maxval = len(functions),
			minval = 0,
			shape = []
		)
		for k, function in enumerate(functions):
			image, label = tf.cond(tf.equal(k, selection), lambda: function(image, label), lambda: (image, label))
		return image, label
	return tf.cond(tf.random.uniform([], 0.0, 1.0) < probability, lambda: _applyOne(image, label, functions = functions), lambda: (image, label))

def augmentBlurer(images): ### modified from https://gist.github.com/blzq/c87d42f45a8c5a53f5b393e27b1f5319
	halfKernel = (settings['GaussianKernel']//2)+1
	kernelRange = tf.range(-halfKernel, halfKernel, dtype = tf.float32)
	xx, yy = tf.meshgrid(kernelRange, kernelRange)
	sigma = tf.random.uniform(
		dtype = tf.float32, 
		maxval = settings['GaussianSigma'][1], 
		minval = settings['GaussianSigma'][0], 
		shape = []
	)
	kernel = tf.exp(-(xx**2 + yy**2)/(2.0*sigma**2))
	kernel = kernel/tf.reduce_sum(kernel)
	kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, settings['channels']])
	kernel = kernel[..., tf.newaxis]
	images = tf.nn.depthwise_conv2d(
		data_format = 'NHWC',
		filter = kernel, 
		input = images, 
		padding = 'SAME', 
		strides = (1, 1, 1, 1)
	)
	return images

def augmentBlur(images, labels, blurs = 16):
	splits = tf.split(
		axis = 0,
		num_or_size_splits = blurs,
		value = images
	)
	for k in range(0, blurs):
		splits[k] = augmentBlurer(splits[k])
	images = tf.concat(splits, axis = 0)
	randomSeed = tf.random.uniform(
		dtype = tf.dtypes.int32,
		maxval = I32,
		minval = 0,
		shape = [2]
	)
	images = tf.random.experimental.stateless_shuffle(
		seed = randomSeed,
		value = images 
	)
	labels = tf.random.experimental.stateless_shuffle(
		seed = randomSeed,
		value = labels 
	)
	return images, labels

def augmentColorJitter00(image, label):
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	return image, label

def augmentColorJitter01(image, label):
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	return image, label

def augmentColorJitter02(image, label):
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	return image, label

def augmentColorJitter03(image, label):
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	return image, label

def augmentColorJitter04(image, label):
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	return image, label

def augmentColorJitter05(image, label):
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	return image, label

def augmentColorJitter06(image, label):
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	return image, label

def augmentColorJitter07(image, label):
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	return image, label

def augmentColorJitter08(image, label):
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	return image, label

def augmentColorJitter09(image, label):
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	return image, label

def augmentColorJitter10(image, label):
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	return image, label

def augmentColorJitter11(image, label):
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	return image, label

def augmentColorJitter12(image, label):
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	return image, label

def augmentColorJitter13(image, label):
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	return image, label

def augmentColorJitter14(image, label):
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	return image, label

def augmentColorJitter15(image, label):
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	return image, label

def augmentColorJitter16(image, label):
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	return image, label

def augmentColorJitter17(image, label):
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	return image, label

def augmentColorJitter18(image, label):
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	return image, label

def augmentColorJitter19(image, label):
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	return image, label

def augmentColorJitter20(image, label):
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	return image, label

def augmentColorJitter21(image, label):
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	return image, label

def augmentColorJitter22(image, label):
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	return image, label

def augmentColorJitter23(image, label):
	image = tf.image.random_saturation(image, lower = settings['saturationRange'][0], upper = settings['saturationRange'][1])
	image = tf.image.random_hue(image, max_delta = settings['hue'])
	image = tf.image.random_contrast(image, lower = settings['contrastRange'][0], upper = settings['contrastRange'][1])
	image = tf.image.random_brightness(image, max_delta = settings['brightnessDelta'])
	return image, label

def augmentFlip(images, labels):
	splits = tf.split(
		axis = 0,
		num_or_size_splits = 2,
		value = images
	)
	splits[0] = tf.image.random_flip_left_right(splits[0])
	splits[1] = tf.image.flip_up_down(splits[1])
	images = tf.concat(splits, axis = 0)
	randomSeed = tf.random.uniform(
		dtype = tf.dtypes.int32,
		maxval = I32,
		minval = 0,
		shape = [2]
	)
	images = tf.random.experimental.stateless_shuffle(
		seed = randomSeed,
		value = images 
	)
	labels = tf.random.experimental.stateless_shuffle(
		seed = randomSeed,
		value = labels 
	)
	return images, labels

def augmentGrayscale(images, labels, scales = 4):
	splits = tf.split(
		axis = 0,
		num_or_size_splits = scales, ### p = 0.25
		value = images
	)
	splits[0] = tf.image.rgb_to_grayscale(splits[0])
	splits[0] = tf.image.grayscale_to_rgb(splits[0])
	images = tf.concat(splits, axis = 0)
	randomSeed = tf.random.uniform(
		dtype = tf.dtypes.int32,
		maxval = I32,
		minval = 0,
		shape = [2]
	)
	images = tf.random.experimental.stateless_shuffle(
		seed = randomSeed,
		value = images 
	)
	labels = tf.random.experimental.stateless_shuffle(
		seed = randomSeed,
		value = labels 
	)
	return images, labels

def augmentNothing(image, label): 
	return image, label

def augmentRotate(images, labels):
	splits = tf.split(
		axis = 0,
		num_or_size_splits = 4,
		value = images
	)
	splits[0] = tf.image.rot90(splits[0], k = 1)
	splits[1] = tf.image.rot90(splits[1], k = 2)
	splits[2] = tf.image.rot90(splits[2], k = 3)
	images = tf.concat(splits, axis = 0)
	randomSeed = tf.random.uniform(
		dtype = tf.dtypes.int32,
		maxval = I32,
		minval = 0,
		shape = [2]
	)
	images = tf.random.experimental.stateless_shuffle(
		seed = randomSeed,
		value = images 
	)
	labels = tf.random.experimental.stateless_shuffle(
		seed = randomSeed,
		value = labels 
	)
	return images, labels

def batchMix(images, labels, PROBABILITY = 1.0): ### based on https://www.kaggle.com/code/yihdarshieh/batch-implementation-of-more-data-augmentations/notebook?scriptVersionId=29767726 and https://www.kaggle.com/code/cdeotte/cutmix-and-mixup-on-gpu-tpu
	mixup = tf.cast(tf.random.uniform([settings['batch']], 0, 1) <= PROBABILITY, tf.int32)
	mix = tf.random.uniform([settings['batch']], 0, 1)*tf.cast(mixup, tf.float32) ### beta distribution with alpha = 1.0
	new = tf.cast(tf.random.uniform([settings['batch']], 0, settings['batch']), tf.int32)
	mixupImages = (1-mix)[:, tf.newaxis, tf.newaxis, tf.newaxis] * images + mix[:, tf.newaxis, tf.newaxis, tf.newaxis] * tf.gather(images, new)
	if settings['remix']:
		decodedLabels = tf.math.argmax(labels, axis = -1)
		decodedNewLabels = tf.math.argmax(tf.gather(labels, new), axis = -1)
		if settings['perBatchRemix']:
			classCount = tf.math.reduce_sum(
				labels, 
				axis = 0, 
				keepdims = False
			)
			remixRatio = tf.gather(classCount, decodedLabels)/tf.gather(classCount, decodedNewLabels)
		else:
			remixRatio = tf.gather(classCounts, decodedLabels)/tf.gather(classCounts, decodedNewLabels)
		newMix = tf.where(tf.math.logical_and(tf.greater_equal(remixRatio, settings['kappa']), tf.less(mix, settings['remixTau'])), 0.0, mix)
		newMix = tf.where(tf.math.logical_and(tf.less_equal(remixRatio, 1/settings['kappa']), tf.less(1-mix, settings['remixTau'])), 1.0, newMix)
		mix = newMix
	mixupLabels = (1-mix)[:, tf.newaxis] * labels + mix[:, tf.newaxis] * tf.gather(labels, new)
	return mixupImages, mixupLabels

def batchRandomResizedCrop(images, labels):
	randomRatios = tf.exp(tf.random.uniform((settings['batch'], ), tf.math.log(settings['resizeRatio'][0]), tf.math.log(settings['resizeRatio'][1]), dtype = tf.float32))
	randomScales = tf.random.uniform((settings['batch'], ), settings['resizeScale'][0], settings['resizeScale'][1], dtype = tf.float32)
	cropHeights = tf.clip_by_value(tf.sqrt(randomScales/randomRatios), 0.0, 1.0)
	cropWidths = tf.clip_by_value(tf.sqrt(randomScales*randomRatios), 0.0, 1.0)
	heightOffsets = tf.random.uniform((settings['batch'], ), 0.0, 1.0-cropHeights, dtype = tf.float32)
	widthOffsets = tf.random.uniform((settings['batch'], ), 0.0, 1.0-cropWidths, dtype = tf.float32)
	boundingBoxes = tf.stack([heightOffsets, widthOffsets, heightOffsets+cropHeights, widthOffsets+cropWidths], axis = 1)
	images = tf.image.crop_and_resize(
		box_indices = tf.range(settings['batch']),
		boxes = boundingBoxes,
		crop_size = (settings['inputSize'], settings['inputSize']),
		extrapolation_value = 0.0,
		image = images,
		method = 'bilinear'
	)
	return images, labels

def decodeTFR(record, train = False):
	if settings['jcrop']:
		feature = {
			'category': tf.io.FixedLenFeature([], tf.int64),
			'images': tf.io.VarLenFeature(tf.string)
		}
		record = tf.io.parse_single_example(record, feature)
		images = tf.sparse.to_dense(record['images'], default_value = '', validate_indices = True)
		if train:
			images = tf.random.shuffle(images)
		image = tf.cast(tf.io.decode_png(
			channels = settings['channels'],
			contents = images[0]
		), tf.float32)
	else:
		feature = {
			'category': tf.io.FixedLenFeature([], tf.int64),
			'image': tf.io.FixedLenFeature([], tf.string)
		}
		record = tf.io.parse_single_example(record, feature)
		image = tf.cast(tf.io.decode_png(
			channels = settings['channels'],
			contents = record['image']
		), tf.float32)
	record['category'] = tf.one_hot(
		depth = settings['outputArray'],
		indices = remapper.lookup(record['category']) if settings['remap'] else record['category']
	)
	return image, record['category']

def unifiedAugmenter(images, labels):
	images, labels = augmentRotate(images, labels)
	images, labels = batchRandomResizedCrop(images, labels)
	# images, labels = augmentFlip(images, labels)
	# images, labels = augmentGrayscale(images, labels, scales = settings['batch']//4)
	# images, labels = augmentBlur(images, labels, blurs = settings['batch']//4)
	return images, labels

### DATASETS
validationData = (
	tf.data.TFRecordDataset(settings['dataValidate'])
	.map(
		lambda x: (decodeTFR(x, train = False)),
		deterministic = False,
		num_parallel_calls = tf.data.AUTOTUNE
	).batch(
		batch_size = settings['batch'],
		deterministic = False,
		drop_remainder = False,
		num_parallel_calls = tf.data.AUTOTUNE
	).prefetch(tf.data.AUTOTUNE)
)

trainData = (
	tf.data.TFRecordDataset(settings['dataTrain'])
	.map(
		lambda x: (decodeTFR(x, train = True)),
		deterministic = False,
		num_parallel_calls = tf.data.AUTOTUNE
	# ).map(
	# 	lambda x, y: (augmentApplyOne(x, y, functions = (augmentColorJitter00, augmentColorJitter01, augmentColorJitter02, augmentColorJitter03, augmentColorJitter04, augmentColorJitter05, augmentColorJitter06, augmentColorJitter07, augmentColorJitter08, augmentColorJitter09, augmentColorJitter10, augmentColorJitter11, augmentColorJitter12, augmentColorJitter13, augmentColorJitter14, augmentColorJitter15, augmentColorJitter16, augmentColorJitter17, augmentColorJitter18, augmentColorJitter19, augmentColorJitter20, augmentColorJitter21, augmentColorJitter22, augmentColorJitter23, augmentNothing, augmentNothing, augmentNothing, augmentNothing, augmentNothing, augmentNothing))),
	# 	deterministic = False,
	# 	num_parallel_calls = tf.data.AUTOTUNE
	).shuffle(
		buffer_size = settings['batch']*48,
		reshuffle_each_iteration = True
	).batch(
		batch_size = settings['batch'],
		deterministic = False,
		drop_remainder = True,
		num_parallel_calls = tf.data.AUTOTUNE
	).map(
		lambda x, y: (unifiedAugmenter(x, y)),
		deterministic = False,
		num_parallel_calls = tf.data.AUTOTUNE
	).map(
		lambda x, y: (batchMix(x, y)),
		deterministic = False,
		num_parallel_calls = tf.data.AUTOTUNE
	).prefetch(tf.data.AUTOTUNE)
)



### READ AND TRAIN MODEL

### training class/functions
class belLoss(tf.keras.losses.Loss): 
	def __init__(self, **kwargs):
		super().__init__(name = 'bel_loss', **kwargs)
		self.cbbs = cbbsLoss
		self.lab = labLoss
	def call(self, y_true, y_pred):
		c = self.cbbs(y_true, y_pred)
#
# c fails...
# 		
		l = self.lab(y_true, y_pred)
		return c+l

class CyclicalLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule): ### Smith (2015; https://arxiv.org/abs/1506.01186) based on https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/cyclical_learning_rate.py
	def __init__(self, initial_learning_rate, maximal_learning_rate, step_size, scale_fn, scale_mode = 'cycle', name = 'CyclicalLearningRate'):
		super().__init__()
		self.initial_learning_rate = initial_learning_rate
		self.maximal_learning_rate = maximal_learning_rate
		self.step_size = step_size
		self.scale_fn = scale_fn
		self.scale_mode = scale_mode
		self.name = name
	def __call__(self, step):
		with tf.name_scope(self.name or 'CyclicalLearningRate'):
			initial_learning_rate = tf.convert_to_tensor(
				self.initial_learning_rate, name = 'initial_learning_rate'
			)
			dtype = initial_learning_rate.dtype
			maximal_learning_rate = tf.cast(self.maximal_learning_rate, dtype)
			step_size = tf.cast(self.step_size, dtype)
			step_as_dtype = tf.cast(step, dtype)
			cycle = tf.floor(1 + step_as_dtype / (2 * step_size))
			x = tf.abs(step_as_dtype / step_size - 2 * cycle + 1)
			mode_step = cycle if self.scale_mode == 'cycle' else step
			return initial_learning_rate + (maximal_learning_rate - initial_learning_rate) * tf.maximum(tf.cast(0, dtype), (1 - x)) * self.scale_fn(mode_step)
	def get_config(self):
		return {
			'initial_learning_rate': self.initial_learning_rate,
			'maximal_learning_rate': self.maximal_learning_rate,
			'scale_fn': self.scale_fn,
			'step_size': self.step_size,
			'scale_mode': self.scale_mode
		}

def cbfLoss(labels, logits): ### class balanced focal loss (arXiv:1901.05555)
	weights = labels*tf.tile(cbLossWeights, [tf.shape(labels)[0], 1])
	weights = tf.reduce_sum(weights, axis = 1)
	weights = tf.expand_dims(weights, 1)
	weights = tf.tile(weights, [1, settings['outputArray']])
	ce = tf.nn.sigmoid_cross_entropy_with_logits(
		labels = labels, 
		logits = logits
	)
	if settings['cbGamma'] == 0.0:
		modulator = 1.0
	else:
		modulator = tf.exp(-settings['cbGamma'] * labels * logits - settings['cbGamma'] * tf.math.log1p(tf.exp(-1.0 * logits)))
	loss = modulator*ce
	focal = tf.reduce_sum(weights*loss)
	focal /= tf.reduce_sum(labels)
	return focal

def cbsLoss(labels, logits): ### class balanced sigmoid loss (arXiv:1901.05555)
	weights = labels*tf.tile(cbLossWeights, [tf.shape(labels)[0], 1])
	weights = tf.reduce_sum(weights, axis = 1)
	weights = tf.expand_dims(weights, 1)
	weights = tf.tile(weights, [1, settings['outputArray']])
	loss = weights*tf.nn.sigmoid_cross_entropy_with_logits(
		labels = labels, 
		logits = logits
	)
	loss = tf.reduce_sum(loss)/tf.reduce_sum(labels)
	return loss

def cbbsLoss(labels, logits): ### per-batch class balanced sigmoid loss (cf. arXiv:1901.05555)
	classCount = tf.math.reduce_sum(
		labels, 
		axis = 0, 
		keepdims = False
	)
	effective = 1.0-tf.math.pow(settings['cbBeta'], classCount)
	weights = (1.0-settings['cbBeta'])/effective
	weights /= tf.reduce_sum(weights)
	weights = tf.expand_dims(weights, 0)
	weights = labels*tf.tile(weights, [tf.shape(labels)[0], 1])
	weights = tf.reduce_sum(weights, axis = 1)
	weights = tf.expand_dims(weights, 1)
	weights = tf.tile(weights, [1, settings['outputArray']])
	loss = weights*tf.nn.sigmoid_cross_entropy_with_logits(
		labels = labels, 
		logits = logits
	)
	loss = tf.reduce_sum(loss)/tf.reduce_sum(labels)
	return loss

def ccmLoss(labels, logits): ### cosine + ce loss + mae loss (cf. arXiv:1901.09054)
	ce = tf.keras.losses.CategoricalCrossentropy(from_logits = True, label_smoothing = 0.2)
	return settings['ccmWeight'][0]*cosLoss(labels, logits) + settings['ccmWeight'][1]*ce(labels, logits) + settings['ccmWeight'][2]*mseLoss(labels, logits)

def cosLoss(labels, logits): ### cosine loss (arXiv:1901.09054)
	normalizedLogits = tf.math.l2_normalize(logits, axis = -1)
	loss = 1.0-tf.math.reduce_sum(normalizedLogits*labels, axis = -1)
	return tf.reduce_mean(loss, axis = 0)

def laLoss(labels, logits): ### logit-adjusted loss (arXiv:2007.07314)
	# loss = tf.nn.sparse_softmax_cross_entropy_with_logits( ### original for int labels
	loss = tf.nn.softmax_cross_entropy_with_logits(
		labels = labels, 
		logits = logits+laLossWeights
	)
	return tf.reduce_mean(loss, axis = 0)

def labLoss(labels, logits): ### per-batch logit-adjusted loss (cf. arXiv:2007.07314)
	classCount = tf.math.reduce_sum(
		labels, 
		axis = 0, 
		keepdims = False
	)
	total = tf.math.reduce_sum(
		labels, 
		axis = None, 
		keepdims = False
	)
	weights = tf.math.log((classCount/total)**settings['laTau'] + EPSILON)
	loss = tf.nn.softmax_cross_entropy_with_logits(
		labels = labels, 
		logits = logits+weights
	)
	return tf.reduce_mean(loss, axis = 0)

def lwbLoss(labels, logits): ### per-batch logit-adjusted loss using inverse normalized weights (cf. arXiv:2007.07314)
	# binary = tf.where(condition = tf.math.greater(labels, 0.5), x = 1.0,	y = 0.0)
	classCount = tf.math.reduce_sum(
		labels, 
		axis = 0, 
		keepdims = False
	)
	total = tf.math.reduce_sum(
		labels, 
		axis = None, 
		keepdims = False
	)
	weights = 1-(classCount/total)
	minimum = tf.math.reduce_min(
		weights,
		axis = None, 
		keepdims = False
	)
	minimum = tf.math.reduce_max(
		tf.stack([minimum, EPSILON]),
		axis = None, 
		keepdims = False
	)
	weights /= minimum
	loss = tf.nn.softmax_cross_entropy_with_logits(
		labels = labels, 
		logits = logits*weights
	)
	return tf.reduce_mean(loss, axis = 0)

def lwLoss(labels, logits): ### logit-adjusted loss using inverse normalized weights (cf. arXiv:2007.07314)
	loss = tf.nn.softmax_cross_entropy_with_logits(
		labels = labels, 
		logits = logits+lwLossWeights
	)
	return tf.reduce_mean(loss, axis = 0)

def maeLoss(labels, logits):
	batch = tf.shape(labels)[0]
	normalizeLogits = tf.nn.l2_normalize(logits, axis = -1)
	mae = tf.math.reduce_sum(tf.math.abs(normalizeLogits-labels))*tf.cast((1/batch), normalizeLogits.dtype)
	return mae

def mseLoss(labels, logits): ### squared Euclidean distance (arXiv:1901.09054)
	squared = tf.math.square(logits-labels)
	return tf.reduce_sum(squared, axis = -1)

def nisLoss(labels, logits): ### inverse normalized weight sigmoid loss
	weights = labels*tf.tile(nisLossWeights, [tf.shape(labels)[0], 1])
	weights = tf.reduce_sum(weights, axis = 1)
	weights = tf.expand_dims(weights, 1)
	weights = tf.tile(weights, [1, settings['outputArray']])
	loss = weights*tf.nn.sigmoid_cross_entropy_with_logits(
		labels = labels, 
		logits = logits
	)
	loss = tf.reduce_sum(loss)/tf.reduce_sum(labels)
	return loss

def scale(x): 
	return 1.0/(2.0**(x-1))

### model
# model = tf.keras.models.load_model(settings['model'], compile = False, custom_objects = {'BiasLayer': BiasLayer, 'IterativeNormalization': IterativeNormalization, 'LayerScale': LayerScale, 'PatchEncoder': PatchEncoder, 'PositionalEmbeddings': PositionalEmbeddings})
model = tf.keras.models.load_model(settings['model'], compile = False, custom_objects = {'IterativeNormalization': IterativeNormalization})

if len(settings['smoothModel']):
	trainable = False
	for layer in model.layers:
		if layer.name == settings['smoothModel']:
			trainable = True
		layer.trainable = trainable

eprint(model.summary())

### loss
loss = None
if settings['loss'] == 'bel':
	loss = belLoss()
elif settings['loss'] == 'cbf':
	loss = cbfLoss
elif settings['loss'] == 'cbs':
	loss = cbsLoss
elif settings['loss'] == 'cbbs':
	loss = cbbsLoss
elif settings['loss'] == 'ccm':
	loss = ccmLoss
elif settings['loss'] == 'ce':
	loss = tf.keras.losses.CategoricalCrossentropy(
		from_logits = True, 
		label_smoothing = 0.2
	)
elif settings['loss'] == 'cos':
	loss = cosLoss
elif settings['loss'] == 'la':
	loss = laLoss
elif settings['loss'] == 'lab':
	loss = labLoss
elif settings['loss'] == 'lw':
	loss = lwLoss
elif settings['loss'] == 'lwb':
	loss = lwbLoss
elif settings['loss'] == 'mse':
	loss = mseLoss
elif settings['loss'] == 'mae':
	loss = maeLoss
elif settings['loss'] == 'nis':
	loss = nisLoss

### metrics
metrics = [
	tf.keras.metrics.CategoricalAccuracy(
		name = 'accuracy'
	),
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
]

### scheduler
clr = CyclicalLearningRate(
	initial_learning_rate = settings['learningRate']/settings['clrInitial'],
	maximal_learning_rate = settings['learningRate'],
	name = 'CyclicalLearningRate',
	scale_fn = scale,
	scale_mode = 'cycle',
	step_size = settings['clrStep']*settings['batch'] 
)

### optimizer
optimizer = None
if settings['optimizer'] == 'a':
	optimizer = tf.keras.optimizers.Adam(
		learning_rate = clr
	)
elif settings['optimizer'] == 'aw':
	optimizer = tf.keras.optimizers.AdamW(
		amsgrad = False,
		beta_1 = settings['beta1'],
		beta_2 = settings['beta2'],
		clipnorm = None,
		clipvalue = None,
		ema_momentum = settings['ema_momentum'],
		ema_overwrite_frequency = None,
		epsilon = EPSILON,
		global_clipnorm = None,
		# jit_compile = True, ### not in 2.16
		learning_rate = clr,
		name = 'AdamW',
		use_ema = False,
		weight_decay = settings['weightDecay']
	)
elif settings['optimizer'] == 'sgd':
	optimizer = tf.keras.optimizers.SGD(
		clipnorm = None,
		clipvalue = None,
		ema_momentum = settings['ema_momentum'],
		ema_overwrite_frequency = None,
		global_clipnorm = None,
		learning_rate = clr,
		momentum = 0.0,
		name = 'sgd',
		nesterov = True,
		use_ema = False,
		weight_decay = settings['weightDecay']
	)

### compile
model.compile(
	loss = loss,
	metrics = metrics,
	optimizer = optimizer
)

### output directory
encoded = json.dumps(settings, ensure_ascii = False, indent = 3, sort_keys = True).encode()
hexMD5 = hashlib.md5(encoded).hexdigest()
directory = os.path.join(settings['outputDirectory'], f"{hexMD5}-intermediate")
if not os.path.exists(directory):
	os.makedirs(directory)

### callbacks
callbacks = []
callbacks.append(tf.keras.callbacks.EarlyStopping(
	baseline = None,
	min_delta = 0,
	mode = 'max',
	# monitor = 'val_f1',
	monitor = 'val_auc',
	patience = settings['leniency'],
	restore_best_weights = True,
	verbose = 0
))
if settings['saveQuantity']:
	callbacks.append(
		tf.keras.callbacks.ModelCheckpoint(
		filepath = os.path.join(directory, 'epoch-{epoch:04d}.keras'),
		verbose = 0,
		save_freq = 'epoch',
		save_best_only = False,
		save_weights_only = False
	))
### train
history = model.fit(
	batch_size = settings['batch'],
	callbacks = callbacks,
	class_weight = settings['classWeight'],
	epochs = settings['epochs'],
	validation_data = validationData,
	x = trainData
)

### save
directory = os.path.join(settings['outputDirectory'], f"{hexMD5}-best")
if not os.path.exists(directory):
	os.makedirs(directory)
model.save(os.path.join(directory, 'best-model.keras'))
np.save(os.path.join(directory, 'training-history.npy'), history.history, allow_pickle = True) ### history = np.load('file', allow_pickle = True).item()
with open(os.path.join(directory, 'training-settings.json'), 'w') as file:
	print(json.dumps(settings, ensure_ascii = False, indent = 3, sort_keys = True).encode(), file = file)



sys.exit(0)
