#!/usr/bin/env python3

### SAFE IMPORTS
import getopt
import numpy as np
import os
import random
import re
import shutil
import sys
import textwrap



### CONSTANTS
ACTIVATIONS = ('elu', 'gelu', 'relu', 'selu', 'swish')
CHANNELS = -1
FILTERS = re.compile('^(?:[0-9]+,?)+$')
VECTORS = re.compile('^(?:-?[0-9]+\.[0-9]+,?)+$')
WRAP = shutil.get_terminal_size().columns
ZCA = ('conv', 'output')



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
settings['activation'] = 'selu'
settings['bands'] = 3
settings['depthwise'] = False
settings['dropout'] = False
settings['extraBottleneck'] = False
settings['filters'] = [16, 24, 24, 16]
settings['generalizer'] = 0.2
settings['inputAdjustment'] = True
settings['inputSize'] = 256
settings['means'] = []
settings['outFile'] = ''
settings['outputArray'] = None
settings['outputPolish'] = 32
settings['randomSeed'] = 123456789
settings['sigmoidLossInit'] = False
settings['softClip'] = False
settings['squeezeExciteChannels'] = False
settings['squeezeExciteSpatial'] = False
settings['transferLayer'] = False
settings['variances'] = []
settings['zcaConv'] = False
settings['zcaIterations'] = 3
settings['zcaOutput'] = False



### OTHER SETTINGS
settings['bias'] = True
settings['dformat'] = 'channels_last'
settings['epsilon'] = 1e-7
settings['initializer'] = 'glorot_uniform'
settings['randomMax'] = 2**32 ### 64 is unsafe (53 is max safe)
settings['randomMin'] = 0
settings['seRatio'] = 4
settings['weightDecay'] = 1e-4



### READ OPTIONS
arrayError = 'Number of elements in the output array (required): -a int | --array=int'
outFileError = 'Output file (required): -o file.keras | --output=file.keras'
try:
	arguments, values = getopt.getopt(sys.argv[1:], 'a:b:Ccdef:g:hi:lm:no:p:r:stv:x:Z:z:', ['array=', 'bands=', 'clip', 'channel', 'depthwise', 'extra', 'function=', 'generalizer=', 'help', 'input=', 'loss', 'mean=', 'none', 'output=', 'polish=' 'random=', 'spatial', 'transfer', 'variance=', 'xfilters', 'ZCA=', 'zca='])
except getopt.error as err:
	eprintWrap(str(err))
	sys.exit(2)
for argument, value in arguments:
	if argument in ('-a', '--array') and int(value) > 0:
		settings['outputArray'] = int(value)
	elif argument in ('-b', '--bands') and int(value) > 0:
		settings['bands'] = int(value)
	elif argument in ('-C', '--clip'):
		settings['softClip'] = True
	elif argument in ('-c', '--channel'):
		settings['squeezeExciteChannels'] = True
	elif argument in ('-d', '--depthwise'):
		settings['depthwise'] = True
	elif argument in ('-e', '--extra'):
		settings['extraBottleneck'] = True
	elif argument in ('-f', '--function') and value in ACTIVATIONS:
		settings['activation'] = value
	elif argument in ('-g', '--generalizer') and float(value) > 0.0 and float(value) < 1.0:
		settings['dropout'] = True
		settings['generalize'] = float(value)
	elif argument in ('-h', '--help'):
		eprint('')
		eprintWrap('A Python3 script to create a TensorFlow 2.13.0 reduced FireNet (DOI:10.1007/S11063-021-10555-1) model.')
		eprintWrap(arrayError)
		eprintWrap(f"Input image bands (optional; default = {settings['bands']}): -b int | --bands=int")
		eprintWrap(f"Preprocess input with tanh and a learnable rescaling (optional; default = {settings['softClip']}): -C | --clip")
		eprintWrap(f"Insert squeeze and excite modules (i.e. channel attention; arXiv:1709.01507; optional; default = {settings['squeezeExciteChannels']}): -c | --channel")
		eprintWrap(f"Modify (arXiv:1907.02157) Fire Module (arXiv:1602.07360) 3x3 convolution to use depth-wise convolution (optional; default = {settings['depthwise']}): -d | --depthwise")
		eprintWrap(f"Add an extra terminal squeezeFire (bottleneck) to the model (optional; default = {settings['extraBottleneck']}): -e | --extra")
		eprintWrap(f"Internal activation function (optional; default = {settings['activation']}): -f {'|'.join(ACTIVATIONS)} | --function={'|'.join(ACTIVATIONS)}")
		eprintWrap(f"Insert central dropout (optional; default = {settings['dropout']}): -g = float | --generalizer=float")
		eprintWrap(f"Input image size (optional; default = {settings['inputSize']}): -i int | --input=int")
		eprintWrap(f"Use sigmoid loss bias initializer (optional; default = {settings['sigmoidLossInit']}; arXiv:1901.05555): -l | --loss")
		eprintWrap('Band means for image normalization (optional): -m float,float,... | --mean float,float,...')
		eprintWrap(f"Do no perform any input adjustments such as rescaling or mean/variance scaling (optional; default = {not settings['inputAdjustment']}): -n | --none")
		eprintWrap(outFileError)
		eprintWrap(f"Number of channels used for output polishing (optional; default = {settings['outputPolish']}): -p int | --polish=int")
		eprintWrap(f"Random seed (optional; default = {settings['randomSeed']}): -r int | --random=int")
		eprintWrap(f"Insert squeeze and excite modules (i.e. spatial attention; arXiv:1803.02579; optional; default = {settings['squeezeExciteSpatial']}): -s | --spatial")
		eprintWrap(f"Insert a layer norm after the output Global Average Pooling (GAP) layer for easier transfer learning (optional; default = {settings['transferLayer']}): -t | --transfer")
		eprintWrap('Band variance for image normalization (optional): -v float,float,... | --variance float,float,...')
		eprintWrap(f"Number of expansion filters per block (optional; default = {','.join([str(x) for x in settings['filters']])}): -x int,int,int,int | --xfilters=int,int,int,int")
		eprintWrap(f"Number of ZCA iterative batch norm layer iterations (optional; default = {settings['zcaIterations']}): -Z int | --ZCA=int")
		eprintWrap(f"Insert a ZCA iterative batch norm layer after the input convolution ('conv') and/or before the output classifier ('output') (arXiv:1904.03441; optional; default = {settings['zcaOutput']}): -z {'|'.join(ZCA)},... | --zca={'|'.join(ZCA)},...")
		eprint('')
		sys.exit(0)
	elif argument in ('-i', '--input') and int(value) > 0:
		settings['inputSize'] = int(value)
	elif argument in ('-l', '--loss'):
		settings['sigmoidLossInit'] = True
	elif argument in ('-m', '--mean') and re.search(VECTORS, value):
		settings['means'] = [float(x) for x in value.split(',')]
	elif argument in ('-n', '--none'):
		settings['inputAdjustment'] = False
	elif argument in ('-o', '--output'):
		settings['outFile'] = value
	elif argument in ('-p', '--polish') and int(value) >= 0:
		settings['outputPolish'] = int(value)
	elif argument in ('-r', '--random') and int(value) >= settings['randomMin'] and int(value) <= settings['randomMax']:
		settings['randomSeed'] = int(value)
	elif argument in ('-s', '--spatial'):
		settings['squeezeExciteSpatial'] = True
	elif argument in ('-t', '--transfer'):
		settings['transferLayer'] = True
	elif argument in ('-v', '--variance') and re.search(VECTORS, value):
		settings['variances'] = [float(x) for x in value.split(',')]
	elif argument in ('-x', '--xfilters') and re.search(FILTERS, value):
		settings['filters'] = [int(x) for x in value.split(',')]
	elif argument in ('-Z', '--ZCA') and int(value) > 0:
		settings['zcaIterations'] = int(value)
	elif argument in ('-z', '--zca'):
		for location in value.split(','):
			if location == 'conv':
				settings['zcaConv'] = True
			elif location == 'output':
				settings['zcaOutput'] = True



### START/END
if not settings['outputArray']:
	eprintWrap(arrayError)
	sys.exit(2)
elif not settings['outFile']:
	eprintWrap(outFileError)
	sys.exit(2)
else:
	eprintWrap('started...')
	for key, value in settings.items():
		eprintWrap(f"{key} = {value}")



### DISABLE GPU, THEN IMPORT TENSORFLOW
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from IterativeNormalization import IterativeNormalization
eprintWrap(f"TensorFlow {tf.version.VERSION}\n")



### INIT
random.seed(settings['randomSeed'])
tf.random.set_seed(random.randint(settings['randomMin'], settings['randomMax']))
settings['regularizer'] = tf.keras.regularizers.L2(
	l2 = settings['weightDecay']
)



### CONV2D
def conv2D(x, activation, dilation, filters, groups, kernel, name, padding, strides):
	return tf.keras.layers.Conv2D(
		activation = activation,
		activity_regularizer = None,
		bias_constraint = None,
		bias_initializer = None,
		bias_regularizer = None,
		data_format = settings['dformat'],
		dilation_rate = dilation,
		filters = filters,
		groups = groups,
		kernel_constraint = None,
		kernel_initializer = settings['initializer'],
		kernel_regularizer = settings['regularizer'],
		kernel_size = kernel,
		name = f"{name}_conv2D",
		padding = padding,
		strides = strides,
		use_bias = settings['bias']
	)(x)

### DENSE
def dense(x, activation, bias, name, units, zeros = True):
	return tf.keras.layers.Dense(
		activation = activation,
		activity_regularizer = None,
		bias_constraint = None,
		bias_initializer = 'zeros' if zeros else tf.constant_initializer(-np.log(settings['outputArray']-1)),
		bias_regularizer = None,
		kernel_constraint = None,
		kernel_initializer = settings['initializer'],
		kernel_regularizer = None,
		name = f"{name}_dense",
		units = units,
		use_bias = bias
	)(x)

### DCONV2D
def dconv2D(x, activation, dilation, kernel, name, padding, strides):
	return tf.keras.layers.DepthwiseConv2D(
		activation = activation,
		activity_regularizer = None,
		bias_constraint = None,
		bias_initializer = None,
		bias_regularizer = None,
		data_format = settings['dformat'],
		depth_multiplier = 1,
		depthwise_constraint = None,
		depthwise_initializer = settings['initializer'],
		depthwise_regularizer = settings['regularizer'],
		dilation_rate = dilation,
		kernel_size = kernel,
		name = f"{name}_dconv2D",
		padding = padding,
		strides = strides,
		use_bias = settings['bias']
	)(x)

### FIRE MODULE (ARXIV:1602.07360) WITH OPTIONAL MODIFICATION (ARXIV:1907.02157)
def fireModule(filters, fire, name, reduce):
	fire = conv2D(
		x = fire,
		activation = settings['activation'],
		dilation = 1,
		filters = filters,
		groups = 1,
		kernel = (1, 1),
		name = f"{name}_squeezeFire",
		padding = 'same',
		strides = 1
	)
	if settings['depthwise']:
		bigExpand = dconv2D(
			x = fire,
			activation = settings['activation'],
			dilation = 1,
			kernel = (3, 3),
			name = f"{name}_bigExpandFire",
			padding = 'same',
			strides = 1
		)
	else:
		bigExpand = conv2D(
			x = fire,
			activation = settings['activation'],
			dilation = 1,
			filters = filters,
			groups = 1,
			kernel = (3, 3),
			name = f"{name}_bigExpandFire",
			padding = 'same',
			strides = 1
		)
	smallExpand = conv2D(
		x = fire,
		activation = settings['activation'],
		dilation = 1,
		filters = filters,
		groups = 1,
		kernel = (1, 1),
		name = f"{name}_smallExpandFire",
		padding = 'same',
		strides = 1
	)
	fire = tf.keras.layers.Concatenate(
		axis = -1,
	)([bigExpand, smallExpand])
	if reduce:
		fire = tf.keras.layers.MaxPool2D(
			name = f"{name}_fire_maxpool2D",
			padding = 'same',
			pool_size = (3, 3),
			strides = 2
		)(fire)
	fire = squeezeExcite(fire, name)
	return fire

### REDUCED FIRENET (DOI:10.1007/S11063-021-10555-1)
def firenet():
	input = tf.keras.layers.Input(
		(settings['inputSize'], settings['inputSize'], settings['bands']),
		name = 'input'
	)
	firenet = input
	if len(settings['means']) and len(settings['variances']):
		firenet = tf.keras.layers.Normalization(
			axis = -1, 
			invert = False,
			mean = settings['means'], 
			variance = settings['variances']
		)(firenet)
	elif settings['inputAdjustment']:
		if settings['softClip']:
			firenet = SoftClipLayer(name = 'softClip')(firenet)
		else:
			firenet = tf.keras.layers.Rescaling(
				name = 'rescale',
				offset = -1,
				scale = 1.0/127.5 
			)(firenet)
	firenet = conv2D(
		x = firenet,
		activation = settings['activation'],
		dilation = 1,
		filters = settings['filters'][0]*2,
		groups = 1,
		kernel = (3, 3),
		name = 'fire0',
		padding = 'same',
		strides = 1
	)
	if settings['zcaConv']:
		firenet = zca(
			x = firenet, 
			name = 'fire0',
			place = 'fire0'
		)
	for k, filters in enumerate(settings['filters']):
		firenet = fireModule(
			filters = filters, 
			fire = firenet, 
			name = f"fire{k+1}", 
			reduce = k < len(settings['filters'])-1
		)
		if k == 1 and settings['dropout']:
			firenet = tf.keras.layers.Dropout(settings['generalizer'])(firenet)
	if settings['extraBottleneck']:
		firenet = conv2D(
			x = firenet,
			activation = settings['activation'],
			dilation = 1,
			filters = settings['filters'][-1],
			groups = 1,
			kernel = (1, 1),
			name = 'terminal_squeezeFire',
			padding = 'same',
			strides = 1
		)
	firenet = gap(
		x = firenet, 
		flatten = True,
		name = 'output'
	)
	if settings['transferLayer']:
		firenet = normalize(
			x = firenet, 
			name = 'output'
		)
	if settings['outputPolish'] > 0:
		firenet = dense(
			x = firenet, 
			activation = settings['activation'], 
			bias = True, 
			name = 'gap_resolve', 
			units = settings['outputPolish']
		)
		if settings['dropout']:
			firenet = tf.keras.layers.Dropout(settings['generalizer'])(firenet)
	if settings['zcaOutput']:
		firenet = zca(
			x = firenet, 
			name = 'output',
			place = 'output'
		)	
	output = dense(
		x = firenet, 
		activation = None, 
		bias = True, 
		name = 'output', 
		units = settings['outputArray'],
		zeros = False if settings['sigmoidLossInit'] else True 
	)
	return tf.keras.Model(inputs = input, outputs = output)

### GLOBAL AVERAGE POOLING
def gap(x, flatten, name):
	return tf.keras.layers.GlobalAveragePooling2D(
		data_format = settings['dformat'], 
		keepdims = not flatten,
		name = f"{name}_gap", 
	)(x)

### LAYER NORMALIZATION
def normalize(x, name):
	return tf.keras.layers.LayerNormalization(
		axis = -1,
		beta_constraint = None,
		beta_initializer = 'zeros',
		beta_regularizer = None,
		center = True,
		epsilon = settings['epsilon'],
		gamma_constraint = None,
		gamma_initializer = 'ones',
		gamma_regularizer = None,
		name = f"{name}_layerNormalization",
		scale = True
	)(x)

### SQUEEZE AND EXCITE
def squeezeExcite(x, name):
	if settings['squeezeExciteChannels'] and settings['squeezeExciteSpatial']:
		skipConnection = x
		x = squeezeExciteSpatial(x, name)
		x += skipConnection
		x = squeezeExciteChannels(x, name)
		x += skipConnection
		return x
	elif settings['squeezeExciteChannels']:
		return squeezeExciteChannels(x, name)
	elif settings['squeezeExciteSpatial']:
		return squeezeExciteSpatial(x, name)
	else:
		return x

### SQUEEZE AND EXCITE CHANNELS (ARXIV:1709.01507)
def squeezeExciteChannels(x, name):
	units = x.shape[CHANNELS]
	se = gap(
		x = x, 
		flatten = False,
		name = f"{name}_squeezeExcite"
	)
	se = dense(
		x = se, 
		activation = settings['activation'], 
		bias = settings['bias'], 
		name = f"{name}_squeeze", 
		units = units//settings['seRatio']
	)
	se = dense(
		x = se, 
		activation = 'sigmoid',
		bias = settings['bias'], 
		name = f"{name}_excite", 
		units = units
	)
	x *= se
	return x

### SQUEEZE AND EXCITE SPATIAL (ARXIV:1709.01507)
def squeezeExciteSpatial(x, name):
	se = conv2D(
		x = x,
		activation = 'sigmoid',
		dilation = 1,
		filters = 1,
		groups = 1,
		kernel = (1, 1),
		name = f"{name}_fuse",
		padding = 'same',
		strides = 1
	)
	x *= se
	return x

### ZCA NORMALIZATION (ARXIV:1904.03441)
def zca(x, name, place):
	group = x.shape[CHANNELS]
	return IterativeNormalization(
		data_format = settings['dformat'],
		epsilon = settings['epsilon'],
		iterations = settings['zcaIterations'],
		membersPerGroup = group if place == 'input' else group//4,
		momentum = 0.9,
		name = f"{name}_zcaBN",
		trainable = True
	)(x)



### OUTPUT
model = firenet()
eprint(model.summary())
model.save(filepath = settings['outFile'], save_format = 'keras')
sys.exit(0)
