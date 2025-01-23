### 
### Iterative Normalization
### https://openaccess.thecvf.com/content_CVPR_2019/papers/Huang_Iterative_Normalization_Beyond_Standardization_Towards_Efficient_Whitening_CVPR_2019_paper.pdf
###
### modified from https://github.com/bhneo/decorrelated_bn/blob/master/common/normalization.py
###

import tensorflow as tf

class IterativeNormalization(tf.keras.layers.Layer):
	def __init__(
		self, 
		data_format = 'channels_last',
		epsilon = 1e-7,
		iterations = 3,
		membersPerGroup = 0,
		momentum = 0.9,
		**kwargs  
	):
		super().__init__(**kwargs)
		self.data_format = data_format
		self.epsilon = epsilon
		self.iterations = iterations
		self.membersPerGroup = membersPerGroup
		self.momentum = momentum

	def _movingMatrixInitializer(self, shape, dtype = tf.float32, partition_info = None):
		movingConvs = []
		for _ in range(shape[0]):
			moving_conv = tf.expand_dims(tf.eye(shape[1], dtype = dtype), 0)
			movingConvs.append(moving_conv)
		movingConvs = tf.concat(movingConvs, 0)
		return movingConvs
	
	def build(self, input_shape):
		assert self.data_format == 'channels_last'
		self.inShape = input_shape
		axis = len(input_shape)-1
		dimensions = input_shape[axis]
		if dimensions is None:
			raise ValueError(f"Axis {axis} of input tensor should have a defined dimension but the layer received an input with shape {input_shape}.")
		if self.membersPerGroup == 0:
			self.membersPerGroup = dimensions
		self.group = dimensions//self.membersPerGroup
		assert (dimensions%self.membersPerGroup == 0), f"dimensions is {dimensions}, m is {self.membersPerGroup}"
		self.movingMean = self.add_weight(
			initializer = 'zeros',
			shape = (dimensions, 1),
			trainable = False
		)
		self.movingMatrix = self.add_weight(
			shape = (self.group, self.membersPerGroup, self.membersPerGroup),
			initializer = self._movingMatrixInitializer,
			trainable = False
		)
		learnableWeightsShape = [dimensions if i == axis else 1 for i in range(len(input_shape))]
		self.gamma = self.add_weight(
			constraint = None,
			initializer = 'ones',
			regularizer = None,
			shape = learnableWeightsShape,
			trainable = True
		)
		self.beta = self.add_weight(
			constraint = None,
			initializer= 'zeros',
			regularizer = None,
			shape = learnableWeightsShape,
			trainable = True
		)

	def call(self, inputs, training = False):

		def groupCOV(f, b, w, h, c):
			ff = []
			for i in range(self.group):
				begin = i*self.membersPerGroup
				end = tf.math.reduce_min(((i+1)*self.membersPerGroup, c))
				centered = f[begin:end, :]
				ff_apr = tf.matmul(centered, centered, transpose_b = True)
				ff_apr = tf.expand_dims(ff_apr, 0)
				ff.append(ff_apr)
			ff = tf.concat(ff, 0)
			ff /= tf.cast(b*w*h, tf.float32)-1.0
			return ff

		def _invertedSquare(matrix): ### iterative normalization decompose function
			trace = tf.linalg.trace(matrix)
			trace = tf.expand_dims(trace, [-1])
			trace = tf.expand_dims(trace, [-1])
			sigmaNorm = matrix/trace
			projection = tf.eye(self.membersPerGroup)
			projection = tf.expand_dims(projection, 0)
			projection = tf.tile(projection, [self.group, 1, 1])
			for _ in range(self.iterations):
				projection = (3 * projection - tf.matmul(tf.matmul(tf.matmul(projection, projection), projection), sigmaNorm)) / 2
			return projection/tf.sqrt(trace)
		
		def _whileTest():
			return (1.0 - self.epsilon) * self.movingMatrix + tf.eye(self.membersPerGroup) * self.epsilon

		def _whileTrain(f, m, b, w, h, c):
			ff = groupCOV(f, b, w, h, c)
			ff = (1.0 - self.epsilon) * ff + tf.expand_dims(tf.eye(self.membersPerGroup) * self.epsilon, 0)
			whiteningMatrix = _invertedSquare(ff)
			self.movingMean.assign_sub((self.movingMean - m) * (1.0 - self.momentum))
			self.movingMatrix.assign_sub((self.movingMatrix - whiteningMatrix) * (1.0 - self.momentum))
			return whiteningMatrix

		### init
		shape = self.inShape
		if len(shape) == 4:
			w, h, c = shape[1:]
		elif len(shape) == 2:
			w, h, c = 1, 1, shape[-1]
			inputs = tf.expand_dims(inputs, 1)
			inputs = tf.expand_dims(inputs, 1)
		else:
			raise ValueError(f"shape not support: {shape}")
		b = tf.shape(inputs)[0]

		### center
		xTranspose = tf.transpose(inputs, (3, 0, 1, 2))
		xFlattened = tf.reshape(xTranspose, (c, -1))
		m = tf.reduce_mean(xFlattened, axis = 1, keepdims = True)
		m = m if training else self.movingMean
		f = xFlattened-m
	
		### whitening
		invertedSquare = _whileTrain(f, m, b, w, h, c) if training else _whileTest()
		f = tf.reshape(f, [self.group, self.membersPerGroup, -1])
		fHat = tf.matmul(invertedSquare, f)
		decorelated = tf.reshape(fHat, (c, b, w, h))
		decorelated = tf.transpose(decorelated, [1, 2, 3, 0])
		if w == 1:
			decorelated = tf.squeeze(decorelated, 1)
		if h == 1:
			decorelated = tf.squeeze(decorelated, 1)
		scale = tf.cast(self.gamma, inputs.dtype)
		decorelated *= scale
		offset = tf.cast(self.beta, inputs.dtype)
		decorelated += offset
		return decorelated