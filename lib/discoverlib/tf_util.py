import numpy

def apply_conv(session, m, im, size=2048, stride=1024, scale=1, channels=1, outputs=None):
	if outputs is None:
		outputs = m.outputs

	output = numpy.zeros((im.shape[0]/scale, im.shape[1]/scale, channels), dtype='float32')
	for x in range(0, im.shape[0] - size, stride) + [im.shape[0] - size]:
		for y in range(0, im.shape[1] - size, stride) + [im.shape[1] - size]:
			conv_input = im[x:x+size, y:y+size, :].astype('float32') / 255.0
			conv_output = session.run(outputs, feed_dict={
				m.is_training: False,
				m.inputs: [conv_input],
			})[0, :, :, :]
			startx = size / 2 - stride / 2
			endx = size / 2 + stride / 2
			starty = size / 2 - stride / 2
			endy = size / 2 + stride / 2
			if x == 0:
				startx = 0
			elif x >= im.shape[0] - size - stride:
				endx = size
			if y == 0:
				starty = 0
			elif y >= im.shape[1] - size - stride:
				endy = size
			output[(x+startx)/scale:(x+endx)/scale, (y+starty)/scale:(y+endy)/scale, :] = conv_output[startx/scale:endx/scale, starty/scale:endy/scale, :]
	return output
