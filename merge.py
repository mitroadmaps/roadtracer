import numpy
from PIL import Image
import scipy.ndimage
import sys

path = sys.argv[1]
region = sys.argv[2]
x_start = int(sys.argv[3])
y_start = int(sys.argv[4])
x_end = int(sys.argv[5])
y_end = int(sys.argv[6])
out_fname = sys.argv[7]

x_len = x_end - x_start
y_len = y_end - y_start

merged_im = numpy.zeros((x_len * 4096, y_len * 4096, 3), dtype='uint8')

for i in xrange(x_len):
	for j in xrange(y_len):
		fname = '{}/{}_{}_{}_sat.png'.format(path, region, x_start + i, y_start + j)
		merged_im[i*4096:(i+1)*4096, j*4096:(j+1)*4096, :] = scipy.ndimage.imread(fname)[:, :, 0:3].swapaxes(0, 1)

Image.fromarray(merged_im.swapaxes(0, 1)).save(out_fname)
