import sys

def progress(count, total, epoch, suffix = ''):
	""" Writes a progress bar to console. """
	bar_len = 40
	filled_len = int(round(bar_len * count / float(total)))

	percents = round(100.0 * count / float(total), 1)
	bar = '=' * filled_len + '-' * (bar_len - filled_len)
	# sys.stdout.write('epoch %s:\r' %(epoch))
	sys.stdout.write('epoch %s: [%s] %s%s ...%s\r' % (epoch, bar, percents, '%', suffix))
	# sys.stdout.write('average_loss: %s\r' % (average_loss))
	sys.stdout.flush()  # As suggested by Rom Ruben