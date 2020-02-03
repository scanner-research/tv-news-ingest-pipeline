candidates = ['Elizabeth Warren', 'Joe Biden', 'Bernie Sanders', 'Pete Buttigieg',
	           'Amy Klobuchar', 'Tom Steyer']

def read_file(path):
	with open(path) as f:
		return f.read().split("\n")

def get_precision_and_recall(truth, estimate, term):
	false_positives = 0
	false_negatives = 0
	true_positives = 0
	for i, (t, e) in enumerate(zip(truth, estimate)):
		if t == term:
			if e == term:
				true_positives += 1
			else:
				false_negatives += 1
		else:
			if e == term:
				false_positives += 1
	return (
		true_positives / (true_positives + false_positives),
		true_positives / (true_positives + false_negatives)
	)

def get_accuracy(truth, estimate):
	correct = 0
	for i, (t, e) in enumerate(zip(truth, estimate)):
		if t == e:
			correct += 1
		# elif e != '':
		# 	print("{}\t'{}'\t'{}'".format(i, t, e))
	return correct / len(truth)

truth = read_file('ground_truth_labels.txt')

# Number of hours of video time an average of 2 faces per frame (rough estimate)
# times 1200 samples per hour
montages_per_dollar = 1000
total_instances = 287885 * 1.5 * 1200
print("Accuracy")
print("Size\tAzure\tAWS\tPrice Estimate")
for size in ['01x01', '02x02', '04x02', '02x04', '04x04', '08x04', '04x08', '08x08']:
	azure = read_file('{}/azure_labels.txt'.format(size))
	aws = read_file('{}/aws_labels.txt'.format(size))
	faces_per_montage = int(size.split('x')[0]) * int(size.split('x')[1])
	price_estimate = total_instances / faces_per_montage / montages_per_dollar
	print("{}\t{:.2f}\t{:.2f}\t${:.2f}".format(
		size,
		get_accuracy(truth, azure),
		get_accuracy(truth, aws),
		price_estimate
	))
print()

size = '08x04'
for estimator in ['azure', 'aws']:
	estimate = read_file('{}/{}_labels.txt'.format(size, estimator))
	print(estimator.upper())
	print("Candidate\tPrecsn\tRecall\t(n)")
	for candidate in candidates:
	    precision, recall = get_precision_and_recall(truth, estimate, candidate)
	    n = sum([t == candidate for t in truth])
	    print("{}\t{:.2f}\t{:.2f}\t{}".format(candidate[:10], precision, recall, n))
	print()

