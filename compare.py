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

def get_percent_labeled_correctly(truth, estimate):
	correct = 0
	incorrect = 0
	for t, e in zip(truth, estimate):
		if t == e:
			correct += 1
		else:
			if t in candidates or e in candidates:
				incorrect += 1
	return (correct / (correct + incorrect))

truth = read_file('ground_truth_labels.txt')
print("True Positive / (True Positive + False Negative + False Positive)")
print("Size\tAzure\tAWS")
for size in ['01x01', '02x02', '04x04', '04x08', '08x04', '08x08']:
	azure = read_file('{}/azure_labels.txt'.format(size))
	aws = read_file('{}/aws_labels.txt'.format(size))
	print("{}\t{:.2f}\t{:.2f}".format(
		size,
		get_percent_labeled_correctly(truth, azure),
		get_percent_labeled_correctly(truth, aws)
	))
print()

size = '04x04'
for estimator in ['azure', 'aws']:
	estimate = read_file('{}/{}_labels.txt'.format(size, estimator))
	print(estimator)
	print("Candidate\tPrecsn\tRecall\t(n)")
	for candidate in candidates:
	    precision, recall = get_precision_and_recall(truth, estimate, candidate)
	    n = sum([t == candidate for t in truth])
	    print("{}\t{:.2f}\t{:.2f}\t{}".format(candidate[:14], precision, recall, n))
	print()

