
def read_file(path):
	with open(path) as f:
		return f.read().split("\n")

def get_precision_and_recall(truth, estimate, term):
	false_positives = 0
	false_negatives = 0
	true_positives = 0
	for t, e in zip(truth, estimate):
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
		elif e != "":
			incorrect += 1
	return (correct / len(truth), incorrect / len(truth))

# options = ['Elizabeth Warren', 'Joe Biden', 'Bernie Sanders', 'Peter Buttigieg',
#            'Amy Klobuchar', 'Tom Steyer']

truth = read_file('ground_truth_labels.txt')
print("Size\tAzure\tAWS")
for size in ['01x01', '02x02', '04x04', '08x08']:
	azure = read_file('{}/azure_labels.txt'.format(size))
	aws = read_file('{}/aws_labels.txt'.format(size))
	print("{}\t{:.2f}\t{:.2f}".format(
		size,
		get_percent_labeled_correctly(truth, azure)[0],
		get_percent_labeled_correctly(truth, aws)[0]
	))
