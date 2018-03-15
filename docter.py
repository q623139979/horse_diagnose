import sys

attributes = ["K", "Na", "CL", "HCO3", "Endotoxin", "Aniongap", "PLA2",
			  "SDH", "GLDH", "TPP", "Breath rate", "PCV", "Pulse rate",
			  "Fibrinogen", "Dimer", "FibPerDim"]


# consume a filename and return a list of list
def contruct_data(filename):
	data = []
	f = open(filename)
	for l in f:
		line = []
		for x in l.split(",")[0:-1]:
			line.append(float(x))
		line.append((l.split(",")[-1]).split(".")[0])
		data.append(line)
	return data


# count the number of horses in different classification
def classification_count(data):
	count = {}
	for r in data:
		classification = r[-1]
		if classification not in count:
			count[classification] = 1
		else:
			count[classification] += 1
	return count


# the binary test
class Test:
	def __init__(self, col, val):
		self.col = col
		self.val = val

	def match(self, case):
		val = case[self.col]
		if val >= self.val:
			return True
		else:
			return False

	def __repr__(self):
		return "{} >= {}".format(attributes[self.col], self.val)


# using the test to splite the data into two different nodes
def partition(data, test):
	true_node = []
	false_node = []
	for r in data:
		if test.match(r):
			true_node.append(r)
		else:
			false_node.append(r)
	return true_node, false_node


# calculate the impurity of current data
def cal_impurity(data):
	count = classification_count(data)
	impurity = 1
	for classification in count:
		Pi = float(count[classification]) / len(data)
		impurity -= Pi**2
	return impurity


# calculte the information gain of current node
def info_gain(left, right, current):
	p_left = float(len(left)) / (len(left) + len(right))
	p_right = float(len(right)) / (len(left) + len(right))
	IG = current - p_left * cal_impurity(left) - p_right * cal_impurity(right)
	return IG


# choose the best question which will get the largest information gain
def best_test(data):
	best_test = None
	best_gain = 0
	current = cal_impurity(data)
	for col in range(16):
		values = list(set([r[col] for r in data]))
		values.sort()
		for i in range(len(values)-1):
			val = (values[i] + values[i+1]) / 2.0
			test = Test(col, val)
			true_node, false_node = partition(data, test)
			if len(true_node) == 0 or len(false_node) == 0:
				continue
			else:
				gain = info_gain(true_node, false_node, current)
				if gain > best_gain:
					best_test = test
					best_gain = gain
	return best_gain, best_test


class Leaf:
	def __init__(self, data):
		self.predictions = classification_count(data)


class Node:
	def __init__(self, test, true_branch, false_branch):
		self.test = test
		self.true_branch = true_branch
		self.false_branch = false_branch


# build the learning tree
def build_tree(data):
	gain, test = best_test(data)
	tree = None
	if gain == 0:
		tree = Leaf(data)
	else:
		true_part, false_part = partition(data, test)
		true_branch = build_tree(true_part)
		false_branch = build_tree(false_part)
		tree = Node(test, true_branch, false_branch)
	return tree


# print the learning tree
def print_tree(node):
	if isinstance(node, Leaf):
		print("Predict: {}".format(node.predictions))
	else:
		print(node.test)
		print("True_Branch: ")
		print_tree(node.true_branch)
		print("False_Branch: ")
		print_tree(node.false_branch)


# using the learning tree to diagnose a horse
def diagnose(row, node):
	if isinstance(node, Leaf):
		return node.predictions
	if node.test.match(row):
		return diagnose(row, node.true_branch)
	else:
		return diagnose(row, node.false_branch)


# give the the horse's classification with the probability of confidence
def diagnose_prob(diag):
	total = 0
	for v in diag.values():
		total += float(v)
	probs = {}
	for atb in diag.keys():
		probs[atb] = diag[atb] / total
	return probs


def main():
	training_data = contruct_data(sys.argv[1])
	tree = build_tree(training_data)
	print_tree(tree)
	print("\n")
	test_data = contruct_data(sys.argv[2])
	for row in test_data:
		print ("Actual: {}. Predicted: {}".format(row[-1], diagnose_prob(diagnose(row, tree))))


main()
