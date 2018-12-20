from utils import *


"""
A leaf node of a decision tree which holds just a result.
"""
class DTLeaf:
    def __init__(self, result):
        self.result = result

    def __call__(self, example):
        return self.result

    def display(self, indent = 0):
        print('Result =', self.result)


"""
An internal node of a decision tree which tests an attribute, along with a
dictionary of branches, one for each of the attribute's values.
"""
class DTInternal:
    """
    Initialize by saying what attribute this node tests.
    """
    def __init__(self, attr, attrName = None, defaultChild = None,
                 branches = None):
        self.attr = attr
        self.attrName = attrName or attr
        self.defaultChild = defaultChild
        self.branches = branches or {}


    """
    Given an example, classify it using the attribute and the branches.
    """
    def __call__(self, example):
        attrVal = example[self.attr]
        if attrVal in self.branches:
            return self.branches[attrVal](example)
        else:
            return self.defaultChild(example)


    """
    Add a branch.
    """
    def add(self, val, subtree):
        self.branches[val] = subtree


    """
    Prints out the decision tree starting at this internal node using recursion.
    """
    def display(self, indent = 1):
        print ('Testing attribute:', self.attrName)
        for (val, subtree) in self.branches.items():
            print(' ' * 4 * indent, self.attrName, '=', val, '==>', end = ' ')
            subtree.display(indent + 1)
        print()


"""
Multiply each number by a constant such that the sum is 1.0
"""
def normalize(dist):
    if isinstance(dist, dict):
        total = sum(dist.values())
        for key in dist:
            dist[key] = dist[key] / total
            assert 0 <= dist[key] <= 1, 'Probabilities must be between 0 and 1.'
        return dist
    total = sum(dist)
    return [(n / total) for n in dist]


"""
Return a copy of the input with all occurrences of item removed.
"""
def removeAll(item, seq):
    if isinstance(seq, str):
        return seq.replace(item, '')
    else:
        return [x for x in seq if x != item]


"""
Randomly shuffle a copy of the input.
"""
def shuffled(seq):
    items = list(seq)
    random.shuffle(items)
    return items


"""
Return an element with the highest value according to key, which is a function.
We break ties by first shuffling the input.
"""
def argmaxRandomTie(seq, key):
    return max(shuffled(seq), key = key)


"""
Count the number of examples that have example[attr] = val.
"""
def count(attr, val, examples):
    return sum(e[attr] == val for e in examples)


"""
[pluralityVal(examples, values, target)] is the most commonly occuring value of the target attribute
[target] in [examples]. The values corresponding to [target] is the set [values[target]].
"""
def pluralityVal(examples, values, target):
    plurality_vals = {}
    for val in values[target]: 
        plurality_vals[val] = 0 
    for val,freq in enumerate(plurality_vals):
        for i in range(len(examples[0])):
            plurality_vals[val] += count(i,val,examples)
    pval_tups = [(v,plurality_vals[v]) for i,v in enumerate(plurality_vals)]
    pval = (argmaxRandomTie(pval_tups, lambda e:e[1]))
    return DTLeaf(pval[0])
        


# Test for pluralityVal
te = [[3,2,1], [1,0,1], [4,3,2], [3,0,0], [5,2,1]]
tv = [[1,2,3,4,5], [0,1,2,3], [0,1,2]]
tt = 2
assert pluralityVal(te, tv, tt).result == 1, 'Failed pluralityVal test'


"""
[allSameClass(examples,target)] returns True if for all examples in [examples]
example[[target]] is equal to some value v. Returns False otherwise.
"""
def allSameClass(examples, target):
    classification = None
    for example in examples:
        if classification == None:
            classification = example[target]
        elif example[target] != classification: 
            return False
    return True

# Tests for allSameClass
te = [[3,2,1], [1,0,1], [4,3,2], [3,0,0], [5,2,1]]
tt = 1
assert not allSameClass(te, tt), 'Failed allSameClass test #1'
te = [[3,0,1], [1,0,1], [4,0,2], [3,0,0], [5,0,1]]
assert allSameClass(te, tt), 'Failed allSameClass test #2'


"""
[entropy(values)] is the entropy value of a given set of values [values].
Preconditons: [values] is a list of nonnegative integers.
Represents the function - Sigma[P(vk)log2(P(vk))] where P(vk) is the probability of 
value vk
"""
def entropy(values):
    if len(values) == 0: 
        return 0
    pvals = normalize(values)
    return -1*sum([(v*math.log2(v) if v != 0 else 0) for v in pvals])


# Tests for entropy
tv = [13, 5, 2, 20, 4, 10, 4]
assert abs(entropy(tv) - 2.4549947941466774) <= 1e-3, 'Failed entropy test #1'
tv = [0, 0, 5, 0, 0, 0, 0]
assert entropy(tv) == 0.0, 'Failed entropy test #2'


"""
Return a list of (value, examples) pairs for each val of attr.
"""
def splitBy(attr, examples, values):
    return [(v, [e for e in examples if e[attr] == v]) for v in values[attr]]


"""
[infoGain(attr, examples, values, target)] is the value of the information gain from 
splitting on attribute [attr] using [examples] [values] and [target]. It is determined
by subtracting the sum of entropies for the sets that result from splitting [examples] on the 
attribute [attr] from the entropy of [examples]
"""
def infoGain(attr, examples, values, target):
    def calculateEntropy(target, values, example):
        freq = []
        for v in values[target]:
            freq.append(count(target, v, example))
        return entropy(freq) 
    
    split_vals = splitBy(attr, examples, values)
    gain = 0
    for (v,split_e) in split_vals: 
        if (len(split_e) != 0):
            gain += (len(split_e)/len(examples))*calculateEntropy(target,values,split_e)
    return calculateEntropy(target,values,examples) - gain
    

# Tests for infoGain
te = [[3,2,1], [1,0,1], [4,3,2], [3,0,0], [5,2,1]]
tv = [[1,2,3,4,5], [0,1,2,3], [0,1,2]]
tt = 2
ta = 0
assert abs(infoGain(ta, te, tv, tt) - 0.9709505944546687) <= 1e-3,        'Failed infoGain test #1'
tt = 1
assert abs(infoGain(ta, te, tv, tt) - 1.121928094887362) <= 1e-3,        'Failed infoGain test #2'


"""
[chooseAttr(attrs, examples, values, target)] is the attribute from [attrs] that it is 
most beneficial to split [examples] on according to the highest information gain produced from that
attribute on [examples] [values] and [target]. 
"""
def chooseAttr(attrs, examples, values, target):
    best_gain = None
    best_attr = None
    for attr in attrs: 
        gain = infoGain(attr, examples, values, target)
        if best_gain is None or gain > best_gain: 
            best_gain = gain
            best_attr = attr
    assert(best_attr is not None)
    return best_attr 


# Test for chooseAttr
te = [[6,7,8,7,0], [7,3,4,0,0], [1,0,2,3,3], [8,5,9,1,1], [9,1,5,4,2],
      [9,5,6,8,3], [4,6,0,1,6], [8,5,4,4,2], [0,3,7,2,6], [7,9,3,1,2]]
tv = [[0,1,2,3,4,5,6,7,8,9] for i in range(10)]
tt = 3
ta = [0,1,2,4]
assert chooseAttr(ta, te, tv, tt) == 2, 'Failed chooseAttr test'

"""
[DTLearner(examples, attrs, values, target, attrNames, parentExamples)] is the tree 
representing the result of applying a greedy selection learning function to a set of 
[examples] with attributes [attrs], [values] and with a target attribute [target].
"""
def DTLearner(examples, attrs, values, target, attrNames, parentExamples = ()):
    if not examples: 
        return DTLeaf(pluralityVals(parentExamples, values, target))
    elif allSameClass(examples, target): 
        return DTLeaf(examples[0][target]) 
    elif not attrs: 
        return DTLeaf(pluralityVals(examples, values, target))
    else:
        to_split = chooseAttr(attrs, examples, values, target)
        attr_values = splitBy(to_split, examples, values)
        node = DTInternal(to_split)
        attrs.remove(to_split)
        for v,e in attr_values:
            node.add(v, DTLearner(e,attrs,values, target, attrNames))
        return node
        


"""
Test for DTLearner

Output should match:

Testing attribute: 7
     7 = 0 ==> Result = 1
     7 = 1 ==> Result = 0
     7 = 2 ==> Testing attribute: 11
         11 = 0 ==> Result = 0
         11 = 1 ==> Result = 2
         11 = 2 ==> Result = 1
"""
te = [[1,1,1,1,1,1,2,2,2,0,2,0,0,1,2,0,0,2,2,0],
      [2,0,1,2,1,1,1,2,2,2,2,1,0,2,2,0,2,2,2,2],
      [1,2,1,1,1,2,2,2,0,1,2,2,2,2,2,1,2,0,2,1],
      [2,2,2,1,0,1,2,2,2,0,2,1,0,1,1,1,0,2,0,2],
      [0,1,1,0,0,0,2,0,1,0,1,2,2,2,2,0,0,0,1,1],
      [0,2,0,0,0,1,0,1,0,2,1,1,2,0,2,2,0,2,0,0],
      [0,0,1,2,0,0,0,0,1,2,0,0,2,0,0,0,0,2,1,1],
      [0,2,2,1,1,0,0,2,2,0,2,1,1,0,0,2,0,2,1,2],
      [1,0,2,0,1,2,2,1,0,1,0,2,1,2,0,0,1,1,2,0],
      [0,2,0,1,2,1,1,1,1,0,1,2,2,0,1,2,1,0,0,0]]
tv = [[j for j in range(3)] for i in range(20)]
tt = 19
ta = [i for i in range(19)]
tn = ta
DTLearner(te, ta, tv, tt, tn).display()

"""
Front-end caller for the decision tree learner. Given an input dataset, it will
extract the necessary information and pass it in to the learner.
"""
def DTLCaller(dataset):
    return DTLearner(dataset.examples, dataset.inputs, dataset.values,
                     dataset.target, dataset.attrNames)


# Load the restaurant dataset from a CSV file
restaurant = Dataset(name = 'restaurant', target = 'Wait',
                     attrNames = 'Alternate Bar Fri/Sat Hungry Patrons Price ' +
                     'Raining Reservation Type WaitEstimate Wait')


# Feed the dataset into the decision tree learner
dtlRestaurant = DTLCaller(restaurant)


# Display the decision tree that is learned
dtlRestaurant.display()


# Try classifying new test data
print(dtlRestaurant(['Yes','No','No','Yes','Full',
                     '$$','No','No','Italian','0-10']))


# We can also use additional datasets. For example, the code block below uses a zoo dataset to train our learner and find a decision tree for our data. The zoo dataset contains a list of animals, each of which has different features about the animal, along with what kind of animal it is. We use our learned decision tree to determine what kind of animal a dragonfly is. Please ensure you have the *zoo.csv* in the same directory as your Jupyter Notebook.

# In[13]:


# Load the zoo dataset from a CSV file
zoo = Dataset(name = 'zoo', target = 'type', exclude = ['name'],
              attrNames = 'name hair feathers eggs milk airborne aquatic ' +
              'predator toothed backbone breathes venomous fins legs tail ' +
              'domestic catsize type')


# Feed the zoo dataset into the decision tree learner
dtlZoo = DTLCaller(zoo)


# Display the decision tree that is learned
dtlZoo.display()


# Try classifying new test data
print(dtlZoo(['dragonfly',0,0,1,0,1,0,1,0,0,1,0,0,6,0,0,0]))
