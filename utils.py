import math
import os.path
import random

from statistics import mean


"""
Input is a string consisting of lines, each line has comma-delimited fields.
Convert this into a list of lists. Blank lines are skipped. Fields that look
like numbers are converted to numbers.
"""
def parseCSV(input, delim = ','):
    lines = [line for line in input.splitlines() if line.strip()]
    return [list(map(numOrStr, line.split(delim))) for line in lines]


"""
Remove duplicate elements from the input. Assumes hashable elements.
"""
def unique(seq):
    return list(set(seq))


"""
The input is a string; convert it to a number if possible, or strip it.
"""
def numOrStr(x):
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return str(x).strip()


"""
Open the CSV file by looking for a directory called 'data' at the current level
that this file (utils.py) is stored.
"""
def openData(name, mode = 'r'):
    root = os.path.dirname(__file__)
    file = os.path.join(root, *[name])
    return open(file, mode = mode)


"""
A dataset for a machine learning problem. It has the following fields:

d.examples   A list of examples. Each one is a list of attribute values.
d.attrs      A list of integers to index into an example, so example[attr]
             gives a value. Normally the same as range(len(d.examples[0])).
d.attrNames  Optional list of mnemonic names for corresponding attrs.
d.target     The attribute that a learning algorithm will try to predict.
             By default the final attribute.
d.inputs     The list of attrs without the target.
d.values     A list of lists: each sublist is the set of possible
             values for the corresponding attribute. If initially None,
             it is computed from the known examples by self.setproblem.
             If not None, an erroneous value raises ValueError.
d.name       Name of the data set (for output display only).
d.exclude    A list of attribute indexes to exclude from d.inputs. Elements
             of this list can either be integers (attrs) or attrnames.

Normally, you call the constructor and you're done; then you just access fields
like d.examples and d.target and d.inputs.
"""
class Dataset:
    def __init__(self, attrs = None, attrNames = None, target = -1,
                 inputs = None, values = None, name = '', source = '',
                 exclude = (), examples = None):
        self.name = name
        self.source = source
        self.values = values
        self.gotValueFlag = bool(values)

        if examples is None:
            self.examples = parseCSV(openData(name + '.csv').read())
        else:
            self.examples = examples

        if self.examples is not None and attrs is None:
            attrs = list(range(len(self.examples[0])))
        self.attrs = attrs

        if isinstance(attrNames, str):
            self.attrNames = attrNames.split()
        else:
            self.attrNames = attrNames or attrs

        self.setProblem(target, inputs = inputs, exclude = exclude)


    """
    Set (or change) the target and/or inputs. This way, one Dataset can be used
    multiple ways. inputs, if specified, is a list of attributes, or specify
    exclude as a list of attributes to not use in inputs. Attributes can be
    -n to n, or an attrname. Also computes the list of possible values, if that
    wasn't done yet.
    """
    def setProblem(self, target, inputs = None, exclude = ()):
        self.target = self.attrNum(target)
        exclude = list(map(self.attrNum, exclude))

        if inputs:
            self.inputs = removeAll(self.target, inputs)
        else:
            self.inputs = [a for a in self.attrs
                           if a != self.target and a not in exclude]

        if not self.values:
          self.updateValues()

        self.validate()


    """
    Updates the values.
    """
    def updateValues(self):
        self.values = list(map(unique, zip(*self.examples)))


    """
    Returns the number used for attr, which can be a name, or -n to n - 1.
    """
    def attrNum(self, attr):
        if isinstance(attr, str):
            return self.attrNames.index(attr)
        elif attr < 0:
            return len(self.attrs) + attr
        else:
            return attr


    """
    Check that the fields make sense.
    """
    def validate(self):
        assert len(self.attrNames) == len(self.attrs)
        assert self.target in self.attrs
        assert self.target not in self.inputs
        assert set(self.inputs).issubset(set(self.attrs))

        if self.gotValueFlag:
            list(map(self.checkExample, self.examples))
