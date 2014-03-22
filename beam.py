#!/opt/python-2.6/bin/python2.6
# Mami Hackl and Nat Byington
# LING 572 HW6 Beam search
# Assign POS tags using a MaxEnt model and beam search algorithm.
# Args: test, boundary, model, sysout, beam_size, topN, topK > acc

import sys
import re
import math

### Classes ###

class Node:
    ''' Store cumulative tag info. '''
    
    def __init__(self, t=[('BOS', 0.0, '', ''), ('BOS', 0.0, '', '')], clp=0.0):
        self.tags = t # a list of (tag, log_prob, instance, true_tag) tuples
        self.c_log_prob = clp # cumulative log_prob
        
    def spawn_node(self, tag, log_prob, instance, true_tag):
        ''' Create and return a new node with added tag info. '''
        t = list(self.tags) # creates copy of self.tags
        t.append((tag, log_prob, instance, true_tag))
        clp = self.c_log_prob + log_prob
        new_node = Node(t, clp)
        return new_node
        
### Main ###

# Get files and values from arguments.
test = open(sys.argv[1])
boundary = open(sys.argv[2])
model_file = open(sys.argv[3])
sys_out = open(sys.argv[4], 'w')
beam_size = int(sys.argv[5])
N = int(sys.argv[6])
K = int(sys.argv[7])

# Create a model hash using model file.
model = {} # hash contains weight values for (feature, class) keys
classes = set() # the set of all classes from model file
features = set() # the set of all features from model file
current_class = ''
class_re = re.compile(r'^FEATURES FOR CLASS ([\S]+)')
feature_re = re.compile(r'^ ([\S]+) ([\S]+)')
for line in model_file.readlines():
    if class_re.match(line):
        current_class = class_re.match(line).group(1)
        classes.add(current_class)
    elif feature_re.match(line):
        feature = feature_re.match(line).group(1)
        weight = float(feature_re.match(line).group(2))
        features.add(feature)
        model[(feature, current_class)] = weight

# Initialize confusion matrix for accuracy output.
MATRIX = dict( [(x, {}) for x in classes] )
for c in classes:
    for c2 in classes:
        MATRIX[c][c2] = 0
        
# Classify POS tag for each word in sentence, outputting to sys_out.
v_count = 0     # total number of vectors
sys_out.write('\n\n%%%%% test data: \n')
for line in boundary.readlines():
    sentence_length = int(line.strip())
    v_count += sentence_length
    topK = [] # list of nodes
    initial_node = Node()
    topK.append(initial_node)
    for i in range(2, sentence_length+2):      
        vector = test.readline()    # reads next line from test data
        instance, true_tag = re.match(r'^([\S]+) ([\S]+) ', vector).group(1,2)
        feats = re.findall(r'([\S]+) [0-9]+', vector)
        new_topK = []
        #print '****** instance ' + instance #debug
        for node in topK:
            Z = 0.0
            results = []
            pt = 'prevT=' + node.tags[i-1][0]
            p2t = 'prevTwoTags=' + node.tags[i-2][0] + '+' + node.tags[i-1][0]
            for c in classes:
                summ = model[('<default>', c)]
                summ += model.get((pt, c), 0.0)
                summ += model.get((p2t, c), 0.0)
                #print 'before feats: ' + str(summ) #debug
                for f in feats:
                    summ += model.get((f, c), 0.0)
                    #print f + ' ' + str(summ) #debug
                result = math.exp(summ)
                #print '&& Result: ' + str(result) + ' ' + c #debug
                Z += result
                results.append((result, c))
            results.sort(reverse=True) 
            results = results[:N] # take top N results
            for r in results:
                log_prob = math.log(r[0] / Z)
                tag = r[1]
                new_node = node.spawn_node(tag, log_prob, instance, true_tag)
                new_topK.append(new_node)
        new_topK.sort(key=lambda obj: obj.c_log_prob, reverse=True) # find node with highest prob
        max_prob = new_topK[0].c_log_prob
        new_topK = new_topK[:K] # prune down to K nodes
        topK = [] # rebuild topK list using beam size
        for node in new_topK:
            if (node.c_log_prob + beam_size) >= max_prob:
                topK.append(node)
    # Output results of top node
    topK.sort(key=lambda obj: obj.c_log_prob, reverse=True)
    best_node = topK[0]
    for i in range(2, sentence_length+2):
        best = best_node.tags[i]
        sys_tag = best[0]
        prob = str(10**best[1])
        instance = best[2]
        true_tag = best[3]
        string = [instance, true_tag, sys_tag, prob, '\n']
        sys_out.write(' '.join(string))
        MATRIX[true_tag][sys_tag] += 1        
          
# Output accuracy results using MATRIX data
sys.stdout.write('class_num=' + str(len(classes)) + ' feat_num=' + str(len(features)) + '\n')
correct = 0.0
print 'Confusion matrix for Test Data:'
print 'row is the truth, column is the system output'
print ''
sys.stdout.write('\t\t')
for c in classes:
    sys.stdout.write(' ' + c)
    correct += MATRIX[c][c]
sys.stdout.write('\n')
for c in classes:
    sys.stdout.write(c)
    for c2 in classes:
        sys.stdout.write(' ' + str(MATRIX[c][c2]))
    sys.stdout.write('\n')
print ''
print 'Test accuracy: ' + str(correct / float(v_count))
print ''
