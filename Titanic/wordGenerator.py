__author__ = 'iwitaly'
import string
import random

f = open('words.txt', 'w')

def id_generator(chars=string.ascii_lowercase):
    size = random.choice(range(5, 15))
    return ''.join(random.choice(chars) for _ in range(size)), random.choice(range(size))

for i in xrange(1000000):
    result = id_generator()
    s = str(result[0]) + ' ' + str(result[1]) + '\n'
    f.write(s)