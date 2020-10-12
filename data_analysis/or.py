import os
import random
begin_end = './Record/test'
f = open(begin_end,'a+')

for i in [1,2,3,4,5,6]:
    f.write(str(1))
f.close()

txtName = "./Record/codingWord.txt"
f=open(txtName, "a+")
for i in range(1,100):
    if i % 2 == 0:
        new_context = "C++" + '\n'
        f.write(new_context)
    # else:
    #     new_context = "Python" + '\n'
    #     f.write(new_context)
f.close()
