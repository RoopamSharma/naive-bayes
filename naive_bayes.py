from collections import Counter
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
import re
import os
import time
from os.path import join
"""
Uncomment to download the package if not installed 
import nltk
nltk.download('punkt')
nltk.download('stopwords')
"""
start = time.clock()
parent_dir = '20_newsgroups'
dirs = os.listdir(parent_dir)
#stemmer = PorterStemmer()
train_files = {}
vocab = set()

# reading the set of files in each directory
for d in dirs:
    train_files[d] = os.listdir(join(parent_dir,d))
test_files = {}

# splitting the files into test and train data
for k in train_files.keys():
    test_files[k] = train_files[k][int(0.5*len(train_files[k])):]
    train_files[k] = train_files[k][:int(0.5*len(train_files[k]))]
    
total = 0
doc_words = {}
doc_words_count = {}

# computing the vocab and count of word in each class type
for k,v in train_files.items():
    total += len(train_files[k])
    doc_words[k] = []   
    for doc in v:
        with open(join(parent_dir,k,doc), 'r') as file:
            raw_data=file.read().replace('\n', ' ')
        for i in "~`!@#$%^&*()_-+=|\}}{[:;\"'<,>.?/":
            raw_data = raw_data.replace(i," ")
        pre_data = re.findall(r"[\w']+",raw_data)
        
        #stop_words = stopwords.words('english')
        #data = [stemmer.stem(w) for w in pre_data if not w.lower() in stop_words and w.isalpha() and len(w)>3]
        data = [w for w in pre_data if w.isalpha()]
        doc_words[k].extend(data)
    cnt = Counter(doc_words[k])
    doc_words_count[k] = cnt    
    vocab.update(doc_words[k])    
#    tmp = {}
#    for key in cnt.keys():
#        if cnt[key]>5:
#            tmp[key] = cnt[key]
#    vocab.update(tmp.keys())
    print("Time since start", time.clock()-start,"iteration",len(doc_words))       


# first find the correct vocabulary for the data
# size 97822       

P_Y = {}
# need to find probability of each document type i.e. P(Y)
for k,v in train_files.items():
    P_Y[k] = len(train_files[k])/total

print("P_Y calculations completed ")    

train_start = time.clock()        
# then need to find probability of each word occuring in a given document type for each type P(X/Y)
P_X_Y = {}
for wclass,values in doc_words.items():
    for word in vocab:
        c = 0.1
        if word in doc_words_count[wclass]:
            c = doc_words_count[wclass][word]
        P_X_Y[(word,wclass)] = c/len(values)
    print("Class: ",wclass)

print("Training time ", time.clock()-train_start)               
# lastly find the argmax of the multiplication of the above two value that will give the probability of the class it most closely belongs to P(Y/X) 
def findmax(d):
    max_val = 0
    max_key = 0
    for key,val in d.items():
        if val>=max_val:
            max_val = val
            max_key = key
    return max_key
 

# classify test data
test_start = time.clock()
res = []
for k,v in test_files.items():
    for doc in v:
        r = {}
        data = []        
        with open(join(parent_dir,k,doc), 'r') as file:
            raw_data=file.read().replace('\n', ' ')
            for i in "~`!@#$%^&*()_-+=|\}}{[:;\"'<,>.?/":
                raw_data = raw_data.replace(i," ")
            pre_data = re.findall(r"[\w']+",raw_data)
            #stop_words = stopwords.words('english')
            #data = [stemmer.stem(w) for w in pre_data if not w.lower() in stop_words and w.isalpha() and len(w)>3]
            data = [w for w in pre_data if w.isalpha()]
        for wclass,values in doc_words.items():
            r[wclass] = P_Y[wclass]
            for word in set(data):
                if not (word,wclass) in P_X_Y:
                    # to keep the probability in range multiply with 1000
                    r[wclass] *= 0.1/len(doc_words[wclass])*1000
                else:
                    # to keep the probability in range multiply with 1000
                    r[wclass] *= P_X_Y[(word,wclass)]*1000
        m = findmax(r)
        res.append((r,k,doc,m))
        if len(res)%500==0:
            print("Example: ",len(res))

print("Test time: ",time.clock()-test_start)     

# compute accuracy
accuracy = 0
for r,k,doc,m in res:
    if k==m:
        accuracy+=1
accuracy = accuracy/len(res)*100
print("Accuracy: ",accuracy)        
print("Total time: ",time.clock()-start)