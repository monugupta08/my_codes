from sklearn.features_extraction.text import TfidfVectorizer
from sklearn.features_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
import os
import fnmatch

input_data_path='data'
data=[]
for dirpath, dirs, files in os.walk(input_data_path):
	for filename in fnmatch.filter(files, '*.txt'):
        	fp=open(os.path.join(dirpath,filename))
        	lines=fp.readlines()
        	content=''
        	for line in lines:
        		content+=line[:-1]
        	fp.close()
        	data.append(content)
vectorizer=TfidfVectorizer(max_df=1.0, max_features=100,
stop_words='english', use_idf=True)
X=vectorizer.fit_transform(data)
km=KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=100, n_init=1,verbose=1)
print "clustering data with %s" % km
res=km.fit(X)
print "Labels: %s" % km.labels_
print "cluster centers: %s" % km.cluster_centers.squeeze()
