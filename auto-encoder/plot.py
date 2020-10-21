import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import fetch_mldata
from sklearn.manifold import TSNE
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import sys
from matplotlib import rcParams

random.seed(10)
np.random.seed(13)
plt.rcParams.update({'font.size': 30})
rcParams['legend.fontsize'] = "25"
plt.tick_params(labelsize=13)

weights1=np.load(sys.argv[1])
print(np.shape(weights1))
weights2=np.load(sys.argv[2])
print(np.shape(weights2))

X=np.concatenate((weights1, weights2), axis=0)
print(np.shape(X))

y=[]
for i in range(np.shape(weights1)[0]):
    y.append("Auto-Encoder Regularization")
    
for i in range(np.shape(weights2)[0]):
    y.append("End to End Training")

feat_cols = [ 'weight'+str(i) for i in range(X.shape[1]) ]
df = pd.DataFrame(X,columns=feat_cols)

print('Size of the dataframe: {}'.format(df.shape))

tsne = TSNE(n_components=2, verbose=1, perplexity=130, n_iter=300)
tsne_results = tsne.fit_transform(df[feat_cols].values)
print("t-SNE done")

df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]

y=np.asarray(y).flatten()        
df['y'] = y
df['Label'] = df['y'].apply(lambda i: str(i))
marker_list = ['o', '^']
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="Label",
    style="Label",
    palette=['red','blue'],
    data=df,
    legend="full",
    markers=marker_list,
    s=150
)
plt.ylabel("tSNE axis 2")
plt.xlabel("tSNE axis 1")
plt.ylim(-28, 25)

plt.savefig('tsne_plot.png')