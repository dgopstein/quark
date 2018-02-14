# Take the output of paragraph-vectors and plot its principle dimensions

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

from tsne import bh_sne

import pandas as pd
import numpy as np
import seaborn as sns

saved_model = 'nginx_source_model.dbow_numnoisewords.2_vecdim.100_batchsize.32_lr.0.001000_epoch.93_loss.0.829367.csv'
df=pd.read_csv(saved_model, sep=',')
labels = df.ix[:,0].tolist()
data = df.ix[:,1:]

dirs = [l.split("/")[2] for l in labels]
dirs = np.array(dirs).reshape(len(dirs))

color_map = dict(zip(set(dirs), sns.color_palette()))


sne2 = bh_sne(data, d=2)
plt.title("Learned Dimensions of Source by Module")
plt.scatter(sne2[:,0],sne2[:,1], color=[color_map[d] for d in dirs])
plt.legend([mpatches.Rectangle((0,0),1,1,fc=color) for color in color_map.values()],
           color_map.keys(), loc=4, title="module")
plt.axis((-10, 15, -10, 10))
plt.show()

#for i in [0,1,2,50,51,52,200,201,202]:
#    plt.annotate(str(i)+": "+labels[i], xy=(vis_x[(i)], vis_y[(i)]))


sne3 = bh_sne(data, d=3)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.text2D(0.05, 0.95, "Learned Dimensions of Source by Module", transform=ax.transAxes)
ax.scatter(sne3[:,0], sne3[:,1], sne3[:,2], color=[color_map[d] for d in dirs])
ax.legend([mpatches.Rectangle((0,0),1,1,fc=color) for color in color_map.values()],
           color_map.keys(), loc=4, title="module")
plt.show()
