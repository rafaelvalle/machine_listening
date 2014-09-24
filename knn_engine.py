"""
KNN related methods
"""
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from time import time
import numpy as np
import pylab as plt

from scipy import stats

from sklearn import metrics, neighbors
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.datasets.samples_generator import make_blobs
from sklearn.decomposition import PCA

from pre_processing import standardize, normalize, pre_process
from machine_learning_tools import compute_residuals_and_rsquared
from data_mining import *

try:
  import rpy2.robjects as robjects
  robjects.r('library(LambertW)')
except Exception, e:
  print 'error trying to import rpy2.robjects and loadling the LambertW library'
  print e

#These drive me crazy!!!
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def set_trace():
  from IPython.core.debugger import Pdb
  import sys
  Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

####################
#  KNN ESTIMATORS  #
####################
pca = None
estimatorK = None
estimatorR = None
estimatorP = None
n_init = 20

#############
#  METHODS  #
#############
def bench_k_means(estimator, name, data, target_labels, sample_size):
  """For benchmarking K-Means estimators. Prints different clustering metrics and train accuracy
  ARGS
    estimator: K-Means clustering algorithm <sklearn.cluster.KMeans>
    name: estimator name <str>
    data: array-like or sparse matrix, shape=(n_samples, n_features)
    target_labels: labels of data points <number array>
    sample_size: size of the sample to use when computing the Silhouette Coefficient <int>
  """ 
  t0 = time()
  estimator.fit(data)

  _, _, train_accuracy = compute_residuals_and_rsquared(estimator.labels_, target_labels)

  print('% 9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
        % (name, (time() - t0), estimator.inertia_,
           metrics.homogeneity_score(target_labels, estimator.labels_),
           metrics.completeness_score(target_labels, estimator.labels_),
           metrics.v_measure_score(target_labels, estimator.labels_),
           metrics.adjusted_rand_score(target_labels, estimator.labels_),
           metrics.adjusted_mutual_info_score(target_labels,  estimator.labels_),
           metrics.silhouette_score(data, estimator.labels_,metric='euclidean',sample_size=sample_size),
           train_accuracy
          )
        )

def estimate_metrics(data, target_labels):
  """Instantiates KMeans clusters, fits the data and prints information for benchmarking
    ARGS
      data: array-like or sparse matrix, shape=(n_samples, n_features)
      target_labels: labels of data points <number array>
    """
  global estimatorK, estimatorR, estimatorP, pca

  n_samples, n_features = data.shape

  classes = np.unique(target_labels)
  n_clusters = len(classes)
  sample_size = 300

  pca = PCA(n_components=n_clusters).fit(data)
  estimatorK = KMeans(init='k-means++', n_clusters=n_clusters, n_init=n_init)
  estimatorR = KMeans(init='random', n_clusters=n_clusters, n_init=n_init)
  estimatorP = KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1)

  print("n_clusters and unique labels: %d, \t n_samples %d" % (n_clusters, n_samples))
  print('% 10s' % 'init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette\tR-squared')
  bench_k_means(estimatorK,name="k-means++", data=data, target_labels=target_labels,sample_size=sample_size)
  bench_k_means(estimatorR,name="random", data=data, target_labels=target_labels,sample_size=sample_size)

  # in this case the seeding of the centers is deterministic
  pca = PCA(n_components=n_clusters).fit(data)
  bench_k_means(estimatorP,name="PCA-based",data=data, target_labels=target_labels,sample_size=sample_size)

def plot_knn_mesh(data, target_labels, cut='none', pre_processing='none'):
  """Plots a KNN Mesh
    ARGS
      data: array-like or sparse matrix, shape=(n_samples, n_features)
      target_labels: labels of data points <number array>
      cut: suffix to filename to indicate if data was prunned <str>
      pre_processing: suffix to filename to indicate pre-processig applied <str>
  """
  if data.shape[1] < 2:
    raise Exception("plot_knn_mess: at least two features are required, got %s" % str(data.shape))

  n_clusters = len(np.unique(target_labels))
  possible_weights = ['uniform', 'distance']
  n_weights = len(possible_weights)
  n_samples, n_features = data.shape
  n_neighbors = 15

  h = .1  # step size in the mesh

  # Create color maps
  cmap_light = plt.cm.get_cmap('Accent')
  cmap_bold = plt.cm.get_cmap('Accent')

  for kdx in range(n_weights):
    for idx in np.arange(0, n_features - n_features % 2, 2):
      X = data[:,idx:idx+2]
      weight = possible_weights[kdx]
      # we create an instance of Neighbours Classifier and fit the data.
      clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight, algorithm='ball_tree')
      clf.fit(X, target_labels)

      plt.figure(kdx + idx/2*n_weights)

      # Plot the decision boundary. For that, we will assign a color to each
      # point in the mesh [x_min, x_max] [y_min, y_max].
      feature_arange = []
      for feature_idx in range(X.shape[1]):
        x_min, x_max = X[:, feature_idx].min() - 1, X[:, feature_idx].max() + 1
        feature_arange.append(np.arange(x_min, x_max, h))

      xx, yy = np.meshgrid(feature_arange[0], feature_arange[1])

      Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

      # Put the result into a color plot
      Z = Z.reshape(xx.shape)
      plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

      # Plot also the training points 
      plt.scatter(X[:, 0], X[:, 1], c=target_labels, cmap=cmap_bold)
      plt.xlim(xx.min(), xx.max())
      plt.ylim(yy.min(), yy.max())
      plt.legend()
      plt.title("Classification (k = %d, weights = '%s', features = %d %d)" % (n_neighbors, possible_weights[kdx], idx, idx+1))
      plt.savefig('knn_mesh_weight<%s>classes<%d>cut<%s>pre_processing<%s>mfccs<%d-%d>' % (weight, n_clusters, cut, pre_processing, idx, idx+1))
    plt.show()

def plot_nearest_k_neighbors_2d(data, target_labels, n_neighbors = 2):
  """Plots the N-Neighbors of points related to all target labels
    ARGS
      data: array-like or sparse matrix, shape=(n_samples, n_features)
      target_labels: labels of data points <number array>
      n_neighbors: number of neughbors to plot <int>
  """
  possible_weights = ['uniform', 'distance']
  n_weights = len(possible_weights)
  n_samples, n_features = data.shape
  n_folds = 2
  labels = np.unique(target_labels)
  n_labels = len(labels)
  sample_size = 300

  print 'Sample size is', len(target_labels)
  folds = KFold(len(target_labels), n_folds=n_folds) #shuffle=True, random_state=4

  #for plotting
  fig_num = 1

  for train_index, test_index in folds:
    X_train = data[train_index]
    y_train = target_labels[train_index]
    X_test = data[test_index]
    y_test = target_labels[test_index]
    test_labels = np.unique(y_test)
    n_components= len(np.unique(y_train))

    for kdx in range(n_weights):
      # Plot the ground truth and add labels
      fig, ax = plt.subplots(1, 1, figsize=(10, 8))
      ax.set_xlabel('fft-1')
      ax.set_ylabel('fft-2')

      for idx in range(len(labels)):
        label = labels[idx]
        color = float(idx)/len(labels)
        ax.text(data[target_labels == label, 0].mean(),
                  data[target_labels == label, 1].mean(),
                  'mean ' + str(label),
                   horizontalalignment='center',
                   bbox=dict(alpha=.7, edgecolor='b', facecolor='w'))
        ax.scatter(data[target_labels == label,0], data[target_labels == label, 1],
                   c=str(color),
                   marker='o', s=100,
                   label=str(label))

      #Fit the model
      weight = possible_weights[kdx]

      # we create an instance of Neighbours Classifier and fit the data.
      clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weight, algorithm='ball_tree')
      clf.fit(X_train, y_train)

      #Plot K-NN of test data
      for jdx in range(len(test_labels)):
        this_label = test_labels[jdx]
        label_color = float(jdx)/len(test_labels)
        X_of_label = X_test[y_test==this_label]
        neighbors_per_vector = clf.kneighbors(X_of_label, n_neighbors=2, return_distance=False)
        neighbors_indices = np.unique(neighbors_per_vector.ravel())
        ax.scatter(X_train[neighbors_indices][:,0], X_train[neighbors_indices][:,1],
                   c=str(label_color),
                   marker='x', s=220,
                   label=str(this_label))
        ax.legend(loc='lower right')

      ax.set_xscale('symlog')
      ax.set_yscale('symlog')

      plt.xlim(data[:,0].min(), data[:,0].max())
      plt.ylim(data[:,1].min(), data[:,1].max())
      plt.show()
      fig.savefig('knn_distance<%s>neighbors<%d>classes<%d>' % (weight, n_neighbors, n_labels), dpi=fig.dpi)

def plot_clustering(data, target_labels, cut='none', pre_processing=''):
  """Plots 3d KMeans clustering from the first three indexed features used
    ARGS
      data: array-like or sparse matrix, shape=(n_samples, n_features)
      target_labels: labels of data points <number array>
      cut: suffix to filename to indicate if data was prunned <str>
      pre_processing: suffix to filename to indicate pre-processig applied <str>
  """
  n_samples, n_features = data.shape

  labels = np.unique(target_labels)
  n_labels = len(labels)
  sample_size = 300

  pca = PCA(n_components=n_labels).fit(data)

  estimators = {'k-means++': KMeans(init='k-means++', n_clusters=n_labels, n_init=n_init),
                'random': KMeans(init='random', n_clusters=n_labels, n_init=n_init),
                'pca': KMeans(init=pca.components_, n_clusters=n_labels, n_init=1)}

  fig_num = 1
  for name, estimator in estimators.items():
    fig = plt.figure(fig_num, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    estimator.fit(data)
    estimated_labels = estimator.labels_

    p = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=estimated_labels.astype(np.float))

    ax.set_xlabel('feature-1')
    ax.set_ylabel('feature-2')
    ax.set_zlabel('feature-3')
    fig.suptitle(name)
    fig_num = fig_num + 1
    fig.colorbar(p)
    plt.show()
    plt.savefig('knn_clustering_<%s>classes<%d>cut<%s>pre_processing<%s>fft<1-2-3>' % (name, n_labels, cut, pre_processing) )

  # Plot the ground truth
  fig = plt.figure(fig_num, figsize=(4, 3))

  plt.clf()
  ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

  plt.cla()

  #add labels to ground truth
  for idx in range(len(labels)):
    label = labels[idx]
    color = float(idx)/len(labels)
    ax.text3D(data[target_labels == label, 0].mean(),
              data[target_labels == label, 1].mean(),
              data[target_labels == label, 2].mean(),
              'lbl ' + str(label),
               horizontalalignment='center', 
               bbox=dict(alpha=.7, edgecolor='r', facecolor='w'))
    ax.scatter(data[target_labels == label,0], data[target_labels == label, 1], data[target_labels == label, 2], c=str(color))

  ax.set_xlabel('feature-1')
  ax.set_ylabel('feature-2')
  ax.set_zlabel('feature-3')
  fig.suptitle("Ground truth")
  plt.show()
  plt.savefig('knn_clustering<%s>classes<%d>cut<%s>pre_processing<%s>mfccs<1-2-3>' % ('truth', n_labels, cut, pre_processing) )

def plot_confusion_matrix(data, target_labels, normalize=False, label='', n_init=20):
  """Instantiates KMeans clusters, fits the data and plots confusion matrices
  ARGS
    data: array-like or sparse matrix, shape=(n_samples, n_features)
    target_labels: labels of data points <number array>
    normalize: plot absolute value or percentage in confusion matrix <bool>
    label: suffix to be added to saved file <str>
    n_init : number of initializations <int>
  """

  try:
    n_samples, n_features = data.shape
  except ValueError:
    n_samples = data.shape[0]
    n_features = 1

  labels = sorted(np.unique(target_labels))
  n_clusters = len(labels)

  #INSTANTIATE ESTIMATORS
  pca = PCA(n_components=n_clusters).fit(data)
  k_est = K_Means()
  k_est.add_estimator(pca, n_clusters, 1)
  k_est.add_estimator('random', n_clusters, n_init)
  k_est.add_estimator('k-means++', n_clusters, n_init)

  #FIT DATA
  for key in k_est.estimators:
    estimator = k_est.estimators[key]
    estimator.fit(data)
    cm = metrics.confusion_matrix(target_labels, estimator.labels_)

    #normalize
    if normalize:
      cm = float(cm)/cm.max()

    fig, ax = plt.subplots()
    cax = ax.matshow(cm)

    #write predictions or probabilities to matrix
    cm_clusters = cm.shape[0]
    if normalize:
      for i in range(0,cm_clusters):
        for j in range(0,cm_clusters):
          if cm[j,i] > 0:
            ax.text(i,j, ("%.2f" % cm[j,i]), va='center', ha='center')
    else:
      for i in range(0,cm_clusters):
        for j in range(0,cm_clusters):
          if cm[j,i] > 0:
            ax.text(i,j, ("%d" % cm[j,i]), va='center', ha='center')

    plt.title('Confusion matrix using %s initialization' % key)
    fig.colorbar(cax)

    #add labels
    #ax.set_xticklabels(labels)
    #ax.set_yticklabels(labels)
    plt.xlabel('Truth')
    plt.ylabel('Guess')
    plt.show()


