import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import model_selection
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import Birch, AffinityPropagation

result = None
xs_all_train, ys_all_train = None, None
xs_all_test, ys_all_test = None, None
clusters_train_kmean, clusters_test_kmean = None, None
clusters_train_birch, clusters_test_birch = None, None
clusters_train_AffinProp, clusters_test_AffinProp = None, None
clusters = None

############################################################
#                        Functions                         #
############################################################


def preprocess_text(data):
    # removing signs by regular expression and keep letters and numbers only
    reg = r'[^\w\s]'
    data = data.replace(reg, '', regex=True)
    # split the data
    data = data.apply(lambda x: " ".join(x.lower() for x in x.split()))
    # removing stop wards from the data
    stop = stopwords.words('english')
    data = data.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    # apply lemmatization in the data
    data = data.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return data


def model1_kmean(data_train, data_test, cluster_train_kmean, cluster_test_kmean):
    # Running clustering algorithm
    kmean = MiniBatchKMeans(n_clusters=10, random_state=1000)
    kmean.fit(data_train)
    testX_predict = kmean.predict(data_test)
    cluster = kmean.labels_.tolist()

    if cluster_train_kmean is None:
        cluster_train_kmean = cluster
        cluster_test_kmean = testX_predict
    else:
        cluster_train_kmean = np.append(cluster_train_kmean, cluster)
        cluster_test_kmean = np.append(cluster_test_kmean, testX_predict)
    return testX_predict, cluster,cluster_train_kmean, cluster_test_kmean


def model2_birch(data_train, data_test, cluster_train_birch, cluster_test_birch):
    # Running clustering algorithm
    birch = Birch(threshold=0.01, n_clusters=10)
    birch.fit(data_train)
    testX_predict = birch.predict(data_test)
    cluster = birch.labels_.tolist()

    if cluster_train_birch is None:
        cluster_train_birch = cluster
        cluster_test_birch = testX_predict
    else:
        cluster_train_birch = np.append(cluster_train_birch, cluster)
        cluster_test_birch = np.append(cluster_test_birch, testX_predict)
    return testX_predict, cluster, cluster_train_birch, cluster_test_birch


def model3_AffinityPropagation(data_train, data_test, cluster_train_AffinProp, cluster_test_AffinProp):
    # Running clustering algorithm
    AffinProp = AffinityPropagation(damping=0.9)
    AffinProp.fit(data_train)
    testX_predict = AffinProp.predict(data_test)
    cluster = AffinProp.labels_.tolist()

    if cluster_train_AffinProp is None:
        cluster_train_AffinProp = cluster
        cluster_test_AffinProp = testX_predict
    else:
        cluster_train_AffinProp = np.append(cluster_train_AffinProp, cluster)
        cluster_test_AffinProp = np.append(cluster_test_AffinProp, testX_predict)
    return testX_predict, cluster, cluster_train_AffinProp, cluster_test_AffinProp


def cosinesimilarity_and_reduce_two_dimensional(x_tfidf, xs_group, ys_group):
    # calculate cosine similarity
    similarity_distance = 1 - cosine_similarity(x_tfidf)

    # Convert two components as we're plotting points in a two-dimensional plane
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(similarity_distance)  # shape (n_components, n_samples)
    xs1, ys1 = pos[:, 0], pos[:, 1]

    if xs_group is None:
        xs_group = xs1
        ys_group = ys1
    else:
        xs_group = np.append(xs_group, xs1)
        ys_group = np.append(ys_group, ys1)
    return xs1, ys1, xs_group, ys_group, similarity_distance


def plot_data(x, y, cluster):
    # Set up colors per clusters using a dict
    cluster_colors = {0: 'blue', 1: 'red', 2: 'green', 3: 'cyan', 4: 'magenta',
                      5: 'yellow', 6: 'darkgoldenrod', 7: 'mediumslateblue', 8: 'gray', 9: 'lime'}

    # set up cluster names using a dict
    cluster_names = {0: '1',
                     1: '2',
                     2: '3',
                     3: '4',
                     4: '5',
                     5: '6',
                     6: '7',
                     7: '8',
                     8: '9',
                     9: '10'}
    # Create data frame that has the result of the MDS and the cluster
    df = pd.DataFrame(dict(x=x, y=y, label=cluster))
    groups = df.groupby('label')
    # Set up plot
    fig, ax = plt.subplots(figsize=(17, 9))

    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=10,
                label=cluster_names[name], color=cluster_colors[name],
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(
            axis='x',
            which='both',
            bottom='off',
            top='off',
            labelbottom='off')
        ax.tick_params(
            axis='y',
            which='both',
            left='off',
            top='off',
            labelleft='off')

    ax.legend(numpoints=1)
    plt.show()


############################################################
#                          MAIN                            #
############################################################

# Read the data
for chunk in pd.read_csv("articles1.csv", chunksize=1000):
    print(chunk)

    # Splitting data into train and validation
    train_x, test_x = model_selection.train_test_split(chunk['content'])

    # create the rank of documents for training data
    ranks_train = []
    for i in range(1, len(train_x) + 1):
        ranks_train.append(i)

    # create the rank of document for testing data
    ranks_test = []
    for i in range(1, len(test_x) + 1):
        ranks_test.append(i)

    train_x = pd.DataFrame(train_x)
    test_x = pd.DataFrame(test_x)
    print("training data before pre processing >>> \n", train_x)
    print("testing data before pre processing >>> \n", test_x)

    # ################################ preprocessing #####################################################
    # calling the preprocessing function for train and test data
    train_x = train_x.apply(preprocess_text)
    print("training data after pre processing >>> \n", train_x)
    test_x = test_x.apply(preprocess_text)
    print("testing data after pre processing >>> \n", test_x)

    # ################################ Feature Extraction ################################################
    # calling feature extraction function to apply TF-IDF
    # change data to Series
    train_x = train_x.squeeze()
    test_x = test_x.squeeze()
    # calling the algorith TF-IDF
    tfidf_vect = TfidfVectorizer()
    # fitting the data
    tfidf_vect.fit(train_x)
    # transform the data
    xtrain_tfidf = tfidf_vect.transform(train_x)
    xtest_tfidf = tfidf_vect.transform(test_x)
    print("TF-IDF for training data >>> \n", xtrain_tfidf.data)
    print("TF-IDF for testing data >>> \n", xtest_tfidf.data)
    print("TF-IDF for training data shape >>> \n", xtrain_tfidf.shape)
    print("TF-IDF for testing data shape>>> \n", xtest_tfidf.shape)

    # ################################ Cosine Similarity ################################################
    # getting the cosine similarity and reduce the dat to two-dimensional array for training data
    result_train = cosinesimilarity_and_reduce_two_dimensional(xtrain_tfidf, xs_all_train, ys_all_train)
    xs_train = result_train[0]
    ys_train = result_train[1]
    xs_all_train = result_train[2]
    ys_all_train = result_train[3]
    print("cosine similarity for training data>>> \n", result_train[4])

    # getting the cosine similarity and reduce the dat to two-dimensional array for testing data
    result_test = cosinesimilarity_and_reduce_two_dimensional(xtest_tfidf, xs_all_test, ys_all_test)
    xs_test = result_test[0]
    ys_test = result_test[1]
    xs_all_test = result_test[2]
    ys_all_test = result_test[3]
    print("cosine similarity for testing data>>> \n", result_test[4])

    # ################################ Model training and visualization #################################

    # ################################ Mini Batch Kmeans ################################################
    # calling model1 (kmean)
    kmeans = model1_kmean(xtrain_tfidf, xtest_tfidf, clusters_train_kmean, clusters_test_kmean)
    xtest_predict = kmeans[0]
    clusters_train_kmean = kmeans[2]
    clusters_test_kmean = kmeans[3]
    print("prediction kmean for testing data Kmean >>> \n", xtest_predict)

    # final clusters for training data
    clusters = kmeans[1]
    complaints_data = {'rank': ranks_train, 'complaints': train_x, 'train cluster kmean': clusters}
    frame = pd.DataFrame(complaints_data, index=[clusters], columns=['rank', 'train cluster kmean'])
    # number of docs per cluster
    print(frame['train cluster kmean'].value_counts())

    # final clusters for testing data
    complaints_data = {'rank': ranks_test, 'complaints': test_x, 'test cluster kmean': xtest_predict}
    frame = pd.DataFrame(complaints_data, index=[xtest_predict], columns=['rank', 'test cluster kmean'])
    # number of docs per cluster
    print(frame['test cluster kmean'].value_counts())

    # plot the data for every chunk for training data
    plot_data(xs_train, ys_train, clusters)

    # plot the data for every chunk for testing data
    plot_data(xs_test, ys_test, xtest_predict)

    # ################################ Birch ############################################################
    # calling model2 (birch)
    birchs = model2_birch(xtrain_tfidf, xtest_tfidf, clusters_train_birch, clusters_test_birch)
    xtest_predict = birchs[0]
    clusters_train_birch = birchs[2]
    clusters_test_birch = birchs[3]
    print("prediction birch for testing data Birch >>> \n", xtest_predict)

    # final clusters for training data
    clusters = birchs[1]
    complaints_data = {'rank': ranks_train, 'complaints': train_x, 'train cluster Birch': clusters}
    frame = pd.DataFrame(complaints_data, index=[clusters], columns=['rank', 'train cluster Birch'])
    # number of docs per cluster
    print(frame['train cluster Birch'].value_counts())

    # final clusters for testing data
    complaints_data = {'rank': ranks_test, 'complaints': test_x, 'test cluster Birch': xtest_predict}
    frame = pd.DataFrame(complaints_data, index=[xtest_predict], columns=['rank', 'test cluster Birch'])
    # number of docs per cluster
    print(frame['test cluster Birch'].value_counts())

    # plot the data for every chunk for training data
    plot_data(xs_train, ys_train, clusters)

    # plot the data for every chunk for testing data
    plot_data(xs_test, ys_test, xtest_predict)

    # ################################ Affinity Propagation #############################################
    # calling model3 (AffinityPropagation)
    AffinProps = model3_AffinityPropagation(xtrain_tfidf, xtest_tfidf, clusters_train_AffinProp,clusters_test_AffinProp)
    xtest_predict = AffinProps[0]
    clusters_train_AffinProp = AffinProps[2]
    clusters_test_AffinProp = AffinProps[3]
    print("prediction Affinity Propagation for testing data Affinity Propagation >>> \n", xtest_predict)

    # final clusters for training data
    clusters = AffinProps[1]
    complaints_data = {'rank': ranks_train, 'complaints': train_x, 'train cluster AffinProps': clusters}
    frame = pd.DataFrame(complaints_data, index=[clusters], columns=['rank', 'train cluster AffinProps'])
    # number of docs per cluster
    print(frame['train cluster AffinProps'].value_counts())

    # final clusters for testing data
    complaints_data = {'rank': ranks_test, 'complaints': test_x, 'test cluster AffinProps': xtest_predict}
    frame = pd.DataFrame(complaints_data, index=[xtest_predict], columns=['rank', 'test cluster AffinProps'])
    # number of docs per cluster
    print(frame['test cluster AffinProps'].value_counts())

    # plot the data for every chunk for training data
    dataframe = pd.DataFrame(dict(x=xs_train, y=ys_train, label=clusters))
    packages = dataframe.groupby('label')
    figs, axs = plt.subplots(figsize=(17, 9))  # set size
    for names, package in packages:
        # create scatter of these samples
        axs.scatter(package.x, package.y, marker='o')
    plt.show()
#   for names, package in packages:
#       # create scatter of these samples
#       axs.plot(package.x, package.y, marker='o')
#   plt.show()

    # plot the data for every chunk for testing data
    dataframe = pd.DataFrame(dict(x=xs_test, y=ys_test, label=xtest_predict))
    packages = dataframe.groupby('label')
    figs, axs = plt.subplots(figsize=(17, 9))
    for names, package in packages:
        # create scatter of these samples
        axs.scatter(package.x, package.y, marker='o')
    plt.show()
#   for names, package in packages:
#       # create scatter of these samples
#       axs.plot(package.x, package.y, marker='o')
#   plt.show()


# plot the data for all training data Mini Batch Kmeans
plot_data(xs_all_train, ys_all_train, clusters_train_kmean)

# plot the data for all testing data Mini Batch Kmeans
plot_data(xs_all_test, ys_all_test, clusters_test_kmean)

# plot the data for all training data Birch
plot_data(xs_all_train, ys_all_train, clusters_train_birch)

# plot the data for all testing data Birch
plot_data(xs_all_test, ys_all_test, clusters_test_birch)

# plot the data for all training data Affinity Propagation
dataframe = pd.DataFrame(dict(x=xs_all_train, y=ys_all_train, label=clusters_train_AffinProp))
packages = dataframe.groupby('label')
figs, axs = plt.subplots(figsize=(17, 9))  # set size
for names, package in packages:
    # create scatter of these samples
    axs.scatter(package.x, package.y, marker='o')
plt.show()
# for names, package in packages:
#     # create scatter of these samples
#     axs.plot(package.x, package.y, marker='o')
# plt.show()

# plot the data for all testing data Affinity Propagation
dataframe = pd.DataFrame(dict(x=xs_all_test, y=ys_all_test, label=clusters_test_AffinProp))
packages = dataframe.groupby('label')
figs, axs = plt.subplots(figsize=(17, 9))  # set size
for names, package in packages:
    # create scatter of these samples
    axs.scatter(package.x, package.y, marker='o')
plt.show()
# for names, package in packages:
#     # create scatter of these samples
#     axs.plot(package.x, package.y, marker='o')
# plt.show()

