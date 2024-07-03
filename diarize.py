from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering

def diarize(embeddings, segments, numSpeakers=2):
    scaler = StandardScaler()
    scaledEmbeddings = scaler.fit_transform(embeddings)
    clusters = SpectralClustering(n_clusters=numSpeakers, affinity='nearest_neighbors', random_state=100)
    labels = clusters.fit_predict(scaledEmbeddings)
    for i in range(len(segments)):
        segments[i]['speaker'] = "SPEAKER" + str(labels[i] + 1)
    return segments