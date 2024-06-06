from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def metrics(X, labels):
    
    SCS = silhouette_score(X=X, labels=labels)
    CHI = calinski_harabasz_score(X=X, labels=labels)
    DBI = davies_bouldin_score(X=X, labels=labels)

    return SCS, CHI, DBI