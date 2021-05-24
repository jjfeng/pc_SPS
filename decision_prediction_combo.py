import numpy as np
from sklearn.ensemble import IsolationForest

class DecisionPredictionModel:
    """
    Cutoff via entropy
    """
    def __init__(self, interval_nn, cost_decline, threshold=None):
        self.nn = interval_nn
        self.cost_decline = cost_decline
        self.threshold = threshold if threshold is not None else cost_decline
        self.get_univar_mapping = self.nn.get_univar_mapping

    def get_accept_prob(self, x):
        return np.array(self.nn.get_univar_mapping(x) < self.threshold, dtype=int)
        #return np.array(self.nn.get_univar_mapping(x) < self.cost_decline, dtype=int)

    def score(self, x, y):
        accept_prob = self.get_accept_prob(x)
        pred_loss = self.nn.get_prediction_loss_obs(x, y)
        print("pred loss dist", np.median(pred_loss), np.mean(pred_loss), np.min(pred_loss), np.max(pred_loss))
        return -np.mean(pred_loss * accept_prob + self.cost_decline * (1 - accept_prob))

    def get_density(self, x, y, max_replicates=100):
        return self.nn.get_density(x, y, max_replicates)

    def get_prediction_interval(self, x, alpha):
        return self.nn.get_prediction_interval(x, alpha)

class EntropyOutlierPredictionModel(DecisionPredictionModel):
    """
    Accept via entropy cutoff AND outlier detector via iso forest
    """
    def __init__(self,
            dec_pred_nn: DecisionPredictionModel,
            cost_decline,
            threshold: float = None,
            eps: float = 0):
        self.dec_pred_nn = dec_pred_nn
        self.nn = dec_pred_nn.nn
        self.cost_decline = cost_decline
        self.threshold = cost_decline if threshold is None else threshold
        self.get_univar_mapping = lambda x: -self.get_accept_prob(x)
        self.iso_forest = IsolationForest(behaviour='new', contamination='auto')
        self.eps = eps

    def iso_score_samples(self, X):
        return 1 + self.iso_forest.score_samples(X).reshape((X.shape[0], 1))

    def fit_decision_model(self, X):
        """
        Fits an IsolationForest outlier classifier based on training data X
        eps parameter in (0,1] controls proportion of observations removed
        """
        X = X.reshape((X.shape[0], -1))
        self.iso_forest.fit(X)
        self.fitted_scores = self.iso_score_samples(X)
        self.od_thres = np.quantile(self.fitted_scores, self.eps)

    def get_accept_prob(self, x):
        accept_inner = self.dec_pred_nn.get_accept_prob(x)
        outlier_accept = self.iso_score_samples(x.reshape(x.shape[0], -1)) > self.od_thres
        return accept_inner * outlier_accept

    def score(self, x, y):
        accept_prob = self.get_accept_prob(x)
        pred_loss = -self.dec_pred_nn.score(x, y)
        return -np.mean(pred_loss * accept_prob + self.cost_decline * (1 - accept_prob))

class EmbeddingEntropyOutlierPredictionModel(EntropyOutlierPredictionModel):
    """
    Accept using entropy cutoff AND outlier detector via learned NN embedding
    """
    def fit_decision_model(self, X):
        """
        Fits an IsolationForest outlier classifier based on training data X
        eps parameter in (0,1] controls proportion of observations removed
        """
        embedding = self.dec_pred_nn.nn.get_embedding(X)
        super().fit_decision_model(embedding)

    def get_accept_prob(self, x):
        embedding = self.dec_pred_nn.nn.get_embedding(x)
        accept_inner = self.dec_pred_nn.get_accept_prob(x)
        outlier_accept = self.iso_score_samples(embedding) > self.od_thres
        return accept_inner * outlier_accept

    def score(self, x, y):
        accept_prob = self.get_accept_prob(x)
        pred_loss = -self.dec_pred_nn.score(x, y)
        return -np.mean(pred_loss * accept_prob + self.cost_decline * (1 - accept_prob))
