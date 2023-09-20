import numpy as np
from sklearn.utils import compute_sample_weight
from sklearn.metrics import roc_curve, auc, confusion_matrix

# For EER calculations
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def calc_eer(fpr:np.ndarray, tpr:np.ndarray, threshold:np.ndarray):
    """
    Discovers the threshold where FPR and FRR intersects.

    Parameters
    ----------
    fpr : np.ndarray
        Array with False Positive Rate from sklearn.metrics.roc_curve
    tpr : np.ndarray
        Array with True Positive Rate from sklearn.metrics.roc_curve
    threshold : np.ndarray
        Array with thresholds from sklearn.metrics.roc_curve

    Returns
    -------
    floats, float
        eer_score, eer_threshold


    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
    >>> eer_score, eer_threshold = calc_eer(fpr, tpr, thresholds)
    

    """
    #Implementation from -> https://yangcha.github.io/EER-ROC/
    # first position is always set to max_threshold+1 (close to 2) by sklearn, 
    # overwrite with 1.0 to avoid EER threshold exceeding 1.0. 
    #threshold[0] = 1.0 
    eer_score = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_thresh = interp1d(fpr, threshold)(eer_score)
    
    return eer_score, float(eer_thresh)

def pred_at_threshold(y_scores, threshold, pos_label=1):
    #Make predictions based on a specific decision threshold
    #y_scores : array with predicted probabilities, similarities, or distances.
    #threshold : the specified threshold to seperate the two classes.
    #pos_label : integer defining the positive class.
    if pos_label == 0:
        return np.array((y_scores <= threshold)).astype(int)
    else: 
        assert(pos_label == 1)
        return np.array((y_scores >= threshold)).astype(int)
    
class MetricMaker():
    def __init__(self, y_true, y_similarity, weighted=True) -> None:
        self.y_true = y_true
        self.y_sims = y_similarity
        self.weighted = weighted

        # call the fit function when instantiated
        self._fit()

    def _fit(self):
        # Create balanced sample weights for imbalanced evaluation
        if self.weighted:
            self.smpl_w = compute_sample_weight(
                class_weight='balanced', 
                y=self.y_true)
        else:
            self.smpl_w = None

        # Compute the FPR, TPR, and Thresholds
        self.fpr, self.tpr, self.thresholds = roc_curve(
            y_true=self.y_true, 
            y_score=self.y_sims,
            sample_weight=self.smpl_w)
        
        # Compute the AUC score
        self.auc = auc(self.fpr,self.tpr)

    def eer(self, output=False):
        self.eer_score, self.eer_thresh = calc_eer(
            fpr=self.fpr, 
            tpr=self.tpr, 
            threshold=self.thresholds)
        
        if output:
            return self.eer_score, self.eer_thresh
    
    def get_fpr_threshold(self, max_fpr):
        return np.interp(max_fpr, self.fpr, self.thresholds)

    def get_cm(self, decision_threshold, normalize='true', breakdown=False):
         """
         Compute and returns the Confusion Matrix at a certain decision threshold

         breakdown: instead returns tn, fp, fn, tp
         """
         # an ugly patch to allow passing 'none' to sklean arg
         if normalize == 'none' : normalize=None

         y_pred = pred_at_threshold(self.y_sims, decision_threshold, pos_label=1)
            
         cm = confusion_matrix(
             y_true=self.y_true,
             y_pred=y_pred,
             sample_weight=self.smpl_w,
             normalize=normalize #type:ignore 
             )
         
         if breakdown:
             tn, fp, fn, tp = cm.ravel()
             return tn, fp, fn, tp
         else:
            return cm
         
def makepretty(styler, **kwargs):
    #https://pandas.pydata.org/docs/user_guide/style.html#Styler-Object-and-Customising-the-Display 
    title = kwargs['title']
    styler.set_caption(f"Stats for '{title}'")
    
    styler.format(precision=4,thousands=".", decimal=",")
    styler.background_gradient(axis=None, subset=['25%','75%'], vmin=0, vmax=1, cmap='Greys')
    styler.hide(subset=['count'], axis=1)
    styler.format_index(str.upper, axis=1)
    
    return styler

def macro_stats(data, le_a, le_m, transform, style=True):
    stats = data.groupby(['algo','metric'])[transform].describe().reset_index()
    stats['algo'] = le_a.inverse_transform(stats['algo'])
    stats['metric'] = le_m.inverse_transform(stats['metric'])
    
    if style:
        stats = stats.style.pipe(makepretty, title=transform)
    
    return stats