import numpy as np
import sklearn.metrics as skm
import scipy.interpolate
import scipy.optimize
import pandas as pd
import joblib

class BC_Model:

    def __init__(self, model):
        self.model = model

    def fit(self, preds, labels):
        r"""
        Fits `model` to data.

        `preds` should be ? $\times$ $m$ for model with $m$ features. `labels` should
        be ? $\times$ 1 and take values in $\{0,1\}$.
        """
        self.model.fit(preds, labels)

    def predict_proba(self, preds):
        r"""
        Predicts signal probability for each element of a dataset ($? \times m$ `numpy` array).

        Returns `numpy` array of with values in $[0,1]$ giving predicted signal probabilities for
        each data point.
        """
        try:
            return self.model.predict_proba(preds)[:,1]
        except:
            print("self.model doesn't have a predict_proba function")

    def predict(self, preds, threshold=None):
        r"""
        Predicts signal ($1$) or background ($0$) for each element of a dataset ($? \times m$ `numpy` array).

        Returns `numpy` array of length with values in $\{0,1\}$ giving predicted classifications.

        Uses the `predict` method built into `scikit-learn` models.
        """
        if threshold is not None:
            probs = self.model.predict_proba(preds)[:,1]
            return np.where(probs > threshold, np.ones_like(probs), np.zeros_like(probs))
        else:
            try:
                return self.model.predict(preds)
            except:
                print("self.model doesn't have a predict function")

    def predict_hist(self, preds, labels, num_bins=100, sepbg=False, sig_norm=1, bg_norm=1, dataframe=False,
                     yAxisUnits='events/bin width'):
        r"""
        Constructs a histogram of predicted signal probabilities for signal and background constituents of
        a dataset ($? \times M$ `numpy` array).

        If `sepbg` is `False` (the default), labels are assumed to take values in $\{0,1\}$. Backgrounds are treated in combination:
        a list of $3$ $?_i \times m$ `numpy` arrays are returned, containing bin edges (partitioning $[0,1]$),
        signal bin contents, and background bin contents.

        If `sepbg` is `True`, labels are assumed to take values in $\{-n,\dots,-1,1\}$ (if there are $n$ backgrounds) while
        `bg_norm` should be a list of length $n$. Backgrounds are then differentiated: a list of $2 + n$ `numpy` arrays of shape
        $?_i \times m$ are returned, containing bin edges (partitioning $[0,1]$), signal bin contents, and
        $n$ background bin contents.

        `yAxisUnits` accepts either the string 'events' or 'events/bin width': in the former case, histograms are generated normally
        (using `density=True` within `np.histogram` and then multipliying bins by `sig_norm` and `bg_norm`) meaning you get the desired
        normalization values by summing the area under the curve, while in the latter case the usual procedure is following BUT then bin heights
        are divided by bin width, thereby ensuring that merely summing the heights of the individual bins yields the desired normalization
        values.
        """
        predictions = self.predict_proba(preds)
        labels = np.array(labels)
        sig_bins, bin_edges = np.histogram(predictions[labels==1], bins=num_bins, density=True)
        binWidthFactor = 1/num_bins if yAxisUnits == 'events/bin width' else 1
        sig_bins *= sig_norm * binWidthFactor
        if sepbg:
            bg_norms = bg_norm
            bg_binss = [
                bg_norm * binWidthFactor * np.histogram(predictions[labels==-(i+1)], bins=num_bins, density=True)[0]
                for i, bg_norm in enumerate(bg_norms)]
            if dataframe:
                return pd.DataFrame(
                    data=[bin_edges, sig_bins] + bg_binss,
                    columns=['Bin Edges', 'Signal'] + ['Background {}'.format(i) for i in range(1, self.num_bgs+1)])
            else:
                return [bin_edges, sig_bins] + bg_binss
        else:
            bg_bins = bg_norm * binWidthFactor * np.histogram(predictions[labels!=1], bins=num_bins, density=True)[0]
            if dataframe:
                return pd.DataFrame(data=[bin_edges, sig_bins, bg_bins], columns=['Bin Edges', 'Signal', 'Background'])
            else:
                return [bin_edges, sig_bins, bg_bins]

    def feature_importance(self):
        """
        Returns the importance of the $M$ features used to train `self.model`.
        """
        try:
            return self.model.feature_importances_
        except:
            try:
                return self.model[-1].feature_importances_
            except:
                print("It looks like self.model doesn't have an attribute 'feature_importances_'")

    def sorted_feature_importance(self, features):
        r"""
        Returns list of features sorted by importance.

        Given arguments `features` and `importances`, lists of length $M$, returns list of size $M \times 2$ where
        the first column gives features and the second their associated importances, sorted by importance.
        """
        importances = self.feature_importance()
        ranked_indices = np.argsort(-np.abs(importances))
        return [[features[i], importances[i]] for i in ranked_indices]

    def accuracy(self, preds, labels, threshold=None):
        r"""
        Computes model accuracy on a dataset ($? x m$ predictors, length $?$ labels).

        Returns value in $[0,1]$ giving model accuracy on the provided predictors and labels.
        """
        predictions = self.predict(preds=preds, threshold=threshold)
        return len(preds) - np.sum(np.abs(predictions - labels)) / len(preds)

    def conf_matrix(self, labels, predictions=None, preds=None):
        r"""
        Computes the confusion matrix of the trained model on a dataset ($? x M$ predictors, length $?$ labels).

        Returns $2 \times 2$ confusion matrix using `sklearn.metrics.confusion_matrix`.

        If predictors `preds` aren't provided, `self.test_preds` is used.
        If `labels` aren't provided, `self.test_labels` is used.
        """
        if predictions is not None:
            return skm.confusion_matrix(labels, predictions, labels=[0,1])
        elif preds is not None:
            return skm.confusion_matrix(labels, self.predict(preds), labels=[0,1])
        else:
            raise ValueError('Either predictions or preds must be passed.')


    def tpr_cm(self, conf_matrix):
        """
        Computes the true positive rate (tpr; correctly identified signal/total signal)
        of a trained model given a confusion matrix.

        Returns value in $[0,1]$.
        """
        return conf_matrix[1,1]/np.sum(conf_matrix[1])

    def fpr_cm(self, conf_matrix):
        """
        Computes the false positive rate (fpr; misidentified background/total background)
        of a trained model given a confusion matrix.

        Returns value in $[0,1]$.
        """
        return conf_matrix[0,1]/np.sum(conf_matrix[0])

    def tpr(self, labels, predictions=None, preds=None):
        r"""
        Computes the true positive rate (tpr; correctly identified signal/total signal)
        of a trained model given predictions and labels (both `numpy` array of length $?$ with values in $\{0,1\}$)

        Returns value in $[0,1]$.
        """
        return self.tpr_cm(self.conf_matrix(labels, predictions=predictions, preds=preds))

    def fpr(self, labels, predictions=None, preds=None):
        r"""
        Computes the false positive rate (fpr; misidentified background/total background)
        of a trained model given predictions and labels (both `numpy` array of length $?$ with values in $\{0,1\}$)

        Returns value in $[0,1]$.
        """
        return self.fpr_cm(self.conf_matrix(labels, predictions=predictions, preds=preds))

    def significance(self, signal, background, tpr, fpr, sepbg=False):
        r"""
        Computes signal significance of a trained model given signal and background yield.

        Returns a positive real number computed by
        $$\frac{S \cdot TPR}{\sqrt{S \cdot TPR + B \cdot FPR}}$$
        which corresponds to signal significance after selecting only datapoints the model identifies as signal.

        If `sepbg` is `False`, `background` should be a single real number and is multiplied by `fpr`. If `sepbg` is `True`,
        `background` should be a list of length `self.num_bgs` where the $i$th element contains background yield of the $i$th
        background type. `fpr`, if passed, is then also a list of length `self.num_bgs` giving false positive rates for each
        of the background types.
        """
        if sepbg:
            backgrounds = background
            fprs = fpr
            fprXbackground = np.sum(np.multiply(fprs, backgrounds), axis=-1)
            return (signal * tpr) / np.sqrt(signal * tpr + fprXbackground + 1e-10)
        else:
            return (signal * tpr) / np.sqrt(signal * tpr + background * fpr + 1e-10)

    def newvar2thresh(self, newvar):
        r"""
        Helper method for `bcml.max_allowable_threshold()`, `bcml.get_tprs_fprs()`, and `bcml.best_threshold()`,
        performing change of variables from `newvar` to `threshold`

        In particular, threshold $= 1 - 10^{\text{newvar}}$
        """
        return 1 - np.power(10, newvar)

    def thresh2newvar(self, thresh):
        r"""
        Helper method for `bcml.max_allowable_threshold()`, `bcml.get_tprs_fprs()`, and `bcml.best_threshold()`,
        performing change of variables from `threshold` to `newvar`

        In particular, newvar $= \log_{10}(1 - \text{threhold})$
        """
        return np.log10(1 - thresh)

    def max_allowable_threshold(self, preds, labels, sigYield):
        """
        Returns the highest threshold such that only labelling elements of `self.test_pred` with predicted
        probabilities higher than that threshold as signal still yields 25 signal.

        To achieve a discovery potential of $5\sigma$, even in the best case scenario ($TPR = 1, FPR = 0$) we still
        require $5^2 = 25$ signal events, hence we cannot chose a threshold so high that we do not keep at least
        25 signal events.
        """

        sig_indx = np.where(np.array(labels)==1)[0]
        preds = np.array(preds)[sig_indx]
        probs = self.predict_proba(preds)
        num_predicts = np.array(
            [np.sum(np.where(probs > self.newvar2thresh(newvar),np.ones_like(probs), np.zeros_like(probs)))
             for newvar in newvars])
        num_sig_yields = (num_predicts / len(preds)) * sigYield
        return [newvars, num_sig_yields]
        # f = scipy.interpolate.interp1d(num_sig_yield, newvars, kind='cubic')
        # return self.newvar2thresh(f(25))

    def get_tprs_fprs(self, preds, labels, sepbg=False):
        """
        Produces (true positive rate, false positive rate) pairs for various thresholds
        for the trained model on data sets.

        If `sepbg` is `True`, labels should take values in $\{-n,\dots,-1,1\}$. Background is combined and a list of length $4$
        is returned containing a list of $L$ sampled newvars (a convenient change of variable to approach arbitrarily close to 1:
        related to thresholds by  `bcml_model.newvar2thresh()`), an $L$-list of tprs associated to those thresholds, an $L$-list of fprs
        related to those thresholds, and an $L$-list of length $?$ `numpy` arrays giving the predicted signal probabilities
        for the given data set.

        If `sepbg` is `Frue`, labels should take values in $\{0,1\}$. Background is split and a list of length $4$ `self.num_bgs`
        is returned containing a  list of $L$ sampled newvars, an $L$-list of tprs associated to those thresholds, an $L$-list of lists of length
        $n$ (number of backgrounds) containing fprs for each background type for each threshold, and an $L$-list of length $?$
        """
        # setting up variables
        min_newvar, max_newvar = [-10, 0]
        newvars = np.concatenate((np.linspace(min_newvar, -2, 10, endpoint=False), np.linspace(-2, max_newvar, 15, endpoint=False)))

        # computing tprs, fprs
        if sepbg:
            num_bgs = len(np.unique(labels)) - 1
            labelTypes = [1] + [-(i+1) for i in range(num_bgs)]
            labelsIndices = [np.where(np.array(labels)==i)[0] for i in labelTypes]
            predss = [preds[indices] for indices in labelsIndices]
            probss = [self.predict_proba(preds) for preds in predss]
            predictionsss = [np.array([np.where(probs > self.newvar2thresh(newvar),
                          np.ones_like(probs), np.zeros_like(probs)) for probs in probss]) for newvar in newvars]
            sig_conf_matrices = [
                self.conf_matrix(labels=np.ones_like(predictionss[0]), predictions=predictionss[0]) for predictionss in predictionsss]
            bg_conf_matricess = [
                [self.conf_matrix(labels=np.zeros_like(predictions), predictions=predictions) for i, predictions in enumerate(predictionss[1:])] for predictionss in predictionsss]
            tprs = np.array([self.tpr_cm(conf_matrix) for conf_matrix in sig_conf_matrices])
            fprss = np.array([[self.fpr_cm(conf_matrix) for conf_matrix in conf_matrices] for conf_matrices in bg_conf_matricess])
            sums = tprs + np.sum(fprss, axis=1)
            cutoff = len(sums) - np.argmax(np.flip(sums)==0) + 1 if 0 in sums else 0
            return [newvars[cutoff:], tprs[cutoff:], fprss[cutoff:], probss]
        else:
            probs = self.predict_proba(preds)
            predictionss = np.array(
                [np.where(probs > self.newvar2thresh(newvar),
                          np.ones_like(probs), np.zeros_like(probs)) for newvar in newvars])
            conf_matrices = [self.conf_matrix(labels=labels, predictions=predictions) for predictions in predictionss]
            tprs = np.array([self.tpr_cm(conf_matrix) for conf_matrix in conf_matrices])
            fprs = np.array([self.fpr_cm(conf_matrix) for conf_matrix in conf_matrices])
            sums = tprs + fprs
            cutoff = len(sums) - np.argmax(np.flip(sums)==0) + 1 if 0 in sums else 0
            return [newvars[cutoff:], tprs[cutoff:], fprs[cutoff:], probs]

    def best_threshold(self, signal, background, preds, labels, sepbg=False):
        """
        Optimizes the threshold on a given data set ($? x M$ predictors, length $?$ labels).
        """
        newvars, tprs, fprs, probs = self.get_tprs_fprs(preds, labels, sepbg)
        significances = -self.significance(signal, background, tprs, fprs, sepbg=sepbg)

        # interpolating significance as a function of threshold, then maximizing
        max_sig = np.amin(significances)
        significances_list = list(significances)
        i = significances_list.index(max_sig)
        min_i, max_i = [max(0,i-4),min(len(significances),i+5)]
        f = scipy.interpolate.interp1d(newvars[min_i:max_i], significances[min_i:max_i], kind='cubic')
        res = scipy.optimize.minimize(f, [0.5 * (newvars[min_i] + newvars[max_i-1])], bounds=[(newvars[min_i] + 1e-1, newvars[max_i-1] - 1e-1)])

        # computing significance, tpr, fpr for optimized threshold
        best_threshold = self.newvar2thresh(res.x[0])
        if sepbg:
            probss = probs
            fprss = fprs
            best_predictss = [np.where(probs > best_threshold, np.ones_like(probs), np.zeros_like(probs)) for probs in probss]
            sig_conf_matrix = self.conf_matrix(labels=np.ones_like(best_predictss[0]), predictions=best_predictss[0])
            bg_conf_matrices = [self.conf_matrix(labels=np.zeros_like(best_predicts), predictions=best_predicts) for i, best_predicts in enumerate(best_predictss[1:])]
            tpr = self.tpr_cm(sig_conf_matrix)
            fprs = [self.fpr_cm(conf_matrix) for conf_matrix in bg_conf_matrices]
            best_sig = self.significance(signal, background, tpr, fprs, sepbg=sepbg)
            return [best_threshold, best_sig, tpr, fprs, tprs, fprss]
        else:
            best_predicts = np.where(probs > best_threshold, np.ones_like(probs), np.zeros_like(probs))
            conf_matrix = self.conf_matrix(labels=labels, predictions=best_predicts)
            tpr = self.tpr_cm(conf_matrix)
            fpr = self.fpr_cm(conf_matrix)
            best_sig = self.significance(signal, background, tpr, fpr, sepbg=sepbg)
            return [best_threshold, best_sig, tpr, fpr, tprs, fprs]

    def req_sig_cs(self, lumi, bg_cs, tpr, fpr, sig=5, sepbg=False):
        """
        Given a luminosity (in fb$^{-1}$), a background cross section (in pb), a true positive rate, a false positive rate,
        and a signal significance, computes the signal cross section required for the signal significance to be achieved.

        If `sepbg` is False, background is combined and a single FPR is used; if `sepbg` is True, it is assumed that
        `bg_cs`, `fpr` are each lists of length $n$ (number of backgrounds) and their vector dot product is used for background yield.
        """
        conv = 10**15 / 10**12
        if sepbg:
            bg = np.sum(np.multiply(bg_cs, fpr))
            coef = [-tpr**2 * lumi * conv**2, sig**2 * tpr * conv, sig**2 * bg * conv]
        else:
            coef = [-tpr**2 * lumi * conv**2, sig**2 * tpr * conv, sig**2 * fpr * bg_cs * conv]
        return np.amax(np.roots(coef))

    def save_model(self, filename):
        """
        Saves the model to `filename.joblib`
        """
        joblib.dump(self.model, filename)
