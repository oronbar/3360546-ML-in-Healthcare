# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """
    {3 points}
    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    CTG_features = CTG_features.drop(columns = [extra_feature], axis=1)
    CTG_features = CTG_features.apply(pd.to_numeric, errors='coerce')
    CTG_features = CTG_features.dropna()
    c_ctg = CTG_features.to_dict('list')
    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """
    {5 points}
    :param CTG_features: Pandas dataframe of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe c_cdf containing the "clean" features
    """

    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    CTG_features = CTG_features.drop(columns = [extra_feature], axis=1)
    CTG_features = CTG_features.apply(pd.to_numeric, errors='coerce')
    for column in CTG_features.columns:
        non_nan_values = CTG_features[column].dropna().values
        nan_loc = CTG_features[column].isna()
        generated_values = np.random.choice(non_nan_values, size=nan_loc.sum())
        CTG_features[column][nan_loc] = generated_values
    c_cdf = CTG_features
    # -------------------------------------------------------------------------
    return c_cdf


def sum_stat(c_feat):
    """
    {3 points}
    :param c_feat: Output of nan2num_samp
    :return: Summary statistics as a dictionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary = {} 
    for column in c_feat.columns:
        d_summary[column] = {}
        d_summary[column]['min'] = c_feat[column].min()
        d_summary[column]['Q1'] = np.percentile(c_feat[column],25)
        d_summary[column]['median'] = c_feat[column].median()
        d_summary[column]['Q3'] = np.percentile(c_feat[column],75)
        d_summary[column]['max'] = c_feat[column].max()

    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """
    {3 points}
    :param c_feat: Output of nan2num_samp
    :param d_summary: Output of sum_stat
    :return: Dataframe containing c_feat with outliers removed
    """
    c_no_outlier = c_feat.copy()
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    for column in c_feat.columns:
        IQR = d_summary[column]['Q3'] - d_summary[column]['Q1']
        topTh = d_summary[column]['Q3'] + 1.5 * IQR
        bottomTh = d_summary[column]['Q1'] - 1.5 * IQR
        outliers = (c_feat[column] > topTh) | (c_feat[column] < bottomTh)
        c_no_outlier[column][outliers] = np.full((outliers.sum()), np.nan)
    # -------------------------------------------------------------------------
    return c_no_outlier


def phys_prior(c_samp, feature, thresh):
    """
    {3 points}
    :param c_samp: Output of nan2num_samp
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    filt_feature = c_samp[feature]
    filt_feature[filt_feature > thresh[1]] = thresh[1]
    filt_feature[filt_feature < thresh[0]] = thresh[0]
    # -------------------------------------------------------------------------
    return np.array(filt_feature)


class NSD:
    """
    {6 points}
    """

    def __init__(self):
        self.max = np.nan
        self.min = np.nan
        self.mean = np.nan
        self.std = np.nan
        self.fit_called = False
    
    def fit(self, CTG_features):
        # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
        self.max = CTG_features.max()
        self.min = CTG_features.min()
        self.mean = CTG_features.mean()
        self.std = CTG_features.std()
        # -------------------------------------------------------------------------
        self.fit_called = True

    def transform(self, CTG_features, mode='none', selected_feat=('LB', 'ASTV'), flag=False):
        """
        Note: x_lbl should only be either: 'Original values [N.U]', 'Standardized values [N.U.]', 'Normalized values [N.U.]' or 'Mean normalized values [N.U.]'
        :param mode: A string determining the mode according to the notebook
        :param selected_feat: A two elements tuple of strings of the features for comparison
        :param flag: A boolean determining whether or not plot a histogram
        :return: Dataframe of the normalized/standardized features called nsd_res
        """
        ctg_features = CTG_features.copy()
        if self.fit_called:
            if mode == 'none':
                nsd_res = ctg_features
                x_lbl = 'Original values [N.U]'
            # ------------------ IMPLEMENT YOUR CODE HERE (for the remaining 3 methods using elif):------------------------------
            elif mode == 'standard':
                nsd_res = (ctg_features - self.mean) / self.std
                x_lbl = 'Standardized values [N.U.]'
            elif mode == 'MinMax':
                nsd_res = (ctg_features - self.min) / (self.max - self.min)
                x_lbl = 'Normalized values [N.U.]'
            elif mode == 'mean':
                nsd_res = (ctg_features - self.mean) / (self.max - self.min)
                x_lbl = 'Mean normalized values [N.U.]'
            else:
                nsd_res = ctg_features
                x_lbl = 'Original values [N.U]'
            # -------------------------------------------------------------------------
            if flag:
                self.plot_hist(nsd_res, mode, selected_feat, x_lbl)
            return nsd_res
        else:
            raise Exception('Object must be fitted first!')

    def fit_transform(self, CTG_features, mode='none', selected_feat=('LB', 'ASTV'), flag=False):
        self.fit(CTG_features)
        return self.transform(CTG_features, mode=mode, selected_feat=selected_feat, flag=flag)

    def plot_hist(self, nsd_res, mode, selected_feat, x_lbl):
        x, y = selected_feat
        if mode == 'none':
            bins = 50
        else:
            bins = 80
            # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
        plt.figure()
        for feat in selected_feat:
            plt.hist(nsd_res[feat], bins=bins)
            plt.xlabel(x_lbl)
            plt.ylabel('Counts')

            # -------------------------------------------------------------------------

# Debugging block!
if __name__ == '__main__':
    from pathlib import Path
    file = Path.cwd().joinpath(
        'messed_CTG.xls')  # concatenates messed_CTG.xls to the current folder that should be the extracted zip folder
    CTG_dataset = pd.read_excel(file, sheet_name='Raw Data')
    CTG_features = CTG_dataset[['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DR', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV',
                                'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance',
                                'Tendency']]
    CTG_morph = CTG_dataset[['CLASS']]
    fetal_state = CTG_dataset[['NSP']]

    extra_feature = 'DR'
    c_ctg = rm_ext_and_nan(CTG_features, extra_feature)