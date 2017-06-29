"""
Functions for explaining text classifiers.
"""
from __future__ import unicode_literals

import itertools
import json
import re

import numpy as np
import scipy as sp
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import math

import explanation
import lime_base

class TimeSeriesDomainMapper(explanation.DomainMapper):
    """Maps feature ids to words or word-positions"""

    def __init__(self, time_series):
        """Initializer.

        Args:
            time_series: the time series getting explained
        """
        self.time_series = time_series

    def map_exp_ids(self, exp):
        """Maps ids to time series points or features

        Args:
            exp: list of tuples [(id, weight), (id,weight)]

        Returns:
            list of tuples (feature_name, weight) or (point_x, weight)
        """
        return exp

    def visualize_instance_html(self, exp, label, div_name, exp_object_name):
        """Adds text with highlighted words to visualization.

        Args:
             exp: list of tuples [(id, weight), (id,weight)]
             label: label id (integer)
             div_name: name of div object to be used for rendering(in js)
             exp_object_name: name of js explanation object
        """

        exp = [(x[0], int(x[1])) for x in exp]
        ret = '''
            %s.show_raw_text(%s, %d, %s, %s);
            ''' % (exp_object_name, json.dumps(exp), label, div_name)
        return ret


class LimeTimeSeriesExplainer(object):
    """Explains text classifiers.
       Currently, we are using an exponential kernel on cosine distance, and
       restricting explanations to words that are present in documents."""

    def __init__(self,
                 kernel_width=25,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 split_expression=r'\W+',
                 bow=True):
        """Init function.

        Args:
            kernel_width: kernel width for the exponential kernel
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            split_expression: strings will be split by this.
            bow: if True (bag of words), will perturb input data by removing
                all ocurrences of individual words.  Explanations will be in
                terms of these words. Otherwise, will explain in terms of
                word-positions, so that a word may be important the first time
                it appears and uninportant the second. Only set to false if the
                classifier uses word order in some way (bigrams, etc).
        """

        # exponential kernel
        def kernel(d): return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.base = lime_base.LimeBase(kernel, verbose)
        self.class_names = class_names
        self.vocabulary = None
        self.feature_selection = feature_selection
        self.bow = bow
        self.split_expression = split_expression

    def explain_instance(self,
                         time_series,
                         classifier_fn,
                         labels=(1,),
                         top_labels=None,
                         num_ranges=2,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='cosine',
                         model_regressor=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly hiding features from
        the instance (see __data_labels_distance_mapping). We then learn
        locally weighted linear models on this neighborhood data to explain
        each of the classes in an interpretable way (see lime_base.py).

        Args:
            time_series: raw time series to be explained.
            classifier_fn: classifier prediction probability function, which
                takes a list of d strings and outputs a (d, k) numpy array with
                prediction probabilities, where k is the number of classes.
                For ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """

        domain_mapper = TimeSeriesDomainMapper(time_series)

        """
        y = time_series.copy().iloc[0]
        x = np.linspace(1, 275, 275)
        print(x)
        #z = np.cos(0.5 * (x[:-1] + x[1:]))  # first derivative

        # Create a colormap for red, green and blue and a norm to color
        # f' < -0.5 red, f' > 0.5 blue, and the rest green
        cmap = ListedColormap(['r', 'g', 'b'])
        norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)

        # Create a set of line segments so that we can color them individually
        # This creates the points as a N x 1 x 2 array so that we can stack points
        # together easily to get the segments. The segments array for line collection
        # needs to be numlines x points per line x 2 (x and y)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create the line collection object, setting the colormapping parameters.
        # Have to set the actual values used for colormapping separately.
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        #lc.set_array(z)
        lc.set_linewidth(3)

        fig1 = plt.figure()
        plt.gca().add_collection(lc)
        plt.xlim(x.min(), x.max())
        #plt.ylim(-1.1, 1.1)
        plt.show()
        """
       
        # visualize ranges
        section_width = math.floor(time_series.shape[1]/num_ranges)
        for section in range(0, num_ranges): 
            tmp = time_series.copy().iloc[0]
            for j in range(1, section_width):
                tmp[section*section_width+j] = 0
            print("FEATURE {}:".format(section))
            tmp.plot(legend=False)    
            plt.show()
        

        data, yss, distances = self.__data_labels_distances(
            time_series, classifier_fn, num_samples, num_ranges,
            distance_metric=distance_metric)

        print(data)
        print(yss)
        print(distances)
        
        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]
        ret_exp = explanation.Explanation(domain_mapper=domain_mapper,
                                          class_names=self.class_names)
        ret_exp.predict_proba = yss[0]
        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score) = self.base.explain_instance_with_data(
                data, yss, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)

        #features = ret_exp.as_list()
        #features.sort(key=lambda x: x[1], reverse=True)

        #self.visualize_range(features[0][0], time_series, num_ranges)

        return ret_exp

    def visualize_range(self, number, time_series, num_ranges):
        tmp = time_series.copy().iloc[0]
        range_width = math.floor(time_series.shape[1]/num_ranges)
        for j in range(1, range_width):
            tmp[number*range_width + j] = 0
        print("FEATURE {}:".format(number))
        tmp.plot(legend=False)    
        plt.show()


    @classmethod
    def __data_labels_distances(cls,
                                time_series,
                                classifier_fn,
                                num_samples,
                                num_ranges,
                                distance_metric='cosine'):
        """Generates a neighborhood around a prediction.

        Generates neighborhood data by randomly set points to zero from
        the instance, and predicting with the classifier. Uses cosine distance
        to compute distances between original and perturbed instances.
        Args:
            time_series: time series to be explained,
            classifier_fn: classifier prediction probability function, which
                takes a string and outputs prediction probabilities. For
                ScikitClassifier, this is classifier.predict_proba.
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity


        Returns:
            A tuple (data, labels, distances), where:
                data: dense num_samples * K binary matrix, where K is the
                    number of tokens in indexed_string. The first row is the
                    original instance, and thus a row of ones.
                labels: num_samples * L matrix, where L is the number of target
                    labels
                distances: cosine distance between the original instance and
                    each perturbed instance (computed in the binary 'data'
                    matrix), times 100.
        """

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0], metric=distance_metric).ravel() * 100

        series_length = time_series.shape[1]
        range_width = math.floor(series_length / num_ranges)
        sample = np.random.randint(1, high=num_ranges, size=num_samples - 1)

        print(sample)
        data = np.ones((num_samples, num_ranges))
        data[0] = np.ones(num_ranges)
        inverse_data = pd.DataFrame(columns=(i for i in range(series_length)))
        inverse_data.loc[0] = time_series.as_matrix()[0]

        for i, size in enumerate(sample, start=1):
            inactive = np.random.choice(range(num_ranges), size, replace=False)
            data[i, inactive] = 0
            time_series_neighbor = time_series.copy().as_matrix()[0]
            
            for nr, active in enumerate(data[i]):
                for j in range(0, range_width):
                    idx = nr * range_width + j
                    if not active:
                        time_series_neighbor[idx] = np.random.uniform(-5.0, 5.0)

            inverse_data.loc[i] = time_series_neighbor

        labels = classifier_fn(inverse_data)
        distances = distance_fn(sp.sparse.csr_matrix(data))
        return data, labels, distances
