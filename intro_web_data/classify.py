#!/usr/bin/env python
# encoding: utf-8
"""
classify.py

Created by Hilary Mason on 2011-02-17.
Copyright (c) 2011 Hilary Mason. All rights reserved.
"""
import re, string

from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

class NaiveBayesClassifier(object):
    """
    Contents of this class object
    
    category_count
        arts: 6
        sports: 6
    feature_count
        again
            sports: 2
        against
            sports: 3
        agenc
            arts: 1
        ahead
            arts: 5
            sports: 1
        also
            arts: 1
            sports: 2
    """
    def __init__(self):
        self.feature_count = {}
        self.category_count = {}
    
    def probability(self, item, category):
        """
        probability: prob that an item (i.e. a whole string) is in a certain category
        According to Bayes Law, 
            the probability of C given S is equal to 
            the probability of S given C, 
            times the probability of C, 
            divided by the probability of S.
        Thus, having
          P_ｃs = [probability that string S belongs to a given category C?]
                  probability that document D is in category C
                  that is, the probability that given the set of words in D, they appear in category C
          ??P_ｃs = [probability that the category C belongs to the features in a string S]
          P_sｃ = [probaility that features in string S belong to category C]
                 probability that for a given category C, the features in string S appear in that category.
          P_s = [probability of finding the features of string S]
          P_c = [probability of finding category C]
        we get
          P_ｃs = P_c x P_sc / P_s

          P_c = category_prob = how many of category C / sum of all categories

          ??P_sc = Having a set of data with category C, what is the probaility that this feature belongs to this category? 
          P_sc = self.document_probability(item, category)
                 calculate the product of the probabilities for each word; that is, the likelihood that each word appears in category C
          ??P_s = should be 1 since the probablility that the feature is present is 100%
                  ?? Given the sample the evidence is a constant and thus scales both posteriors equally. It therefore does not affect classification and can be ignored.
 
        """
        category_prob = self.get_category_count(category) / sum(self.category_count.values())
        return self.document_probability(item, category) * category_prob
    
    def document_probability(self, item, category):
        features = self.get_features(item)
        
        p = 1
        for feature in features:
#             print "%s - %s - %s" % (feature, category, self.weighted_prob(feature, category))
            p *= self.weighted_prob(feature, category)
            
        return p
        
    def train_from_data(self, data):
        for category, documents in data.items():
            for doc in documents:
                self.train(doc, category)
                
#         print self.feature_count
        
        
    # def get_features(self, document):
    #     all_words = word_tokenize(document)
    #     all_words_freq = FreqDist(all_words)
    #     
    #     # print sorted(all_words_freq.items(), key=lambda(w,c):(-c, w))
    #     return all_words_freq
        
    def get_features(self, document):
        document = re.sub('[%s]' % re.escape(string.punctuation), '', document) # removes punctuation
        document = document.lower() # make everything lowercase
        all_words = [w for w in word_tokenize(document) if len(w) > 3 and len(w) < 16]
        p = PorterStemmer()
        all_words = [p.stem(w) for w in all_words]
        all_words_freq = FreqDist(all_words)
        
#         print sorted(all_words_freq.items(), key=lambda(w,c):(-c, w))
        return all_words_freq
        
    def increment_feature(self, feature, category):
        # This function 
        self.feature_count.setdefault(feature,{})
        self.feature_count[feature].setdefault(category, 0)
        self.feature_count[feature][category] += 1
        
    def increment_cat(self, category):
        """
        Increments category_count every time you call train()
        """
        self.category_count.setdefault(category, 0)
        self.category_count[category] += 1
        
    def get_feature_count(self, feature, category):
        """
        How many times do you find this category in this feature?
        """
        if feature in self.feature_count and category in self.feature_count[feature]:
            return float(self.feature_count[feature][category])
        else:
            return 0.0
            
    def get_category_count(self, category):
        """
        Just returns number of categories
        """
        if category in self.category_count:
            return float(self.category_count[category])
        else:
            return 0.0
    
    def feature_prob(self, f, category): # Pr(A|B)
        """
        In how many lines of our training data does this feature show up at all?
        Calculate number of occurances in the set of lines defined
        as representing this category divided by total number of lines in this category.
        --> Having a set of data with category C, what is the probaility
            that this feature belongs to this category?
        """
        if self.get_category_count(category) == 0:
            return 0
        fp = self.get_feature_count(f, category) / self.get_category_count(category)
        print "Feature: %s | Feature count: %s | Category count: %s | Feature probability: %s" % (f, self.get_feature_count(f, category), self.get_category_count(category), fp)
        return fp
        
    def weighted_prob(self, f, category, weight=1.0, ap=0.5):
        """
        ?? Smoothing?
        
        ??Increase probability for features occuring in both categories 
        
        Example 1:
            Feature: player
            Feature count: 2   (in category 'arts')
            Feature count: 6   (in category 'sports')
            totals (total number of occurances in training data for all categories): 8.0
            Category count: 6.0
            Feature probability: 0.333333333333
            Weighted Probability player: 0.351851851852
            self.category_count.keys(): ['arts', 'sports']
        
        Example 2:
            Feature: break 
            Feature count: 1.0  (in category 'arts')
            Feature count: 1.0  (in category 'sports')
            totals (total number of occurances in training data for all categories): 2.0
            Category count: 6.0 
            Feature probability: 0.166666666667
            Weighted Probability: 0.277777777778 
            self.category_count.keys(): ['arts', 'sports'] 
        """
        basic_prob = self.feature_prob(f, category)
        totals = sum([self.get_feature_count(f, category) for category in self.category_count.keys()])
        w_prob = ((weight*ap) + (totals * basic_prob)) / (weight + totals)
        print "Weighted Probability %s: %s | self.category_count.keys(): %s | totals: %s" % (f, w_prob, self.category_count.keys(), totals)
        return w_prob
            
    def train(self, item, category):
        features = self.get_features(item)
        
        for f in features:
            self.increment_feature(f, category)
        
        self.increment_cat(category)

if __name__ == '__main__':
    labels = ['arts', 'sports'] # these are the categories we want
    data = {}
    for label in labels:
        f = open(label, 'r')
        data[label] = f.readlines()
#         print len(data[label])
        f.close()

    nb = NaiveBayesClassifier()
    nb.train_from_data(data)
    print nb.probability("Early Friday afternoon, the lead negotiators for the N.B.A. and the players union will hold a bargaining session in Beverly Hills — the latest attempt to break a 12-month stalemate on a new labor deal.", 'arts')
    print nb.probability("Early Friday afternoon, the lead negotiators for the N.B.A. and the players union will hold a bargaining session in Beverly Hills — the latest attempt to break a 12-month stalemate on a new labor deal.", 'sports')


