'''
File: fuzzy_matcher.py
Project: Projects
File Created: Saturday, 21st November 2020 2:22:53 am
Author: Sparsh Dutta (sparsh.dtt@gmail.com)
-----
Last Modified: Saturday, 21st November 2020 5:56:57 am
Modified By: Sparsh Dutta (sparsh.dtt@gmail.com>)
-----
Copyright 2020 Sparsh Dutta
'''
from tqdm.notebook import tqdm
from rapidfuzz import fuzz

import copy
import functools
import numpy as np
import pandas as pd
import textdistance


class FuzzyMatcher:
    """
        Generalized fuzzy matcher class 
        with built-in memoization support
    """
    def __init__(self, config):
        """
            Config structure required for the fuzzy matcher :-
            
                fuzzy_config = {
                                    'base' : {
                                        'data' : df_1,
                                        'identifier' : 'sess_id' ,
                                        'column' : 'combined_drug'
                                    },
                                    'comparator' : {
                                        'data' : df_2,
                                        'identifier' : 'drug_id' ,
                                        'column' : 'asset_name'
                                    },
                                    'top_n' : 3, 
                                    'metric' : 'jaro-winkler'
                }

        """
        # Initializing attributes from config
        ## Initializing base attributes
        self.base_data = copy.deepcopy(config['base']['data'])
        self.base_data_id = config['base']['identifier']
        self.base_data_column = config['base']['column']

        ## Initializing comparator attributes
        self.comparator_data = copy.deepcopy(config['comparator']['data'])
        self.comparator_data_id = config['comparator']['identifier']
        self.comparator_data_column = config['comparator']['column']

        ## Initializing general attributes
        self.top_n = config['top_n']
        self.metric = config['metric']

        # Dropping duplicates and resetting index
        self.base_data = self.base_data[[self.base_data_id, self.base_data_column]].drop_duplicates().reset_index(drop=True)
        self.comparator_data = self.comparator_data[[self.comparator_data_id, self.comparator_data_column]].drop_duplicates().reset_index(drop=True)

        # Clearing the cache
        self.get_fuzzy_distance.cache_clear()

    @functools.lru_cache(maxsize=100000)
    def get_fuzzy_distance(self, 
                           string_1, 
                           string_2, 
                           metric='jarowinkler'):
        """Function to calculate the fuzzy distance
           between two strings and return the match distance

        Args:
            string_1 ([str]): [base string]
            string_2 ([str]): [comparator string]
            metric (str, optional): ['jaro-winkler', 'lev-partial', 'lev-ratio']. Defaults to 'jaro-winkler'.

        Raises:
            NotImplementedError: [description]

        Returns:
            [float]: [similarity distance based on the chosen metric]
        """
        string_1 = str(string_1).strip().upper()
        string_2 = str(string_2).strip().upper()
        
        if string_1 == 'UNKNOWN' or string_1 == '-' or string_1 == 'NAN' or string_1 == '':
            return 0.0
        else:
            if string_2 == 'UNKNOWN' or string_2 == '-' or string_2 == 'NAN' or string_2 == '':
                return 0.0
            else:
                if metric == 'jaro-winkler':
                    return round(textdistance.jaro_winkler.similarity(string_1, string_2), 2)
                elif metric == 'lev-partial':
                    return round(fuzz.partial_ratio(string_1, string_2) / 100, 2)
                elif metric == 'lev-ratio':
                    return round(fuzz.ratio(string_1, string_2) / 100, 2)
                else:
                    print_this = "Please use available metric from given list -['jaro-winkler', 'lev-partial', 'lev-ratio']."
                    raise NotImplementedError(print_this)

    def calculate_fuzzy_distance(self):
        """
        """
        # First getting comparator strings and string identifier
        comparator_string_list = list(self.comparator_data[self.comparator_data_column].values)
        comparator_string_unique_id = list(self.comparator_data[self.comparator_data_id].values)

        # Output list
        output_list = list()


        for _, row in tqdm(self.base_data.iterrows(), total=self.base_data.shape[0]):
            
            # Creating the list for score, string_2, string_2_id
            score_list = list()
            string_2_list = list()
            string_2_id_list = list()

            for string_2_id, string_2 in zip(comparator_string_unique_id, comparator_string_list):
                score = self.get_fuzzy_distance(string_1=row[self.base_data_column], string_2=string_2, metric=self.metric)

                # Appending the items into the list
                score_list.append(score)
                string_2_list.append(string_2)
                string_2_id_list.append(string_2_id)
            
            # Selecting the top-n based on the score list
            top_n_idx = np.argsort(score_list)[::-1][:self.top_n]

            # Type casting list to array for faster indexing
            score_list = np.array(score_list)
            string_2_list = np.array(string_2_list)
            string_2_id_list = np.array(string_2_id_list)

            # Selecting the top n score, strings and there ids
            top_n_string_2_score_list = score_list[top_n_idx]
            top_n_string_2_list = string_2_list[top_n_idx]
            top_n_string_2_id_list = string_2_id_list[top_n_idx]
            
            temp_list = list()

            for string_2_id, string_2, string_2_score in zip(top_n_string_2_id_list, top_n_string_2_list, top_n_string_2_score_list):
                temp_list.append(string_2_id)
                temp_list.append(string_2)
                temp_list.append(string_2_score)

            output_list.append(temp_list)

        
        # Creating the final data feed
        self.result_df = copy.deepcopy(self.base_data)
        self.result_df['Output'] = output_list

        column_list = list()

        for index in range(self.top_n):
            column_list.append(str(self.comparator_data_id) + '_Match_' + str(index+1))
            column_list.append(str(self.comparator_data_column) + '_Match_' + str(index+1))
            column_list.append('Score_Match_' + str(index+1))

        # Final Output Creation
        self.result_df[column_list] = self.result_df['Output'].apply(pd.Series)
        self.result_df = self.result_df.drop(['Output'],axis=1).drop_duplicates().reset_index(drop=True)

    def run_accelerator(self):
        self.calculate_fuzzy_distance()
