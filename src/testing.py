import pandas as pd    
csv_path = "/home/ht65490/Desktop/SPFlow-H/src/spn/RSPMN_MDP_Datasets/ComputerDiagnostician/Computer_diagnostician.tsv"
df = pd.read_csv(csv_path, sep='\t')
partial_order = [['System_State'], ['Rework_Decision'],
                 ['Logic_board_fail', 'IO_board_fail', 'Rework_Outcome', 
                 'Rework_Cost']]
utility_node = ['Rework_Cost']
decision_nodes = ['Rework_Decision']
feature_names = ['System_State', 'Rework_Decision', 'Logic_board_fail', 
                'IO_board_fail', 'Rework_Outcome', 'Rework_Cost']

from spn.structure.StatisticalTypes import MetaType
# Utility variable is the last variable. Other variables are of discrete type
meta_types = [MetaType.DISCRETE]*5+[MetaType.UTILITY]  
from spn.algorithms.SPMNDataUtil import align_data
import numpy as np

df1, column_titles = align_data(df, partial_order)  # aligns data in partial order sequence
col_ind = column_titles.index(utility_node[0]) 

df_without_utility = df1.drop(df1.columns[col_ind], axis=1)
from sklearn.preprocessing import LabelEncoder
# transform categorical string values to categorical numerical values
df_without_utility_categorical = df_without_utility.apply(LabelEncoder().fit_transform)  
df_utility = df1.iloc[:, col_ind]
df = pd.concat([df_without_utility_categorical, df_utility], axis=1, sort=False)

train_data = df.values
from spn.algorithms.SPMNDataUtil import align_data
import numpy as np

df1, column_titles = align_data(df, partial_order)  # aligns data in partial order sequence
col_ind = column_titles.index(utility_node[0]) 

df_without_utility = df1.drop(df1.columns[col_ind], axis=1)
from sklearn.preprocessing import LabelEncoder
# transform categorical string values to categorical numerical values
df_without_utility_categorical = df_without_utility.apply(LabelEncoder().fit_transform)  
df_utility = df1.iloc[:, col_ind]
df = pd.concat([df_without_utility_categorical, df_utility], axis=1, sort=False)

train_data = df.values
from spn.algorithms.SPMN import SPMN
spmn = SPMN(partial_order , decision_nodes, utility_node, feature_names, 
            meta_types, cluster_by_curr_information_set=True,
            util_to_bin = False)
spmn_structure = spmn.learn_spmn(train_data)    
from spn.io.Graphics import plot_spn
plot_spn(spmn_structure, "computer_diagonistic.pdf", feature_labels=['SS', 'DD', 'LBF', 'IBF', 'RO', 'RC'])
from spn.algorithms.MEU import meu
test_data = [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]
meu = meu(spmn_structure, test_data)