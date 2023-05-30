import pandas as pd
import numpy as np
import logging
from spn.structure.StatisticalTypes import MetaType
from spn.algorithms.RSPMNnewAlgo import RSPMNnewAlgo
from spn.io.Graphics import plot_spn
logging.basicConfig(filename="FLNADC2.log", level=logging.DEBUG)

csv_path = "/home/ht65490/Desktop/SPFlow-H/src/spn/RSPMN_MDP_Datasets/FrozenLake/FrozenLake.csv"
df = pd.read_csv(csv_path, sep=",", header=None)
train_data = df.values

#Parameter Setting
partial_order = [["state"],["action"],["reward"]]
decision_nodes = ["action"]
utility_nodes = ["reward"]
feature_names = [var for var_set in partial_order for var in var_set]
meta_types = [MetaType.DISCRETE]*2+[MetaType.UTILITY]

#Structure Learning
rspmn = RSPMNnewAlgo(partial_order, decision_nodes, utility_nodes, feature_names, meta_types, cluster_by_curr_information_set=True, util_to_bin=False)
wrapped_two_timestep_data = rspmn.InitialTemplate.wrap_sequence_into_two_time_steps(train_data)
spmn_structure_two_time_steps, top_network, initial_template_network = rspmn.InitialTemplate.build_initial_template(wrapped_two_timestep_data)


#Learn Template Network
print("IT & TP Obtained, Learning IT to obtain FT")
template = rspmn.InitialTemplate.template_network
template = rspmn.hard_em(train_data, template, False)

#Plotting Learned Structure
plot_spn(spmn_structure_two_time_steps, "/home/ht65490/Desktop/SPFlow-H/src/spn/RSPMN_Plots/FL2StepNADC2.pdf", feature_labels=["State0", "Action0", "Reward0", "State1", "Action1", "Reward1"])
plot_spn(top_network, "/home/ht65490/Desktop/SPFlow-H/src/spn/RSPMN_Plots/FLTopNADC2.pdf", feature_labels=["State", "Action", "Reward"])
plot_spn(initial_template_network, "/home/ht65490/Desktop/SPFlow-H/src/spn/RSPMN_Plots/FLTemplateNADC2.pdf", feature_labels=["State", "Action", "Reward"])

#MEU
print("MEU Calculations via value iteration")
num_of_iterations = 300
meu_list, lls_list = rspmn.value_iteration(template, num_of_iterations)
test_data = [0, np.nan, np.nan]
test_data = np.array(test_data).reshape(1, len(test_data))
meu = rspmn.meu_of_state(rspmn.template, test_data, meu_list, lls_list)[0]
print(meu)