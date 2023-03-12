import uproot
import numpy as np

import sys
import json


config_path = sys.argv[1]
with open(config_path) as config_f:
    config = json.load(config_f)

train_path = config['path_to_train']

f = uproot.open(train_path)
tree = f['Low_Tree']
nevents = tree.num_entries
print('total number of events:', nevents)

variable_transform = {}

track_variables = [
    'track_parent_idx', 
    'track_d0',
    'track_z0',
    'sinu_track_phi',
    'cosin_track_phi',
    'track_theta',
    'track_eta_layer_0',
    'track_eta_layer_1',
    'track_eta_layer_2',
    'track_eta_layer_3',
    'track_eta_layer_4',
    'track_eta_layer_5',
    'sinu_track_phi_layer_0',
    'sinu_track_phi_layer_1',
    'sinu_track_phi_layer_2',
    'sinu_track_phi_layer_3',
    'sinu_track_phi_layer_4',
    'sinu_track_phi_layer_5',
    'cosin_track_phi_layer_0',
    'cosin_track_phi_layer_1',
    'cosin_track_phi_layer_2',
    'cosin_track_phi_layer_3',
    'cosin_track_phi_layer_4',
    'cosin_track_phi_layer_5',
    'track_qoverp'
]

track_inputs = [
    'track_d0',
    'track_z0',
    'track_pt',
    'track_eta',
    'track_phi',
    'sinu_track_phi',
    'cosin_track_phi',
    'track_eta_layer_0',
    'track_eta_layer_1',
    'track_eta_layer_2',
    'track_eta_layer_3',
    'track_eta_layer_4',
    'track_eta_layer_5',
    'track_phi_layer_0',
    'track_phi_layer_1',
    'track_phi_layer_2',
    'track_phi_layer_3',
    'track_phi_layer_4',
    'track_phi_layer_5',
    'sinu_track_phi_layer_0',
    'sinu_track_phi_layer_1',
    'sinu_track_phi_layer_2',
    'sinu_track_phi_layer_3',
    'sinu_track_phi_layer_4',
    'sinu_track_phi_layer_5',
    'cosin_track_phi_layer_0',
    'cosin_track_phi_layer_1',
    'cosin_track_phi_layer_2',
    'cosin_track_phi_layer_3',
    'cosin_track_phi_layer_4',
    'cosin_track_phi_layer_5'
]


cell_variables = ['cell_x','cell_y','cell_z','cell_e','cell_eta','cosin_cell_phi','sinu_cell_phi','cell_particle_target','cell_parent_idx','cell_topo_idx']
        
cell_inputs = ['cell_x','cell_y','cell_z','cell_eta','cell_phi','cosin_cell_phi','sinu_cell_phi','cell_e']
        
particle_variables = ['particle_pdgid','particle_px','particle_py','particle_pz','particle_e',
                                    'particle_prod_x','particle_prod_y','particle_prod_z']



full_data_array = {}

# loading the part needed for the truth attention weights
full_data_array["particle_to_node_weight"] = tree["particle_to_node_weight"].array(library='np',entry_stop=nevents)
full_data_array["particle_to_node_idx"] = tree["particle_to_node_idx"].array(library='np',entry_stop=nevents)
print('loading data:')

# needed for track selection
full_data_array["track_not_reg"] = tree["track_not_reg"].array(library='np',entry_stop=nevents)
full_data_array["particle_to_track"] = tree["particle_to_track"].array(library='np',entry_stop=nevents)
full_data_array["particle_pdgid_noC"] =  np.copy(tree["particle_pdgid"].array(library='np',entry_stop=nevents))
full_data_array["particle_to_track"] = np.concatenate( full_data_array["particle_to_track"] )

# transform in -1 and 1
full_data_array["particle_to_track"] = np.where(full_data_array["particle_to_track"]==-1, full_data_array["particle_to_track"],1)


for var in  cell_variables+particle_variables+track_variables:
    newvar = ""
    if "cosin_" in var or "sinu_" in var:
        replace = ""
        if "cosin_" in var: replace = "cosin_"
        if "sinu_" in var:  replace = "sinu_"
        newvar = var.replace(replace, '')
        full_data_array[var] = np.copy(tree[newvar].array(library='np',entry_stop=nevents))
    else: 
        full_data_array[var] = tree[var].array(library='np',entry_stop=nevents)

    if "track" in var or "particle" in var:
        if var == "track_parent_idx":
            full_data_array["track_isMuon"] = np.copy(full_data_array["track_parent_idx"])

    if var == 'cell_x':
        n_cells = [len(x) for x in full_data_array[var]]
    elif var=='track_d0':
        n_tracks = [len(x) for x in full_data_array[var]]
    elif var=='particle_pdgid':
        n_particles = [len(x) for x in full_data_array[var]]
       
    # flatten the arrays
    full_data_array[var] = np.concatenate( full_data_array[var] )

    if newvar in ['cell_phi']:
        full_data_array[var][full_data_array[var] > np.pi] = full_data_array[var][full_data_array[var] > np.pi]-2*np.pi

    if "cosin_" in var: full_data_array[var] = np.cos(full_data_array[var]) 
    if "sinu_" in var: full_data_array[var]  = np.sin(full_data_array[var]) 

    if var in ['track_d0','track_z0']:
        full_data_array[var] = np.sign(full_data_array[var])*np.log(1+50.0*abs(full_data_array[var]))

    if var in ['cell_e','particle_e']:
        full_data_array[var] = np.log(full_data_array[var])

    if var in ['track_theta']:
        full_data_array['track_eta'] =  -np.log( np.tan( full_data_array[var]/2 )) 

    if var in ['track_qoverp']:
        full_data_array['track_pt'] =  np.log(np.abs(1./full_data_array["track_qoverp"]) * np.sin(full_data_array["track_theta"]))


# adding the raw phis as well. Computing it from cosin and sin, coz it guarentees that phi will be in (-pi, +pi)
for var in cell_inputs + track_inputs:
    if 'phi' in var and ('sinu' not in var and 'cosin' not in var):
        full_data_array[var] = np.arctan2(full_data_array['sinu_'+var], full_data_array['cosin_'+var])


# particle properties
particle_phi   = np.arctan2(full_data_array['particle_py'],full_data_array['particle_px'])
particle_p     = np.linalg.norm(np.column_stack([full_data_array['particle_px'],full_data_array['particle_py'],full_data_array['particle_pz']]),axis=1)
particle_theta = np.arccos( full_data_array['particle_pz']/particle_p)
particle_eta   =  -np.log( np.tan( particle_theta/2 )) 
particle_xhat = np.cos(particle_phi )
particle_yhat = np.sin(particle_phi )

particle_pt = particle_p*np.sin(particle_theta)
particle_pt = np.log(particle_pt)

full_data_array['particle_phi'] = particle_phi
full_data_array['particle_pt'] = particle_pt
full_data_array['particle_eta'] = particle_eta

cell_cumsum = np.cumsum([0] + n_cells)




# usual meanstd
for v in ['particle_pt','particle_eta','particle_phi'] + track_inputs + cell_inputs: 

    variable_transform[v] ={
        'mean' : np.mean(full_data_array[v].mean()),
        'std' : np.std(full_data_array[v].std())
    } 

    print('"'+v+'"'+': {"mean":',np.round(np.mean(full_data_array[v].mean()),5), ', "std":',np.round(np.mean(full_data_array[v].std()),5),' },')


# combined meanstd
combined_pt = []; combined_phi = []; combined_eta = []
for v in ['particle_pt','particle_eta','particle_phi'] + track_inputs + cell_inputs:
    if ('phi' in v) and ('cosin' not in v) and ('sinu' not in v):
        combined_phi.extend(full_data_array[v].tolist())
    elif 'eta' in v:
        combined_eta.extend(full_data_array[v].tolist())
    elif 'pt' in v:
        combined_pt.extend(full_data_array[v].tolist())

combined_pt, combined_eta, combined_phi = np.array(combined_pt), np.array(combined_eta), np.array(combined_phi)

print('"combined_pt": {"mean":', np.round(combined_pt.mean(),5), ',', '"std":', np.round(combined_pt.std(),5), "}")
print('"combined_eta": {"mean":', np.round(combined_eta.mean(),5), ',', '"std":', np.round(combined_eta.std(),5), "}")
print('"combined_phi": {"mean":', np.round(combined_phi.mean(),5), ',', '"std":', np.round(combined_phi.std(),5), "}")




# # topo meanstd
# topo_es = []
# for idx in range(nevents):
#     cell_start, cell_end = cell_cumsum[idx], cell_cumsum[idx+1]

#     cell_topo_idx  = full_data_array['cell_topo_idx'][cell_start:cell_end] - 1
#     n_topoclusters = int(max(cell_topo_idx)) + 1

#     cell_e  = np.exp(full_data_array['cell_e'][cell_start:cell_end])
#     topo_e = np.log(np.bincount(cell_topo_idx, weights=cell_e))

#     topo_es.append(topo_e)

# topo_es = np.hstack(topo_es)
# print('"topo_e": {"mean":', np.round(topo_es.mean(),5), ',', '"std":', np.round(topo_es.std(),5), "}")
