# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: neurolang
#     language: python
#     name: neurolang
# ---

# +
#import logging
#from neurolang import probabilistic
#logger = logging.getLogger()
#fhandler = logging.FileHandler(filename='mylog.log', mode='a')
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#fhandler.setFormatter(formatter)
#logger.addHandler(fhandler)
#logger.setLevel(logging.DEBUG)

# +
from nilearn import datasets, image, plotting
import numpy as np
import nibabel as nib

julich_ontology_l = datasets.utils._fetch_files(
    datasets.utils._get_dataset_dir('julich_ontology'),
    [
        (
            'julich_ontology_l.xml',
            'https://github.com/NeuroLang/neurolang_data/raw/main/Julich-Brain/WB/22/MPM/'
            'JulichBrain_MPMAtlas_l_N10_nlin2Stdicbm152asym2009c_publicDOI_3f6407380a69007a54f5e13f3c1ba2e6.xml',
            {'move': 'julich_ontology_l.xml'}
        )
    ]
)[0]

julich_ontology_r = datasets.utils._fetch_files(
    datasets.utils._get_dataset_dir('julich_ontology'),
    [
        (
            'julich_ontology_r.xml',
            'https://github.com/NeuroLang/neurolang_data/raw/main/Julich-Brain/WB/22/MPM/'
            'JulichBrain_MPMAtlas_r_N10_nlin2Stdicbm152asym2009c_publicDOI_14622b49a715338ce96e96611d395646.xml',
            {'move': 'julich_ontology_r.xml'}
        )
    ]
)[0]

ontology = datasets.utils._fetch_files(
    datasets.utils._get_dataset_dir('julich_ontology'),
    [
        (
            'jubrain-ontology_22.json',
            'https://github.com/NeuroLang/neurolang_data/raw/main/Julich-Brain/WB/22/jubrain-ontology_22.json',
            {'move': 'jubrain-ontology_22.json'}
        )
    ]
)[0]

def parse_region(elem, id_2_num, father=None, triples=[]):
    name = elem['name']
    if 'labelIndex' in elem:
        if elem['labelIndex'] is not None:
            index = int(elem['labelIndex'])
            if index in id_2_num:
                num = id_2_num[index]
                triples.append((name, num))
            else:
                print(f'Este no esta: {index, name}')
        
    for c in elem['children']:
        parse_region(c, id_2_num, father=name, triples=triples)
        
    return triples


import json
import pandas as pd
import xml.etree.ElementTree as ET

tree = ET.parse(julich_ontology_l)

id_2_num = {}
for a in tree.iter():
    if a.tag == 'Structure':
        num = int(a.attrib['grayvalue'])
        id_ = int(a.attrib['id'])
        id_2_num[id_] = num

tree = ET.parse(julich_ontology_r)

for a in tree.iter():
    if a.tag == 'Structure':
        num = int(a.attrib['grayvalue'])
        id_ = int(a.attrib['id'])
        id_2_num[id_] = num


with open(ontology) as f:
    data = json.load(f)

regions = data['properties']['regions']
for elem in regions:
    triples = parse_region(elem, id_2_num)
    
    for n, r in [
        (13, 'GapMap Frontal-I (GapMap)'),
        (32, 'GapMap Frontal-to-Occipital (GapMap)'),
        (59, 'GapMap Temporal-to-Parietal (GapMap)'),
        (89, 'GapMap Frontal-II (GapMap)'),
        (95, 'GapMap Frontal-to-Temporal (GapMap)')
    ]:
        triples.append((r, n))
        
    f.close()   
    regions = pd.DataFrame(triples, columns=['r_name', 'r_number']).astype({'r_number': 'int32'}).sort_values('r_number')
    regions.drop_duplicates(inplace=True)
    

# +
regions2 = regions.copy()
regions2['r_number'] = regions2['r_number'] + 1000
regions2['hemis'] = 'l'
regions['hemis'] = 'r'

regions = pd.concat((regions, regions2))
# -

regions[regions.r_name.str.contains('IFG')]

analized_region_number = 1067
analized_region_name = 'Area 45 (IFG)'

# +
wb22_l = datasets.utils._fetch_files(
    datasets.utils._get_dataset_dir('julich'),
    [
        (
            'wb22_l.nii.gz',
            'https://github.com/NeuroLang/neurolang_data/raw/main/Julich-Brain/WB/22/MPM/'
            'JulichBrain_MPMAtlas_l_N10_nlin2Stdicbm152asym2009c_publicDOI_3f6407380a69007a54f5e13f3c1ba2e6.nii.gz',
            {'move': 'wb22_l.nii.gz'}
        )
    ]
)[0]

wb22_r = datasets.utils._fetch_files(
    datasets.utils._get_dataset_dir('julich'),
    [
        (
            'wb22_r.nii.gz',
            'https://github.com/NeuroLang/neurolang_data/raw/main/Julich-Brain/WB/22/MPM/'
            'JulichBrain_MPMAtlas_r_N10_nlin2Stdicbm152asym2009c_publicDOI_14622b49a715338ce96e96611d395646.nii.gz',
            {'move': 'wb22_r.nii.gz'}
        )
    ]
)[0]

# +
img_r = image.load_img(wb22_r)
img_l = image.load_img(wb22_l)
img_l_data = img_l.get_fdata()
img_r_data = img_r.get_fdata()
img_l_unmaskes = np.nonzero(img_l_data)

for v in zip(*img_l_unmaskes):
    value = img_l_data[v[0]][v[1]][v[2]]
    ex_value = img_r_data[v[0]][v[1]][v[2]]
    if ex_value == 0:
        img_r_data[v[0]][v[1]][v[2]] = value + 1000
    
conc_img = nib.spatialimages.SpatialImage(img_r_data, img_r.affine)
plotting.plot_roi(conc_img)

# +
mni_t1 = nib.load(datasets.fetch_icbm152_2009()['t1'])
mni_t1 = image.resample_img(mni_t1, np.eye(3) * 4)

conc_img = image.resample_img(
    conc_img, mni_t1.affine, interpolation='nearest'
)

conc_img_data = conc_img.get_fdata()
conc_img_unmaskes = np.nonzero(conc_img_data)

julich_brain = []
for v in zip(*conc_img_unmaskes):
    julich_brain.append((v[0], v[1], v[2], conc_img_data[v[0]][v[1]][v[2]]))
# -

d = conc_img.get_fdata()
mask = d != analized_region_number
d[mask] = 0
#d[193//2:] = 0
wb = nib.spatialimages.SpatialImage(d, conc_img.affine)

mask = wb.get_fdata() > 0
labels = wb.get_fdata()[mask]

# +
from scipy.ndimage import binary_dilation
import nibabel as nib

new_img = nib.spatialimages.SpatialImage(
    binary_dilation(wb.get_fdata()), 
    wb.affine
)

plotting.plot_roi(new_img)

# +
import numpy as np

jl_data = new_img.get_fdata()
jl_unmaskes = np.nonzero(jl_data)

xyz_to_jl_region = []
for v in zip(*jl_unmaskes):
    xyz_to_jl_region.append(tuple(v))
# -

cogAt = datasets.utils._fetch_files(
    datasets.utils._get_dataset_dir('CogAt'),
    [
        (
            'cogat.xml',
            'http://data.bioontology.org/ontologies/COGAT/download?'
            'apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb&download_format=rdf',
            {'move': 'cogat.xml'}
        )
    ]
)[0]

# +
from neurolang.frontend import NeurolangPDL

nl = NeurolangPDL()
nl.load_ontology(cogAt)

j_brain = nl.add_tuple_set(
    julich_brain,
    name='julich_brain'
)

j_regions = nl.add_tuple_set(
    regions.values,
    name='julich_regions'
)

dilated_ifg44 = nl.add_tuple_set(
    tuple(xyz_to_jl_region),
    name='dilated_ifg44'
)
# -

with nl.scope as e:
    e.local_areas[e.area, e.id] = (
        e.julich_brain[e.i, e.j, e.k, e.id] &
        e.dilated_ifg44[e.i, e.j, e.k,] &
        e.julich_regions[e.area, e.id, e.hemis]
    )

    res = nl.solve_all()
    local_areas = res['local_areas'].as_pandas_dataframe()


# +
#local_areas['area'].unique()

# +
for v in zip(*img_l_unmaskes):
    value = img_l_data[v[0]][v[1]][v[2]]
    ex_value = img_r_data[v[0]][v[1]][v[2]]
    if ex_value == 0:
        img_r_data[v[0]][v[1]][v[2]] = value + 1000
    
conc_img = nib.spatialimages.SpatialImage(img_r_data, img_r.affine)

mask = conc_img.get_fdata() > 0
labels = np.unique(conc_img.get_fdata()[mask])
# -

print(set(labels) - set(regions.r_number.unique()))
print(set(regions.r_number.unique()) - set(labels))

# +
for v in zip(*img_l_unmaskes):
    value = img_l_data[v[0]][v[1]][v[2]]
    ex_value = img_r_data[v[0]][v[1]][v[2]]
    if ex_value == 0:
        img_r_data[v[0]][v[1]][v[2]] = value + 1000
    
conc_img = nib.spatialimages.SpatialImage(img_r_data, img_r.affine)
conc_img = image.resample_img(
    conc_img, mni_t1.affine, interpolation='nearest'
)

func_conn = datasets.utils._fetch_files(
    datasets.utils._get_dataset_dir('neurosynth'),
    [
        (
            'functional_connectivity_-48_24_-10.nii.gz',
            'https://github.com/NeuroLang/neurolang_data/raw/main/NS'
            '/Locations/functional_connectivity_-48_24_-10.nii.gz',
            {'move': 'functional_connectivity_-48_24_-10.nii.gz'}
        )
    ]
)[0]

ns_conn = image.load_img(func_conn)
ns_conn = image.resample_img(
    ns_conn, mni_t1.affine, interpolation='nearest'
)


jl_data = conc_img.get_fdata()
#jl_unmaskes = np.nonzero(jl_data)

ns_data = ns_conn.get_fdata()
#ns_unmaskes = np.nonzero(ns_data)
jl_temp = jl_data[:48, :57, :46]
reg_pos = []
for reg in regions.r_number.values:
    mask = jl_temp == reg
    name = regions[regions.r_number == reg].r_name.values[0]
    reg_mean = np.arctan(ns_data[mask]).mean()
    reg_pos.append((name, reg, reg_mean))
    ns_data[mask] = reg_mean

# -

corr44 = pd.DataFrame(reg_pos, columns=['r_name', 'r_number', 'corr']).sort_values('corr', ascending=False)

corr44.head()

for l in local_areas['area'].unique():
    corr44 = corr44[corr44.r_name != l]
long_regions = corr44[corr44['corr'] >= corr44['corr'].quantile(.95)].sort_values('corr', ascending=False)
long_regions

# +
# Remover Etiqueta 103.0
#regions = regions[regions.r_number != 103]
#regions = regions[regions.r_number != 1103]

#from nilearn.input_data import NiftiLabelsMasker

#subjects = [101309, 104416,111009,119833, 114318,137431,147636,155938,165941,186141,
#            206828, 123420,223929,365343, 132017,445543,536647,656657,952863, 175439, 
#            196851, 299760, 727654, 800941, 141119, 150928, 159441, 169949, 178849,
#           190031, 200109, 210415, 251833, 320826]
#masker = NiftiLabelsMasker(labels_img=conc_img, standardize=True)

#params = zip(subjects, [masker] * len(subjects))

#import multiprocessing
#import os

#def load_hcp(params):
#    s, masker = params
#    print(s)
#    img = 
#    return masker.fit_transform(img)

#pool = multiprocessing.Pool(os.cpu_count())
#time_series = pool.map(load_hcp, params)

# +
#from nilearn.connectome import ConnectivityMeasure

#connectome_measure = ConnectivityMeasure(kind='correlation')
#correlation_matrices = connectome_measure.fit_transform([ns_conn])[0]

#del time_series

# +
# Display the correlation matrix
#import numpy as np
#from nilearn import plotting
#import matplotlib.pyplot as plt

# Mask out the major diagonal
#np.fill_diagonal(correlation_matrices, 0)
#plotting.plot_matrix(correlation_matrices, labels=regions.r_name.values, colorbar=True, figure=(10, 8))


# +
#correlation_matrices.shape

# +
#regions.r_name.values.shape

# +
#if analized_region_number > 1000:
#    ar = 124 + analized_region_number - 1001
#else:
#    ar = analized_region_number - 1

#for n, r in enumerate(regions.r_name.values):
#    if n == ar:
#        print(r)
#        print(analized_region_name)

# +
#regions[regions.r_name == analized_region_name]

# +
#area_44_con = correlation_matrices[ar]

# +
#regions['r_name'] = regions.r_name.apply(lambda x: x.replace('.', ''))
# -

#corr44 = pd.DataFrame(zip(area_44_con, regions.r_name.values, regions.r_number.values), columns=['corr', 'r_name', 'r_number'])


# +
#corr44[corr44['corr'] >= corr44['corr'].quantile(.95)].sort_values('corr', ascending=False)

# +
#for l in local_areas['area'].unique():
#    corr44 = corr44[corr44.r_name != l]

# +
#long_regions = corr44[corr44['corr'] >= corr44['corr'].quantile(.95)].sort_values('corr', ascending=False)
# -

long_regions = long_regions[['r_name', 'r_number']]
long_regions = long_regions.append(regions[regions.r_number == analized_region_number])

long_regions

# +
long_regions = long_regions[['r_name','r_number']]

long_areas = nl.add_tuple_set(
    long_regions.values,
    name='long_areas'
)

# +
ns_database_fn, ns_features_fn = datasets.utils._fetch_files(
    datasets.utils._get_dataset_dir('neurosynth'),
    [
        (
            'database.txt',
            'https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz',
            {'uncompress': True}
        ),
        (
            'features.txt',
            'https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz',
            {'uncompress': True}
        ),
    ]
)

ns_database = pd.read_csv(ns_database_fn, sep=f'\t')
ijk_positions = (
    np.round(nib.affines.apply_affine(
        np.linalg.inv(mni_t1.affine),
        ns_database[['x', 'y', 'z']].values.astype(float)
    )).astype(int)
)
ns_database['i'] = ijk_positions[:, 0]
ns_database['j'] = ijk_positions[:, 1]
ns_database['k'] = ijk_positions[:, 2]

ns_features = pd.read_csv(ns_features_fn, sep=f'\t')
ns_docs = ns_features[['pmid']].drop_duplicates()
ns_terms = (
    pd.melt(
            ns_features,
            var_name='term', id_vars='pmid', value_name='TfIdf'
       )
    .query('TfIdf > 1e-3')[['pmid', 'term']]
)

activations = nl.add_tuple_set(ns_database.values, name='activations')
terms = nl.add_tuple_set(ns_terms.values, name='terms')
docs = nl.add_uniform_probabilistic_choice_over_set(
        ns_docs.values, name='docs'
)

from rdflib import RDFS

part_of = nl.new_symbol(name='http://www.obofoundry.org/ro/ro.owl#part_of')
subclass_of = nl.new_symbol(name=str(RDFS.subClassOf))
label = nl.new_symbol(name=str(RDFS.label))
hasTopConcept = nl.new_symbol(name='http://www.w3.org/2004/02/skos/core#hasTopConcept')

# +
from typing import Iterable

@nl.add_symbol
def word_lower(name: str) -> str:
    return name.lower()


# -

# ## Long network

with nl.scope as e:
    e.ontology_terms[e.onto_name] = (
        hasTopConcept[e.uri, e.cp] &
        label[e.uri, e.onto_name]
    )
    
    e.filtered_terms[e.term] = (
        e.term == 'motor'
    )

    e.filtered_terms[e.term] = (
        e.ontology_terms[e.onto_term] &
        (e.term == word_lower[e.onto_term])
    )

    e.long_regions[e.d, e.id] = (
        e.long_areas[e.area, e.id] &
        e.julich_brain[e.i, e.j, e.k, e.id] &
        e.activations[
            e.d, ..., ..., ..., ..., 'MNI', ..., ..., ..., ...,
            ..., ..., ..., e.i, e.j, e.k
        ]
    )
    
    e.long_network[e.d, e.id] = (
        e.long_regions[e.d, analized_region_number] &
        e.long_regions[e.d, e.id] &
        (e.id != analized_region_number)
    )
    
    e.local_regions[e.d, e.id] = (
        e.dilated_ifg44[e.i, e.j, e.k,] &
        e.julich_regions[e.area, e.id] &
        e.julich_brain[e.i, e.j, e.k, e.id] &
        e.activations[
            e.d, ..., ..., ..., ..., 'MNI', ..., ..., ..., ...,
            ..., ..., ..., e.i, e.j, e.k
        ]
    )
    
    e.local_network[e.d, e.id] = (
        e.local_regions[e.d, e.id] &
        e.local_regions[e.d, analized_region_number] &
        (e.id != analized_region_number)
    )
    
    #e.just_local[e.d, e.id, e.id2] = (
    #    e.local_network[e.d, e.id] &
    #    (~e.long_network[e.d, e.id2]) &
    #    e.julich_regions[..., e.id2, ...]
    #)
    
    #e.just_long[e.d, e.id] = (
    #    e.long_network[e.d, e.id] &
    #    (~e.local_network[e.d, e.id2]) &
    #    e.julich_regions[..., e.id2, ...]
    #)
    
    #e.result_local[e.t, e.PROB[e.t]] = (
    #   (e.terms[e.d, e.t] & e.filtered_terms[e.t]  ) // 
    #    (e.docs[e.d] & e.just_local[e.d, e.id, e.id2])  
    #)
    
    #e.result_long[e.t, e.PROB[e.t]] = (
    #   (e.terms[e.d, e.t]   ) // 
    #    (e.docs[e.d] & e.just_long[e.d, e.id])  
    #)
    
    res = nl.solve_all()


filtered_terms = nl.add_tuple_set(res['filtered_terms'].as_pandas_dataframe().values, name='filtered_terms')
local_network = nl.add_tuple_set(res['local_network'].as_pandas_dataframe().values, name='local_network')
long_network = nl.add_tuple_set(res['long_network'].as_pandas_dataframe().values, name='long_network')

with nl.scope as e:
    
    e.just_long[e.d, e.id] = (
        e.long_network[e.d, e.id] &
        (~e.local_network[e.d, e.id2]) &
        e.julich_regions[..., e.id2, ...]
    )
    
    #e.result_long[e.t, e.PROB[e.t]] = (
    #   (e.terms[e.d, e.t] & e.filtered_terms[e.t] ) // 
    #    (e.docs[e.d] & e.just_long[e.d, e.id])  
    #)
    
    res = nl.solve_all()



# +
for v in zip(*img_l_unmaskes):
    value = img_l_data[v[0]][v[1]][v[2]]
    ex_value = img_r_data[v[0]][v[1]][v[2]]
    if ex_value == 0:
        img_r_data[v[0]][v[1]][v[2]] = value + 1000
    
conc_img = nib.spatialimages.SpatialImage(img_r_data, img_r.affine)

d = conc_img.get_fdata()
mask = None
num = 1
for a in res['just_long'].as_pandas_dataframe()['id'].unique():
    temp = d == a
    d[temp] = num
    num = num + 1
    if mask is None:
        mask = temp
    else:
        mask = mask + temp
d[~mask] = 0
wb = nib.spatialimages.SpatialImage(d, conc_img.affine)

#plotting.plot_roi(wb, display_mode='x', threshold=0.1, alpha=0.9, cmap='tab20b', cut_coords=np.linspace(-63, 63, 4))
plotting.view_img(wb)
# -

c_long = res['result_long']._container.copy()
c_long.rename({0:'t', 1:'PROB'}, inplace=True, axis=1)
c_long.drop_duplicates(['t'], inplace=True)
c_long = c_long[['t', 'PROB']]
final_local = c_long[c_long['PROB'] >= c_long['PROB'].quantile(.95)].sort_values('PROB', ascending=False)
final_local.reset_index(drop=True)

# +
import matplotlib.pyplot as plt

plt.plot(c_long.sort_values(['PROB'])['PROB'].values.T)
plt.axhline(c_long.PROB.quantile(.95))
# -





# ## Local network

with nl.scope as e:
    e.ontology_terms[e.onto_name] = (
        hasTopConcept[e.uri, e.cp] &
        label[e.uri, e.onto_name]
    )
    
    e.filtered_terms[e.term] = (
        e.term == 'motor'
    )

    e.filtered_terms[e.term] = (
        e.ontology_terms[e.onto_term] &
        (e.term == word_lower[e.onto_term])
    )

    e.long_regions[e.d, e.id] = (
        e.long_areas[e.area, e.id] &
        e.julich_brain[e.i, e.j, e.k, e.id] &
        e.activations[
            e.d, ..., ..., ..., ..., 'MNI', ..., ..., ..., ...,
            ..., ..., ..., e.i, e.j, e.k
        ]
    )
    
    e.long_network[e.d, e.id] = (
        e.long_regions[e.d, analized_region_number] &
        e.long_regions[e.d, e.id] &
        (e.id != analized_region_number)
    )
    
    e.local_regions[e.d, e.id] = (
        e.dilated_ifg44[e.i, e.j, e.k,] &
        e.julich_regions[e.area, e.id] &
        e.julich_brain[e.i, e.j, e.k, e.id] &
        e.activations[
            e.d, ..., ..., ..., ..., 'MNI', ..., ..., ..., ...,
            ..., ..., ..., e.i, e.j, e.k
        ]
    )
    
    e.local_network[e.d, e.id] = (
        e.local_regions[e.d, e.id] &
        e.local_regions[e.d, analized_region_number] &
        (e.id != analized_region_number)
    )
    
    e.just_local[e.d, e.id, e.id2] = (
        e.local_network[e.d, e.id] &
        (~e.long_network[e.d, e.id2]) &
        e.julich_regions[..., e.id2, ...]
    )
    
    e.result_local[e.t, e.PROB[e.t]] = (
       (e.terms[e.d, e.t] & e.filtered_terms[e.t]  ) // 
        (e.docs[e.d] & e.just_local[e.d, e.id, e.id2])  
    )
    
    res = nl.solve_all()


c_local = res['result_local']._container.copy()
c_local.rename({0:'t', 1:'PROB'}, inplace=True, axis=1)
c_local.drop_duplicates(['t'], inplace=True)
c_local = c_local[['t', 'PROB']]
final_local = c_local[c_local['PROB'] >= c_local['PROB'].quantile(.95)].sort_values('PROB', ascending=False)
final_local.reset_index(drop=True)

# +
import matplotlib.pyplot as plt

plt.plot(c_local.sort_values(['PROB'])['PROB'].values.T)
plt.axhline(c_local.PROB.quantile(.95))

# +
for v in zip(*img_l_unmaskes):
    value = img_l_data[v[0]][v[1]][v[2]]
    ex_value = img_r_data[v[0]][v[1]][v[2]]
    if ex_value == 0:
        img_r_data[v[0]][v[1]][v[2]] = value + 1000
    
conc_img = nib.spatialimages.SpatialImage(img_r_data, img_r.affine)

d = conc_img.get_fdata()
mask = None
num = 1
for a in res['just_local'].as_pandas_dataframe()['id'].unique():
    temp = d == a
    d[temp] = num
    num = num + 1
    if mask is None:
        mask = temp
    else:
        mask = mask + temp
d[~mask] = 0
wb = nib.spatialimages.SpatialImage(d, conc_img.affine)

#plotting.plot_roi(wb, display_mode='y', threshold=0.1, alpha=0.9, cmap='tab20b', cut_coords=np.linspace(-70, 70, 8))
plotting.view_img(wb)
