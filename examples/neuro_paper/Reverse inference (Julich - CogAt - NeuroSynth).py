# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: neurolang
#     language: python
#     name: neurolang
# ---

# %%
import json

import pandas as pd
import nibabel as nib
import numpy as np
import xml.etree.ElementTree as ET

from nilearn import datasets, image
from neurolang.frontend import NeurolangPDL
from typing import Iterable
from sklearn.model_selection import KFold
from rdflib import RDFS

# %%
n_folds = 25
resample = 4
random_state = 42


# %%
def parse_region(elem, id_2_num, father=None, triples=[]):
    name = elem['name']
    if 'labelIndex' in elem:
        if elem['labelIndex'] is not None:
            index = int(elem['labelIndex'])
            if index in id_2_num:
                num = id_2_num[index]
                triples.append((name, num))
        
    for c in elem['children']:
        parse_region(c, id_2_num, father=name, triples=triples)
        
    return triples


# %%
# Ontology
julich_ontology_l = datasets.utils._fetch_files(
    datasets.utils._get_dataset_dir('julich_ontology'),
    [
        (
            'julich_ontology_l.xml',
            'https://raw.githubusercontent.com/NeuroLang/neurolang_data/main/Julich-Brain/WB/22/MPM/'
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
            'https://raw.githubusercontent.com/NeuroLang/neurolang_data/main/Julich-Brain/WB/22/MPM/'
            'JulichBrain_MPMAtlas_l_N10_nlin2Stdicbm152asym2009c_publicDOI_3f6407380a69007a54f5e13f3c1ba2e6.xml',
            {'move': 'julich_ontology_r.xml'}
        )
    ]
)[0]

jubrain_ontology = datasets.utils._fetch_files(
    datasets.utils._get_dataset_dir('julich_ontology'),
    [
        (
            'jubrain_ontology.xml',
            'https://raw.githubusercontent.com/NeuroLang/neurolang_data/main/Julich-Brain/WB/22/jubrain-ontology_22.json',
            {'move': 'jubrain_ontology.xml'}
        )
    ]
)[0]

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


with open(jubrain_ontology) as f:
    data = json.load(f)

regions = data['properties']['regions']
for elem in regions:
    triples = parse_region(elem, id_2_num)
    
    #for n, r in [
    #    (13, 'GapMap Frontal-I (GapMap)'),
    #    (32, 'GapMap Frontal-to-Occipital (GapMap)'),
    #    (59, 'GapMap Temporal-to-Parietal (GapMap)'),
    #    (89, 'GapMap Frontal-II (GapMap)'),
    #    (95, 'GapMap Frontal-to-Temporal (GapMap)')
    #]:
    #    triples.append((r, n))
        
    f.close()   
    regions = pd.DataFrame(triples, columns=['r_name', 'r_number']).astype({'r_number': 'int32'}).sort_values('r_number')
    regions.drop_duplicates(inplace=True)


# %%
regions2 = regions.copy()
regions2['r_number'] = regions2['r_number'] + 1000
regions2['hemis'] = 'l'
regions['hemis'] = 'r'

regions = pd.concat((regions, regions2))

# %%
## Atlas
mni_t1 = nib.load(datasets.fetch_icbm152_2009()['t1'])
mni_t1_4mm = image.resample_img(mni_t1, np.eye(3) * resample)

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

conc_img = image.resample_img(
    conc_img, mni_t1_4mm.affine, interpolation='nearest'
)

conc_img_data = conc_img.get_fdata()
conc_img_unmaskes = np.nonzero(conc_img_data)

julich_brain = []
for v in zip(*conc_img_unmaskes):
    julich_brain.append((v[0], v[1], v[2], conc_img_data[v[0]][v[1]][v[2]]))

# %%
# NeuroSynth
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
    nib.affines.apply_affine(
        np.linalg.inv(mni_t1_4mm.affine),
        ns_database[['x', 'y', 'z']]
    ).astype(int)
)
ns_database['i'] = ijk_positions[:, 0]
ns_database['j'] = ijk_positions[:, 1]
ns_database['k'] = ijk_positions[:, 2]

ns_features = pd.read_csv(ns_features_fn, sep=f'\t')
ns_terms = (
    pd.melt(
            ns_features,
            var_name='term', id_vars='pmid', value_name='TfIdf'
       )
    .query('TfIdf > 1e-3')[['pmid', 'term']]
)
ns_docs = ns_features[['pmid']].drop_duplicates()

# %%
# CogAt
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

# %%
nl = NeurolangPDL()
nl.load_ontology(cogAt)

@nl.add_symbol
def agg_max(i: Iterable) -> float:
    return np.max(i)

@nl.add_symbol
def mean(iterable: Iterable) -> float:
    return np.mean(iterable)


@nl.add_symbol
def std(iterable: Iterable) -> float:
    return np.std(iterable)


part_of = nl.new_symbol(name='http://www.obofoundry.org/ro/ro.owl#part_of')
subclass_of = nl.new_symbol(name=str(RDFS.subClassOf))
label = nl.new_symbol(name=str(RDFS.label))
hasTopConcept = nl.new_symbol(name='http://www.w3.org/2004/02/skos/core#hasTopConcept')

@nl.add_symbol
def word_lower(name: str) -> str:
    return name.lower()


kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

ns_doc_folds = pd.concat(
    ns_docs.iloc[train].assign(fold=[i] * len(train))
    for i, (train, _) in enumerate(kfold.split(ns_docs))
)
doc_folds = nl.add_tuple_set(ns_doc_folds, name='doc_folds')


activations = nl.add_tuple_set(ns_database.values, name='activations')
terms = nl.add_tuple_set(ns_terms.values, name='terms')
docs = nl.add_uniform_probabilistic_choice_over_set(
        ns_docs.values, name='docs'
)

terms_det = nl.add_tuple_set(
        ns_terms.term.unique(), name='terms_det'
)

j_brain = nl.add_tuple_set(
    julich_brain,
    name='julich_brain'
)

j_regions = nl.add_tuple_set(
    regions.values,
    name='julich_regions'
)

# %%
with nl.scope as e:

    e.ontology_terms[e.onto_name] = (
    hasTopConcept[e.uri, e.cp] &
    label[e.uri, e.onto_name]
    )

    e.lower_terms[e.term] = (
        e.ontology_terms[e.onto_term] &
        (e.term == word_lower[e.onto_term])
    )

    e.filtered_terms[e.d, e.t] = (
        e.terms[e.d, e.t] &
        e.lower_terms[e.t]
    )

    f_term = nl.solve_all()

# %%
filtered = f_term['filtered_terms'].as_pandas_dataframe()
filtered_terms = nl.add_tuple_set(filtered.values, name='filtered_terms')

terms_det = nl.add_tuple_set(
        filtered.t.unique(), name='terms_det'
)

# %%
regions = regions[regions.r_number != 103] #Remove amygdala (error)

# %%
import datetime

from tqdm.notebook import tqdm
import os.path

for name, id_region, _ in tqdm(regions.values):
    print(f'{id_region} - {name}')
    start_time = datetime.datetime.now()
    if os.path.isfile('reverse_inference_results/neuro_paper_ri_no_term_probs_region{id_region}_{n_folds}folds.hdf'):
        end_time = datetime.datetime.now()
        print(f'{id_region} - {name}: {end_time - start_time}')
        print('--------------')
        continue
    
    with nl.scope as e:

        e.act_regions[e.d, id_region] = (
            e.julich_brain[e.i, e.j, e.k, id_region] &
            e.activations[
                e.d, ..., ..., ..., ..., 'MNI', ..., ..., ..., ...,
                ..., ..., ..., e.i, e.j, e.k
            ]
        )
        
        e.no_act_regions[e.d, e.id] = (
            ~(e.act_regions[e.d, e.id]) &
             e.doc_folds[e.d, ...] &
            e.julich_regions[..., e.id]
        )
        
        e.term_prob[e.t, e.fold, e.PROB[e.t, e.fold]] = (
            (
                e.filtered_terms[e.d, e.t]
            ) // (
                e.act_regions[e.d, id_region] &
                e.doc_folds[e.d, e.fold] &
                e.docs[e.d]
            )
        )

        e.no_term_prob[e.t, e.fold, e.PROB[e.t, e.fold]] = (
           (
                e.filtered_terms[e.d, e.t]
            ) // (
                e.no_act_regions[e.d, id_region] &
                e.doc_folds[e.d, e.fold] &
                e.docs[e.d]
            )
        )

        
        res = nl.solve_all()
        
        end_time = datetime.datetime.now()
        print(f'{id_region} - {name}: {end_time - start_time}')
        
        pss = res['term_prob'].as_pandas_dataframe()
        
        pss.to_hdf(f'reverse_inference_results/neuro_paper_ri_term_probs_region{id_region}_{n_folds}folds.hdf', key=f'results')
        
        pss = res['no_term_prob'].as_pandas_dataframe()
        
        pss.to_hdf(f'reverse_inference_results/neuro_paper_ri_no_term_probs_region{id_region}_{n_folds}folds.hdf', key=f'results')

# %%
