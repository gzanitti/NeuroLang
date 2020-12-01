# -*- coding: utf-8 -*-
r'''
NeuroLang Example based Implementing a NeuroSynth Query
====================================================

'''

import logging
import sys
from typing import Iterable
import warnings

from neurolang import frontend as fe
from neurolang.frontend import probabilistic_frontend as pfe
import nibabel as nib
from nilearn import datasets, image, plotting
import numpy as np
import pandas as pd

logger = logging.getLogger('neurolang.frontend')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stderr))
warnings.filterwarnings("ignore")


"""
Data preparation
----------------
"""

###############################################################################
# Load the MNI template and resample it to 4mm voxels

mni_t1 = nib.load(datasets.fetch_icbm152_2009()['t1'])
mni_t1_4mm = image.resample_img(mni_t1, np.eye(3) * 4)

""
julich_ontology_l = datasets.utils._fetch_files(
    datasets.utils._get_dataset_dir('julich_ontology'),
    [
        (
            'julich_ontology_l.xml',
            'https://raw.githubusercontent.com/NeuroLang/neurolang_data/main/Julich-Brain/WB/MPM/'
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
            'https://raw.githubusercontent.com/NeuroLang/neurolang_data/main/Julich-Brain/WB/MPM/'
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
            'https://raw.githubusercontent.com/NeuroLang/neurolang_data/main/Julich-Brain/WB/jubrain-ontology_22.json',
            {'move': 'jubrain_ontology.xml'}
        )
    ]
)[0]


""
def parse_region(elem, id_2_num, father=None, triples=[]):
    name = elem['name']
    if 'labelIndex' in elem:
        if elem['labelIndex'] is not None:
            index = int(elem['labelIndex'])
            if index in id_2_num:
                num = id_2_num[index]
                triples.append((name, 'labelIndex', num))
        
    for c in elem['children']:
        parse_region(c, id_2_num, father=name, triples=triples)
        
    return triples


###############################################################################
# Load Destrieux's atlas
import xml.etree.ElementTree as ET
import json

#path_julich = '/Users/gzanitti/Projects/INRIA/neurolang_data/Julich-Brain/julich_brain.hdf'
#path_julich = '/Users/gzanitti/Projects/INRIA/neurolang_data/Julich-Brain/julich_brain_resampled.hdf'
path_julich = datasets.utils._fetch_files(
    datasets.utils._get_dataset_dir('julich_brain'),
    [
        (
            'julich_brain_resampled.hdf',
            'https://github.com/NeuroLang/neurolang_data/raw/main/Julich-Brain/julich_brain_resampled.hdf',
            {'move': 'julich_brain_resampled.hdf'}
        )
    ]
)[0]

df_julich = pd.read_hdf(path_julich, key='data')


tree = ET.parse(julich_ontology_l)

id_2_num = {}
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


###############################################################################
# Load the NeuroSynth database

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
        np.linalg.inv(mni_t1_4mm.affine),
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


###############################################################################
# Load the CogAt ontology

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

###############################################################################
# Probabilistic Logic Programming in NeuroLang
# --------------------------------------------

nl = pfe.ProbabilisticFrontend()
nl.load_ontology(cogAt)


###############################################################################
# Adding new aggregation function to build a region overlay
from rdflib import RDFS

@nl.add_symbol
def agg_create_region_overlay(
    i: Iterable, j: Iterable, k: Iterable, p: Iterable
) -> fe.ExplicitVBR:
    voxels = np.c_[i, j, k]
    return fe.ExplicitVBROverlay(
        voxels, mni_t1_4mm.affine, p,
        image_dim=mni_t1_4mm.shape
    )


@nl.add_symbol
def agg_max(i: Iterable) -> float:
    return np.max(i)


part_of = nl.new_symbol(name='http://www.obofoundry.org/ro/ro.owl#part_of')
subclass_of = nl.new_symbol(name=str(RDFS.subClassOf))
label = nl.new_symbol(name=str(RDFS.label))
hasTopConcept = nl.new_symbol(name='http://www.w3.org/2004/02/skos/core#hasTopConcept')

@nl.add_symbol
def word_lower(name: str) -> str:
    return name.lower()


""
activations = nl.add_tuple_set(ns_database.values, name='activations')
terms = nl.add_tuple_set(ns_terms.values, name='terms')
docs = nl.add_uniform_probabilistic_choice_over_set(
        ns_docs.values, name='docs'
)

julich_ontology = nl.add_tuple_set(
    triples,
    name="julich_ontology",
)


for set_symbol in (
    'activations', 'terms', 'docs', 'julich_ontology'
):
    print(f"#{set_symbol}: {len(nl.symbols[set_symbol].value)}")

""
areas = df_julich.Area.unique()

""
#df_julich = df_julich[['PROB', 'Area', 'Hemis', 'ref', 'i', 'j', 'k']]
df_julich = df_julich[['Area', 'Hemis', 'ref', 'i', 'j', 'k']]

""
df_julich.head()

""
from glob import glob
paths = glob('./vinod/*')
print(paths)

for path in paths:
    name = path.split('/')[2].split('_1mm')[0]
    print(name)

""
from glob import glob
paths = glob('./vinod/*')

for path in paths:
    name = path.split('/')[2].split('_1mm')[0]
    print(name)
    
    img = image.load_img(path)
    img = image.resample_img(
        img, mni_t1_4mm.affine, interpolation='nearest'
    )

    img_data = img.get_fdata()
    img_unmaskes = np.nonzero(img_data)

    coords = []
    for v in zip(*img_unmaskes):
        prob = img_data[v[0]][v[1]][v[2]]
        print(v, prob)
        coords.append((tuple(v) + tuple([prob])))

    df_img = pd.DataFrame(coords, columns=['i', 'j', 'k', 'value'])
    
    vinod_image = nl.add_tuple_set(
        df_img.values,
        name='vinod_image'
    )
    
    with nl.scope as e:

        e.ontology_terms[e.onto_name] = (
            hasTopConcept[e.uri, e.cp] &
            label[e.uri, e.onto_name]
        )

        e.filtered_terms[e.lower_name] = (
            e.ontology_terms[e.term] &
            (e.lower_name == word_lower[e.term])
        )

        e.filtered_regions[e.d, e.i, e.j, e.k] = (
            e.vinod_image[e.i, e.j, e.k, ...] &
            e.activations[
                e.d, ..., ..., ..., ..., 'MNI', ..., ..., ..., ...,
                ..., ..., ..., e.i, e.j, e.k
            ]
        )

        e.term_prob[e.t, e.PROB[e.t]] = (
            e.filtered_regions[e.d, e.i, e.j, e.k]
            & e.terms[e.d, e.t]
            & e.docs[e.d]
        )

        e.result[e.term, e.PROB] = (
            e.filtered_terms[e.term] &
            e.term_prob[e.term, e.PROB]
        )


        res = nl.solve_all()
        c = res['result'].as_pandas_dataframe()
        
    c.to_hdf('vinod_filtered_results.hdf', key=name)

""
pd.read_hdf('vinod_filtered_results.hdf', key='L_PPC_6mm_-44_-42_52')

""

