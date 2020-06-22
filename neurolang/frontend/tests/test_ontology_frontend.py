import os
import urllib.request

import numpy as np
import pandas as pd
from neurolang import frontend as fe
from neurolang.frontend.ontology_frontend import NeurolangOntologyDL
from nilearn import datasets
from nilearn.datasets import utils


def test_someValuesFrom():
    d_onto = utils._get_dataset_dir("ontologies", data_dir="neurolang_data")

    if not os.path.exists(d_onto + "/COGAT.owl"):
        url = "http://data.bioontology.org/ontologies/COGAT/submissions/7/download?apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb"
        urllib.request.urlretrieve(url, d_onto + "/COGAT.owl")

    onto_paths = d_onto + "/COGAT.owl"
    nl = NeurolangOntologyDL()
    nl.load_ontology(onto_paths)

    part_of = nl.new_symbol(name="http://www.obofoundry.org/ro/ro.owl#part_of")

    with nl.scope as e:
        e.part[e.a, e.b] = part_of(e.a, e.b)

        res = nl.solve_query()

    a = 1
