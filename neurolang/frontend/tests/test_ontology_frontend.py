import os
import urllib.request
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from neurolang import frontend as fe
from neurolang.datalog.ontologies_parser import OntologyParser
from neurolang.frontend.ontology_frontend import NeurolangOntologyDL
from nilearn import datasets, plotting
from nilearn.datasets import utils
from rdflib import RDFS


def destrieux_name_to_fma_relations():
    return [
        ("l_g_and_s_frontomargin", "Left frontomarginal gyrus"),
        ("l_g_and_s_occipital_inf", "Left inferior occipital gyrus"),
        ("l_g_and_s_paracentral", "Left paracentral lobule"),
        ("l_g_and_s_subcentral", "Left subcentral gyrus"),
        (
            "l_g_and_s_transv_frontopol",
            "Left superior transverse frontopolar gyrus",
        ),
        ("l_g_and_s_cingul_ant", "Left anterior cingulate gyrus"),
        ("l_g_and_s_cingul_mid_ant", "Left anterior middle cingulate gyrus"),
        ("l_g_and_s_cingul_mid_post", "Left posterior middle cingulate gyrus"),
        (
            "l_g_cingul_post_dorsal",
            "Dorsal segment of left posterior middle cingulate gyrus",
        ),
        (
            "l_g_cingul_post_ventral",
            "Ventral segment of left posterior middle cingulate gyrus",
        ),
        ("l_g_cuneus", "Left cuneus"),
        (
            "l_g_front_inf_opercular",
            "Opercular part of left inferior frontal gyrus",
        ),
        (
            "l_g_front_inf_orbital",
            "Orbital part of left inferior frontal gyrus",
        ),
        (
            "l_g_front_inf_triangul",
            "Triangular part of left inferior frontal gyrus",
        ),
        ("l_g_front_middle", "Left middle frontal gyrus"),
        ("l_g_front_sup", "Left superior frontal gyrus"),
        ("l_g_ins_lg_and_s_cent_ins", "Left central insular sulcus"),
        ("l_g_ins_lg_and_s_cent_ins", "Left long insular gyrus"),
        ("l_g_insular_short", "Short insular gyrus"),
        ("l_g_occipital_middleLeft", " 	Left lateral occipital gyrus"),
        ("l_g_occipital_sup", "Left superior occipital gyrus"),
        ("l_g_oc_temp_lat_fusifor", "Left fusiform gyrus"),
        ("l_g_oc_temp_med_lingual", "Left lingual gyrus"),
        ("l_g_oc_temp_med_parahip", "Left parahippocampal gyrus"),
        ("l_g_orbital", "Left orbital gyrus"),
        ("l_g_pariet_inf_angular", "Left angular gyrus"),
        ("l_g_pariet_inf_supramar", "Left supramarginal gyrus"),
        ("l_g_parietal_sup", "Left superior parietal lobule"),
        ("l_g_postcentral", "Left postcentral gyrus"),
        ("l_g_precentral", "Left precentral gyrus"),
        ("l_g_precuneus", "Left precuneus"),
        ("l_g_rectus", "Left straight gyrus"),
        ("l_g_subcallosal", "Left paraterminal gyrus"),
        ("l_g_temp_sup_g_t_transv", "Left transverse temporal gyrus"),
        ("l_g_temp_sup_lateral", "Left superior temporal gyrus"),
        ("l_g_temp_sup_plan_polar", "Left superior temporal gyrus"),
        ("l_g_temp_sup_plan_tempo", "Left superior temporal gyrus"),
        ("l_g_temporal_inf", "Left inferior temporal gyrus"),
        ("l_g_temporal_middle", "Left middle temporal gyrus"),
        (
            "l_lat_fis_ant_horizont",
            "Anterior horizontal limb of left lateral sulcus",
        ),
        (
            "l_lat_fis_ant_vertical",
            "Anterior ascending limb of left lateral sulcus",
        ),
        ("l_lat_fis_post", "Posterior ascending limb of left lateral sulcus"),
        ("l_lat_fis_post", "Left lateral sulcus"),
        ("l_pole_occipital", "Left occipital pole"),
        ("l_pole_temporal", "Left temporal pole"),
        ("l_s_calcarine", "Left Calcarine sulcus"),
        ("l_s_central", "Left central sulcus"),
        ("l_s_cingul_marginalis", "Left marginal sulcus"),
        ("l_s_circular_insula_ant", "Circular sulcus of left insula"),
        ("l_s_circular_insula_inf", "Circular sulcus of left insula"),
        ("l_s_circular_insula_sup", "Circular sulcus of left insula"),
        ("l_s_collat_transv_ant", "Left collateral sulcus"),
        ("l_s_collat_transv_post", "Left collateral sulcus"),
        ("l_s_front_inf", "Left inferior frontal sulcus"),
        ("l_s_front_sup", "Left superior frontal sulcus"),
        ("l_s_intrapariet_and_p_trans", "Left intraparietal sulcus"),
        ("l_s_oc_middle_and_lunatus", "Left lunate sulcus"),
        ("l_s_oc_sup_and_transversal", "Left transverse occipital sulcus"),
        ("l_s_occipital_ant", "Left anterior occipital sulcus"),
        ("l_s_oc_temp_lat", "Left occipitotemporal sulcus"),
        ("l_s_oc_temp_med_and_lingual", "Left intralingual sulcus"),
        ("l_s_orbital_lateral", "Left orbital sulcus"),
        ("l_s_orbital_med_olfact", "Left olfactory sulcus"),
        ("l_s_orbital_h_shaped", "Left transverse orbital sulcus"),
        ("l_s_orbital_h_shaped", "Left orbital sulcus"),
        ("l_s_parieto_occipital", "Left parieto-occipital sulcus"),
        ("l_s_pericallosal", "Left callosal sulcus"),
        ("l_s_postcentral", "Left postcentral sulcus"),
        ("l_s_precentral_inf_part", "Left precentral sulcus"),
        ("l_s_precentral_sup_part", "Left precentral sulcus"),
        ("l_s_suborbital", "Left fronto-orbital sulcus"),
        ("l_s_subparietal", "Left subparietal sulcus"),
        ("l_s_temporal_inf", "Left inferior temporal sulcus"),
        ("l_s_temporal_sup", "Left superior temporal sulcus"),
        ("l_s_temporal_transverse", "Left transverse temporal sulcus"),
        ("r_g_and_s_frontomargin", "Right frontomarginal gyrus"),
        ("r_g_and_s_occipital_inf", "Right inferior occipital gyrus"),
        ("r_g_and_s_paracentral", "Right paracentral lobule"),
        ("r_g_and_s_subcentral", "Right subcentral gyrus"),
        (
            "r_g_and_s_transv_frontopol",
            "Right superior transverse frontopolar gyrus",
        ),
        ("r_g_and_s_cingul_ant", "Right anterior cingulate gyrus"),
        ("r_g_and_s_cingul_mid_ant", "Right anterior middle cingulate gyrus"),
        (
            "r_g_and_s_cingul_mid_post",
            "Right posterior middle cingulate gyrus",
        ),
        (
            "r_g_cingul_post_dorsal",
            "Dorsal segment of right posterior middle cingulate gyrus",
        ),
        (
            "r_g_cingul_post_ventral",
            "Ventral segment of right posterior middle cingulate gyrus",
        ),
        ("r_g_cuneus", "Right cuneus"),
        (
            "r_g_front_inf_opercular",
            "Opercular part of right inferior frontal gyrus",
        ),
        (
            "r_g_front_inf_orbital",
            "Orbital part of right inferior frontal gyrus",
        ),
        (
            "r_g_front_inf_triangul",
            "Triangular part of right inferior frontal gyrus",
        ),
        ("r_g_front_middle", "Right middle frontal gyrus"),
        ("r_g_front_sup", "Right superior frontal gyrus"),
        ("r_g_ins_lg_and_s_cent_ins", "Right central insular sulcus"),
        ("r_g_ins_lg_and_s_cent_ins", "Right long insular gyrus"),
        ("r_g_insular_short", "Right short insular gyrus"),
        ("r_g_occipital_middle", "Right lateral occipital gyrus"),
        ("r_g_occipital_sup", "Right superior occipital gyrus"),
        ("r_g_oc_temp_lat_fusifor", "Right fusiform gyrus"),
        ("r_g_oc_temp_med_lingual", "Right lingual gyrus"),
        ("r_g_oc_temp_med_parahip", "Right parahippocampal gyrus"),
        ("r_g_orbital", "Right orbital gyrus"),
        ("r_g_pariet_inf_angular", "Right angular gyrus"),
        ("r_g_pariet_inf_supramar", "Right supramarginal gyrus"),
        ("r_g_parietal_sup", "Right superior parietal lobule"),
        ("r_g_postcentral", "Right postcentral gyrus"),
        ("r_g_precentral", "Right precentral gyrus"),
        ("r_g_precuneus", "Right precuneus"),
        ("r_g_rectus", "Right straight gyrus"),
        ("r_g_subcallosal", "Right paraterminal gyrus"),
        ("r_g_temp_sup_g_t_transv", "Right transverse temporal gyrus"),
        ("r_g_temp_sup_lateral", "Right superior temporal gyrus"),
        ("r_g_temp_sup_plan_polar", "Right superior temporal gyrus"),
        ("r_g_temp_sup_plan_tempo", "Right superior temporal gyrus"),
        ("r_g_temporal_inf", "Right inferior temporal gyrus"),
        ("r_g_temporal_middle", "Right middle temporal gyrus"),
        (
            "r_lat_fis_ant_horizont",
            "Anterior horizontal limb of right lateral sulcus",
        ),
        (
            "r_lat_fis_ant_vertical",
            "Anterior ascending limb of right lateral sulcus",
        ),
        ("r_lat_fis_post", "Right lateral sulcus"),
        ("r_lat_fis_post", "Posterior ascending limb of right lateral sulcus"),
        ("r_pole_occipital", "Right occipital pole"),
        ("r_pole_temporal", "Right temporal pole"),
        ("r_s_calcarine", "Right Calcarine sulcus"),
        ("r_s_central", "Right central sulcus"),
        ("r_s_cingul_marginalis", "Right marginal sulcus"),
        ("r_s_circular_insula_ant", "Circular sulcus of Right insula"),
        ("r_s_circular_insula_inf", "Circular sulcus of Right insula"),
        ("r_s_circular_insula_sup", "Circular sulcus of Right insula"),
        ("r_s_collat_transv_ant", "Right collateral sulcus"),
        ("r_s_collat_transv_post", "Right collateral sulcus"),
        ("r_s_front_inf", "Right inferior frontal sulcus"),
        ("r_s_front_sup", "Right superior frontal sulcus"),
        ("r_s_intrapariet_and_p_trans", "Right intraparietal sulcus"),
        ("r_s_oc_middle_and_lunatus", "Right lunate sulcus"),
        ("r_s_oc_sup_and_transversal", "Right transverse occipital sulcus"),
        ("r_s_occipital_ant", "Right anterior occipital sulcus"),
        ("r_s_oc_temp_lat", "Right occipitotemporal sulcus"),
        ("r_s_oc_temp_med_and_lingual", "Right intralingual sulcus"),
        ("r_s_orbital_lateral", "Right orbital sulcus"),
        ("r_s_orbital_med_olfact", "Right olfactory sulcus"),
        ("r_s_orbital_h_shaped", "Right orbital sulcus"),
        ("r_s_orbital_h_shaped", "Right transverse orbital sulcus"),
        ("r_s_parieto_occipital", "Right parieto-occipital sulcus"),
        ("r_s_pericallosal", "Right callosal sulcus"),
        ("r_s_postcentral", "Right postcentral sulcus"),
        ("r_s_precentral_inf_part", "Right precentral sulcus"),
        ("r_s_precentral_sup_part", "Right precentral sulcus"),
        ("r_s_suborbital", "Right fronto-orbital sulcus"),
        ("r_s_subparietal", "Right subparietal sulcus"),
        ("r_s_temporal_inf", "Right inferior temporal sulcus"),
        ("r_s_temporal_sup", "Right superior temporal sulcus"),
        ("r_s_temporal_transverse", "Right transverse temporal sulcus"),
    ]


def test_entailment():
    d_fma = utils._get_dataset_dir("neuro_fma", data_dir="neurolang_data")

    if not os.path.exists(d_fma + "/neurofma_fma3.0.owl"):
        url = "http://data.bioontology.org/ontologies/NeuroFMA/submissions/1/download?apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb"
        urllib.request.urlretrieve(url, d_fma + "/neurofma_fma3.0.owl")

    nl = NeurolangOntologyDL()
    nl.load_ontology(d_fma + "/neurofma_fma3.0.owl")

    label = nl.new_symbol(name=str(RDFS.label))
    subclass_of = nl.new_symbol(name=str(RDFS.subClassOf))
    regional_part = nl.new_symbol(
        name="http://sig.biostr.washington.edu/fma3.0#regional_part_of"
    )

    d_neurosynth = utils._get_dataset_dir(
        "neurosynth", data_dir="neurolang_data"
    )

    f_neurosynth = utils._fetch_files(
        d_neurosynth,
        [
            (
                f,
                "https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz",
                {"uncompress": True},
            )
            for f in ("database.txt", "features.txt")
        ],
        verbose=True,
    )

    database = pd.read_csv(f_neurosynth[0], sep="\t")
    features = pd.read_csv(f_neurosynth[1], sep="\t")

    features_normalised = features.melt(
        id_vars=features.columns[0],
        var_name="term",
        value_vars=features.columns[1:],
        value_name="tfidf",
    ).query("tfidf > 0")

    nsh = fe.neurosynth_utils.NeuroSynthHandler()
    ns_ds = nsh.ns_load_dataset()
    it = ns_ds.image_table
    vox_ids, study_ids_ix = it.data.nonzero()
    study_ids = ns_ds.image_table.ids[study_ids_ix]
    study_id_vox_id = np.transpose([study_ids, vox_ids])
    masked_ = it.masker.unmask(np.arange(it.data.shape[0]))
    nnz = masked_.nonzero()
    vox_id_MNI = np.c_[
        masked_[nnz].astype(int),
        nib.affines.apply_affine(it.masker.volume.affine, np.transpose(nnz)),
        [
            fe.ExplicitVBR(
                [v],
                affine_matrix=it.masker.volume.affine,
                image_dim=it.masker.volume.shape,
            )
            for v in zip(*nnz)
        ],
    ]

    from nilearn import image

    dd = datasets.fetch_atlas_destrieux_2009()
    destrieux_to_ns_mni = image.resample_to_img(
        dd["maps"], it.masker.volume, interpolation="nearest"
    )
    dd_data = destrieux_to_ns_mni.get_fdata()
    dd_unmaskes = np.where(destrieux_to_ns_mni.get_fdata() > 0)

    xyz_to_dd_region = []
    for v in zip(*dd_unmaskes):
        region = dd_data[v[0]][v[1]][v[2]]
        xyz_to_dd_region.append((v, region))

    dd_labels = []
    for n, name in dd["labels"]:
        dd_labels.append(
            (
                n,
                name.decode("UTF-8")
                .replace(" ", "_")
                .replace("-", "_")
                .lower(),
            )
        )

    xyz_to_ns_region = []
    for n, _, _, _, region in vox_id_MNI:
        xyz_to_ns_region.append((tuple(region.voxels[0]), n))

    @nl.add_symbol
    def agg_count(x: Iterable) -> int:
        return len(x)

    @nl.add_symbol
    def agg_sum(x: Iterable) -> float:
        return x.sum()

    @nl.add_symbol
    def agg_mean(x: Iterable) -> float:
        return x.mean()

    @nl.add_symbol
    def agg_create_region(
        x: Iterable, y: Iterable, z: Iterable
    ) -> fe.ExplicitVBR:
        mni_t1 = it.masker.volume
        voxels = nib.affines.apply_affine(
            np.linalg.inv(mni_t1.affine), np.c_[x, y, z]
        )
        return fe.ExplicitVBR(voxels, mni_t1.affine, image_dim=mni_t1.shape)

    ns_pmid_term_tfidf = nl.add_tuple_set(
        features_normalised.values, name="ns_pmid_term_tfidf"
    )
    ns_activations = nl.add_tuple_set(
        database[["id", "x", "y", "z", "space"]].values, name="ns_activations"
    )
    ns_activations_by_id = nl.add_tuple_set(
        study_id_vox_id, name="ns_activations_by_id"
    )
    ns_vox_id_MNI = nl.add_tuple_set(vox_id_MNI, name="ns_vox_id_MNI")

    xyz_ns = nl.add_tuple_set(xyz_to_ns_region, name="xyz_ns")
    xyz_dd = nl.add_tuple_set(xyz_to_dd_region, name="xyz_dd")
    dd_label = nl.add_tuple_set(dd_labels, name="dd_label")

    ds = destrieux_name_to_fma_relations()
    nl.add_tuple_set(
        [(dsname, onto) for dsname, onto in ds], name="relation_name"
    )

    with nl.scope as e:
        e.fma_related_region[e.fma_subregion, e.fma_name] = (
            label(e.xfma_entity_name, e.fma_name)
            & regional_part(e.fma_region, e.xfma_entity_name)
            & subclass_of(e.fma_subregion, e.fma_region)
        )

        e.fma_related_region[e.recursive_region, e.fma_name] = subclass_of(
            e.recursive_region, e.fma_subregion
        ) & e.fma_related_region(e.fma_subregion, e.fma_name)

        e.fma_to_destrieux[e.fma_name, e.destrieux_name] = label(
            e.fma_name, e.fma_uri
        ) & e.relation_name(e.destrieux_name, e.fma_uri)

        e.dd_to_ns[e.dd_name, e.ns_id, e.xyz] = (
            xyz_dd[e.xyz, e.dd_name] & xyz_ns[e.xyz, e.ns_id]
        )

        e.term_docs[e.term, e.pmid] = (
            ns_pmid_term_tfidf[e.pmid, e.term, e.tfidf]
            & (e.term == "auditory")
            & (e.tfidf > 1e-3)
        )

        e.filtered_xyz[e.voxid] = (
            e.fma_related_region(e.fma_subregions, "Temporal lobe")
            & e.fma_to_destrieux(e.fma_subregions, e.dd_name)
            & dd_label[e.dd_id, e.dd_name]
            & e.dd_to_ns[e.dd_id, e.voxid, e.xyz]
        )

        res = nl.solve_query()

    a = 1
