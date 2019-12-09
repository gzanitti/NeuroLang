import rdflib
import pandas as pd
import nibabel as nib
from nilearn import datasets
from .ontologies_rewriter import RightImplication
from ..expressions import Symbol, ExpressionBlock, Constant
# I think we should move this RegionMixin outside frontend
from neurolang.frontend.query_resolution import RegionMixin

C_ = Constant
S_ = Symbol
EB_ = ExpressionBlock
RI_ = RightImplication


class OntologiesParser():
    def __init__(self, paths, namespaces):
        self.namespaces_dic = None
        self.owl_dic = None
        if isinstance(paths, list):
            self._load_ontology(paths, namespaces)
        else:
            self._load_ontology([paths], [namespaces])

    def _load_ontology(self, paths, namespaces):
        self._create_graph(paths)
        self._process_properties(namespaces)

    def _create_graph(self, paths):
        self.df = pd.DataFrame()
        temp = []
        for path in paths:
            g = rdflib.Graph()
            g.load(path)
            gdf = pd.DataFrame(iter(g))
            gdf = gdf.astype(str)
            gdf.columns = ['Entity', 'Property', 'Value']
            temp.append(gdf)

        self.df = self.df.append(temp)

    # @dev
    def _get_graph(self):
        return self.df

    def _process_properties(self, namespaces):
        namespaces_properties = self.df[~self.df.Property.str.
                                        contains('#')].Property.unique()
        namespaces_properties = list(
            filter(
                lambda x: (x in n for n in namespaces), namespaces_properties
            )
        )
        namespaces_prop = list(
            map(
                lambda x: x[0] + '_' + x[1],
                list(map(lambda y: y.split('/')[-2:], namespaces_properties))
            )
        )
        self.namespaces_dic = dict(zip(namespaces_properties, namespaces_prop))

        owl_properties = self.df[self.df.Property.str.contains('#')
                                 ].Property.unique()
        owl_rdf = list(
            map(
                lambda a: list(
                    map(
                        lambda s: s.replace('-', '_'),
                        a.split('/')[-1].split('#')
                    )
                ), owl_properties
            )
        )
        owl_rdf = list(map(lambda x: x[0] + '_' + x[1], owl_rdf))
        self.owl_dic = dict(zip(owl_properties, owl_rdf))

    def _replace_property(self, prop):
        if prop in self.owl_dic:
            new_prop = self.owl_dic[prop]
        elif prop in self.namespaces_dic:
            new_prop = self.namespaces_dic[prop]
        else:
            new_prop = prop

        if new_prop in [
            'rdf_schema_subClassOf', 'rdf_schema_subPropertyOf',
            'owl_onProperty', 'owl_someValuesFrom'
        ]:
            new_prop = new_prop + '2'

        return new_prop

    def parse_ontology(self, neurolangDL, destrieux_relations=False):
        self.eb = EB_(())
        self.neurolangDL = neurolangDL
        self._load_domain()
        self._load_properties()

        # Maybe we should put this outside the class
        if destrieux_relations:
            relations_list = self.get_destrieux_relations()
            #relations = S_('relations')

            destrieux_region = S_('destrieux_region')
            destrieux_name = S_('destrieux_name')
            fma_region = S_('fma_name')

            symbols_list = tuple([RightImplication(fma_region(S_(fma)), destrieux_name(S_(destrieux))) for destrieux, fma in relations_list])

            destrieux_dataset = datasets.fetch_atlas_destrieux_2009()
            destrieux_map = nib.load(destrieux_dataset['maps'])

            destrieux = []
            for label_number, name in destrieux_dataset['labels']:
                if label_number == 0:
                    continue
                name = name.decode()
                region = RegionMixin.create_region(
                    destrieux_map, label=label_number
                )
                if region is None:
                    continue
                name = name.replace('-', '_').replace(' ', '_').lower()
                destrieux.append((name, region))

            regions_list = tuple([RightImplication(destrieux_name(S_(name)), destrieux_region(S_(region))) for name, region in destrieux])
            self.eb = ExpressionBlock(self.eb.expressions + symbols_list + regions_list)


        self.neurolangDL.load_constraints(self.eb)
        return self.neurolangDL

    def _load_properties(self):
        all_props = list(self.owl_dic.keys()
                         ) + list(self.namespaces_dic.keys())

        w = S_('w')
        x = S_('x')
        y = S_('y')
        z = S_('z')
        triple = S_('triple')

        symbols = ()
        for prop in all_props:
            symbol_name = prop
            #name = self._replace_property(prop)
            #symbol_name = name.replace(':', '_')
            symbol = S_(symbol_name)
            symbols += (RightImplication(triple(x, symbol, z), symbol(x, z)), )

        self.eb = ExpressionBlock(self.eb.expressions + symbols)

        self._parse_subproperties()
        self._parse_subclasses()

        owl_disjointWith = S_('owl_disjointWith')
        rdf_schema_subClassOf = S_('rdf_schema_subClassOf')
        disjoint = RI_(
            owl_disjointWith(x, y) & rdf_schema_subClassOf(w, x) &
            rdf_schema_subClassOf(z, y), owl_disjointWith(w, z)
        )

        self.eb = ExpressionBlock(self.eb.expressions + (disjoint, ))

        self._parse_somevalue_properties()

    def _parse_somevalue_properties(self):
        w = S_('w')
        x = S_('x')
        y = S_('y')
        z = S_('z')

        owl_onProperty = S_('owl_onProperty')
        owl_onProperty2 = S_('owl_onProperty2')
        onProperty = RI_(owl_onProperty2(x, y), owl_onProperty(x, y))

        owl_someValuesFrom = S_('owl_someValuesFrom')
        owl_someValuesFrom2 = S_('owl_someValuesFrom2')
        someValueFrom = RI_(
            owl_someValuesFrom2(x, y), owl_someValuesFrom(x, y)
        )

        pointer = S_('pointer')
        rdf_schema_subClassOf = S_('rdf_schema_subClassOf')

        temp_triple = RI_(
            pointer(w) & owl_someValuesFrom(w, z) & owl_onProperty(w, y) &
            rdf_schema_subClassOf(x, w), y(x, z)
        )

        self.eb = ExpressionBlock(
            self.eb.expressions + (
                onProperty,
                someValueFrom,
                temp_triple,
            )
        )

    def _parse_subclasses(self):
        rdf_schema_subClassOf = S_('rdf_schema_subClassOf')
        rdf_schema_subClassOf2 = S_('rdf_schema_subClassOf2')
        w = S_('w')
        x = S_('x')
        y = S_('y')
        z = S_('z')

        subClass = RI_(
            rdf_schema_subClassOf2(x, y), rdf_schema_subClassOf(x, y)
        )
        subClass2 = RI_(
            rdf_schema_subClassOf2(x, y) & rdf_schema_subClassOf(y, z),
            rdf_schema_subClassOf(x, z)
        )

        rdf_syntax_ns_rest = S_('rdf_syntax_ns_rest')
        ns_rest = RI_(
            rdf_schema_subClassOf(x, y) & rdf_syntax_ns_rest(w, x) &
            rdf_syntax_ns_rest(z, y), rdf_schema_subClassOf(w, z)
        )

        rdf_syntax_ns_type = S_('rdf_syntax_ns_type')
        class_sim = RI_(
            rdf_syntax_ns_type(x, S_('http://www.w3.org/2002/07/owl#Class')),
            rdf_schema_subClassOf(x, x)
        )

        self.eb = ExpressionBlock(
            self.eb.expressions + (
                subClass,
                subClass2,
                ns_rest,
                class_sim,
            )
        )

    def _parse_subproperties(self):
        rdf_schema_subPropertyOf = S_('rdf_schema_subPropertyOf')
        rdf_schema_subPropertyOf2 = S_('rdf_schema_subPropertyOf2')
        w = S_('w')
        x = S_('x')
        y = S_('y')
        z = S_('z')

        subProperty = RI_(
            rdf_schema_subPropertyOf2(x, y), rdf_schema_subPropertyOf(x, y)
        )
        subProperty2 = RI_(
            rdf_schema_subPropertyOf2(x, y) & rdf_schema_subPropertyOf(y, z),
            rdf_schema_subPropertyOf(x, z)
        )

        owl_inverseOf = S_('owl_inverseOf')
        inverseOf = RI_(
            rdf_schema_subPropertyOf(x, y) & owl_inverseOf(w, x) &
            owl_inverseOf(z, y), rdf_schema_subPropertyOf(w, z)
        )

        rdf_syntax_ns_type = S_('rdf_syntax_ns_type')
        objectProperty = RI_(
            rdf_syntax_ns_type(
                x, S_('http://www.w3.org/2002/07/owl#ObjectProperty')
            ), rdf_schema_subPropertyOf(x, x)
        )

        self.eb = ExpressionBlock(
            self.eb.expressions + (
                subProperty,
                subProperty2,
                inverseOf,
                objectProperty,
            )
        )

    def _load_domain(self):
        triple = S_('triple')
        #triples = tuple([triple(S_(e1), S_(self._replace_property(e2)), S_(e3)) for e1, e2, e3 in self.df.values])
        triples = tuple([
            triple(S_(e1), S_(e2), S_(e3)) for e1, e2, e3 in self.df.values
        ])

        pointers = self.df.loc[~self.df.Entity.str.
                               contains('http')].Entity.unique()
        pointer = S_('pointer')
        pointer_list = tuple([pointer(S_(e)) for e in pointers])

        dom = S_('dom')
        x = S_('x')
        y = S_('y')
        z = S_('z')

        dom1 = RightImplication(triple(x, y, z), dom(x))
        dom2 = RightImplication(triple(x, y, z), dom(y))
        dom3 = RightImplication(triple(x, y, z), dom(z))

        self.eb = EB_(
            self.eb.expressions + triples + pointer_list + (dom1, dom2, dom3)
        )

    def get_destrieux_relations(self):
        return [
            ('l_g_and_s_frontomargin', 'Left frontomarginal gyrus'),
            ('l_g_and_s_occipital_inf', 'Left inferior occipital gyrus'),
            ('l_g_and_s_paracentral', 'Left paracentral lobule'),
            ('l_g_and_s_subcentral', 'Left subcentral gyrus'),
            (
                'l_g_and_s_transv_frontopol',
                'Left superior transverse frontopolar gyrus'
            ),
            ('l_g_and_s_cingul_ant', 'Left anterior cingulate gyrus'),
            (
                'l_g_and_s_cingul_mid_ant',
                'Left anterior middle cingulate gyrus'
            ),
            (
                'l_g_and_s_cingul_mid_post',
                'Left posterior middle cingulate gyrus'
            ),
            (
                'l_g_cingul_post_dorsal',
                'Dorsal segment of left posterior middle cingulate gyrus'
            ),
            (
                'l_g_cingul_post_ventral',
                'Ventral segment of left posterior middle cingulate gyrus'
            ),
            ('l_g_cuneus', 'Left cuneus'),
            (
                'l_g_front_inf_opercular',
                'Opercular part of left inferior frontal gyrus'
            ),
            (
                'l_g_front_inf_orbital',
                'Orbital part of left inferior frontal gyrus'
            ),
            (
                'l_g_front_inf_triangul',
                'Triangular part of left inferior frontal gyrus'
            ),
            ('l_g_front_middle', 'Left middle frontal gyrus'),
            ('l_g_front_sup', 'Left superior frontal gyrus'),
            ('l_g_ins_lg_and_s_cent_ins', 'Left central insular sulcus'),
            ('l_g_ins_lg_and_s_cent_ins', 'Left long insular gyrus'),
            ('l_g_insular_short', 'Short insular gyrus'),
            ('l_g_occipital_middleLeft', ' 	Left lateral occipital gyrus'),
            ('l_g_occipital_sup', 'Left superior occipital gyrus'),
            ('l_g_oc_temp_lat_fusifor', 'Left fusiform gyrus'),
            ('l_g_oc_temp_med_lingual', 'Left lingual gyrus'),
            ('l_g_oc_temp_med_parahip', 'Left parahippocampal gyrus'),
            ('l_g_orbital', 'Left orbital gyrus'),
            ('l_g_pariet_inf_angular', 'Left angular gyrus'),
            ('l_g_pariet_inf_supramar', 'Left supramarginal gyrus'),
            ('l_g_parietal_sup', 'Left superior parietal lobule'),
            ('l_g_postcentral', 'Left postcentral gyrus'),
            ('l_g_precentral', 'Left precentral gyrus'),
            ('l_g_precuneus', 'Left precuneus'),
            ('l_g_rectus', 'Left straight gyrus'),
            ('l_g_subcallosal', 'Left paraterminal gyrus'),
            ('l_g_temp_sup_g_t_transv', 'Left transverse temporal gyrus'),
            ('l_g_temp_sup_lateral', 'Left superior temporal gyrus'),
            ('l_g_temp_sup_plan_polar', 'Left superior temporal gyrus'),
            ('l_g_temp_sup_plan_tempo', 'Left superior temporal gyrus'),
            ('l_g_temporal_inf', 'Left inferior temporal gyrus'),
            ('l_g_temporal_middle', 'Left middle temporal gyrus'),
            (
                'l_lat_fis_ant_horizont',
                'Anterior horizontal limb of left lateral sulcus'
            ),
            (
                'l_lat_fis_ant_vertical',
                'Anterior ascending limb of left lateral sulcus'
            ),
            (
                'l_lat_fis_post',
                'Posterior ascending limb of left lateral sulcus'
            ),
            ('l_lat_fis_post', 'Left lateral sulcus'),
            ('l_pole_occipital', 'Left occipital pole'),
            ('l_pole_temporal', 'Left temporal pole'),
            ('l_s_calcarine', 'Left Calcarine sulcus'),
            ('l_s_central', 'Left central sulcus'),
            ('l_s_cingul_marginalis', 'Left marginal sulcus'),
            ('l_s_circular_insula_ant', 'Circular sulcus of left insula'),
            ('l_s_circular_insula_inf', 'Circular sulcus of left insula'),
            ('l_s_circular_insula_sup', 'Circular sulcus of left insula'),
            ('l_s_collat_transv_ant', 'Left collateral sulcus'),
            ('l_s_collat_transv_post', 'Left collateral sulcus'),
            ('l_s_front_inf', 'Left inferior frontal sulcus'),
            ('l_s_front_sup', 'Left superior frontal sulcus'),
            ('l_s_intrapariet_and_p_trans', 'Left intraparietal sulcus'),
            ('l_s_oc_middle_and_lunatus', 'Left lunate sulcus'),
            ('l_s_oc_sup_and_transversal', 'Left transverse occipital sulcus'),
            ('l_s_occipital_ant', 'Left anterior occipital sulcus'),
            ('l_s_oc_temp_lat', 'Left occipitotemporal sulcus'),
            ('l_s_oc_temp_med_and_lingual', 'Left intralingual sulcus'),
            ('l_s_orbital_lateral', 'Left orbital sulcus'),
            ('l_s_orbital_med_olfact', 'Left olfactory sulcus'),
            ('l_s_orbital_h_shaped', 'Left transverse orbital sulcus'),
            ('l_s_orbital_h_shaped', 'Left orbital sulcus'),
            ('l_s_parieto_occipital', 'Left parieto-occipital sulcus'),
            ('l_s_pericallosal', 'Left callosal sulcus'),
            ('l_s_postcentral', 'Left postcentral sulcus'),
            ('l_s_precentral_inf_part', 'Left precentral sulcus'),
            ('l_s_precentral_sup_part', 'Left precentral sulcus'),
            ('l_s_suborbital', 'Left fronto-orbital sulcus'),
            ('l_s_subparietal', 'Left subparietal sulcus'),
            ('l_s_temporal_inf', 'Left inferior temporal sulcus'),
            ('l_s_temporal_sup', 'Left superior temporal sulcus'),
            ('l_s_temporal_transverse', 'Left transverse temporal sulcus'),
            ('r_g_and_s_frontomargin', 'Right frontomarginal gyrus'),
            ('r_g_and_s_occipital_inf', 'Right inferior occipital gyrus'),
            ('r_g_and_s_paracentral', 'Right paracentral lobule'),
            ('r_g_and_s_subcentral', 'Right subcentral gyrus'),
            (
                'r_g_and_s_transv_frontopol',
                'Right superior transverse frontopolar gyrus'
            ),
            ('r_g_and_s_cingul_ant', 'Right anterior cingulate gyrus'),
            (
                'r_g_and_s_cingul_mid_ant',
                'Right anterior middle cingulate gyrus'
            ),
            (
                'r_g_and_s_cingul_mid_post',
                'Right posterior middle cingulate gyrus'
            ),
            (
                'r_g_cingul_post_dorsal',
                'Dorsal segment of right posterior middle cingulate gyrus'
            ),
            (
                'r_g_cingul_post_ventral',
                'Ventral segment of right posterior middle cingulate gyrus'
            ),
            ('r_g_cuneus', 'Right cuneus'),
            (
                'r_g_front_inf_opercular',
                'Opercular part of right inferior frontal gyrus'
            ),
            (
                'r_g_front_inf_orbital',
                'Orbital part of right inferior frontal gyrus'
            ),
            (
                'r_g_front_inf_triangul',
                'Triangular part of right inferior frontal gyrus'
            ),
            ('r_g_front_middle', 'Right middle frontal gyrus'),
            ('r_g_front_sup', 'Right superior frontal gyrus'),
            ('r_g_ins_lg_and_s_cent_ins', 'Right central insular sulcus'),
            ('r_g_ins_lg_and_s_cent_ins', 'Right long insular gyrus'),
            ('r_g_insular_short', 'Right short insular gyrus'),
            ('r_g_occipital_middle', 'Right lateral occipital gyrus'),
            ('r_g_occipital_sup', 'Right superior occipital gyrus'),
            ('r_g_oc_temp_lat_fusifor', 'Right fusiform gyrus'),
            ('r_g_oc_temp_med_lingual', 'Right lingual gyrus'),
            ('r_g_oc_temp_med_parahip', 'Right parahippocampal gyrus'),
            ('r_g_orbital', 'Right orbital gyrus'),
            ('r_g_pariet_inf_angular', 'Right angular gyrus'),
            ('r_g_pariet_inf_supramar', 'Right supramarginal gyrus'),
            ('r_g_parietal_sup', 'Right superior parietal lobule'),
            ('r_g_postcentral', 'Right postcentral gyrus'),
            ('r_g_precentral', 'Right precentral gyrus'),
            ('r_g_precuneus', 'Right precuneus'),
            ('r_g_rectus', 'Right straight gyrus'),
            ('r_g_subcallosal', 'Right paraterminal gyrus'),
            ('r_g_temp_sup_g_t_transv', 'Right transverse temporal gyrus'),
            ('r_g_temp_sup_lateral', 'Right superior temporal gyrus'),
            ('r_g_temp_sup_plan_polar', 'Right superior temporal gyrus'),
            ('r_g_temp_sup_plan_tempo', 'Right superior temporal gyrus'),
            ('r_g_temporal_inf', 'Right inferior temporal gyrus'),
            ('r_g_temporal_middle', 'Right middle temporal gyrus'),
            (
                'r_lat_fis_ant_horizont',
                'Anterior horizontal limb of right lateral sulcus'
            ),
            (
                'r_lat_fis_ant_vertical',
                'Anterior ascending limb of right lateral sulcus'
            ),
            ('r_lat_fis_post', 'Right lateral sulcus'),
            (
                'r_lat_fis_post',
                'Posterior ascending limb of right lateral sulcus'
            ),
            ('r_pole_occipital', 'Right occipital pole'),
            ('r_pole_temporal', 'Right temporal pole'),
            ('r_s_calcarine', 'Right Calcarine sulcus'),
            ('r_s_central', 'Right central sulcus'),
            ('r_s_cingul_marginalis', 'Right marginal sulcus'),
            ('r_s_circular_insula_ant', 'Circular sulcus of Right insula'),
            ('r_s_circular_insula_inf', 'Circular sulcus of Right insula'),
            ('r_s_circular_insula_sup', 'Circular sulcus of Right insula'),
            ('r_s_collat_transv_ant', 'Right collateral sulcus'),
            ('r_s_collat_transv_post', 'Right collateral sulcus'),
            ('r_s_front_inf', 'Right inferior frontal sulcus'),
            ('r_s_front_sup', 'Right superior frontal sulcus'),
            ('r_s_intrapariet_and_p_trans', 'Right intraparietal sulcus'),
            ('r_s_oc_middle_and_lunatus', 'Right lunate sulcus'),
            (
                'r_s_oc_sup_and_transversal',
                'Right transverse occipital sulcus'
            ),
            ('r_s_occipital_ant', 'Right anterior occipital sulcus'),
            ('r_s_oc_temp_lat', 'Right occipitotemporal sulcus'),
            ('r_s_oc_temp_med_and_lingual', 'Right intralingual sulcus'),
            ('r_s_orbital_lateral', 'Right orbital sulcus'),
            ('r_s_orbital_med_olfact', 'Right olfactory sulcus'),
            ('r_s_orbital_h_shaped', 'Right orbital sulcus'),
            ('r_s_orbital_h_shaped', 'Right transverse orbital sulcus'),
            ('r_s_parieto_occipital', 'Right parieto-occipital sulcus'),
            ('r_s_pericallosal', 'Right callosal sulcus'),
            ('r_s_postcentral', 'Right postcentral sulcus'),
            ('r_s_precentral_inf_part', 'Right precentral sulcus'),
            ('r_s_precentral_sup_part', 'Right precentral sulcus'),
            ('r_s_suborbital', 'Right fronto-orbital sulcus'),
            ('r_s_subparietal', 'Right subparietal sulcus'),
            ('r_s_temporal_inf', 'Right inferior temporal sulcus'),
            ('r_s_temporal_sup', 'Right superior temporal sulcus'),
            ('r_s_temporal_transverse', 'Right transverse temporal sulcus'),
        ]
