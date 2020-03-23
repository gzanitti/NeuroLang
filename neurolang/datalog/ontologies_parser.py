import rdflib
import pandas as pd
import nibabel as nib
from nilearn import datasets
from .ontologies_rewriter import RightImplication
from ..expressions import Symbol, ExpressionBlock, Constant

C_ = Constant
S_ = Symbol
EB_ = ExpressionBlock
RI_ = RightImplication


class OntologiesParser():
    def __init__(self, paths, namespaces, load_format='xml'):
        self.namespaces_dic = None
        self.owl_dic = None
        if isinstance(paths, list):
            self._load_ontology(paths, namespaces, load_format)
        else:
            self._load_ontology([paths], [namespaces], load_format)

    def _load_ontology(self, paths, namespaces, load_format):
        self._create_graph(paths, load_format)
        self._process_properties(namespaces)

    def _create_graph(self, paths, load_format):
        self.df = pd.DataFrame()
        temp = []
        for path in paths:
            g = rdflib.Graph()
            g.load(path, format=load_format)
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

    def parse_ontology(self, neurolangDL):
        self.eb = EB_(())
        self.neurolangDL = neurolangDL
        self._load_domain()
        self._load_properties()

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
            const = C_(symbol_name)
            symbols += (RightImplication(triple(x, const, z), symbol(x, z)), )

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
            rdf_syntax_ns_type(x, C_('http://www.w3.org/2002/07/owl#Class')),
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
                x, C_('http://www.w3.org/2002/07/owl#ObjectProperty')
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
            triple(C_(e1), C_(e2), C_(e3)) for e1, e2, e3 in self.df.values
        ])

        self.triples = triples

        pointers = self.df.loc[~self.df.Entity.str.
                               contains('http')].Entity.unique()
        pointer = S_('pointer')
        pointer_list = tuple([pointer(C_(e)) for e in pointers])

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
