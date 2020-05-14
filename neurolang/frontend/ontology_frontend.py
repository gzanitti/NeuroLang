from .query_resolution_datalog import QueryBuilderDatalog
from . import RegionFrontendDatalogSolver


class NeurolangOntologyDL(QueryBuilderDatalog):
    def __init__(self, path, solver=None):
        if solver is None:
            solver = RegionFrontendDatalogSolver()

        paths = ["./data/neurofma_fma3.0.owl"]
        self.onto = OntologyParser(paths)
        d_pred, u_constraints = onto.parse_ontology()
        self.d_pred = d_pred
        self.u_constraints = u_constraints

        super().__init__(solver, chase_class=Chase)

    def solve_query(self, symbol_prob, symbol_prior):
        prob_terms, prob_voxels, prob_terms_voxels = (
            self.load_neurosynth_database()
        )
        self.prob_terms = prob_terms
        self.prob_voxels = prob_voxels
        self.prob_terms_voxels = prob_terms_voxels

        eB2 = self.rewrite_database_with_ontology()
        dl = self.load_facts(eB2)
        sol = self.build_chase_solution(dl, symbol_prob)
        dlProb = self.load_probabilistic_facts(sol)
        result = self.solve_probabilistic_query(dlProb)

        return result

    def load_neurosynth_database(self):
        prob_terms = pd.read_hdf("./data/neurosynth_prob.h5", key="terms")
        prob_voxels = pd.read_hdf("./data/neurosynth_prob.h5", key="voxels")
        prob_terms_voxels = pd.read_hdf(
            "./data/neurosynth_prob.h5", key="terms_voxels"
        )

        prob_terms_voxels = prob_terms_voxels[
            prob_terms_voxels.index.get_level_values("term") == "auditory"
        ]
        prob_terms = prob_terms[prob_terms["index"] == "auditory"]

        prob_terms = prob_terms[["proba", "index"]]

        prob_voxels.reset_index(inplace=True)
        prob_voxels.rename(columns={0: "prob"}, inplace=True)
        prob_voxels = prob_voxels[["prob", "index"]]

        prob_terms_voxels.reset_index(inplace=True)
        prob_terms_voxels = prob_terms_voxels[["prob", "term", "variable"]]

        return prob_terms, prob_voxels, prob_terms_voxels

    def rewrite_database_with_ontology(self):
        orw = OntologyRewriter(self.get_expressions(), self.u_constraints)
        rewrite = orw.Xrewrite()

        eB2 = ()
        for imp in rewrite:
            eB2 += (imp[0],)

        return Union(eB2)

    def load_facts(self, eB2):
        destrieux_to_voxels = Symbol("destrieux_to_voxels")
        relation_name = Symbol("relation_name")

        with open("./data/regiones.pickle", "rb") as fp:
            inter_regions = pickle.load(fp)

        relations_list = destrieux_name_to_fma_relations()
        r_name = tuple(
            [
                relation_name(Constant(destrieux), Constant(fma))
                for destrieux, fma in relations_list
            ]
        )

        dl = Datalog()
        dl.add_extensional_predicate_from_tuples(
            self.onto.get_triples_symbol(), d_pred[onto.get_triples_symbol()]
        )
        dl.add_extensional_predicate_from_tuples(
            self.onto.get_pointers_symbol(), d_pred[onto.get_pointers_symbol()]
        )
        dl.add_extensional_predicate_from_tuples(
            destrieux_to_voxels,
            [(ns, ds, region) for ns, ds, region in inter_regions],
        )
        dl.add_extensional_predicate_from_tuples(
            relation_name, [(a.args[0].value, a.args[1].value) for a in r_name]
        )

        dl.walk(eB2)

        return dl

    def build_chase_solution(self, dl, symbol):

        dc = Chase(dl)
        solution_instance = dc.build_chase_solution()
        list_regions = list(
            solution_instance[symbol.name].value.unwrapped_iter()
        )

        return list_regions

    def load_probabilistic_facts(self, list_regions):
        dlProb = ProbDatalogProgram()

        term = Symbol("term")
        neurosynth_data = Symbol("neurosynth_data")
        region_contains_voxel = Symbol("region_contains_voxel")

        dlProb.add_extensional_predicate_from_tuples(
            region_contains_voxel, set(list_regions)
        )
        dlProb.add_probfacts_from_tuples(
            term, set(self.prob_terms.itertuples(index=False, name=None))
        )
        dlProb.add_probfacts_from_tuples(
            neurosynth_data,
            set(self.prob_terms_voxels.itertuples(index=False, name=None)),
        )

        return dlProb

    def solve_probabilistic_query(self, dlProb):
        dt2 = DatalogTranslator()
        eb = dt2.walk(self.get_prob_expressions())
        dlProb.walk(eb)

        probability_voxel = Symbol("probability_voxel")
        z = Symbol("z")

        dl_program = probdatalog_to_datalog(dlProb, datalog=DatalogRegions)
        dc = Chase(dl_program)
        solution_instance = dc.build_chase_solution()
        grounded = build_grounding(dlProb, solution_instance)

        gm = TranslateGroundedProbDatalogToGraphicalModel().walk(grounded)
        query = SuccQuery(probability_voxel(z))
        solver = QueryGraphicalModelSolver(gm)
        result = solver.walk(query)

        return result

    def get_expressions(self):
        exp = ()
        for e in self.current_program:
            if not str.startswith(
                e.expression.consequent.functor.name, "probability"
            ):
                exp = exp + (e.expression,)

        return Union(exp)

    def get_prob_expressions(self):
        exp = ()
        for e in self.current_program:
            if str.startswith(
                e.expression.consequent.functor.name, "probability"
            ):
                exp = exp + (e.expression,)

        return Union(exp)
