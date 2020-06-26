from ..datalog.aggregation import (
    AggregationApplication,
    Chase,
    DatalogWithAggregationMixin,
)
from ..datalog.chase import (
    ChaseGeneral,
    ChaseNaive,
    ChaseNamedRelationalAlgebraMixin,
    ChaseSemiNaive,
)
from ..datalog.constraints_representation import DatalogConstraintsProgram
from ..datalog.expression_processing import (
    extract_logic_predicates,
    reachable_code,
)
from ..datalog.expressions import TranslateToLogic
from ..datalog.ontologies_parser import OntologyParser
from ..datalog.ontologies_rewriter import OntologyRewriter
from ..exceptions import (
    NeuroLangFrontendException,
    NeuroLangNotImplementedError,
)
from ..expression_walker import ExpressionBasicEvaluator
from ..logic import Union
from ..region_solver import RegionSolver
from ..regions import ExplicitVBR
from . import RegionFrontendDatalogSolver
from .query_resolution_datalog import QueryBuilderDatalog


class ChaseFrontend(Chase, ChaseNamedRelationalAlgebraMixin, ChaseGeneral):
    pass


class DatalogRegions(
    TranslateToLogic,
    RegionSolver,
    DatalogWithAggregationMixin,
    DatalogConstraintsProgram,
    ExpressionBasicEvaluator,
):
    pass


class NeurolangOntologyDL(QueryBuilderDatalog):
    def __init__(self, solver=None):
        if solver is None:
            solver = DatalogRegions()

        self.ontology_loaded = False

        super().__init__(solver, chase_class=ChaseFrontend)

    def load_ontology(self, paths, load_format="xml"):
        onto = OntologyParser(paths, load_format)
        d_pred, u_constraints = onto.parse_ontology()
        self.solver.walk(u_constraints)
        self.solver.add_extensional_predicate_from_tuples(
            onto.get_triples_symbol(), d_pred[onto.get_triples_symbol()]
        )
        self.solver.add_extensional_predicate_from_tuples(
            onto.get_pointers_symbol(), d_pred[onto.get_pointers_symbol()]
        )

        self.ontology_loaded = True

    def _separate_deterministic_probabilistic_code(
        self, query_pred=None, det_symbols=None, prob_symbols=None
    ):
        if det_symbols is None:
            det_symbols = set()
        if prob_symbols is None:
            prob_symbols = set()
        if query_pred is None:
            query_reachable_code = self._union_of_idb()
        else:
            query_reachable_code = reachable_code(query_pred, self.solver)
        deterministic_symbols = (
            set(self.solver.extensional_database().keys())
            | set(det_symbols)
            | set(self.solver.builtins().keys())
        )
        deterministic_program = list()
        probabilistic_symbols = (
            self.solver.pfact_pred_symbs
            | self.solver.pchoice_pred_symbs
            | set(prob_symbols)
        )
        probabilistic_program = list()
        unclassified_code = list(query_reachable_code.formulas)
        unclassified = 0
        initial_unclassified_length = len(unclassified_code) + 1
        while (
            len(unclassified_code) > 0
            and unclassified <= initial_unclassified_length
        ):
            pred = unclassified_code.pop(0)
            preds_antecedent = set(
                p.functor
                for p in extract_logic_predicates(pred.antecedent)
                if p.functor != pred.consequent.functor
            )
            if not probabilistic_symbols.isdisjoint(preds_antecedent):
                probabilistic_symbols.add(pred.consequent.functor)
                probabilistic_program.append(pred)
                unclassified = 0
                initial_unclassified_length = len(unclassified_code) + 1
            elif deterministic_symbols.issuperset(preds_antecedent):
                deterministic_symbols.add(pred.consequent.functor)
                deterministic_program.append(pred)
                unclassified = 0
                initial_unclassified_length = len(unclassified_code) + 1
            else:
                unclassified_code.append(pred)
                unclassified += 1
        if not probabilistic_symbols.isdisjoint(deterministic_symbols):
            raise NeuroLangFrontendException(
                "An atom was defined as both deterministic and probabilistic"
            )
        if len(unclassified_code) > 0:
            raise NeuroLangFrontendException("There are unclassified atoms")
        return Union(deterministic_program), Union(probabilistic_program)

    def solve_query(self):
        # det, prob = self._separate_deterministic_probabilistic_code()

        det = []
        for p in self.current_program:
            det.append(p.expression)

        det = Union(det)

        if self.ontology_loaded:
            eB = self.rewrite_database_with_ontology(det)
            self.solver.walk(eB)

        dc = self.chase_class(self.solver)
        solution_instance = dc.build_chase_solution()

        # dlProb = self.load_probabilistic_facts(sol)
        # result = self.solve_probabilistic_query(dlProb, symbol_prob)

        return solution_instance

    def rewrite_database_with_ontology(self, deterministic_program):
        orw = OntologyRewriter(
            deterministic_program, self.solver.constraints()
        )
        rewrite = orw.Xrewrite()

        eB2 = ()
        for imp in rewrite:
            eB2 += (imp[0],)

        return Union(eB2)

    # TODO This should be updated to the latest version.
    """
    def solve_probabilistic_query(self, dlProb, symbol):
        dt2 = DatalogTranslator()
        eb = dt2.walk(self.get_prob_expressions())
        dlProb.walk(eb)

        z = Symbol.fresh()

        dl_program = probdatalog_to_datalog(dlProb, datalog=DatalogRegions)
        dc = Chase(dl_program)
        solution_instance = dc.build_chase_solution()
        grounded = build_grounding(dlProb, solution_instance)

        gm = TranslateGroundedProbDatalogToGraphicalModel().walk(grounded)
        sym = symbol.expression.formulas[0].consequent.function
        query = SuccQuery(sym(z))
        solver = QueryGraphicalModelSolver(gm)
        result = solver.walk(query)

        return result
        """