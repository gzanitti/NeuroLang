import pytest

from ..constraints_representation import DatalogConstraintsProgram
from ..ontologies_parser import OntologiesParser
from ...expression_walker import ExpressionBasicEvaluator


class Datalog(DatalogConstraintsProgram, ExpressionBasicEvaluator):
    pass


@pytest.mark.skip(reason="No way of currently testing this")
def test_load_ontology():
    paths = ['neurofma_fma3.0.owl']
    namespaces = ['http://sig.biostr.washington.edu/fma3.0']
    onto = OntologiesParser(paths, namespaces)
    dl = Datalog()
    datalog_program = onto.load_ontology(dl, destrieux_relations=False)