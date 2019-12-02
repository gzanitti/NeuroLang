import pytest

from .. import DatalogProgram
from ..ontologies_parser import OntologiesParser
from ...expression_walker import ExpressionBasicEvaluator

class Datalog(
    DatalogProgram,
    ExpressionBasicEvaluator
):
    pass


def test_load_ontology():
    paths = ['./neurolang/datalog/tests/neurofma_fma3.0.owl']
    namespaces = ['http://sig.biostr.washington.edu/fma3.0']
    onto = OntologiesParser(paths, namespaces)
    dl = Datalog()
    datalog_program = onto.load_ontology(dl, destrieux_relations=False)