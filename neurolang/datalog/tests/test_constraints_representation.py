import pytest

from ..constraints_representation import DatalogConstraintsProgram
from ..ontologies_rewriter import RightImplication
from ...expression_walker import ExpressionBasicEvaluator
from ...expressions import ExpressionBlock, Symbol

class Datalog(DatalogConstraintsProgram, ExpressionBasicEvaluator):
    pass

S_ = Symbol
RI_ =RightImplication
EB_ = ExpressionBlock

def test_load_constraints():
    project = S_('project')
    inArea = S_('inArea')
    hasCollaborator = S_('hasCollaborator')
    collaborator = S_('collaborator')

    x = S_('x')
    y = S_('y')
    z = S_('z')

    sigma1 = RI_(project(x) & inArea(x, y), hasCollaborator(z, y, x))
    sigma2 = RI_(hasCollaborator(x, y, z), collaborator(x))

    dl = Datalog()
    dl.load_constraints(EB_((sigma1, sigma2)))
    assert dl.symbol_table['__constraints__'] == EB_((sigma1, sigma2))

def test_protected_keywords():
    dl = Datalog()

    assert '__constraints__' in dl.protected_keywords