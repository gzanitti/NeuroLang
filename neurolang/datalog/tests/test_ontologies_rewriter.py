import pytest

from neurolang.logic import ExistentialPredicate, Implication, FunctionApplication
from neurolang.expressions import Symbol, Constant, ExpressionBlock
from neurolang.datalog.ontologies_rewiter import RightImplication, Rewriter

S_ = Symbol
C_ = Constant
EP_ = ExistentialPredicate
EB_ = ExpressionBlock
FA_ = FunctionApplication
I_ = Implication
RI_ = RightImplication


def test_rewriting_step():
    project = S_('project')
    inArea = S_('inArea')
    hasCollaborator = S_(name='hasCollaborator')
    p = S_('p')

    x = S_('x')
    y = S_('y')
    z = S_('z')
    a = S_('a')
    b = S_('b')
    db = C_('db')

    sigma = RI_(project(x) & inArea(x, y), hasCollaborator(z, y, x))
    q = I_(p(b), hasCollaborator(a, db, b))

    qB = EB_((q,))
    sigmaB = EB_((sigma,))

    orw = Rewriter(qB, sigmaB)
    rewrite = orw.Xrewrite()