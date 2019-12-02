import pytest

from neurolang.logic import ExistentialPredicate, Implication, FunctionApplication
from neurolang.expressions import Symbol, Constant, ExpressionBlock
from neurolang.datalog.ontologies_rewriter import RightImplication, OntologyRewriter

S_ = Symbol
C_ = Constant
EP_ = ExistentialPredicate
EB_ = ExpressionBlock
FA_ = FunctionApplication
I_ = Implication
RI_ = RightImplication


def test_normal_rewriting_step():
    project = S_('project')
    inArea = S_('inArea')
    hasCollaborator = S_('hasCollaborator')
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

    assert len(rewrite) == 2
    imp1 = rewrite.pop()
    imp2 = rewrite.pop()
    assert imp1[0] == q or imp2[0] == q
    q2 = I_(p(b), project(b) & inArea(b, db))
    assert imp1[0] == q2 or imp2[0] == q2


def test_more_than_one_free_variable():
    project = S_('project')
    inArea = S_('inArea')
    hasCollaborator = S_('hasCollaborator')
    p = S_('p')

    w = S_('w')
    x = S_('x')
    y = S_('y')
    z = S_('z')
    a = S_('a')
    b = S_('b')
    c = S_('c')
    db = C_('db')

    sigma = RI_(project(x) & inArea(x, y), hasCollaborator(w, z, y, x))
    q = I_(p(b), hasCollaborator(c, a, db, b))

    qB = EB_((q,))
    sigmaB = EB_((sigma,))

    orw = Rewriter(qB, sigmaB)
    rewrite = orw.Xrewrite()

    assert len(rewrite) == 2
    imp1 = rewrite.pop()
    imp2 = rewrite.pop()
    assert imp1[0] == q or imp2[0] == q
    q2 = I_(p(b), project(b) & inArea(b, db))
    assert imp1[0] == q2 or imp2[0] == q2



def test_unsound_rewriting_step_constant():
    project = S_('project')
    inArea = S_('inArea')
    hasCollaborator = S_('hasCollaborator')
    p = S_('p')

    x = S_('x')
    y = S_('y')
    z = S_('z')
    b = S_('b')
    db = C_('db')
    c = C_('c')

    sigma = RI_(project(x) & inArea(x, y), hasCollaborator(z, y, x))
    q = I_(p(b), hasCollaborator(c, db, b))

    qB = EB_((q,))
    sigmaB = EB_((sigma,))

    orw = Rewriter(qB, sigmaB)
    rewrite = orw.Xrewrite()

    assert len(rewrite) == 1
    imp = rewrite.pop()
    assert imp[0] == q

def test_unsound_rewriting_step_shared():
    project = S_('project')
    inArea = S_('inArea')
    hasCollaborator = S_('hasCollaborator')
    p = S_('p')

    x = S_('x')
    y = S_('y')
    z = S_('z')
    b = S_('b')
    db = C_('db')

    sigma = RI_(project(x) & inArea(x, y), hasCollaborator(z, y, x))
    q = I_(p(b), hasCollaborator(b, db, b))

    qB = EB_((q,))
    sigmaB = EB_((sigma,))

    orw = Rewriter(qB, sigmaB)
    rewrite = orw.Xrewrite()

    assert len(rewrite) == 1
    imp = rewrite.pop()
    assert imp[0] == q


def test_outside_variable():
    s = S_('s')
    r = S_('r')
    t = S_('t')
    p = S_('p')

    x = S_('x')
    y = S_('y')
    z = S_('z')

    a = S_('a')
    b = S_('b')
    c = S_('c')
    e = S_('e')

    sigma = RI_(s(x) & r(x, y), t(x, y, z))
    q2 = I_(p(a), s(c) & t(a, b, c) & t(a, e, c))

    qB = EB_((q2,))
    sigmaB = EB_((sigma,))

    orw = Rewriter(qB, sigmaB)
    rewrite = orw.Xrewrite()

    assert len(rewrite) == 2
    factorized = [x for x in rewrite if x[1] == 'f']
    assert len(factorized) == 0


def test_example_4_3():
    project = S_('project')
    inArea = S_('inArea')
    hasCollaborator = S_('hasCollaborator')
    collaborator = S_('collaborator')
    p = S_('p')

    x = S_('x')
    y = S_('y')
    z = S_('z')
    a = S_('a')
    b = S_('b')
    c = S_('c')

    sigma1 = RI_(project(x) & inArea(x, y), hasCollaborator(z, y, x))
    sigma2 = RI_(hasCollaborator(x, y, z), collaborator(x))

    q = I_(p(b, c), hasCollaborator(a, b, c) & collaborator(a))

    qB = EB_((q,))
    sigmaB = EB_((sigma1, sigma2,))

    orw = Rewriter(qB, sigmaB)
    rewrite = orw.Xrewrite()

    assert len(rewrite) == 4
