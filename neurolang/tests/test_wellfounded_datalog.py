import pytest
from operator import invert, and_, or_

from .. import solver_datalog_extensional_db
from .. import solver_datalog_naive as sdb
from .. import expression_walker
from .. import expressions
from ..expressions import (
    ExpressionBlock, Query, Lambda, FunctionApplication,
)

from ..type_system import Unknown, _Unknown

from ..wellfounded_datalog import (
    WellFoundedRewriter, WellFoundedEvaluator,
    WellFoundedDatalog
)
from ..solver_datalog_naive import (
    SolverNonRecursiveDatalogNaive, DatalogBasic, Implication, Fact
)

from ..existential_datalog import (
    ExistentialDatalog, SolverNonRecursiveExistentialDatalog, Implication
)

C_ = expressions.Constant
S_ = expressions.Symbol
F_ = expressions.FunctionApplication
Eb_ = expressions.ExpressionBlock

def test_symbol_rewrite():

    x = S_('x')
    new_symbols = {}
    new_symbols['x'] = invert(x)

    solver = WellFoundedRewriter(new_symbols)
    wx = solver.walk(x)

    assert wx.functor.value == invert
    assert wx.args[0] == x


    x = S_('x')
    new_symbols = {}
    new_symbols['x'] = x

    solver = WellFoundedRewriter(new_symbols)
    wx = solver.walk(x)

    assert wx == x


def test_implication_rewrite():

    x = S_('x')
    y = S_('y')

    new_symbols = {}
    new_symbols['x'] = invert(x)
    new_symbols['y'] = invert(y)

    imp = Implication(x, y)

    solver = WellFoundedRewriter(new_symbols)
    wx = solver.walk(imp)

    assert not hasattr(wx.consequent, 'functor')
    assert wx.consequent == x

    assert wx.antecedent.functor.value == invert
    assert wx.antecedent.args[0] == y

    P = S_('P')
    Px = P(x)
    new_symbols['P'] = invert(Px)

    imp1 = Implication(Px, y)
    imp2 = Implication(y, Px)

    w1 = solver.walk(imp1)
    w2 = solver.walk(imp2)

    assert w1.consequent.functor == P
    assert w1.consequent == Px
    assert w1.consequent.args[0] == x
    assert w1.antecedent.functor.value == invert
    assert w1.antecedent.args[0] == y

    assert not hasattr(w2.consequent, 'functor')
    assert w2.consequent == y
    assert w2.antecedent.functor.value == invert
    assert w2.antecedent.args[0] == Px
    assert w2.antecedent.args[0].args[0] == x


def test_function_application_rewrite():

    P = S_('P')
    x = S_('x')
    new_symbols = {}
    new_symbols['P'] = invert(P)

    Px = FunctionApplication(P, (x,))

    solver = WellFoundedRewriter(new_symbols)
    wx = solver.walk(Px)

    assert wx.functor.value == invert
    assert wx.args[0] == Px
    assert wx.args[0].functor == P
    assert wx.args[0].args[0] == x


    new_symbols = {}
    new_symbols['P'] = P

    Px = FunctionApplication(P, (x,))

    solver = WellFoundedRewriter(new_symbols)
    wx = solver.walk(Px)

    assert wx == Px


def test_constant_eval():

    t = C_(True)
    f = C_(False)

    solver = WellFoundedEvaluator({})

    assert solver.walk(t) == 1
    assert solver.walk(f) == 0
    

def test_symbol_eval():

    x = S_('x')

    new_symbols = {}
    new_symbols['x'] = x
    solver = WellFoundedEvaluator(new_symbols)
    wx = solver.walk(x)

    assert wx == 1


    new_symbols = {}
    solver = WellFoundedEvaluator(new_symbols)
    wx = solver.walk(x)

    assert wx == 1/2


def test_function_application_eval():

    x = S_('x')
    F = S_('F')
    Fx = F(x)
    nFx = invert(F(x))

    new_symbols = {}
    new_symbols['F'] = F
    new_symbols['x'] = x
    
    solver = WellFoundedEvaluator(new_symbols)
    wx = solver.walk(Fx)
    wnx = solver.walk(nFx)

    assert wx == 1
    assert wnx == 0

    xax = and_(x, x)
    xaFx = and_(x, Fx)
    xanFx = and_(x, nFx)
    FxanFx = and_(Fx, nFx)
    nFxaFx = and_(nFx, Fx)
    nFxanFx = and_(nFx, nFx)

    assert solver.walk(xax) == 1
    assert solver.walk(xaFx) == 1
    assert solver.walk(xanFx) == 0
    assert solver.walk(FxanFx) == 0
    assert solver.walk(nFxaFx) == 0
    assert solver.walk(nFxanFx) == 0

    xox = or_(x, x)
    xoFx = or_(x, Fx)
    xonFx = or_(x, nFx)
    FxonFx = or_(Fx, nFx)
    nFxoFx = or_(nFx, Fx)
    nFxonFx = or_(nFx, nFx)

    assert solver.walk(xox) == 1
    assert solver.walk(xoFx) == 1
    assert solver.walk(xonFx) == 1
    assert solver.walk(FxonFx) == 1
    assert solver.walk(nFxoFx) == 1
    assert solver.walk(nFxonFx) == 0

    
def test_function_application_unknow_eval():

    x = S_('x')
    F = S_('F')
    Fx = F(x)
    nFx = invert(F(x))

    new_symbols = {}
    new_symbols['F'] = F
    
    solver = WellFoundedEvaluator(new_symbols)
    wx = solver.walk(Fx)
    wnx = solver.walk(nFx)

    assert wx == 1
    assert wnx == 0

    xax = and_(x, x)
    xaFx = and_(x, Fx)
    xanFx = and_(x, nFx)
    FxanFx = and_(Fx, nFx)
    nFxaFx = and_(nFx, Fx)
    nFxanFx = and_(nFx, nFx)

    assert solver.walk(xax) == 1/2
    assert solver.walk(xaFx) == 1/2
    assert solver.walk(xanFx) == 0
    assert solver.walk(FxanFx) == 0
    assert solver.walk(nFxaFx) == 0
    assert solver.walk(nFxanFx) == 0

    xox = or_(x, x)
    xoFx = or_(x, Fx)
    xonFx = or_(x, nFx)
    FxonFx = or_(Fx, nFx)
    nFxoFx = or_(nFx, Fx)
    nFxonFx = or_(nFx, nFx)

    assert solver.walk(xox) == 1/2
    assert solver.walk(xoFx) == 1
    assert solver.walk(xonFx) == 1/2
    assert solver.walk(FxonFx) == 1
    assert solver.walk(nFxoFx) == 1
    assert solver.walk(nFxonFx) == 0


def test_implication_eval():

    t = C_(True)
    x = S_('x')
    P = S_('P')

    Px = P(x)
    nPx = invert(Px)

    y = S_('y')
    R = S_('R')

    Ry = R(y)
    nRy = invert(Ry)

    xiy = Implication(x, y)
    xiPx = Implication(x, Px)
    nPxiPx = Implication(nPx, Px)
    PxiRy = Implication(Px, Ry)
    PxinRy = Implication(Px, nRy)
    Pxiy = Implication(Px, y)

    new_symbols = {}
    new_symbols['x'] = x
    new_symbols['P'] = P
    new_symbols['R'] = nRy
    
    solver = WellFoundedEvaluator(new_symbols)

    wx1 = solver.walk(xiy)
    wx2 = solver.walk(xiPx)
    wx3 = solver.walk(nPxiPx)
    wx4 = solver.walk(PxiRy)
    wx5 = solver.walk(PxinRy)
    wx6 = solver.walk(Pxiy)

    assert wx1 == ()
    assert wx2 == (x,)
    assert wx3 == (nPx,)
    assert wx4 == (Px,)
    assert wx5 == (nPx,)
    assert wx6 == ()


def test_wellfounded_datalog():

    true = C_(True)
    unknown = C_[Unknown]('unk')

    p = S_('p')
    q = S_('q')
    s = S_('s')
    t = S_('t')
    u = S_('u')

    pitrue = Implication(p(), true)
    qitp = Implication(q(), and_(true, p()))
    qiunk = Implication(q(), unknown)
    tiqunk = Implication(t(), and_(q(), unknown))
    uiunkps = Implication(u(), and_(and_(unknown, p()), s()))

    class Datalog(
        sdb.SolverNonRecursiveDatalogNaive,
        solver_datalog_extensional_db.ExtensionalDatabaseSolver,
        expression_walker.ExpressionBasicEvaluator
    ):
        pass

    dl = Datalog()
    program = Eb_((
        pitrue, qitp, qiunk, tiqunk, uiunkps
    ))

    print(dl)
    eb = dl.walk(program)
    
    wfDatalog = WellFoundedDatalog()
    wfDatalog.solve(eb, dl)
    