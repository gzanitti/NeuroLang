import typing

import pytest

from ....datalog.expressions import Fact
from ....exceptions import NeuroLangException
from ....expressions import Constant, Symbol
from ....logic import Conjunction, Implication, Union
from ...expressions import ProbabilisticPredicate
from ..program import CPLogicProgram

P = Symbol("P")
Q = Symbol("Q")
R = Symbol("R")
Z = Symbol("Z")
Y = Symbol("Y")
p = Symbol("p")
p_1 = Symbol("p_1")
p_2 = Symbol("p_2")
x = Symbol("x")
y = Symbol("y")
z = Symbol("z")
a = Constant("a")
b = Constant("b")
c = Constant("c")
bernoulli = Symbol("bernoulli")


def test_probfact():
    probfact = Implication(
        ProbabilisticPredicate(Constant[float](0.2), P(x)),
        Constant[bool](True),
    )
    assert probfact.consequent.probability == Constant[float](0.2)
    assert probfact.consequent.body == P(x)

    with pytest.raises(
        NeuroLangException, match=r"must be a symbol or constant"
    ):
        Implication(ProbabilisticPredicate(0.3, P(x)), Constant[bool](True))


def test_deterministic_program():
    code = Union(
        (Implication(Z(x), Conjunction((P(x), Q(x)))), Fact(Q(a)), Fact(P(a)),)
    )
    cpl_program = CPLogicProgram()
    cpl_program.walk(code)
    assert cpl_program.extensional_database() == {
        Q: Constant(frozenset({(a,)})),
        P: Constant(frozenset({(a,)})),
    }
    assert cpl_program.intensional_database() == {
        Z: Union((Implication(Z(x), Conjunction((P(x), Q(x)))),))
    }


def test_cplogic_program():
    cpl = CPLogicProgram()
    code = Union(
        [
            Implication(
                ProbabilisticPredicate(Constant[float](0.5), P(x)),
                Constant[bool](True),
            ),
            Implication(Q(x), Conjunction((P(x), Z(x)))),
            Fact(Z(a)),
            Fact(Z(b)),
        ]
    )
    cpl.walk(code)
    assert cpl.extensional_database() == {
        Z: Constant(frozenset({Constant((a,)), Constant((b,))}))
    }
    assert cpl.intensional_database() == {
        Q: Union((Implication(Q(x), Conjunction((P(x), Z(x)))),))
    }
    assert cpl.probabilistic_facts() == {
        P: Union(
            [
                Implication(
                    ProbabilisticPredicate(Constant[float](0.5), P(x)),
                    Constant[bool](True),
                )
            ]
        )
    }


def test_multiple_probfact_same_pred_symb():
    cpl = CPLogicProgram()
    code = Union(
        [
            Implication(
                ProbabilisticPredicate(Constant[float](0.5), P(a)),
                Constant[bool](True),
            ),
            Implication(
                ProbabilisticPredicate(Constant[float](0.2), P(b)),
                Constant[bool](True),
            ),
            Implication(Q(x), Conjunction((P(x), Z(x)))),
            Fact(Z(a)),
            Fact(Z(b)),
        ]
    )
    cpl.walk(code)
    assert cpl.extensional_database() == {
        Z: Constant(frozenset({Constant((a,)), Constant((b,))}))
    }
    assert cpl.intensional_database() == {
        Q: Union((Implication(Q(x), Conjunction((P(x), Z(x)))),))
    }
    assert len(cpl.probabilistic_facts()) == 1
    assert P in cpl.probabilistic_facts()
    probfacts = cpl.probabilistic_facts()[P]
    assert isinstance(probfacts, Constant[typing.AbstractSet])


def test_add_probfacts_from_tuple():
    cpl = CPLogicProgram()
    cpl.walk(Union(tuple()))
    cpl.add_probabilistic_facts_from_tuples(
        P, {(0.3, "hello", "gaston"), (0.7, "hello", "antonia"),},
    )
    assert P in cpl.pfact_pred_symbs
    assert (
        Constant[float](0.7),
        Constant[str]("hello"),
        Constant[str]("antonia"),
    ) in cpl.symbol_table[P].value


def test_add_probfacts_from_tuple_no_probability():
    cpl = CPLogicProgram()
    cpl.walk(Union(tuple()))
    with pytest.raises(NeuroLangException, match=r"probability"):
        cpl.add_probabilistic_facts_from_tuples(
            P, {("hello", "gaston"), ("hello", "antonia"),},
        )


def test_add_probchoice_from_tuple():
    probchoice_as_tuples_iterable = {
        (0.5, "a", "a"),
        (0.2, "a", "b"),
        (0.3, "b", "b"),
    }
    cpl = CPLogicProgram()
    cpl.add_probabilistic_choice_from_tuples(P, probchoice_as_tuples_iterable)
    assert P in cpl.symbol_table
    assert (
        Constant[float](0.2),
        Constant[str]("a"),
        Constant[str]("b"),
    ) in cpl.symbol_table[P].value


def test_add_probchoice_from_tuple_no_probability():
    cpl = CPLogicProgram()
    with pytest.raises(NeuroLangException, match=r"probability"):
        cpl.add_probabilistic_choice_from_tuples(P, {("a", "b"), ("b", "b")})


def test_add_probchoice_from_tuple_twice_same_pred_symb():
    probchoice_as_tuples_iterable = {(1.0, "a", "a")}
    cpl = CPLogicProgram()
    cpl.add_probabilistic_choice_from_tuples(P, probchoice_as_tuples_iterable)
    with pytest.raises(NeuroLangException):
        cpl.add_probabilistic_choice_from_tuples(
            P, probchoice_as_tuples_iterable
        )


def test_add_probchoice_does_not_sum_to_one():
    probchoice_as_tuples_iterable = {
        (0.5, "a", "a"),
        (0.2, "a", "b"),
        (0.1, "b", "b"),
    }
    cpl = CPLogicProgram()
    with pytest.raises(NeuroLangException, match=r"sum"):
        cpl.add_probabilistic_choice_from_tuples(
            P, probchoice_as_tuples_iterable
        )