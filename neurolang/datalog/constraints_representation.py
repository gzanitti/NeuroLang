"""
Compiler for the intermediate representation of a Datalog program.
The :class:`DatalogProgram` class processes the
intermediate representation of a program and extracts
the extensional, intensional, and builtin
sets and has support for constraints.
"""

from .basic_representation import DatalogProgram
from ..expression_walker import PatternWalker, add_match
from ..expressions import (Constant, Expression, FunctionApplication,
                           NeuroLangException, Symbol)
from .ontologies_rewriter import RightImplication
from .expressions import Union

class DatalogConstraintsProgram(DatalogProgram):

    protected_keywords = set({'__constraints__'})

    @add_match(RightImplication(
        FunctionApplication[bool](Symbol, ...),
        Expression
    ))
    def statement_intensional(self, expression):
        consequent = expression.consequent
        antecedent = expression.antecedent

        self._validate_implication_syntax(consequent, antecedent)

        #if consequent.functor in self.symbol_table:
        #    disj = self._new_intensional_internal_representation(consequent)
        #else:
        disj = tuple()

        if expression not in disj:
            disj += (expression,)

        self.symbol_table['__constraints__'] = Union(disj)

        return expression

    def load_constraints(self, expression_block):
        datalog_program = self.walk(expression_block)
        self.symbol_table['__constraints__'] = datalog_program

    def get_constraints(self):
        return self.symbol_table['__constraints__']
