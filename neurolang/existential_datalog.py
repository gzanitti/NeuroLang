from .expressions import (
    NeuroLangException, ExistentialPredicate, Symbol,
    ExpressionBlock, FunctionApplication, Lambda
)
from .solver_datalog_naive import (
    SolverNonRecursiveDatalogNaive, DatalogBasic, Implication,
)

from .expression_pattern_matching import add_match

__all__ = [
    'Implication',
    'ExistentialDatalog',
    'SolverNonRecursiveExistentialDatalog',
]


class ExistentialDatalog(DatalogBasic):
    def existential_intensional_database(self):
        return {
            k: v
            for k, v in self.symbol_table.items()
            if (
                k not in self.protected_keywords and
                isinstance(v, ExpressionBlock) and all(
                    isinstance(expression, Implication) and
                    isinstance(expression.consequent, ExistentialPredicate)
                    for expression in v.expressions
                )
            )
        }

    def intensional_database(self):
        return {
            k: v
            for k, v in self.symbol_table.items()
            if (
                k not in self.protected_keywords and
                isinstance(v, ExpressionBlock) and not any(
                    isinstance(expression, Implication) and
                    isinstance(expression.consequent, ExistentialPredicate)
                    for expression in v.expressions
                )
            )
        }

    @add_match(Implication(ExistentialPredicate, ...))
    def add_existential_implication_to_symbol_table(self, expression):
        """
        Add implication with a \u2203-quantified function application
        consequent to the symbol table
        """
        consequent_body, eq_variables = (
            parse_implication_with_existential_consequent(expression)
        )
        consequent_name = consequent_body.functor.name
        for eq_variable in eq_variables:
            if eq_variable in expression.antecedent._symbols:
                raise NeuroLangException(
                    '\u2203-quantified variable cannot occur in antecedent'
                )
            if eq_variable not in consequent_body._symbols:
                raise NeuroLangException(
                    "\u2203-quantified variable must occur "
                    "in consequent's body"
                )
        if consequent_name in self.symbol_table:
            if consequent_name in self.intensional_database():
                raise NeuroLangException(
                    'A rule cannot be both in IDB and E-IDB'
                )
            expressions = self.symbol_table[consequent_name].expressions
        else:
            expressions = tuple()
        expressions += (expression, )
        self.symbol_table[consequent_name] = ExpressionBlock(expressions)
        return expression


class SolverNonRecursiveExistentialDatalog(
    SolverNonRecursiveDatalogNaive, ExistentialDatalog
):
    @add_match(
        FunctionApplication(Implication(ExistentialPredicate, ...), ...)
    )
    def existential_consequent_implication_resolution(self, expression):
        consequent_body, eq_vars = (
            parse_implication_with_existential_consequent(expression.functor)
        )
        return self.walk(
            FunctionApplication(
                Lambda(consequent_body.args, expression.functor.antecedent),
                expression.args
            )
        )


def parse_implication_with_existential_consequent(expression):
    if not isinstance(expression, Implication):
        raise NeuroLangException('Not an implication')
    if not isinstance(expression.consequent, ExistentialPredicate):
        raise NeuroLangException('No existential consequent')
    eq_variables = set()
    e = expression.consequent
    while isinstance(e, ExistentialPredicate):
        eq_variables.add(e.head)
        e = e.body
    if not isinstance(e, FunctionApplication):
        raise NeuroLangException(
            'Expected core of consequent to be a function application'
        )
    if not all(isinstance(arg, Symbol) for arg in e.args):
        raise NeuroLangException(
            'Expected core of consequent to be '
            'a function application on symbols only'
        )
    if not isinstance(e.functor, Symbol):
        raise NeuroLangException(
            'Core of consequent functor expected to be a symbol'
        )
    return e, eq_variables