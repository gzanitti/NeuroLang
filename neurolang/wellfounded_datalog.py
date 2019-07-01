from .expressions import (
    FunctionApplication, Constant, NonConstant, NeuroLangException, is_leq_informative,
    Symbol, Lambda, ExpressionBlock, Expression, Definition,
    Query, ExistentialPredicate, Quantifier, Unknown,
)

from operator import invert, and_, or_

from .solver_datalog_naive import (
    SolverNonRecursiveDatalogNaive, DatalogBasic, Implication, Fact,
)

from .expression_walker import (
    add_match, PatternWalker, ExpressionBasicEvaluator, ReplaceSymbolWalker,
)

class WellFoundedDatalog():
    '''
    Implementation of negation in Datalog through Well-Founded Semantics.

    Given a program P, the intuition behind the solution is the following:
        - The sequence starts with an overestimate of the negative facts in the answer, called I0
        (it contains all negative facts).
        - From this overestimate, we compute I1 which includes all positive facts that can be
        infered from I0. This is an overestime of the positive facts in the answer, so the set of
        negative facts in I1 is an underestimate of the negative facts in the answer.
        - By continuing the process, we see that the even-indexed instance provide underestimates
        of the positive facts in th answer and the odd-indexed instances provide underestimates
        of the negative facts in the answer.
        - Finally, the 3-valued instance consisting of the facts known in both sequences becomes
        the well-founded semantics of P (all positive and negative facts belonging to all 3-stable
        model of P).
    '''

    _converged_positive = {}
    _overestimation_positive = {}
    _converged_negative = {}
    _overestimation_negative = {}

    def _deny_fact(self, expression: Expression) -> None:
        '''
        This function receives one expression that has to be negated and includes it in 
        the set of negated facts needed to form I0.

        Parameters
        ----------
        expression
            The Expression to be negated.

        Returns
        -------
        None

        '''
        for symbol in expression._symbols:
            self._overestimation_negative[symbol.name] = invert(symbol)

    def solve(self, eb: ExpressionBlock, datalog_program) -> tuple:
        '''
        Given an Expression Block and a Datalog Program, this function calculate the Well-Founded Semantics
        using the WellFoundedEvaluator and iterating between the overestime of the positive and negative facts
        until converge

        Parameters
        ----------
        datalog_program
            A datalog program.
        eb
            The expression block of the Datalog program
            

        Returns
        -------
        result
            A tuple of symbols
        '''

        for v in datalog_program.intensional_database().values():
            for rule in v.expressions:
                self._deny_fact(rule)

        not_converged = True

        while (not_converged):

            self._converged_positive = self._overestimation_positive.copy()

            wfEval = WellFoundedEvaluator(self._overestimation_negative)
            evaluated = wfEval.walk(eb)
            self._converged_negative = self._overestimation_negative.copy()

            self._overestimation_positive = {}
            for elem in evaluated:
                if hasattr(elem.functor, 'value') and elem.functor.value == invert:
                    name = elem.args[0].functor.name
                    self._overestimation_positive[name] = elem
                else:    
                    self._overestimation_positive[elem.functor.name] = elem

            wfEval = WellFoundedEvaluator(self._overestimation_positive)
            evaluated = wfEval.walk(eb)

            self._overestimation_negative = {}
            for elem in evaluated:
                if hasattr(elem.functor, 'value') and elem.functor.value == invert:
                    name = elem.args[0].functor.name
                    self._overestimation_negative[name] = elem
                else:    
                    self._overestimation_negative[elem.functor.name] = elem

            not_converged = (self._converged_positive != self._overestimation_positive
            or self._converged_negative != self._overestimation_negative)

        result = ()
        for k, v in self._converged_negative:
            if self._converged_positive[k] == v:
                result += (v,) 
        
        return result


class WellFoundedEvaluator(PatternWalker):
    '''
        This solver implement de iteration step of the fixed resolution to 
        solve a Well-Founded semantic. It receives a list of symbols 
        (It can include inverted symbols) and walks for every implication 
        evaluating them. It return a tuple of symbols and inverted symbols 
        depends on the result of this evaluations.

        Parameters
        ----------
        new_symbols
            The list of symbols.
        '''

    def __init__(self, new_symbols):
        self._new_symbols = new_symbols


    @add_match(Implication)
    def eval_implication(self, expression):
        new_antecedent = self.walk(expression.antecedent)

        if new_antecedent == 0:
            return (invert(expression.consequent),)
        elif new_antecedent == 1/2:
            return ()
        elif new_antecedent == 1:
            return (expression.consequent,)

    
    @add_match(Fact)
    def eval_fact(self, expression):
        return (expression.consequent,)

    
    @add_match(FunctionApplication(Constant(and_), ...))
    def eval_function_application_and(self, expression):
        eval_args = []
        for arg in expression.args:
            eval_args.append(self.walk(arg))
        return min(eval_args)


    @add_match(FunctionApplication(Constant(or_), ...))
    def eval_function_application_or(self, expression):
        eval_args = []
        for arg in expression.args:
            eval_args.append(self.walk(arg))
        return max(eval_args)


    @add_match(FunctionApplication(Expression, ...))
    def eval_function_application(self, expression):
        f_name = expression.functor.name
        if f_name in self._new_symbols:
            if hasattr(self._new_symbols[f_name].functor, 'value') and self._new_symbols[f_name].functor.value == invert:
                return 0
            else:
                return 1
        else:
            return 1/2

    
    @add_match(Symbol)
    def eval_symbol(self, expression):
        if expression.name in self._new_symbols:
            sym = self._new_symbols[expression.name]
            if hasattr(sym, 'value') and sym.value == invert:
                return 0
            else:
                return 1
        else:
            return 1/2


    @add_match(Constant)
    def eval_constant(self, expression):
        if expression.type == bool:
            return int(expression.value)
        else:
            return 1/2


    @add_match(ExpressionBlock)
    def eval_expression_block(self, expression):
        new_exps = ()
        for exp in expression.expressions:
            new_exps += self.walk(exp)

        return new_exps