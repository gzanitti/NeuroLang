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

    overestimate_negative = {}
    overestimate_positive = {}

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
            self.overestimate_negative[symbol.name] = invert(symbol)

    def solve(self, datalog_program):

        
        for v in datalog_program.intensional_database().values():
            for rule in v.expressions:
                self._deny_fact(rule)

        wfEval = WellFoundedEvaluator(self.overestimate_negative)
        res = wfEval.walk(datalog_program)

        print(res)


class WellFoundedEvaluator(PatternWalker):


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
        if hasattr(expression.functor, 'value') and expression.functor.value == invert:
            if expression.args[0].functor.name in self._new_symbols:
                return 0
            else:
                return 1/2
        else:
            if expression.functor.name in self._new_symbols:
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


    @add_match(Constant[bool])
    def eval_constant(self, expression):
        return int(expression.value)


    @add_match(ExpressionBlock)
    def eval_expression_block(self, expression):
        new_exps = ()
        for exp in expression.expressions:
            new_exps += self.walk(exp)

        return new_exps

class WellFoundedRewriter(PatternWalker):


    def __init__(self, new_symbols):
        self._new_symbols = new_symbols


    @add_match(Symbol)
    def rewrite_symbol(self, expression):
        return self._new_symbols[expression.name]


    @add_match(FunctionApplication(Expression, ...))
    def rewrite_function_application(self, expression):
        functor = expression.functor
        if functor in self._new_symbols and hasattr(self._new_symbols[functor.name], 'functor'):
            return invert(
                FunctionApplication(functor, expression.args)
            )
        else:
            return FunctionApplication(functor, expression.args)


    @add_match(FunctionApplication(Constant(and_), ...))
    def rewrite_function_application_and(self, expression):
        new_args = ()
        for arg in expression.args:
            new_args += (self.walk(arg),)

        return FunctionApplication(Constant(or_), new_args)


    @add_match(FunctionApplication(Constant(or_), ...))
    def rewrite_function_application_or(self, expression):
        new_args = ()
        for arg in expression.args:
            new_args += (self.walk(arg),)

        return FunctionApplication(Constant(and_), new_args)


    @add_match(Implication)
    def rewrite_implication(self, expression):
        new_antecedent = self.walk(expression.antecedent)

        return Implication(expression.consequent, new_antecedent)


    @add_match(ExpressionBlock)
    def rewrite_expression_block(self, expression):
        new_exps = ()
        for exp in expression.expressions:
            new_exps += self.walk(exp)

        return ExpressionBlock(new_exps)
