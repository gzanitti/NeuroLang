import typing

from ..logic import LogicOperator, Constant, FunctionApplication, Implication
from ..logic.unification import most_general_unifier, apply_substitution
from ..expression_walker import ReplaceSymbolWalker, add_match
from ..logic.expression_processing import ExtractFreeVariablesWalker




class RightImplication(LogicOperator):
    '''This class defines implications to the right. They are used to define
    constraints derived from ontologies. The functionality is the same as
    that of an implication, but with body and head inverted in position'''

    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent
        self.consequent = consequent
        self._symbols = consequent._symbols | antecedent._symbols

    def __repr__(self):
        return 'RightImplication{{{} \u2192 {}}}'.format(
            repr(self.consequent), repr(self.antecedent)
        )

class ExtractFreeVariablesRightImplicationWalker(ExtractFreeVariablesWalker):
    @add_match(RightImplication)
    def extract_variables_s(self, expression):
        return (
            self.walk(expression.consequent) -
            self.walk(expression.antecedent)
        )

class Rewriter():

    def __init__(self, dl, owl):
        self.dl = dl
        self.owl = owl

    def Xrewrite(self):
        '''Algorithm based on the one proposed in
        G. Gottlob, G. Orsi, and A. Pieris,
        “Query Rewriting and Optimization for Ontological Databases,”
        ACM Transactions on Database Systems, vol. 39, May 2014.'''
        i = 0
        Q_rew = set({})
        for t in self.dl.expressions:
            Q_rew.add((t, 'r', 'u'))

        sigma_free_vars = []
        for sigma in self.owl.expressions:
            efvw = ExtractFreeVariablesRightImplicationWalker()
            free_vars = efvw.walk(sigma)
            sigma_free_vars.append((sigma, free_vars))

        Q_temp = set({})
        while Q_rew != Q_temp:
            Q_temp = Q_rew.copy()
            for q in Q_temp:
                if q[2] == 'e':
                    continue
                q0 = q[0]
                for sigma in sigma_free_vars:
                    # rewriting step
                    body_q = q0.antecedent
                    S_applicable = self._get_applicable(sigma, body_q)
                    for S in S_applicable:
                        i += 1
                        sigma_i = self._rename(sigma[0], i)
                        qS = most_general_unifier(sigma_i.consequent, S)
                        if qS is not None:
                            new_q0 = self._combine(q0.consequent, qS, sigma_i.antecedent)
                            if (new_q0, 'r', 'u') not in Q_rew and (new_q0, 'r', 'e') not in Q_rew:
                                Q_rew.add((new_q0, 'r', 'u'))

                    # factorization step
                    body_q = q0.antecedent
                    S_factorizable = self._get_factorizable(sigma, body_q)
                    if len(S_factorizable) > 1:
                        for S in S_factorizable:
                            qS = most_general_unifier(S, body_q)
                            if qS is not None:
                                new_q0 = self._combine(q0.consequent, qS, sigma[0].antecedent)
                                if (
                                    (new_q0, 'r', 'u') not in Q_rew and (new_q0, 'r', 'e') not in Q_rew and
                                    (new_q0, 'f', 'u') not in Q_rew and (new_q0, 'f', 'e') not in Q_rew
                                ):
                                    Q_rew.add((new_q0, 'f', 'u'))
                # query is now explored
                Q_rew.remove(q)
                Q_rew.add((q[0], q[1], 'e'))


        return {x for x in Q_rew if x[2] == 'e'}

    def _get_factorizable(self, sigma, q):
        factorizable = []
        for free_var in sigma[1]._list:
            existential_position = self._get_position_existential(sigma[0].consequent, free_var)
            for S in self._get_term(q, sigma[0].consequent):
                if self._is_applicable(sigma, q, S) and self._var_same_position(existential_position, free_var, q, S):
                    factorizable.append(S)

        return factorizable

    def _var_same_position(self, pos, free_var, q, S):
        if self._free_var_other_term(free_var, q, S):
            return False

        if self._free_var_same_term_other_position(free_var, pos, q, S):
            return False

        return True

    def _free_var_other_term(self, free_var, q, S):
        if isinstance(q.args[0], FunctionApplication):
            for arg in q.args:
                if arg != S and free_var in arg.args:
                    return True
        else:
            if q != S and free_var in q.args:
                    return True

        return False

    def _free_var_same_term_other_position(self, free_var, pos, q, S):
        i = 0
        if isinstance(q.args[0], FunctionApplication):
            for arg in q.args:
                i = 0
                for sub_arg in arg.args:
                    if sub_arg == free_var and i != pos:
                        return True
                    i += 1
        else:
            for arg in q.args:
                if arg == free_var and i != pos:
                    return True
                i += 1

        return False

    def _get_applicable(self, sigma, q):
        applicable = []
        for S in self._get_term(q, sigma[0].consequent):
            if self._is_applicable(sigma, q, S):
                applicable.append(S)

        return applicable

    def _get_term(self, q, sigma_con):
        q_args = []
        if isinstance(q.args[0], FunctionApplication):
            for arg in q.args:
                if arg.functor == sigma_con.functor:
                    q_args.append(arg)
        else:
            if q.functor == sigma_con.functor:
                q_args.append(q)

        return q_args

    def _is_applicable(self, sigma, q, S):
        if (
            self._unifies(S, sigma[0].consequent) and
            self._not_in_existential(q, S, sigma)
        ):
            return True

        return False

    def _unifies(self, S, sigma):
        if most_general_unifier(S, sigma) is None:
            return False

        return True

    def _not_in_existential(self, q, S, sigma):
        for free_var in sigma[1]._list:
            existential_position = self._get_position_existential(sigma[0].consequent, free_var)
            if self._position_shared_or_constant(q, S, existential_position):
                return False

        return True

    def _get_position_existential(self, sigma, free_var):
        positions = []
        count = 0
        for symbol in sigma.args:
            if symbol == free_var:
                positions.append(count)
                count += 1

        return positions

    def _position_shared_or_constant(self, q, S, positions):
        for pos in positions:
            a = S.args[pos]
            if isinstance(a, Constant) or self._is_shared(a, q):
                return True

        return False

    def _is_shared(self, a, q):
        count = 0
        for term in q.args:
            if isinstance(term, FunctionApplication):
                if a in term.args:
                    count += 1
            else:
                if a == term:
                    count += 1

        if count > 1:
            return True

        return False

    def _rename(self, sigma, index):
        renamed = set({})
        a, renamed = self._replace(sigma.antecedent, index, renamed)
        b, renamed = self._replace(sigma.consequent, index, renamed)
        sus = {**a, **b}
        rsw = ReplaceSymbolWalker(sus)
        sigma = rsw.walk(sigma)

        return sigma

    def _replace(self, sigma, index, renamed):
        new_args = {}

        if isinstance(sigma.args[0], FunctionApplication):
            for app in sigma.args:
                new_arg, renamed = self._replace(app, index, renamed)
                new_args = {**new_args, **new_arg}
        else:
            for arg in sigma.args:
                if arg not in renamed:
                    temp = arg.fresh()
                    temp.name = arg.name + str(index)
                    new_args[arg] = temp
                renamed.add(arg)
        return new_args, renamed

    def _change_args(self, term, args, renamed):
        new_args = []

        if isinstance(term.args[0], FunctionApplication):
            for t in term.args:
                new_arg, renamed = self._change_args(t, args, renamed)
                new_args.append(new_arg)
        else:
            for arg in term.args:
                if arg not in renamed:
                    arg = args[arg]
                new_args.append(arg)
                renamed.add(arg)
        term.args = tuple(new_args)
        return term, renamed

    def _combine(self, q_cons, qS, sigma_ant):

        sigma_ant = apply_substitution(sigma_ant, qS[0])
        q0 = Implication(q_cons, sigma_ant)

        return q0