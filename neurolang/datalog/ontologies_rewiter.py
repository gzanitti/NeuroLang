import typing

from ..logic import LogicOperator, ExistentialPredicate, Constant, FunctionApplication, Implication
from ..logic.unification import most_general_unifier, apply_substitution


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
        Q_temp = set({})
        while Q_rew != Q_temp:
            Q_temp = Q_rew.copy()
            for q in Q_temp:
                if q[2] == 'e':
                    break
                q0 = q[0]
                for sigma in self.owl.expressions:
                    # rewriting step
                    body_q = q0.antecedent
                    S_applicable = self._get_applicable(sigma, body_q)
                    for S in S_applicable:
                        i += 1
                        sigma_i = self._rename(sigma, i)
                        qS = most_general_unifier(sigma_i.consequent, S)
                        if qS is not None:
                            new_q0 = self._combine(q0.consequent, qS, sigma_i.antecedent)
                            if (new_q0, 'r', 'u') not in Q_temp and (new_q0, 'r', 'e') not in Q_temp:
                                Q_rew.add((new_q0, 'r', 'u'))

                    # factorization step
                    body_q = q0.antecedent
                    S_factorizable = self._get_factorizable(sigma, body_q)
                    for S in S_factorizable:
                        qS = most_general_unifier(S, body_q)
                        if qS is not None:
                            new_q0 = self._combine(q0.consequent, qS, sigma_i.antecedent)
                            if (
                                (new_q0, 'r', 'u') not in Q_temp and (new_q0, 'r', 'e') not in Q_temp and
                                (new_q0, 'f', 'u') not in Q_temp and (new_q0, 'f', 'e') not in Q_temp
                            ):
                                Q_rew.add((new_q0, 'f', 'u'))
                Q_rew.remove(q)
                Q_rew.add((q[0], q[1], 'e'))


        return {x for x in Q_rew if x[2] == 'e'}

    def _get_factorizable(self, sigma, q):
        factorizable = []
        for S in self._get_term(q):
            factorizable.append(S)

        return factorizable

    def _get_applicable(self, sigma, q):
        applicable = []
        for S in self._get_term(q):
            if self._is_applicable(sigma, q, S):
                applicable.append(S)

        return applicable

    #Maybe there is a more elegante way to do this.
    def _get_term(self, q):
        if isinstance(q.args[0], FunctionApplication):
            return q.args

        return (q,)

    def _is_applicable(self, sigma, q, S):
        if (
            self._unifies(S, sigma.consequent) and
            self._not_in_existential(q, S, sigma.consequent)
        ):
            return True

        return False

    def _unifies(self, S, sigma_head):
        if isinstance(sigma_head, ExistentialPredicate):
            sigma_head = sigma_head.body
        if most_general_unifier(S, sigma_head) is None:
            return False

        return True

    def _not_in_existential(self, q, S, sigma_head):
        existential_position = self._get_position_existential(sigma_head)

        if self._position_not_shared_or_constant(q, S, existential_position):
            return False

        return True

    def _get_position_existential(self, sigma):
        positions = []
        if isinstance(sigma, ExistentialPredicate):
            head = sigma.head
            count = 0
            for symbol in sigma.body.args:
                if symbol == head:
                    positions.append(count)
                    count += 1
        else:
            #Exception?
            return positions

        return positions

    def _position_not_shared_or_constant(self, q, S, positions):
        for pos in positions:
            a = S.args[pos]
            if isinstance(a, Constant) or not self._is_shared(a, q):
                return False

        return True

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

    # Maybe there is a more elegante way to do this (probably with a solver).
    def _rename(self, sigma, index):
        renamed = set({})
        sigma.antecedent, renamed = self._replace(sigma.antecedent, index, renamed)
        sigma.consequent, renamed = self._replace(sigma.consequent, index, renamed)

        return sigma

    def _replace(self, sigma, index, renamed):
        new_args = []
        if isinstance(sigma, ExistentialPredicate):
            sigma = sigma.body

        if isinstance(sigma.args[0], FunctionApplication):
            for app in sigma.args:
                new_arg, renamed = self._replace(app, index, renamed)
                new_args.append(new_arg)
        else:
            for arg in sigma.args:
                if arg not in renamed:
                    arg.name = arg.name + str(index)
                new_args.append(arg)
                renamed.add(arg)
        sigma.args = tuple(new_args)
        return sigma, renamed

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