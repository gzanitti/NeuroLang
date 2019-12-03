"""
Compiler for the intermediate representation of a Datalog program.
The :class:`DatalogProgram` class processes the
intermediate representation of a program and extracts
the extensional, intensional, and builtin
sets and has support for constraints.
"""

from .basic_representation import DatalogProgram


class DatalogConstraintsProgram(DatalogProgram):

    protected_keywords = set({'__constraints__'})

    def load_constraints(self, expression_block):
        self.symbol_table['__constraints__'] = expression_block
