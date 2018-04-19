import sys
sys.path.insert(0, '../grammars-v4/java/')

import collections
import pprint
pp = pprint.PrettyPrinter(indent=2)

from antlr4 import *
from antlr4.tree.Tree import TerminalNodeImpl
from JavaLexer import JavaLexer
from JavaParser import JavaParser
from JavaParserListener import JavaParserListener

class PrintListener(JavaParserListener):
    def enterCompilationUnit(self, cu):
        print("getChildren: %s" % list(cu.getChildren()))

def to_tree(node):
    ret = None
    if node.getChildCount() > 0:
        ret = [to_tree(child) for child in node.getChildren()]
    else:
        ret = []
    ret.insert(0, node)
    return ret

def java_file_to_tree(filename):
    input = FileStream(filename)
    lexer = JavaLexer(input)
    stream = CommonTokenStream(lexer)
    parser = JavaParser(stream)
    tree = parser.compilationUnit()
    return to_tree(tree)

def head_children(lst):
    head = lst[0]
    children = lst[1:]
    return (head, children)

def leaves(tree):
    head, children = head_children(tree)
    if children:
        return [child_leaf for child in children for child_leaf in leaves(child)]
    else:
        return [head]

def map_tree(fn, tree):
    head, children = head_children(tree)
    ret = [map_tree(fn, child) for child in children]
    ret.insert(0, fn(head))
    return ret

def flatten_prefix(tree):
    lst = []
    for item in tree:
        if isinstance(item, collections.Sequence):
            lst += flatten_prefix(item)
        else:
            lst += [item]
    return lst

tree = java_file_to_tree("../java/HelloWalker.java")
root = tree[0]
flat_tree_prefix = flatten_prefix(tree)
leaf = leaves(tree)[0]

map_tree(lambda n: n.getText(), [leaf])

map_tree(lambda n: n.getText(), tree)

root.getText()
root.qualifiedName()
dir(root)

pp.pprint(tree)
pp.pprint(flat_tree_prefix)

[print(node) for node in flat_tree_prefix]
[print(node) for node in leaves(tree)]

if __name__ == '__main__':
    print(leaf)
