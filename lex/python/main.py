import sys
sys.path.insert(0, '../grammars-v4/java/')

from antlr4 import *
from antlr4.tree.Tree import TerminalNodeImpl
from JavaLexer import JavaLexer
from JavaParser import JavaParser
from JavaParserListener import JavaParserListener

class PrintListener(JavaParserListener):
    def enterCompilationUnit(self, cu):
        print("getChildren: %s" % list(cu.getChildren()))

    #def visitTerminal(self, node):
    #    print("getSymbol: %s" % node.getSymbol())

    #def enterEveryRule(self, ctx):
    #    #print("Dir ctx: %s" % dir(ctx))
    #    print("getText: %s" % ctx.getText())
    #    #print("getToken: %s" % ctx.getToken())
    #    #print("getTokens: %s" % ctx.getTokens())
    #    print("getPayload: %s" % ctx.getPayload())
    #    #print("toString: %s" % ctx.toString())
    #    print("toStringTree: %s" % ctx.toStringTree())
    #    #print("Enter EVERY rule: %s" % ctx.ID())

def toTree(node):
    ret = None
    if node.getChildCount() > 0:
        ret = [toTree(child) for child in node.getChildren()]
    else:
        ret = []
    ret.insert(0, node)
    return ret

def main(argv):
    input = FileStream("../java/HelloWalker.java")

    #input = "hello abc"
    lexer = JavaLexer(input)
    stream = CommonTokenStream(lexer)
    parser = JavaParser(stream)
    tree = parser.compilationUnit()
    return toTree(tree)
    #printer = PrintListener()
    #walker = ParseTreeWalker()
    #walker.walk(printer, tree)

if __name__ == '__main__':
    print(main(sys.argv))
