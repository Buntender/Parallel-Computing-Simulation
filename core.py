import numpy
import random

class core():
    def __init__(self):
        self.vars = {}
        self.recbuffer = []
        self.sendbuffer = {}

    def boundbuffer(self, name, buffer):
        self.sendbuffer[name] = buffer

    def send(self, targetname, item):
        if len(self.sendbuffer[targetname]) == 0:
            self.sendbuffer[targetname].append(item)

    def recieve(self):
        return self.recbuffer.pop(0)

    def assign(self, name, value):
        self.vars[name] = value


class corepack(list):
    def __init__(self, totalcores):
        super().__init__([core() for i in range(totalcores)])
        self.log = {}
        self.funcgraph = [None] * len(self)
        self.lazycompute = []
        self.laststepname = None

    def __call__(self, coreslist, func, stepname='__NOP__'):
        if stepname not in self.log:
            self.log[stepname] = []
            if self.laststepname == None:
                self.laststepname = stepname

        if self.laststepname != stepname or any([self.funcgraph[core] for core in coreslist]):
            random.shuffle(self.lazycompute)
            [self.funcgraph[core](self[core]) for core in self.lazycompute]
            self.log[self.laststepname].append(1 - (self.funcgraph.count(None)) / len(self))
            self.funcgraph = [None] * len(self)
            self.lazycompute = []

        for core in coreslist:
            self.funcgraph[core] = func
        self.lazycompute += list(coreslist)

        self.laststepname = stepname

    def halt(self):
        self([], None, '__NOP__')
        self.log.pop('__NOP__')

    def printresult(self):
        ops = {key:[len(self.log[key]), sum(self.log[key]) / len(self.log[key])] for key in self.log}
        for item in ops:
            print(f'{item}: \t step:{ops[item][0]} \t usage:{format(ops[item][1], ".4f")}')
        # print(self.ratiocount)