import copy

import numpy
from core import corepack
import functools

if __name__ == '__main__':
    edge = 512
    coresperedge = 512
    # coresperedge = 1/2/4/8/16/32/64
    totalcores = coresperedge * coresperedge
    blockedge = edge / coresperedge

    print('------------    INFO    ------------')
    print(f'mat:{edge}x{edge}      cores:{coresperedge}x{coresperedge}')

    # A = numpy.array([numpy.array(list(range(edge))) + i * edge for i in range(edge)])
    # B = numpy.eye(edge, edge)

    A = numpy.random.rand(edge, edge)
    B = numpy.random.rand(edge, edge)

    C = numpy.zeros((edge, edge))

    # print('------------    A    ------------')
    # print(A)
    # print('------------    B    ------------')
    # print(B)

    cores = corepack(totalcores)

    All = numpy.array(list(range(totalcores)))
    row0 = numpy.array(list(range(coresperedge)))
    col0 = numpy.array(list(range(0, totalcores, coresperedge)))


    #feed data
    for i in range(totalcores):
        cores[i].assign('A', [A[(i//coresperedge) * (edge//coresperedge):(i//coresperedge+1) * (edge//coresperedge), (i%coresperedge) * (edge//coresperedge):(i%coresperedge+1) * (edge//coresperedge)]])
        cores[i].assign('B', [B[(i//coresperedge) * (edge//coresperedge):(i//coresperedge+1) * (edge//coresperedge), (i%coresperedge) * (edge//coresperedge):(i%coresperedge+1) * (edge//coresperedge)]])
        cores[i].assign('C', C[(i//coresperedge) * (edge//coresperedge):(i//coresperedge+1) * (edge//coresperedge), (i%coresperedge) * (edge//coresperedge):(i%coresperedge+1) * (edge//coresperedge)])
        cores[i].boundbuffer('up', cores[(i - coresperedge) % totalcores].recbuffer)
        cores[i].boundbuffer('down', cores[(i + coresperedge) % totalcores].recbuffer)
        cores[i].boundbuffer('left', cores[(i - 1) % coresperedge + (i // coresperedge)*coresperedge].recbuffer)
        cores[i].boundbuffer('right', cores[(i + 1) % coresperedge + (i // coresperedge) * coresperedge].recbuffer)

    #calculate
    for round in range(coresperedge - 1):
        cores(All, lambda x: x.send('left', x.vars['A'][-1]), f"send_{blockedge}x{blockedge}")
        cores(All, lambda x: x.assign('A', x.vars['A'] + [x.recieve()]), f"rec_{blockedge}x{blockedge}")

    for round in range(coresperedge - 1):
        cores(All, lambda x: x.send('up', x.vars['B'][-1]), f"send_{blockedge}x{blockedge}")
        cores(All, lambda x: x.assign('B', x.vars['B'] + [x.recieve()]), f"rec_{blockedge}x{blockedge}")

    for round in range(0, coresperedge):
        f_inner = lambda round, x: {x.assign('A', numpy.concatenate(x.vars['A'][-round:] + x.vars['A'][:-round], axis = 1))}
        f = functools.partial(f_inner, round)
        cores(col0 + round, f)
    for round in range(0, coresperedge):
        f_inner = lambda round, x: {x.assign('B', numpy.concatenate(x.vars['B'][-round:] + x.vars['B'][:-round], axis = 0))}
        f = functools.partial(f_inner, round)
        cores(row0 + round * coresperedge, f)

    cores(All, lambda x: {x.assign('C', numpy.matmul(x.vars['A'], x.vars['B']))}, f"mul_{blockedge}x{blockedge * coresperedge}")

    #get result
    cores.halt()
    for i in range(totalcores):
        C[(i//coresperedge) * (edge//coresperedge):(i//coresperedge+1) * (edge//coresperedge), (i%coresperedge) * (edge//coresperedge):(i%coresperedge+1) * (edge//coresperedge)]\
            = cores[i].vars['C']

    # print('------------    Ground Truth    ------------')
    # print(numpy.matmul(A, B))

    # print('------------    OURS    ------------')
    # print(C)

    print('------------    DELTA    ------------')
    print(numpy.sum(numpy.power(numpy.matmul(A, B) - C, 2)))

    print('------------    OPS    ------------')
    cores.printresult()


