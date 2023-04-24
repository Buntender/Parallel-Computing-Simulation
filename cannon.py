import numpy
from core import corepack

if __name__ == '__main__':
    edge = 3
    coresperedge = 3
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
    TMP = numpy.zeros((edge, edge))

    print('------------    A    ------------')
    print(A)
    print('------------    B    ------------')
    print(B)

    cores = corepack(totalcores)

    All = numpy.array(list(range(totalcores)))
    row0 = numpy.array(list(range(coresperedge)))
    col0 = numpy.array(list(range(0, totalcores, coresperedge)))


    #feed data
    for i in range(totalcores):
        cores[i].assign('A', A[(i//coresperedge) * (edge//coresperedge):(i//coresperedge+1) * (edge//coresperedge), (i%coresperedge) * (edge//coresperedge):(i%coresperedge+1) * (edge//coresperedge)])
        cores[i].assign('B', B[(i//coresperedge) * (edge//coresperedge):(i//coresperedge+1) * (edge//coresperedge), (i%coresperedge) * (edge//coresperedge):(i%coresperedge+1) * (edge//coresperedge)])
        cores[i].assign('C', C[(i//coresperedge) * (edge//coresperedge):(i//coresperedge+1) * (edge//coresperedge), (i%coresperedge) * (edge//coresperedge):(i%coresperedge+1) * (edge//coresperedge)])
        cores[i].assign('TMP', TMP[(i//coresperedge) * (edge//coresperedge):(i//coresperedge+1) * (edge//coresperedge), (i%coresperedge) * (edge//coresperedge):(i%coresperedge+1) * (edge//coresperedge)])
        cores[i].boundbuffer('up', cores[(i - coresperedge) % totalcores].recbuffer)
        cores[i].boundbuffer('down', cores[(i + coresperedge) % totalcores].recbuffer)
        cores[i].boundbuffer('left', cores[(i - 1) % coresperedge + (i // coresperedge)*coresperedge].recbuffer)
        cores[i].boundbuffer('right', cores[(i + 1) % coresperedge + (i // coresperedge) * coresperedge].recbuffer)

    #calculate
    for round in range((coresperedge + 1) // 2):
        for i in range(round + 1, (coresperedge + 1) // 2):
            cores(row0 + i * coresperedge, lambda x: x.send('left', x.vars['A']), f"send_{blockedge}x{blockedge}")
        for i in range(coresperedge - 1 - round, (coresperedge + 1) // 2 - 1, -1):
            cores(row0 + i * coresperedge, lambda x: x.send('right', x.vars['A']), f"send_{blockedge}x{blockedge}")

        for i in range(round + 1, (coresperedge + 1) // 2):
            cores(row0 + i * coresperedge, lambda x: x.assign('A', x.recieve()), f"rec_{blockedge}x{blockedge}")
        for i in range(coresperedge - 1 - round, (coresperedge + 1) // 2 - 1, -1):
            cores(row0 + i * coresperedge, lambda x: x.assign('A', x.recieve()), f"rec_{blockedge}x{blockedge}")

    for round in range((coresperedge + 1) // 2):
        for i in range(round + 1, (coresperedge + 1) // 2):
            cores(col0 + i, lambda x: x.send('up', x.vars['B']), f"send_{blockedge}x{blockedge}")
        for i in range(coresperedge - 1 - round, (coresperedge + 1) // 2 - 1, -1):
            cores(col0 + i, lambda x: x.send('down', x.vars['B']), f"send_{blockedge}x{blockedge}")

        for i in range(round + 1, (coresperedge + 1) // 2):
            cores(col0 + i, lambda x: x.assign('B', x.recieve()), f"rec_{blockedge}x{blockedge}")
        for i in range(coresperedge - 1 - round, (coresperedge + 1) // 2 - 1, -1):
            cores(col0 + i, lambda x: x.assign('B', x.recieve()), f"rec_{blockedge}x{blockedge}")

    cores(All, lambda x: x.assign('C', numpy.matmul(x.vars['A'], x.vars['B'])), f"mul_{blockedge}x{blockedge}")
    for i in range(coresperedge-1):
        cores(All, lambda x: x.send('left', x.vars['A']), f"send_{blockedge}x{blockedge}")
        cores(All, lambda x: x.assign('A', x.recieve()), f"rec_{blockedge}x{blockedge}")
        cores(All, lambda x: x.send('up', x.vars['B']), f"send_{blockedge}x{blockedge}")
        cores(All, lambda x: x.assign('B', x.recieve()), f"rec_{blockedge}x{blockedge}")

        cores(All, lambda x: x.assign('TMP', numpy.matmul(x.vars['A'], x.vars['B'])), f"mul_{blockedge}x{blockedge}")
        cores(All, lambda x: x.assign('C', x.vars['TMP'] + x.vars['C']), f"add_{blockedge}x{blockedge}")

    #get result
    cores.halt()
    for i in range(totalcores):
        C[(i//coresperedge) * (edge//coresperedge):(i//coresperedge+1) * (edge//coresperedge), (i%coresperedge) * (edge//coresperedge):(i%coresperedge+1) * (edge//coresperedge)]\
            = cores[i].vars['C']

    print('------------    Ground Truth    ------------')
    print(numpy.matmul(A, B))

    print('------------    OURS    ------------')
    print(C)

    print('------------    DELTA    ------------')
    print(numpy.sum(numpy.power(numpy.matmul(A, B) - C, 2)))

    print('------------    OPS    ------------')
    cores.printresult()


