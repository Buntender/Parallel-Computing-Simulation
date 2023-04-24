import numpy
from core import corepack
import functools

if __name__ == '__main__':
    edge = 512
    coresperedge = 64
    # coresperedge = 1/2/4/8/16
    totalcores = coresperedge * coresperedge * coresperedge
    blockedge = edge / coresperedge

    print('------------    INFO    ------------')
    print(f'mat:{edge}x{edge}      cores:{coresperedge}x{coresperedge}x{coresperedge}')

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
    col0 = numpy.array(list(range(0, totalcores // coresperedge, coresperedge)))
    hei0 = numpy.array(list(range(0, totalcores, coresperedge * coresperedge)))
    layer0 = numpy.array(list(range(coresperedge * coresperedge)))


    #feed data
    for i in range(coresperedge * coresperedge):
        cores[i].assign('A', A[(i//coresperedge) * (edge//coresperedge):(i//coresperedge+1) * (edge//coresperedge), (i%coresperedge) * (edge//coresperedge):(i%coresperedge+1) * (edge//coresperedge)])
        cores[i].assign('B', B[(i//coresperedge) * (edge//coresperedge):(i//coresperedge+1) * (edge//coresperedge), (i%coresperedge) * (edge//coresperedge):(i%coresperedge+1) * (edge//coresperedge)])

    for i in range(coresperedge * coresperedge, totalcores):
        cores[i].assign('A', numpy.zeros((edge // coresperedge, edge // coresperedge)))
        cores[i].assign('B', numpy.zeros((edge // coresperedge, edge // coresperedge)))

    for i in range(totalcores):
        cores[i].boundbuffer('right', cores[(i + 1) % coresperedge + (i // coresperedge) * coresperedge].recbuffer)
        cores[i].boundbuffer('left', cores[(i - 1) % coresperedge + (i // coresperedge) * coresperedge].recbuffer)

        cores[i].boundbuffer('back', cores[(i + coresperedge) % (coresperedge * coresperedge) + (i // (coresperedge * coresperedge))*coresperedge * coresperedge].recbuffer)
        cores[i].boundbuffer('forward', cores[(i - coresperedge) % (coresperedge * coresperedge) + (i // (coresperedge * coresperedge))*coresperedge * coresperedge].recbuffer)

        cores[i].boundbuffer('up', cores[(i + coresperedge * coresperedge) % totalcores].recbuffer)
        cores[i].boundbuffer('down', cores[(i - coresperedge * coresperedge) % totalcores].recbuffer)

    #calculate
    # cores(layer0, lambda x: x.send('up', x.vars['A']), f"send_{blockedge}x{blockedge}")
    # cores(layer0 + coresperedge * coresperedge, lambda x: x.assign('A', x.recieve()), f"rec_{blockedge}x{blockedge}")
    # for sellayer in range(1, coresperedge-1):
    #     cores(layer0 + sellayer * coresperedge * coresperedge, lambda x: x.send('up', x.vars['A']), f"send_{blockedge}x{blockedge}")
    #     cores(layer0 + (sellayer - 1) * coresperedge * coresperedge, lambda x: x.send('up', x.vars['B']), f"send_{blockedge}x{blockedge}")
    #
    #     cores(layer0 + (sellayer + 1) * coresperedge * coresperedge, lambda x: x.assign('A', x.recieve()), f"rec_{blockedge}x{blockedge}")
    #     cores(layer0 + sellayer * coresperedge * coresperedge, lambda x: x.assign('B', x.recieve()), f"rec_{blockedge}x{blockedge}")
    #
    # cores(layer0 + (coresperedge - 2) * coresperedge * coresperedge, lambda x: x.send('up', x.vars['B']), f"send_{blockedge}x{blockedge}")
    # cores(layer0 + (coresperedge - 1) * coresperedge * coresperedge, lambda x: x.assign('B', x.recieve()), f"rec_{blockedge}x{blockedge}")

    wavefronts = {'A':{'up':None, 'down': None}, 'B':{'up':None, 'down': None}}
    wavefrontnames = [['A', 'up'], ['A', 'down'], ['B', 'up'], ['B', 'down']]

    step = 0
    while step < 3 or any([wavefronts[name[0]][name[1]] is not None for name in wavefrontnames]):
        if step < 4:
            wavefronts[wavefrontnames[step][0]][wavefrontnames[step][1]] = 0
            step += 1

        for name in wavefronts.keys():
            if wavefronts[name]['up'] is None:
                wavefronts[name]['down'] = None
            elif wavefronts[name]['down'] is not None and wavefronts[name]['up'] == (wavefronts[name]['down'] - 1) % coresperedge:
                wavefronts[name]['up'] = None
                wavefronts[name]['down'] = None
            elif wavefronts[name]['down'] is not None and wavefronts[name]['up'] + 1 == (wavefronts[name]['down'] - 1) % coresperedge:
                wavefronts[name]['up'] = None

            if wavefronts[name]['up'] is not None:
                f_inner = lambda name, x: {x.send('up', x.vars[name])}
                f = functools.partial(f_inner, name)
                cores(layer0 + wavefronts[name]['up'] * coresperedge * coresperedge, f,
                      f"send_{blockedge}x{blockedge}")

            if wavefronts[name]['down'] is not None:
                f_inner = lambda name, x: {x.send('down', x.vars[name])}
                f = functools.partial(f_inner, name)
                cores(layer0 + wavefronts[name]['down'] * coresperedge * coresperedge, f,
                      f"send_{blockedge}x{blockedge}")

        for name in wavefronts.keys():
            if wavefronts[name]['up'] is not None:
                wavefronts[name]['up'] = (wavefronts[name]['up'] + 1) % coresperedge
                f_inner = lambda name, x: {x.assign(name, x.recieve())}
                f = functools.partial(f_inner, name)
                cores(layer0 + wavefronts[name]['up'] * coresperedge * coresperedge, f,
                      f"rec_{blockedge}x{blockedge}")

            if wavefronts[name]['down'] is not None:
                wavefronts[name]['down'] = (wavefronts[name]['down'] - 1) % coresperedge
                f_inner = lambda name, x: {x.assign(name, x.recieve())}
                f = functools.partial(f_inner, name)
                cores(layer0 + wavefronts[name]['down'] * coresperedge * coresperedge, f,
                      f"rec_{blockedge}x{blockedge}")

    for round in range((coresperedge + 1) // 2):
        if round < coresperedge // 2:
            for layer in range(coresperedge):
                cores((col0 + (layer + round)) % (coresperedge * coresperedge) + layer * coresperedge * coresperedge,
                      lambda x: x.send('right', x.vars['A']), f"send_{blockedge}x{blockedge}")
        if round > 0:
            for layer in range(coresperedge):
                cores((col0 + (layer - round + 1)) % (coresperedge * coresperedge) + layer * coresperedge * coresperedge,
                      lambda x: x.send('left', x.vars['A']), f"send_{blockedge}x{blockedge}")

        if round < coresperedge // 2:
            for layer in range(coresperedge):
                cores((col0 + (layer + round + 1)) % (
                            coresperedge * coresperedge) + layer * coresperedge * coresperedge,
                      lambda x: x.assign('A', x.recieve()), f"rec_{blockedge}x{blockedge}")
        if round > 0:
            for layer in range(coresperedge):
                cores(
                    (col0 + (layer - round)) % (coresperedge * coresperedge) + layer * coresperedge * coresperedge,
                    lambda x: x.assign('A', x.recieve()), f"rec_{blockedge}x{blockedge}")

    for round in range((coresperedge + 1) // 2):
        if round < coresperedge // 2:
            for layer in range(coresperedge):
                cores((row0 + (layer + round) * coresperedge) % (coresperedge * coresperedge) + layer * coresperedge * coresperedge, lambda x: x.send('back', x.vars['B']), f"send_{blockedge}x{blockedge}")
        if round > 0:
            for layer in range(coresperedge):
                cores((row0 + (layer - round + 1) * coresperedge) % (coresperedge * coresperedge) + layer * coresperedge * coresperedge, lambda x: x.send('forward', x.vars['B']), f"send_{blockedge}x{blockedge}")

        if round < coresperedge // 2:
            for layer in range(coresperedge):
                cores((row0 + (layer + round + 1) * coresperedge) % (coresperedge * coresperedge) + layer * coresperedge * coresperedge, lambda x: x.assign('B', x.recieve()), f"rec_{blockedge}x{blockedge}")
        if round > 0:
            for layer in range(coresperedge):
                cores((row0 + (layer - round) * coresperedge) % (coresperedge * coresperedge) + layer * coresperedge * coresperedge, lambda x: x.assign('B', x.recieve()), f"rec_{blockedge}x{blockedge}")

    cores(All, lambda x: x.assign('A', numpy.matmul(x.vars['A'], x.vars['B'])), f"mul_{blockedge}x{blockedge}")

    for round in range((coresperedge + 1) // 2):
        if round >= coresperedge % 2:
            cores(((layer0 + ((coresperedge + 1) // 2 + round - (coresperedge % 2)) * coresperedge * coresperedge) % totalcores),
                  lambda x: x.send('up', x.vars['A']), f"send_{blockedge}x{blockedge}")
        if round < coresperedge // 2 and round < (coresperedge - 1) // 2:
            cores(((layer0 + ((coresperedge - 1) // 2 - round) * coresperedge * coresperedge) % totalcores),
                  lambda x: x.send('down', x.vars['A']), f"send_{blockedge}x{blockedge}")

        if round >= coresperedge % 2:
            cores(((layer0 + ((coresperedge + 1) // 2 + round - (coresperedge % 2) + 1) * coresperedge * coresperedge) % totalcores),
                  lambda x: x.assign('B', x.recieve()), f"rec_{blockedge}x{blockedge}")
        if round < coresperedge // 2 and round < (coresperedge - 1) // 2:
            cores(((layer0 + ((coresperedge - 1) // 2 - round - 1) * coresperedge * coresperedge) % totalcores),
                  lambda x: x.assign('B', x.recieve()), f"rec_{blockedge}x{blockedge}")

        if round >= coresperedge % 2:
            cores(((layer0 + ((coresperedge + 1) // 2 + round - (coresperedge % 2) + 1) * coresperedge * coresperedge) % totalcores),
                  lambda x: x.assign('A', x.vars['A'] + x.vars['B']), f"add_{blockedge}x{blockedge}")
        if round < coresperedge // 2 and round < (coresperedge - 1) // 2:
            cores(((layer0 + ((coresperedge - 1) // 2 - round - 1) * coresperedge * coresperedge) % totalcores),
                  lambda x: x.assign('A', x.vars['A'] + x.vars['B']), f"add_{blockedge}x{blockedge}")

    #get result
    cores.halt()
    for i in range(coresperedge * coresperedge):
        C[(i//coresperedge) * (edge//coresperedge):(i//coresperedge+1) * (edge//coresperedge), (i%coresperedge) * (edge//coresperedge):(i%coresperedge+1) * (edge//coresperedge)]\
            = cores[i].vars['A']

    # print('------------    Ground Truth    ------------')
    # print(numpy.matmul(A, B))
    #
    # print('------------    OURS    ------------')
    # print(C)

    print('------------    DELTA    ------------')
    print(numpy.sum(numpy.power(numpy.matmul(A, B) - C, 2)))

    print('------------    OPS    ------------')
    cores.printresult()