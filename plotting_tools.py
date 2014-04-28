import matplotlib.pyplot as plt

def hash_string(keyword, n_buckets):
    n = 0
    for letter in keyword:
        n = n + ord(letter)*89
    return n % n_buckets

def hash_to_rgb(n):
    try:
        assert(0 <= n and n < 1000)
    except:
        raise Exception("hash must be between 0 and 1000")

    r = n % 10
    n -= r
    n /= 10

    g = n % 10
    n -= g
    n /= 10

    b = n % 10
    n -= b
    n /= 10

    return (r/10., g/10., b/10.)

def make_graph(matrix, labels, title="untitled",
                lw=0, ms=5, marker='o', save_file=None):
    x = range(len(matrix))

    fig = plt.figure(figsize=(24,13.5), dpi=80)
    ax = fig.add_subplot(111, title=title)
    lines = []
    lbls = []

    for i, k in enumerate(labels):
        c = hash_to_rgb(hash_string(k, 1000))
        y = [line[i] for line in matrix]

        lines.append(ax.plot(x, y, marker=marker, lw=lw, mfc=c, mec=c, ms=ms))
        lbls.append(k)

    # this is necessary because matplotlib.figure.plot returns single-value
    # 2DLine object tuple
    lines = [line[0] for line in lines]
    fig.legend(tuple(lines), tuple(lbls), "right")

    if save_file is None:
        plt.show()
    else:
        print "Saving graph to file:\n{}".format(save_file)
        plt.savefig(save_file)