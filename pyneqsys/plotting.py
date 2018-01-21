# -*- coding: utf-8 -*-


def plot_series(xres, varied_data, indices=None, info=None,
                fail_vline=None, plot_kwargs_cb=None,
                ls=('-', '--', ':', '-.'),
                c=('k', 'r', 'g', 'b', 'c', 'm', 'y'), labels=None,
                ax=None, names=None, latex_names=None):
    """ Plot the values of the solution vector vs the varied parameter.

    Parameters
    ----------
    xres : array
        Solution vector of shape ``(varied_data.size, x0.size)``.
    varied_data : array
        Numerical values of the varied parameter.
    indices : iterable of integers, optional
        Indices of variables to be plotted. default: all
    fail_vline : bool
        Show vertical lines where the solver failed.
    plot_kwargs_cb : callable
        Takes the index as single argument, returns a dict passed to
        the plotting function
    ls : iterable of str
        Linestyles.
    c : iterable of str
        Colors.
    labels : iterable of str
    ax : matplotlib Axes instance
    names : iterable of str
    latex_names : iterable of str

    """
    import matplotlib.pyplot as plt

    if indices is None:
        indices = range(xres.shape[1])

    if fail_vline is None:
        if info is None:
            fail_vline = False
        else:
            fail_vline = True

    if ax is None:
        ax = plt.subplot(1, 1, 1)

    if labels is None:
        labels = names if latex_names is None else ['$%s$' % ln.strip('$') for ln in latex_names]

    if plot_kwargs_cb is None:
        def plot_kwargs_cb(idx, labels=None):
            kwargs = {'ls': ls[idx % len(ls)],
                      'c': c[idx % len(c)]}
            if labels:
                kwargs['label'] = labels[idx]
            return kwargs
    else:
        plot_kwargs_cb = plot_kwargs_cb or (lambda idx: {})

    for idx in indices:
        ax.plot(varied_data, xres[:, idx], **plot_kwargs_cb(idx, labels=labels))

    if fail_vline:
        for i, nfo in enumerate(info):
            if not nfo['success']:
                ax.axvline(varied_data[i], c='k', ls='--')
    return ax


def mpl_outside_legend(ax, **kwargs):
    """ Places a legend box outside a matplotlib Axes instance. """
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), **kwargs)
