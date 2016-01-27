# -*- coding: utf-8 -*-


def plot_series(xres, varied_data, indices=None, info_dicts=None,
                fail_vline=None, plot=None, plot_kwargs_cb=None,
                ls=('-', '--', ':', '-.'),
                c=('k', 'r', 'g', 'b', 'c', 'm', 'y'), labels=None,
                ax=None):
    """ Plot the values of the solution vector vs the varied parameter """
    if indices is None:
        indices = range(xres.shape[1])

    if fail_vline is None:
        if info_dicts is None:
            fail_vline = False
        else:
            fail_vline = True

    if plot is None:
        if ax is None:
            from matplotlib.pyplot import plot
        else:
            plot = ax.plot

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
        plot(varied_data, xres[:, idx], **plot_kwargs_cb(
            idx, labels=labels))

    if fail_vline:
        if ax is None:
            from matplotlib.pyplot import axvline
        else:
            axvline = ax.axvline
        for i, sol in enumerate(info_dicts):
            if not sol['success']:
                axvline(varied_data[i], c='k', ls='--')


def mpl_outside_legend(ax, **kwargs):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), **kwargs)
