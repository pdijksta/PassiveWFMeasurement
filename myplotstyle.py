import sys
import os
import re
import colorsys
import matplotlib.pyplot as plt

mute = False

def myprint(*args, **kwargs):
    if not mute:
        print(*args, **kwargs)


def set_fontsizes(fontsize):
    plt.rc('font', size=fontsize)

def figure(title='', figsize=(12, 10), **kwargs):
    fig = plt.figure(figsize=figsize, **kwargs)
    fig.canvas.manager.set_window_title(title)
    fig.patch.set_facecolor('w')
    plt.suptitle(title, fontsize=16)
    #fig.subplots_adjust(left=0.07, right=0.80, wspace=0.5)
    return fig

def sciy(sp=None):
    if sp is None:
        sp = plt.gca()
    sp.ticklabel_format(style='sci', scilimits=(0,0),axis='y')

def scix(sp=None):
    if sp is None:
        sp = plt.gca()
    sp.ticklabel_format(style='sci', scilimits=(0,0),axis='x')

def comb_legend(sp1, sp2, *args, **kwargs):
    """
    Combine legends for twinx()ed subplots
    """
    lines, labels = sp1.get_legend_handles_labels()
    lines2, labels2 = sp2.get_legend_handles_labels()
    sp2.legend(lines + lines2, labels + labels2, *args, **kwargs)

def colorprog(ctr, max_ctr_or_list, colormap='hsv'):
    if hasattr(max_ctr_or_list, '__len__'):
        max_ctr_or_list = len(max_ctr_or_list)

    if colormap == 'hsv':
        return colorsys.hsv_to_rgb(ctr/max_ctr_or_list, .9, 1.)
    else:
        return plt.get_cmap(colormap)(ctr/max_ctr_or_list)

def saveall(basepath, hspace=None, wspace=None, trim=True, figs=None, ending='.png', transparent=False, empty_suptitle=True, bottom=None, top=None, left=None, right=None, **kwargs):

    if figs is None:
        numbs = plt.get_fignums()
    else:
        numbs = [fig.number for fig in figs]

    for num in numbs:
        fig = plt.figure(num)
        fig.subplots_adjust(hspace=hspace, wspace=wspace, bottom=bottom, top=top, left=left, right=right)

        title = fig.canvas.get_window_title()
        if empty_suptitle:
            plt.suptitle('')
        path = os.path.expanduser(basepath)+'_%i%s' % (num, ending)
        fig.savefig(path, transparent=transparent, bbox_inches='tight', pad_inches=0, **kwargs)

        if 'pdf' not in ending and 'svg' not in ending and trim:
            cmd = 'convert -trim %s %s' % (path, path)
            myprint(cmd)
            os.system(cmd)
        myprint('Saved fig %i with title %s in %s' % (num, title, path))

def subplot_factory(ny, nx, grid=True):
    _sciy = sciy
    _scix = scix

    def subplot(x, grid=grid, title=None, xlabel=None, ylabel=None, sciy=False, scix=False, sharex=None, sharey=None, title_fs=None, **kwargs):
        sp = plt.subplot(ny, nx, x, sharex=sharex, sharey=sharey, **kwargs)
        if grid:
            sp.grid(True)
        if title:
            sp.set_title(title, fontsize=title_fs)
        if xlabel:
            sp.set_xlabel(xlabel)
        if ylabel:
            sp.set_ylabel(ylabel)
        if sciy:
            _sciy(sp)
        if scix:
            _scix(sp)
        return sp
    return subplot

pdijksta_dir = os.path.expanduser('~/plots/')
re_script = re.compile('^(\d{3}[a-z]?_.{4})')

def get_file_title(fig, title=None):
    if title is None:
        title = fig.canvas.get_window_title()
    script_title = os.path.basename(sys.argv[0])
    info = re_script.match(script_title)
    if info is not None:
        script_number = info.group(1)
    else:
        myprint('Warning! Script number could not be identified! %s' % script_title)
        script_number = script_title[:5]

    title = '_'.join([script_number, title, str(fig.number)])
    return title.replace(' ','_').replace('.','') + '.png'

def pdijksta(fig, title=None, figsize=None):
    if figsize is not None:
        myprint('mystyle.pdijksta: figsize is deprecated')

    if hasattr(fig, '__len__'):
        for f in fig: pdijksta(f, title=title, figsize=figsize)
    else:
        file_title = get_file_title(fig, title)
        save_path = pdijksta_dir + file_title
        #fig.set_size_inches(figsize)
        #fig.subplots_adjust(left=0.07, right=0.90, wspace=0.42)
        fig.savefig(save_path, dpi=200)
        myprint('Saved in\n%s' % save_path)

def saveall_pdijksta(figsize=None):
    for num in plt.get_fignums():
        fig = plt.figure(num)
        pdijksta(fig, figsize=figsize)

def saveall_dir(dir_, title=None, figsize=None):
    for num in plt.get_fignums():
        fig = plt.figure(num)
        if title is None:
            title2 = get_file_title(fig)
        else:
            title2 = title + '_%i.png' % num
        path = os.path.join(dir_, title2)
        fig.savefig(path)
        myprint('Figure %i saved in %s' % (num, path))

def closeall():
    nums = plt.get_fignums()
    for num in nums:
        plt.figure(num).clf()
    plt.close('all')

def show():
    plt.show()

