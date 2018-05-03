import os
from collections import OrderedDict
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from scipy import stats


PRESET_CMAPS = {
    'iupac': {'A': '#009e73', 'C': '#0072b2',
              'G': '#f0e442', 'T': '#d55e00',
              'U': '#d55e00'},
    'basepair': {'A': '#0072b2', 'C': '#d55e00',
                 'G': '#d55e00', 'T': '#0072b2',
                 'U': '#d55e00'},
    'purine_pyrimidine': {'A': '#0072b2', 'C': '#d55e00',
                          'G': '#0072b2', 'T': '#d55e00',
                          'U': '#d55e00'},
    'aa_hydrophobicity': {'R': '#0072b2', 'K': '#0072b2', 'D': '#0072b2',
                          'E': '#0072b2', 'N': '#0072b2', 'Q': '#0072b2',
                          'S': '#009e73', 'G': '#009e73', 'H': '#009e73',
                          'T': '#009e73', 'A': '#009e73', 'P': '#009e73',
                          'Y': '#f0e442', 'V': '#f0e442', 'M': '#f0e442',
                          'C': '#f0e442', 'L': '#f0e442', 'F': '#f0e442',
                          'I': '#f0e442', 'W': '#f0e442', 'X': '#f0e442'},
    'aa_chemistry': {'G': '#009e73', 'S': '#009e73', 'T': '#009e73',
                     'Y': '#009e73', 'C': '#009e73', 'N': '#703be7',
                     'Q': '#703be7', 'K': '#0072b2', 'R': '#0072b2',
                     'H': '#0072b2', 'D': '#d55e00', 'E': '#d55e00',
                     'P': '#f0e442', 'A': '#f0e442', 'W': '#f0e442',
                     'F': '#f0e442', 'L': '#f0e442', 'I': '#f0e442',
                     'M': '#f0e442', 'V': '#f0e442'}
}


ALPHABETS = {
    'dna': OrderedDict([
        ('A', np.array([1, 0, 0, 0])),
        ('C', np.array([0, 1, 0, 0])),
        ('G', np.array([0, 0, 1, 0])),
        ('T', np.array([0, 0, 0, 1])),
        ('R', np.array([0.5, 0., 0.5, 0.])),
        ('Y', np.array([0., 0.5, 0., 0.5])),
        ('S', np.array([0., 0.5, 0.5, 0.])),
        ('W', np.array([0.5, 0., 0., 0.25])),
        ('K', np.array([0., 0., 0.5, 0.5])),
        ('M', np.array([0.5, 0.5, 0., 0.])),
        ('B', np.array([0., 1./3, 1./3, 1./3])),
        ('D', np.array([1./3, 0., 1./3, 1./3])),
        ('H', np.array([1./3, 1./3, 0., 1./3])),
        ('V', np.array([1./3, 1./3, 1./3, 0.])),
        ('N', np.array([0.25, 0.25, 0.25, 0.25])),
    ]),
    'rna': OrderedDict([
        ('A', np.array([1, 0, 0, 0])),
        ('C', np.array([0, 1, 0, 0])),
        ('G', np.array([0, 0, 1, 0])),
        ('U', np.array([0, 0, 0, 1])),
        ('R', np.array([0.5, 0., 0.5, 0.])),
        ('Y', np.array([0., 0.5, 0., 0.5])),
        ('S', np.array([0., 0.5, 0.5, 0.])),
        ('W', np.array([0.5, 0., 0., 0.25])),
        ('K', np.array([0., 0., 0.5, 0.5])),
        ('M', np.array([0.5, 0.5, 0., 0.])),
        ('B', np.array([0., 1./3, 1./3, 1./3])),
        ('D', np.array([1./3, 0., 1./3, 1./3])),
        ('H', np.array([1./3, 1./3, 0., 1./3])),
        ('V', np.array([1./3, 1./3, 1./3, 0.])),
        ('N', np.array([0.25, 0.25, 0.25, 0.25])),
    ]),
    'protein': OrderedDict([
        ('A', np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ('C', np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ('D', np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ('E', np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ('F', np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ('G', np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ('H', np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ('I', np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ('K', np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ('L', np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ('M', np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
        ('N', np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])),
        ('P', np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])),
        ('Q', np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])),
        ('R', np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])),
        ('S', np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])),
        ('T', np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])),
        ('V', np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])),
        ('W', np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])),
        ('Y', np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])),
        ('X', np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                       0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])),
    ])
}


def calculate_normalised_counts(seqs, alphabet):
    seqlen = len(seqs[0])
    # only use non-ambiguous alphabet letters
    alphalen = len(_get_unambiguous(alphabet))
    counts = np.zeros((seqlen, alphalen), dtype='f')
    for seq in seqs:
        if len(seq) != seqlen:
            raise ValueError('Not all sequences are of the same length')
        for i, base in enumerate(seq):
            counts[i] += alphabet[base]
    return counts / counts.sum(1)[:, np.newaxis]


def calculate_entropy(counts):
    seqlen, alphalen = counts.shape
    return np.asarray([np.log(alphalen) - stats.entropy(x)
                       for x in counts])


def calculate_bits(counts, entropy):
    total_heights = entropy * (1 / np.log(2))
    return total_heights[:, np.newaxis] * counts


def _make_alphabet(characters):
    alphabet = {}
    for char, row in zip(characters, np.eye(len(characters))):
        alphabet[char] = row
    return alphabet


def _get_unambiguous(alphabet):
    return [base for base, arr in alphabet.items()
            if np.array_equal(arr, arr.astype(bool))]


def _get_base_imgs(alphabet, path=None, cmap=None):
    if path is None:
        path = os.path.join(os.path.split(__file__)[0], 'data')
    if cmap is None:
        if set(alphabet).issubset(set(PRESET_CMAPS['iupac'])):
            cmap = PRESET_CMAPS['iupac']
        elif set(alphabet).issubset(set(PRESET_CMAPS['aa_chemisty'])):
            cmap = PRESET_CMAPS['aa_chemistry']
        else:
            pal = sns.color_palette(n_colors=len(alphabet))
            cmap = {a: colors.to_hex(c) for a, c in zip(alphabet, pal)}
    elif isinstance(cmap, str):
        try:
            cmap = PRESET_CMAPS[cmap]
        except KeyError:
            raise KeyError('{} is not in preset cmaps. Try {}'.format(
                cmap, ','.join(PRESET_CMAPS)))
    base_imgs = {}
    for base in alphabet:
        fn = os.path.join(path, '{}.pk'.format(base))
        img = np.load(fn)
        try:
            img += colors.to_rgba(cmap[base], alpha=0)
        except KeyError:
            raise KeyError('cmap does not contain hex value for {}'.format(base))
        base_imgs[base] = img
    return base_imgs


def draw_logo(seqs,
              alphabet='dna',
              y_format='bits',
              fig_height=2,
              fig_width_per_base=1,
              cmap=None,
              ax=None,
              pad=0.05,
              base_imgs_path=None):
    '''
    Draw a simple sequence logo using matplotlib

    Parameters
    ----------
    seqs: list of str, required
        List of sequences to plot logo for.

    alphabet: str, optional, default: dna
        One of "dna", "rna" or "protein", or a string containing the (alphabetic only)
        members of the alphabet.

    y_format: str, optional, default: bits
        The metric to plot on the y axis of the logo. Can be bits or probability.

    fig_height: int or float, optional, default: 2
        Height of the logo in inches, passed to plt.subplots

    fig_width_per_base: int or float, optional, default: 1
        Width of the logo per base of input sequence, passed to plt.subplots

    cmap: None, str or dict, optional, default: None
        Defines colour of each base or amino acid. If cmap is None, the default cmap
        will be used (iupac for dna or rna, aa_chemistry for protein). If cmap is a
        string the corresponding preset cmap will be used (use cmap_avail to list these).
        If cmap is a dict, it must contain a hex value for each letter in the alphabet used.

    ax: None or pyplot.Axes instance, optional, default: None
        A matplotlib axis to draw the logo onto. If ax is None, a new figure will be created.

    pad: float, optional, default: 0.05
        The size of the gap between letters in the logo.

    base_imgs_path: str, optional, default: None
        A directory containing alternative alphabet images to use in plot. Images must be pickled
        greyscale rgba images and named e.g. A.pk C.pk G.pk T.pk for DNA logos

    Returns
    -------
    
    A matplotlib.pyplot.Axes instance
    '''
    try:
        alphabet = ALPHABETS[alphabet.lower()]
    except KeyError:
        warnings.warn(
            'Alphabet is not one of {}, assuming it is a string of alphabet members'.format(
                ','.join(ALPHABETS)), RuntimeWarning)
        alphabet = make_alphabet(alphabet)
    counts = calculate_normalised_counts(seqs, alphabet)
    entropy = calculate_entropy(counts)
    if y_format == 'bits':
        heights = calculate_bits(counts, entropy)
        ylim = heights.sum(1).max()
    elif y_format == 'probability':
        heights = counts
        ylim = 1
    else:
        raise ValueError(
            'y_format "{}" is not recognised, use "bits" or "probability"'.format(y_format))
    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_width_per_base * len(entropy), fig_height))
    imgs = _get_base_imgs(_get_unambiguous(alphabet), base_imgs_path, cmap)
    for i, position_heights in enumerate(heights):
        order = np.argsort(position_heights)
        base_order = [list(alphabet)[i] for i in order]
        position_heights_sorted = position_heights[order]
        y = 0
        for base_idx, h in zip(base_order, position_heights_sorted):
            if h:
                ax.imshow(imgs[base_idx],
                          origin='upper',
                          extent=[i + pad / 2, i + 1 - pad / 2, y, y + h],
                          interpolation='bilinear')
            y += h
    ax.set_xlim(0, i + 1)
    ax.set_ylim(0, ylim)
    ax.set_ylabel(y_format.capitalize())
    ax.set_xticks([])
    ax.set_aspect('auto')
    return ax
