from functools import partial

import pandas as pd
import scipy
from statsmodels import robust
from tqdm import tqdm
import numpy as np

tqdm.pandas()

statistic_functions = dict(
    [
        ('MIN', np.min),
        ('MAX', np.max),
        ('MEAN', np.mean),
        ('STD', np.std),
        ('VAR', np.var),
        ('MAD', partial(robust.mad, c=1)),
        ('SKEW', partial(scipy.stats.skew, bias=False)),
        ('KURT', partial(scipy.stats.kurtosis, bias=False))
    ] + [
        ('%dPERC' % perc, partial(np.percentile, q=perc)) for perc in range(10, 100, 10)
    ]
)

tcp_flags = dict(
    [(flg, (enc0, enc1)) for flg, enc0, enc1 in zip(
        ['FIN', 'SYN', 'RST', 'PSH', 'ACK', 'URG', 'ECE', 'CWR'],
        [1, 2, 4, 8, 16, 32, 64, 128],
        ['F', 'S', 'R', 'P', 'A', 'U', 'E', 'C']
    )]
)


def transform_flg_int(flg):
    """
    Given a flag number in int (viz. base 10 of the TCP flag bytes), this function return the int for each hot flag
    :param flg: base 10 of the TCP flag bytes, e.g. 132
    :return: list of int for each hot flag, e.g. [4, 128]
    """
    for i, f in enumerate(list(bin(flg)[2:][::-1])):
        if int(f):
            yield 2 ** i


def transform_flg(flg):
    """
    Given a flag representation return the set of flags fall in this representation based on the type
    :param flg: string sequence of flags (e.g., 'SA' == SYN+ACK) or the base 10 version (e.g., 18 == SYN+ACK)
    :return: the list of each flag in flg, e.g. from 'SA' is returned ['S', 'A'], from 18 is returned [2, 16]
    """
    if isinstance(flg, str):
        return list(flg)
    return transform_flg_int(flg)


def get_statistics(data, features=None, label_col='LABEL', num_packets=20):
    if features is None:
        features = [feature for feature in list(data.columns) if feature != label_col]

    print('\nComputing number of packets...')
    data.loc[:, 'NUM_PKTS'] = data['PL'].progress_apply(
        lambda x: min(sum([p != 0 for p in x]), num_packets)).values

    print('\nComputing the total number of bytes...')
    data_stats_cols = {
        'NUM_PKTS': data['NUM_PKTS'].values,
        'TOTAL_PL': data['PL'].progress_apply(
            lambda x: sum(x[:num_packets])).values
    }

    print('\nComputing per-feature statistics...')
    for feature in tqdm(features):
        print('\nElaborating feature %s...' % feature)
        for statistic, statistic_function in tqdm(statistic_functions.items()):
            print('\nComputing %s stats...' % statistic)
            start = 0
            if feature == 'IAT':
                start = 1
            data_stats_cols['%s_%s' % (feature, statistic)] = data[[feature, 'NUM_PKTS']].progress_apply(
                lambda x: statistic_function(
                    x[feature][start:min(num_packets, x['NUM_PKTS'])]
                ) if x['NUM_PKTS'] - start else np.nan, axis=1).values

    print('\nComputing byte-rate...')
    data_stats_cols['BYTE_RATE'] = data[['IAT', 'PL']].progress_apply(
        lambda x: sum(x['PL'][:num_packets]) / sum(x['IAT'][:num_packets]) if sum(x['IAT'][:num_packets]) else 0,
        axis=1).values

    print('\nCounting TCP flags...')
    for tcp_flag, encodings in tqdm(tcp_flags.items()):
        print('\nElaborating flag %s...' % tcp_flag)
        data_stats_cols['%s_COUNT' % tcp_flag] = data['FLG'].progress_apply(
            lambda x: sum([
                sum([v0 in encodings for v0 in transform_flg(v)]) for v in x[:num_packets]])).values

    # The labels are set as last column
    data_stats_cols[label_col] = data[label_col].values

    # Eventual NaN values are pushed to zero
    data_stats_df = pd.DataFrame(data_stats_cols)
    return data_stats_df.fillna(0)
