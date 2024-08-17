#


#
import numpy
import pandas


#
from merrill_feature.feature_treatment.treat import standard_treat, lag, wise_drop, add_by_substring


#

def gentry_load_2nd(bench, target0, target1, n_lags=4):
    #
    """
    Data
    """
    g = './data/series_2nd.xlsx'
    series_data = pandas.read_excel(g)
    series_data = series_data.set_index("date").sort_index()
    series_data.index = series_data.index + pandas.offsets.MonthBegin() - pandas.offsets.MonthBegin()

    d = './data/_gentry.xlsx'

    stats_data = pandas.read_excel(d)
    stats_data = stats_data.set_index("date").sort_index()

    data = pandas.concat((series_data, stats_data), axis=1)
    min_ix_obs = data.index.min()
    min_ix_nna = data.dropna().index.min()
    data = data.query("index >= '{0}'".format(min_ix_nna))

    print('Min available date in the data: {0}'.format(min_ix_obs))
    print('Date cutoff: {0}'.format(min_ix_nna))

    """
    Feature Treatment
    """

    data = data.dropna()

    # gentry

    data['T5YIEM_2more'] = (data['T5YIEM'] > 0.02).astype(dtype=int)
    data['MPMI_th'] = (data['MPMI'] > 50).astype(dtype=int)
    data['NPMI_th'] = (data['MPMI'] > 50).astype(dtype=int)
    data['VIX_th'] = (data['VIX'] > 30).astype(dtype=int)
    data['VIXM_th'] = (data['VIXM'] > 30).astype(dtype=int)
    data['KCFSI_th0'] = (data['KCFSI'] > 0).astype(dtype=int)
    data['KCFSI_th1'] = (data['KCFSI'] > 1).astype(dtype=int)
    data['CBCONSCONF_th'] = (data['CBCONSCONF'] > 100).astype(dtype=int)

    no = ['T5YIEM_2more', 'MPMI_th', 'NPMI_th', 'VIX_th', 'VIXM_th', 'KCFSI_th0', 'KCFSI_th1',
          'CBCONSCONF_th']
    diff = []
    b100diff = ['UNRATE']
    b100 = ['CBCONSCONF']

    no__ = []
    diff__ = []
    b100diff__ = []
    b100__ = []

    cols = data.columns.values
    no = add_by_substring(cols, no__, no)
    diff = add_by_substring(cols, diff__, diff)
    b100diff = add_by_substring(cols, b100diff__, b100diff)
    b100 = add_by_substring(cols, b100__, b100)
    pct = [x for x in data.columns.values if x not in no + diff + b100diff + b100]

    data_pct = standard_treat(data=data, no=no, diff=diff, b100=b100, b100diff=b100diff, pct=pct)

    # 4 and more subsequent months
    t5yiem_maxi = lag(data=data[['T5YIEM']], n_lags=3)
    data['T5YIEM_4chup'] = (t5yiem_maxi.min(axis=1) > 0).astype(dtype=int)
    data['T5YIEM_4chdw'] = (t5yiem_maxi.max(axis=1) < 0).astype(dtype=int)
    t5yiem_maxi = lag(data=data[['UNRATE']], n_lags=3)
    data['UNRATE_4chup'] = (t5yiem_maxi.min(axis=1) > 0).astype(dtype=int)
    data['UNRATE_4chdw'] = (t5yiem_maxi.max(axis=1) < 0).astype(dtype=int)

    data_pct_lagged = lag(data=data_pct, n_lags=n_lags)
    data_pct_lagged = data_pct_lagged.dropna()

    na_1l = ['WTISPLC', 'CPIAUCSL', 'UNRATE', 'MPMI', 'NPMI', 'PAYEMS',
             'CCSA', 'IRLTLT01DEM156N', 'KCFSI']
    na_2l = []
    dead_space = []

    excluded = [] + \
        [x + '_LAG1' for x in na_1l] + \
        [x + '_LAG2' for x in na_1l + na_2l] + \
        dead_space

    # gentry
    wise = []

    x_factors = [x for x in data_pct_lagged.columns.values if
                 ('LAG' in x) and (not any([y in x for y in excluded + wise]))]

    X = data_pct_lagged.loc[:, x_factors].values
    Y = data_pct_lagged.loc[:, [target0, target1]].values
    X_ = data_pct_lagged.loc[:, [target0, target1]].values
    Y_ = data_pct_lagged.loc[:, [target0, target1]].values

    tt = data_pct_lagged.index.values

    bench_series = data_pct_lagged[bench].values

    return X, Y, X_, Y_, tt, bench_series, data_pct_lagged.columns.values, x_factors


def gentry_load_1st(bench, target0, target1, n_lags=4):
    #
    """
    Data
    """
    g = './data/series_1st.xlsx'
    series_data = pandas.read_excel(g)
    series_data = series_data.set_index("date").sort_index()
    series_data.index = series_data.index + pandas.offsets.MonthBegin() - pandas.offsets.MonthBegin()

    d = './data/gentry.xlsx'

    stats_data = pandas.read_excel(d)
    stats_data = stats_data.set_index("date").sort_index()

    data = pandas.concat((series_data, stats_data), axis=1)
    min_ix_obs = data.index.min()
    min_ix_nna = data.dropna().index.min()
    data = data.query("index >= '{0}'".format(min_ix_nna))

    print('Min available date in the data: {0}'.format(min_ix_obs))
    print('Date cutoff: {0}'.format(min_ix_nna))

    """
    Feature Treatment
    """

    data = data.dropna()

    # gentry

    data['T5YIEM_2more'] = (data['T5YIEM'] > 0.02).astype(dtype=int)
    data['MPMI_th'] = (data['MPMI'] > 50).astype(dtype=int)
    data['NPMI_th'] = (data['MPMI'] > 50).astype(dtype=int)
    data['VIX_th'] = (data['VIX'] > 30).astype(dtype=int)
    data['VIXM_th'] = (data['VIXM'] > 30).astype(dtype=int)
    data['KCFSI_th0'] = (data['KCFSI'] > 0).astype(dtype=int)
    data['KCFSI_th1'] = (data['KCFSI'] > 1).astype(dtype=int)
    data['CBCONSCONF_th'] = (data['CBCONSCONF'] > 100).astype(dtype=int)

    no = ['T5YIEM_2more', 'MPMI_th', 'NPMI_th', 'VIX_th', 'VIXM_th', 'KCFSI_th0', 'KCFSI_th1',
          'CBCONSCONF_th']
    diff = []
    b100diff = ['UNRATE']
    b100 = ['CBCONSCONF']

    no__ = []
    diff__ = []
    b100diff__ = []
    b100__ = []

    cols = data.columns.values
    no = add_by_substring(cols, no__, no)
    diff = add_by_substring(cols, diff__, diff)
    b100diff = add_by_substring(cols, b100diff__, b100diff)
    b100 = add_by_substring(cols, b100__, b100)
    pct = [x for x in data.columns.values if x not in no + diff + b100diff + b100]

    data_pct = standard_treat(data=data, no=no, diff=diff, b100=b100, b100diff=b100diff, pct=pct)

    # 4 and more subsequent months
    t5yiem_maxi = lag(data=data[['T5YIEM']], n_lags=3)
    data['T5YIEM_4chup'] = (t5yiem_maxi.min(axis=1) > 0).astype(dtype=int)
    data['T5YIEM_4chdw'] = (t5yiem_maxi.max(axis=1) < 0).astype(dtype=int)
    t5yiem_maxi = lag(data=data[['UNRATE']], n_lags=3)
    data['UNRATE_4chup'] = (t5yiem_maxi.min(axis=1) > 0).astype(dtype=int)
    data['UNRATE_4chdw'] = (t5yiem_maxi.max(axis=1) < 0).astype(dtype=int)

    data_pct_lagged = lag(data=data_pct, n_lags=n_lags)
    data_pct_lagged = data_pct_lagged.dropna()

    na_1l = ['WTISPLC', 'CPIAUCSL', 'UNRATE', 'MPMI', 'NPMI', 'PAYEMS',
             'CCSA', 'IRLTLT01DEM156N', 'KCFSI']
    na_2l = []
    dead_space = []

    excluded = [] + \
        [x + '_LAG1' for x in na_1l] + \
        [x + '_LAG2' for x in na_1l + na_2l] + \
        dead_space

    # gentry
    wise = []

    x_factors = [x for x in data_pct_lagged.columns.values if
                 ('LAG' in x) and (not any([y in x for y in excluded + wise]))]

    X = data_pct_lagged.loc[:, x_factors].values
    Y = data_pct_lagged.loc[:, [target0, target1]].values
    X_ = data_pct_lagged.loc[:, [target0, target1]].values
    Y_ = data_pct_lagged.loc[:, [target0, target1]].values

    tt = data_pct_lagged.index.values

    bench_series = data_pct_lagged[bench].values

    return X, Y, X_, Y_, tt, bench_series, data_pct_lagged.columns.values, x_factors


def gentry_load_15th(bench, target0, target1, n_lags=4):
    #
    """
    Data
    """
    g = './data/series_15th.xlsx'
    series_data = pandas.read_excel(g)
    series_data = series_data.set_index("date").sort_index()
    series_data.index = series_data.index + pandas.offsets.MonthBegin() - pandas.offsets.MonthBegin()

    d = './data/gentry.xlsx'

    stats_data = pandas.read_excel(d)
    stats_data = stats_data.set_index("date").sort_index()

    data = pandas.concat((series_data, stats_data), axis=1)
    min_ix_obs = data.index.min()
    min_ix_nna = data.dropna().index.min()
    data = data.query("index >= '{0}'".format(min_ix_nna))

    print('Min available date in the data: {0}'.format(min_ix_obs))
    print('Date cutoff: {0}'.format(min_ix_nna))

    """
    Feature Treatment
    """

    data = data.dropna()

    # gentry

    data['T5YIEM_2more'] = (data['T5YIEM'] > 0.02).astype(dtype=int)
    data['MPMI_th'] = (data['MPMI'] > 50).astype(dtype=int)
    data['NPMI_th'] = (data['MPMI'] > 50).astype(dtype=int)
    data['VIX_th'] = (data['VIX'] > 30).astype(dtype=int)
    data['VIXM_th'] = (data['VIXM'] > 30).astype(dtype=int)
    data['KCFSI_th0'] = (data['KCFSI'] > 0).astype(dtype=int)
    data['KCFSI_th1'] = (data['KCFSI'] > 1).astype(dtype=int)
    data['CBCONSCONF_th'] = (data['CBCONSCONF'] > 100).astype(dtype=int)

    no = ['T5YIEM_2more', 'MPMI_th', 'NPMI_th', 'VIX_th', 'VIXM_th', 'KCFSI_th0', 'KCFSI_th1',
          'CBCONSCONF_th']
    diff = []
    b100diff = ['UNRATE']
    b100 = ['CBCONSCONF']

    no__ = []
    diff__ = []
    b100diff__ = []
    b100__ = []

    cols = data.columns.values
    no = add_by_substring(cols, no__, no)
    diff = add_by_substring(cols, diff__, diff)
    b100diff = add_by_substring(cols, b100diff__, b100diff)
    b100 = add_by_substring(cols, b100__, b100)
    pct = [x for x in data.columns.values if x not in no + diff + b100diff + b100]

    data_pct = standard_treat(data=data, no=no, diff=diff, b100=b100, b100diff=b100diff, pct=pct)

    # 4 and more subsequent months
    t5yiem_maxi = lag(data=data[['T5YIEM']], n_lags=3)
    data['T5YIEM_4chup'] = (t5yiem_maxi.min(axis=1) > 0).astype(dtype=int)
    data['T5YIEM_4chdw'] = (t5yiem_maxi.max(axis=1) < 0).astype(dtype=int)
    t5yiem_maxi = lag(data=data[['UNRATE']], n_lags=3)
    data['UNRATE_4chup'] = (t5yiem_maxi.min(axis=1) > 0).astype(dtype=int)
    data['UNRATE_4chdw'] = (t5yiem_maxi.max(axis=1) < 0).astype(dtype=int)

    data_pct_lagged = lag(data=data_pct, n_lags=n_lags)
    data_pct_lagged = data_pct_lagged.dropna()

    na_1l = []
    na_2l = []
    dead_space = []

    excluded = [] + \
        [x + '_LAG1' for x in na_1l] + \
        [x + '_LAG2' for x in na_1l + na_2l] + \
        dead_space

    # gentry
    wise = []

    x_factors = [x for x in data_pct_lagged.columns.values if
                 ('LAG' in x) and (not any([y in x for y in excluded + wise]))]

    X = data_pct_lagged.loc[:, x_factors].values
    Y = data_pct_lagged.loc[:, [target0, target1]].values
    X_ = data_pct_lagged.loc[:, [target0, target1]].values
    Y_ = data_pct_lagged.loc[:, [target0, target1]].values

    tt = data_pct_lagged.index.values

    bench_series = data_pct_lagged[bench].values

    return X, Y, X_, Y_, tt, bench_series, data_pct_lagged.columns.values, x_factors


def standard_load_1st(bench, target0, target1, n_lags=4):
    #
    """
    Data
    """
    g = './data/series_1st.xlsx'
    series_data = pandas.read_excel(g)
    series_data = series_data.set_index("date").sort_index()
    series_data.index = series_data.index + pandas.offsets.MonthBegin() - pandas.offsets.MonthBegin()

    d = './data/statz.xlsx'

    stats_data = pandas.read_excel(d)
    stats_data = stats_data.set_index('date').sort_index()

    data = pandas.concat((series_data, stats_data), axis=1)
    min_ix_obs = data.index.min()
    min_ix_nna = data.dropna().index.min()
    data = data.query("index >= '{0}'".format(min_ix_nna))

    print('Min available date in the data: {0}'.format(min_ix_obs))
    print('Date cutoff: {0}'.format(min_ix_nna))

    """
    Feature Treatment
    """

    data = data.dropna()

    # statz

    no = []
    diff = []
    b100diff = ['JTU5300QUL', 'EMVMACRORE', 'EMRATIO', 'FEDFUNDS', 'AAA', 'CPF1M', 'EMVFINCRISES', 'BAA',
                'KCFSI', 'MICH', 'UNRATE', 'RECPROUSM156N', 'PSAVERT', 'GS20',
                'GS10', 'GS3', 'GS2', 'GS1', 'GS3M'] # , 'GS1M']
    b100 = ['GS1M']

    no__ = []
    diff__ = []
    b100diff__ = ['INTDSR']
    b100__ = ['SPASTT']

    cols = data.columns.values
    no = add_by_substring(cols, no__, no)
    diff = add_by_substring(cols, diff__, diff)
    b100diff = add_by_substring(cols, b100diff__, b100diff)
    b100 = add_by_substring(cols, b100__, b100)
    pct = [x for x in data.columns.values if x not in no + diff + b100diff + b100]

    data_pct = standard_treat(data=data, no=no, diff=diff, b100=b100, b100diff=b100diff, pct=pct)
    data_pct_lagged = lag(data=data_pct, n_lags=n_lags)
    data_pct_lagged = data_pct_lagged.dropna()

    na_1l = ['A229RX0', 'AWHMAN', 'BOPGSTB', 'CE16OV', 'CES3000000008',
             'CPIAUCSL', 'EMRATIO', 'KCFSI', 'MANEMP', 'MICH', 'PAYEMS',
             'PCEC96', 'PCUOMFGOMFG', 'PMSAVE', 'PSAVERT', 'RECPROUSM156N',
             'SPASTT01AUM657N', 'SPASTT01BRM657N', 'SPASTT01CNM657N',
             'SPASTT01DEM657N', 'SPASTT01EZM657N', 'SPASTT01GBM657N',
             'SPASTT01INM657N', 'SPASTT01KRM657N', 'SPASTT01MXM657N',
             'SPASTT01RUM657N', 'SPASTT01TRM657N', 'SPASTT01USM657N',
             'SPASTT01ZAM657N', 'STDSL', 'UNRATE', 'USEPUINDXM']
    na_2l = ['EMVFINCRISES', 'EMVMACRORE', 'JTU5300QUL']
    dead_space = ['GEPUPPP', 'INTDSRBRM193N', 'INTDSRCNM193N', 'INTDSRINM193N',
                  'INTDSRJPM193N', 'INTDSRTRM193N', 'INTDSRUSM193N']

    excluded = ['FXT', 'EFA', 'EEM'] + \
        [x + '_LAG1' for x in na_1l] + \
        [x + '_LAG2' for x in na_1l + na_2l] + \
        dead_space

    # statz
    wise = []

    x_factors = [x for x in data_pct_lagged.columns.values if
                 ('LAG' in x) and (not any([y in x for y in excluded + wise]))]

    X = data_pct_lagged.loc[:, x_factors].values
    Y = data_pct_lagged.loc[:, [target0, target1]].values
    X_ = data_pct_lagged.loc[:, [target0, target1]].values
    Y_ = data_pct_lagged.loc[:, [target0, target1]].values

    tt = data_pct_lagged.index.values

    bench_series = data_pct_lagged[bench].values

    return X, Y, X_, Y_, tt, bench_series, data_pct_lagged.columns.values, x_factors


def standard_load_15th(bench, target0, target1, n_lags=4):
    #
    """
    Data
    """
    g = './data/series_15th.xlsx'
    series_data = pandas.read_excel(g)
    series_data = series_data.set_index("date").sort_index()
    series_data.index = series_data.index - pandas.offsets.MonthBegin()

    d = './data/statz.xlsx'

    stats_data = pandas.read_excel(d)
    stats_data = stats_data.set_index('date').sort_index()

    data = pandas.concat((series_data, stats_data), axis=1)
    min_ix_obs = data.index.min()
    min_ix_nna = data.dropna().index.min()
    data = data.query("index >= '{0}'".format(min_ix_nna))

    print('Min available date in the data: {0}'.format(min_ix_obs))
    print('Date cutoff: {0}'.format(min_ix_nna))

    """
    Feature Treatment
    """

    data = data.dropna()

    # statz

    no = []
    diff = []
    b100diff = ['JTU5300QUL', 'EMVMACRORE', 'EMRATIO', 'FEDFUNDS', 'AAA', 'CPF1M', 'EMVFINCRISES', 'BAA',
                'KCFSI', 'MICH', 'UNRATE', 'RECPROUSM156N', 'PSAVERT', 'GS20',
                'GS10', 'GS3', 'GS2', 'GS1', 'GS3M'] # , 'GS1M']
    b100 = ['GS1M']

    no__ = []
    diff__ = []
    b100diff__ = ['INTDSR']
    b100__ = ['SPASTT']

    cols = data.columns.values
    no = add_by_substring(cols, no__, no)
    diff = add_by_substring(cols, diff__, diff)
    b100diff = add_by_substring(cols, b100diff__, b100diff)
    b100 = add_by_substring(cols, b100__, b100)
    pct = [x for x in data.columns.values if x not in no + diff + b100diff + b100]

    data_pct = standard_treat(data=data, no=no, diff=diff, b100=b100, b100diff=b100diff, pct=pct)
    data_pct_lagged = lag(data=data_pct, n_lags=n_lags)
    data_pct_lagged = data_pct_lagged.dropna()

    na_1l = ['A229RX0',
             'EMVFINCRISES', 'EMVMACRORE', 'JTU5300QUL',
             'MICH',
             'PCEC96', 'PMSAVE', 'PSAVERT',
             'STDSL']
    na_2l = ['JTU5300QUL']
    dead_space = ['GEPUPPP', 'INTDSRBRM193N', 'INTDSRCNM193N', 'INTDSRINM193N',
                  'INTDSRJPM193N', 'INTDSRTRM193N', 'INTDSRUSM193N']

    excluded = ['FXT', 'EFA', 'EEM'] + \
        [x + '_LAG1' for x in na_1l] + \
        [x + '_LAG2' for x in na_1l + na_2l] + \
        dead_space

    # statz
    wise = []

    x_factors = [x for x in data_pct_lagged.columns.values if
                 'LAG' in x and ~any([y in x for y in excluded + wise])]

    X = data_pct_lagged.loc[:, x_factors].values
    Y = data_pct_lagged.loc[:, [target0, target1]].values
    X_ = data_pct_lagged.loc[:, [target0, target1]].values
    Y_ = data_pct_lagged.loc[:, [target0, target1]].values

    tt = data_pct_lagged.index.values

    bench_series = data_pct_lagged[bench].values

    return X, Y, X_, Y_, tt, bench_series, data_pct_lagged.columns.values, x_factors


def _eld_standard_load(stats, bench, target0, target1, n_lags=4):
    #
    """
    Data
    """
    g = './data/series.xlsx'
    series_data = pandas.read_excel(g)
    series_data = series_data.set_index("date").sort_index()
    series_data.index = series_data.index + pandas.offsets.MonthBegin()

    if stats == 'stats':
        d = './data/stats.xlsx'
    elif stats == 'statz':
        d = './data/statz.xlsx'
    else:
        raise ValueError("Data Loading Stage Error: 'stats' parameter valid values are either 'stats' or 'statz'")
    stats_data = pandas.read_excel(d)
    stats_data = stats_data.set_index('date').sort_index()

    data = pandas.concat((series_data, stats_data), axis=1)
    min_ix_obs = data.index.min()
    min_ix_nna = data.dropna().index.min()
    data = data.query("index >= '{0}'".format(min_ix_nna))

    print('Min available date in the data: {0}'.format(min_ix_obs))
    print('Date cutoff: {0}'.format(min_ix_nna))

    """
    Feature Treatment
    """

    data = data.dropna()

    if stats == 'stats':
        # stats
        no, diff, b100diff, b100, no__, diff__, b100diff__, b100__ = [], [], [], ['GS1M'], [], [], [], []

        cols = data.columns.values
        no = add_by_substring(cols, no__, no)
        diff = add_by_substring(cols, diff__, diff)
        b100diff = add_by_substring(cols, b100diff__, b100diff)
        b100 = add_by_substring(cols, b100__, b100)
        pct = [x for x in data.columns.values if x not in no + diff + b100diff + b100]

        data_pct = standard_treat(data=data, no=no, diff=diff, b100=b100, b100diff=b100diff, pct=pct)
        data_pct_lagged = lag(data=data_pct, n_lags=n_lags)
        data_pct_lagged = data_pct_lagged.dropna()

        excluded = ['FXT', 'EFA', 'EEM', 'GS1M']

        # stats
        wise = []

        x_factors = [x for x in data_pct_lagged.columns.values if 'LAG' in x and ~any(
            [y in x for y in excluded + wise])]

        X = data_pct_lagged.loc[:, x_factors].values
        Y = data_pct_lagged.loc[:, [target0, target1]].values
        X_ = data_pct_lagged.loc[:, [target0, target1]].values
        Y_ = data_pct_lagged.loc[:, [target0, target1]].values

        tt = data_pct_lagged.index.values

    elif stats == 'statz':
        # statz

        no = []
        diff = []
        b100diff = ['JTU5300QUL', 'EMVMACRORE', 'EMRATIO', 'FEDFUNDS', 'AAA', 'CPF1M', 'EMVFINCRISES', 'BAA',
                    'KCFSI', 'MICH', 'UNRATE', 'RECPROUSM156N', 'PSAVERT', 'GS20',
                    'GS10', 'GS3', 'GS2', 'GS1', 'GS3M'] # , 'GS1M']
        b100 = ['GS1M']

        no__ = []
        diff__ = []
        b100diff__ = ['INTDSR']
        b100__ = ['SPASTT']

        cols = data.columns.values
        no = add_by_substring(cols, no__, no)
        diff = add_by_substring(cols, diff__, diff)
        b100diff = add_by_substring(cols, b100diff__, b100diff)
        b100 = add_by_substring(cols, b100__, b100)
        pct = [x for x in data.columns.values if x not in no + diff + b100diff + b100]

        data_pct = standard_treat(data=data, no=no, diff=diff, b100=b100, b100diff=b100diff, pct=pct)
        data_pct_lagged = lag(data=data_pct, n_lags=n_lags)
        data_pct_lagged = data_pct_lagged.dropna()

        excluded = ['FXT', 'EFA', 'EEM']

        # statz
        wise = []

        x_factors = [x for x in data_pct_lagged.columns.values if
                     'LAG' in x and ~any([y in x for y in excluded + wise])]

        X = data_pct_lagged.loc[:, x_factors].values
        Y = data_pct_lagged.loc[:, [target0, target1]].values
        X_ = data_pct_lagged.loc[:, [target0, target1]].values
        Y_ = data_pct_lagged.loc[:, [target0, target1]].values

        tt = data_pct_lagged.index.values

    else:
        raise ValueError("Data Loading Stage Error: 'stats' parameter valid values are either 'stats' or 'statz'")

    bench_series = data_pct_lagged[bench].values

    return X, Y, X_, Y_, tt, bench_series, data_pct_lagged.columns.values, x_factors
