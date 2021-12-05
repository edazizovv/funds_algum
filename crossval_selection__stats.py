#


#
import numpy
import pandas

from sklearn.tree import DecisionTreeRegressor

import torch
from torch import nn

#
from the_skeleton.diezel import DiStill
from the_skeleton.losses import loss_202012var_party
from the_skeleton.models import WrappedNN
from the_skeleton.func import make_params, simple_search_report, extended_risk_report, hedge_management_report

from core0.feature_selection.cut_off import cut_fwd__pair
from core0.feature_selection.func import pearson, granger

from core0.macro.data_util.load_data import standard_load

#
X, Y, X_, Y_, tt, bench, data_pct_lagged_cols, x_factors = standard_load(stats='stats', bench='GS1M',
                                                                         n_lags=4, target0=0, target1=1)

start_bounds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
thresh_bounds = [45, 55, 65, 75, 85, 95, 105, 115, 125, 135]
end_bounds = [-97, -87, -77, -67, -57, -47, -37, -27, -17, -7]

reported = []

for se in range(len(start_bounds)):
    start = start_bounds[se]
    thresh = thresh_bounds[se]
    end = end_bounds[se]

    X_train = torch.tensor(X[start:thresh, :], dtype=torch.float)
    Y_train = torch.tensor(Y[start:thresh, :], dtype=torch.float)
    X_train_ = torch.tensor(X_[start:thresh, :], dtype=torch.float)
    Y_train_ = torch.tensor(Y_[start:thresh, :], dtype=torch.float)
    X_test = torch.tensor(X[thresh:end, :], dtype=torch.float)
    Y_test = torch.tensor(Y[thresh:end, :], dtype=torch.float)
    X_test_ = torch.tensor(X_[thresh:end, :], dtype=torch.float)
    Y_test_ = torch.tensor(Y_[thresh:end, :], dtype=torch.float)

    tt_train, tt_test = tt[start:thresh], tt[thresh:end]
    bench_train, bench_test = bench[start:thresh], bench[thresh:end]

    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    print(tt_train.shape)
    print(tt_test.shape)

    feature_selectors = [pearson]
    fs_thresholds = [0.5]

    for fs_thresh in fs_thresholds:
        for feature_selector in feature_selectors:

            if feature_selector is granger:
                includes_targets = [any([y in x for y in data_pct_lagged_cols[[0, 1]]]) for x in x_factors]
                excludes_targets = [not x for x in includes_targets]

                fs_mask = cut_fwd__pair(X_train.numpy()[:, excludes_targets], Y_train.numpy(),
                                        method=feature_selector, thresh=fs_thresh)
                X_train_ = torch.tensor(numpy.concatenate((X_train[:, excludes_targets][:, fs_mask],
                                                           X_train[:, includes_targets]), axis=1),
                                        dtype=torch.float)
                X_test_ = torch.tensor(numpy.concatenate((X_test[:, excludes_targets][:, fs_mask],
                                                          X_test[:, includes_targets]), axis=1),
                                       dtype=torch.float)

            else:
                fs_mask = cut_fwd__pair(X_train.numpy()[:, :], Y_train.numpy(),
                                        method=feature_selector, thresh=fs_thresh)
                X_train_ = torch.tensor(X_train[:, fs_mask], dtype=torch.float)
                X_test_ = torch.tensor(X_test[:, fs_mask], dtype=torch.float)

            model = WrappedNN

            layer_type = nn.Linear
            n_multiplier = 2
            verse = 'dec'
            depth = 3
            act = nn.ReLU
            drop = 0.3
            optima = torch.optim.Adamax
            lr = 0.002
            loss = loss_202012var_party
            ep = 2000

            nn_kwargs = make_params(layer_type=layer_type,
                                    n_multiplier=n_multiplier,
                                    verse=verse,
                                    depth=depth,
                                    act=act,
                                    drops=drop,
                                    optima=optima,
                                    lr=lr,
                                    loss=loss,
                                    eps=ep)

            for r in range(10):
                distill_model = DecisionTreeRegressor
                distill_kwargs = {'max_depth': 2}

                still = DiStill(nn_model=model, nn_kwargs=nn_kwargs,
                                distill_model=distill_model, distill_kwargs=distill_kwargs,
                                commi=0.01)

                still.still(X_train_, Y_train, X_test_, Y_test)

                summary = still.plot(X_train=X_train_, Y_train=Y_train, tt_train=tt_train, bench_train=bench_train,
                                     X_test=X_test_, Y_test=Y_test, tt_test=tt_test, bench_test=bench_test,
                                     report=simple_search_report,
                                     on='filt', do_plot=False)

                # pyplot.figure()
                # plot_tree(still.distill_model_fit, feature_names=x_factors)

                report = pandas.DataFrame(data={'r': r,
                                                'start': [start],
                                                'end': [end],
                                                'thresh': [thresh],
                                                'thr': [fs_thresh],
                                                'trans': [feature_selector],
                                                'layer': [layer_type],
                                                'multiplier': [n_multiplier],
                                                'verse': [verse],
                                                'depth': [depth],
                                                'activator': [act],
                                                'drop': [drop],
                                                'loss': [loss],
                                                'ep': [ep],
                                                'optima': [optima],
                                                'lr': [lr],
                                                'Yield on train HERO': [summary[0].values[0, 0]],
                                                'Yield on test HERO': [summary[1].values[0, 0]],
                                                'VaR 99 on train HERO': [summary[0].values[0, 3]],
                                                'VaR 99 on test HERO': [summary[1].values[0, 3]],
                                                'Yield on train C1': [summary[0].values[2, 0]],
                                                'Yield on test C1': [summary[1].values[2, 0]],
                                                'VaR 99 on train C1': [summary[0].values[2, 3]],
                                                'VaR 99 on test C1': [summary[1].values[2, 3]],
                                                'Yield on train C2': [summary[0].values[3, 0]],
                                                'Yield on test C2': [summary[1].values[3, 0]],
                                                'VaR 99 on train C2': [summary[0].values[3, 3]],
                                                'VaR 99 on test C2': [summary[1].values[3, 3]]
                                                }
                                          )
                reported.append(report)
reported = pandas.concat(reported, axis=0, ignore_index=True)
