from .detector import *
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model


def rapp(**kwargs):
    # RaPP: https://openreview.net/pdf?id=HkgeGeBYDB
    class RaPP(Detector):
        def __init__(self, input_size, red_fact):
            i = Input(input_size)
            e = i
            # Auxiliary outputs will contain the Encoder layer outputs
            aux_outputs = []
            for rf in red_fact:
                e = Dense(np.ceil(input_size * rf), activation='relu')(e)
                aux_outputs.append(e)
            d = e
            for rf in red_fact[::-1][1:]:
                d = Dense(np.ceil(input_size * rf), activation='relu')(d)
            d = Dense(input_size, activation='sigmoid')(d)
            aux_outputs.append(d)

            self.model = Model(i, d)
            self.aux_model = Model(i, aux_outputs)

            self.mu = {}
            self.s = {}
            self.v = {}

            self.model.compile(optimizer=optimizer, loss=loss)

        def pathaway_diffs(self, X, mode='all'):
            # First pass through the DAE
            outs = self.aux_model.predict(X)
            # Second pass through the DAE
            aux_outs = self.aux_model.predict(outs[-1])
            if mode in ['all', 'ext']:
                diffs = [outs[-1] - X]
            else:
                diffs = []
            if mode in ['all', 'int']:
                for x, recon_x in zip(outs[:-1], aux_outs[:-1]):
                    diff = recon_x - x
                    diffs.append(diff)
            diffs = np.concatenate(diffs, axis=1)
            return diffs

        def fit(self, X, y, **kwargs):
            super().fit(X, y, epochs=epochs, **kwargs)
            # At the end of train, the mu, s, and v are computed.
            # self.on_fit_ends(X)

        def on_fit_ends(self, X, s_critical=1e-10):
            for mode in ['int', 'ext', 'all']:
                diffs = self.pathaway_diffs(X, mode)
                self.mu[mode] = np.mean(diffs, axis=0, keepdims=True)
                _, self.s[mode], vh = np.linalg.svd(diffs - self.mu[mode], full_matrices=False)
                n_critical_val = np.sum(self.s[mode] <= s_critical)
                if n_critical_val:
                    from sklearn.utils.extmath import randomized_svd
                    zm_diffs = diffs - self.mu[mode]
    #                 try:
    #                     zm_diffs = zm_diffs.get()
    #                 except: # zm_diffs is a NumPy array, not CuPy.
    #                     pass
                    _, self.s[mode], vh = randomized_svd(zm_diffs, n_components=len(self.s[mode]) - n_critical_val)
                self.v[mode] = np.array(vh.T)

        def score(self, X, **kwargs):
            
            mode = 'int' if rapp_mode.startswith('i_') else 'ext' if rapp_mode.startswith('e_') else 'all'

            ret_scores = None
            # RaPP scores
            rapp_scores = self.pathaway_diffs(X, mode)
            if rapp_mode == 'sap' or rapp_mode.endswith('rmse'):
                ret_scores = np.mean(rapp_scores ** 2, axis=1)
            if rapp_mode == 'nap' or rapp_mode.endswith('mhln'):
                ret_scores = np.nanmean(
                    (np.matmul(rapp_scores - self.mu[mode], self.v[mode]) / self.s[mode]) ** 2, axis=1)
#             try:
#                 return ret_scores.get()
#             except: # ret_scores is a NumPy array, not CuPy.
            return ret_scores

    input_size = kwargs['input_size']
    red_fact = kwargs.get('red_fact', [.75, .5])
    rapp_mode = kwargs.get('rapp_mode', 'e_rmse')
    epochs = kwargs.get('epochs', 100)

    optimizer = kwargs.get('optimizer', 'adam')
    loss = kwargs.get('loss', 'mse')

    return RaPP(input_size, red_fact)
