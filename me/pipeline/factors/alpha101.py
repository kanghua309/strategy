
from zipline.pipeline.factors import Latest
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import CustomFactor, AverageDollarVolume, Returns, RSI, VWAP

from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import CustomFactor



import pandas as pd
import numpy as np
from scipy import stats
# This frame is only for the preview
#  (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))
'''
class Alpha58(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    sectors_in = morningstar.asset_classification.morningstar_sector_code.latest
    sectors_in.window_safe = True
    inputs = [USEquityPricing.volume, sectors_in, vwap_in]
    window_length = 17
'''


def demean_by_group(signal, grouping):
    """Calculates and subtracts the group mean from the signal.
    Both inputs are 1-day np arrays.  Returns 1-day np array of demeaned values."""
    values_to_return = np.empty(signal.shape[0])
    options = set(grouping)
    for option in options:
        logical = grouping == option
        mean_by_group = signal[logical].sum() / logical.size
        values_to_return[logical] = signal[logical] - mean_by_group

    return values_to_return

class Alpha1(CustomFactor):
    inputs = [USEquityPricing.close, Returns(window_length=2)]
    window_length = 25

    def compute(self, today, assets, out, close, returns):
        v000 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v000000 = returns[-i0]
            v000001 = np.full(out.shape[0], 0.0)
            v00000 = v000000 < v000001
            v000010 = np.empty((20, out.shape[0]))
            for i1 in range(1, 21):
                v000010[-i1] = returns[- i0 -i1]
            v00001 = np.std(v000010, axis=0)
            v00002 = close[-i0]
            v0000lgcl = np.empty(out.shape[0])
            v0000lgcl[v00000] = v00001[v00000]
            v0000lgcl[~v00000] = v00002[~v00000]
            v0000 = v0000lgcl
            v0001 = np.full(out.shape[0], 2.0)
            v000[-i0] = np.power(v0000, v0001)
        v00 = np.argmax(v000, axis=0)
        v0 = stats.rankdata(v00)
        v1 = np.full(out.shape[0], 0.5)
        out[:] = v0 - v1

# (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
class Alpha2(CustomFactor):
    inputs = [USEquityPricing.volume, USEquityPricing.close, USEquityPricing.open]
    window_length = 9

    def compute(self, today, assets, out, volume, close, open):
        v0 = np.full(out.shape[0], -1.0)
        v10 = np.empty((6, out.shape[0]))
        for i0 in range(1, 7):
            v1000 = np.empty((3, out.shape[0]))
            for i1 in range(1, 4):
                v10000 = volume[- i0 -i1]
                v1000[-i1] = np.log(v10000)
            v100 = v1000[-1] - v1000[-3]
            v10[-i0] = stats.rankdata(v100)
        v11 = np.empty((6, out.shape[0]))
        for i0 in range(1, 7):
            v11000 = close[-i0]
            v11001 = open[-i0]
            v1100 = v11000 - v11001
            v1101 = open[-i0]
            v110 = v1100 / v1101
            v11[-i0] = stats.rankdata(v110)
        v1 = pd.DataFrame(v10).rolling(window=6).corr(pd.DataFrame(v11)).tail(1).as_matrix()[-1]
        out[:] = v0 * v1

# (-1 * correlation(rank(open), rank(volume), 10))
class Alpha3(CustomFactor):
    inputs = [USEquityPricing.volume, USEquityPricing.open]
    window_length = 10

    def compute(self, today, assets, out, volume, open):
        v0 = np.full(out.shape[0], -1.0)
        v10 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v100 = open[-i0]
            v10[-i0] = stats.rankdata(v100)
        v11 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v110 = volume[-i0]
            v11[-i0] = stats.rankdata(v110)
        v1 = pd.DataFrame(v10).rolling(window=10).corr(pd.DataFrame(v11)).tail(1).as_matrix()[-1]
        out[:] = v0 * v1

# (-1 * Ts_Rank(rank(low), 9))
class Alpha4(CustomFactor):
    inputs = [USEquityPricing.low]
    window_length = 9

    def compute(self, today, assets, out, low):
        v0 = np.full(out.shape[0], -1.0)
        v10 = np.empty((9, out.shape[0]))
        for i0 in range(1, 10):
            v100 = low[-i0]
            v10[-i0] = stats.rankdata(v100)
        v1 = pd.DataFrame(v10).rank().tail(1).as_matrix()[-1]
        out[:] = v0 * v1

# (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
class Alpha5(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    inputs = [USEquityPricing.close, USEquityPricing.open, vwap_in]
    window_length = 10

    def compute(self, today, assets, out, close, open, vwap):
        v000 = open[-1]
        v00100 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v00100[-i0] = vwap[-i0]
        v0010 = v00100.sum(axis=0)
        v0011 = np.full(out.shape[0], 10.0)
        v001 = v0010 / v0011
        v00 = v000 - v001
        v0 = stats.rankdata(v00)
        v10 = np.full(out.shape[0], -1.0)
        v11000 = close[-1]
        v11001 = vwap[-1]
        v1100 = v11000 - v11001
        v110 = stats.rankdata(v1100)
        v11 = np.abs(v110)
        v1 = v10 * v11
        out[:] = v0 * v1

# (-1 * correlation(open, volume, 10))
class Alpha6(CustomFactor):
    inputs = [USEquityPricing.volume, USEquityPricing.open]
    window_length = 10

    def compute(self, today, assets, out, volume, open):
        v0 = np.full(out.shape[0], -1.0)
        v10 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v10[-i0] = open[-i0]
        v11 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v11[-i0] = volume[-i0]
        v1 = pd.DataFrame(v10).rolling(window=10).corr(pd.DataFrame(v11)).tail(1).as_matrix()[-1]
        out[:] = v0 * v1

# ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))
class Alpha7(CustomFactor):
    adv20_in = AverageDollarVolume(window_length=20)
    adv20_in.window_safe = True
    inputs = [USEquityPricing.volume, USEquityPricing.close, adv20_in]
    window_length = 68

    def compute(self, today, assets, out, volume, close, adv20):
        v00 = adv20[-1]
        v01 = volume[-1]
        v0 = v00 < v01
        v100 = np.full(out.shape[0], -1.0)
        v1010 = np.empty((60, out.shape[0]))
        for i0 in range(1, 61):
            v101000 = np.empty((8, out.shape[0]))
            for i1 in range(1, 9):
                v101000[-i1] = close[- i0 -i1]
            v10100 = v101000[-1] - v101000[-8]
            v1010[-i0] = np.abs(v10100)
        v101 = pd.DataFrame(v1010).rank().tail(1).as_matrix()[-1]
        v10 = v100 * v101
        v1100 = np.empty((8, out.shape[0]))
        for i0 in range(1, 9):
            v1100[-i0] = close[-i0]
        v110 = v1100[-1] - v1100[-8]
        v11 = np.sign(v110)
        v1 = v10 * v11
        v20 = np.full(out.shape[0], -1.0)
        v21 = np.full(out.shape[0], 1.0)
        v2 = v20 * v21
        vlgcl = np.empty(out.shape[0])
        vlgcl[v0] = v1[v0]
        vlgcl[~v0] = v2[~v0]
        out[:] = vlgcl

# (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))
class Alpha8(CustomFactor):
    inputs = [Returns(window_length=2), USEquityPricing.open]
    window_length = 16

    def compute(self, today, assets, out, returns, open):
        v0 = np.full(out.shape[0], -1.0)
        v10000 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v10000[-i0] = open[-i0]
        v1000 = v10000.sum(axis=0)
        v10010 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v10010[-i0] = returns[-i0]
        v1001 = v10010.sum(axis=0)
        v100 = v1000 * v1001
        v101000 = np.empty((5, out.shape[0]))
        for i0 in range(11, 16):
            v101000[ 10 -i0] = open[-i0]
        v10100 = v101000.sum(axis=0)
        v101010 = np.empty((5, out.shape[0]))
        for i0 in range(11, 16):
            v101010[ 10 -i0] = returns[-i0]
        v10101 = v101010.sum(axis=0)
        v1010 = v10100 * v10101
        v101 = v1010 # delay
        v10 = v100 - v101
        v1 = stats.rankdata(v10)
        out[:] = v0 * v1

# ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
class Alpha9(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 7

    def compute(self, today, assets, out, close):
        v00 = np.full(out.shape[0], 0.0)
        v010 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v0100 = np.empty((2, out.shape[0]))
            for i1 in range(1, 3):
                v0100[-i1] = close[- i0 -i1]
            v010[-i0] = v0100[-1] - v0100[-2]
        v01 = np.min(v010, axis=0)
        v0 = v00 < v01
        v10 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v10[-i0] = close[-i0]
        v1 = v10[-1] - v10[-2]
        v2000 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v20000 = np.empty((2, out.shape[0]))
            for i1 in range(1, 3):
                v20000[-i1] = close[- i0 -i1]
            v2000[-i0] = v20000[-1] - v20000[-2]
        v200 = np.max(v2000, axis=0)
        v201 = np.full(out.shape[0], 0.0)
        v20 = v200 < v201
        v210 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v210[-i0] = close[-i0]
        v21 = v210[-1] - v210[-2]
        v220 = np.full(out.shape[0], -1.0)
        v2210 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v2210[-i0] = close[-i0]
        v221 = v2210[-1] - v2210[-2]
        v22 = v220 * v221
        v2lgcl = np.empty(out.shape[0])
        v2lgcl[v20] = v21[v20]
        v2lgcl[~v20] = v22[~v20]
        v2 = v2lgcl
        vlgcl = np.empty(out.shape[0])
        vlgcl[v0] = v1[v0]
        vlgcl[~v0] = v2[~v0]
        out[:] = vlgcl

# rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))
class Alpha10(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 6

    def compute(self, today, assets, out, close):
        v000 = np.full(out.shape[0], 0.0)
        v0010 = np.empty((4, out.shape[0]))
        for i0 in range(1, 5):
            v00100 = np.empty((2, out.shape[0]))
            for i1 in range(1, 3):
                v00100[-i1] = close[- i0 -i1]
            v0010[-i0] = v00100[-1] - v00100[-2]
        v001 = np.min(v0010, axis=0)
        v00 = v000 < v001
        v010 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v010[-i0] = close[-i0]
        v01 = v010[-1] - v010[-2]
        v02000 = np.empty((4, out.shape[0]))
        for i0 in range(1, 5):
            v020000 = np.empty((2, out.shape[0]))
            for i1 in range(1, 3):
                v020000[-i1] = close[- i0 -i1]
            v02000[-i0] = v020000[-1] - v020000[-2]
        v0200 = np.max(v02000, axis=0)
        v0201 = np.full(out.shape[0], 0.0)
        v020 = v0200 < v0201
        v0210 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v0210[-i0] = close[-i0]
        v021 = v0210[-1] - v0210[-2]
        v0220 = np.full(out.shape[0], -1.0)
        v02210 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v02210[-i0] = close[-i0]
        v0221 = v02210[-1] - v02210[-2]
        v022 = v0220 * v0221
        v02lgcl = np.empty(out.shape[0])
        v02lgcl[v020] = v021[v020]
        v02lgcl[~v020] = v022[~v020]
        v02 = v02lgcl
        v0lgcl = np.empty(out.shape[0])
        v0lgcl[v00] = v01[v00]
        v0lgcl[~v00] = v02[~v00]
        v0 = v0lgcl
        out[:] = stats.rankdata(v0)

# ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
class Alpha11(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    inputs = [USEquityPricing.volume, USEquityPricing.close, vwap_in]
    window_length = 4

    def compute(self, today, assets, out, volume, close, vwap):
        v0000 = np.empty((3, out.shape[0]))
        for i0 in range(1, 4):
            v00000 = vwap[-i0]
            v00001 = close[-i0]
            v0000[-i0] = v00000 - v00001
        v000 = np.max(v0000, axis=0)
        v00 = stats.rankdata(v000)
        v0100 = np.empty((3, out.shape[0]))
        for i0 in range(1, 4):
            v01000 = vwap[-i0]
            v01001 = close[-i0]
            v0100[-i0] = v01000 - v01001
        v010 = np.min(v0100, axis=0)
        v01 = stats.rankdata(v010)
        v0 = v00 + v01
        v100 = np.empty((4, out.shape[0]))
        for i0 in range(1, 5):
            v100[-i0] = volume[-i0]
        v10 = v100[-1] - v100[-4]
        v1 = stats.rankdata(v10)
        out[:] = v0 * v1

# (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
class Alpha12(CustomFactor):
    inputs = [USEquityPricing.volume, USEquityPricing.close]
    window_length = 2

    def compute(self, today, assets, out, volume, close):
        v000 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v000[-i0] = volume[-i0]
        v00 = v000[-1] - v000[-2]
        v0 = np.sign(v00)
        v10 = np.full(out.shape[0], -1.0)
        v110 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v110[-i0] = close[-i0]
        v11 = v110[-1] - v110[-2]
        v1 = v10 * v11
        out[:] = v0 * v1

# (-1 * rank(covariance(rank(close), rank(volume), 5)))
class Alpha13(CustomFactor):
    inputs = [USEquityPricing.volume, USEquityPricing.close]
    window_length = 5

    def compute(self, today, assets, out, volume, close):
        v0 = np.full(out.shape[0], -1.0)
        v100 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v1000 = close[-i0]
            v100[-i0] = stats.rankdata(v1000)
        v101 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v1010 = volume[-i0]
            v101[-i0] = stats.rankdata(v1010)
        v10 = pd.DataFrame(v100).rolling(window=5).cov(pd.DataFrame(v101)).tail(1).as_matrix()[-1]
        v1 = stats.rankdata(v10)
        out[:] = v0 * v1

# ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
class Alpha14(CustomFactor):
    inputs = [USEquityPricing.volume, Returns(window_length=2), USEquityPricing.open]
    window_length = 10

    def compute(self, today, assets, out, volume, returns, open):
        v00 = np.full(out.shape[0], -1.0)
        v0100 = np.empty((4, out.shape[0]))
        for i0 in range(1, 5):
            v0100[-i0] = returns[-i0]
        v010 = v0100[-1] - v0100[-4]
        v01 = stats.rankdata(v010)
        v0 = v00 * v01
        v10 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v10[-i0] = open[-i0]
        v11 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v11[-i0] = volume[-i0]
        v1 = pd.DataFrame(v10).rolling(window=10).corr(pd.DataFrame(v11)).tail(1).as_matrix()[-1]
        out[:] = v0 * v1

# (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
class Alpha15(CustomFactor):
    inputs = [USEquityPricing.high, USEquityPricing.volume]
    window_length = 6

    def compute(self, today, assets, out, high, volume):
        v0 = np.full(out.shape[0], -1.0)
        v10 = np.empty((3, out.shape[0]))
        for i0 in range(1, 4):
            v1000 = np.empty((3, out.shape[0]))
            for i1 in range(1, 4):
                v10000 = high[- i0 -i1]
                v1000[-i1] = stats.rankdata(v10000)
            v1001 = np.empty((3, out.shape[0]))
            for i1 in range(1, 4):
                v10010 = volume[- i0 -i1]
                v1001[-i1] = stats.rankdata(v10010)
            v100 = pd.DataFrame(v1000).rolling(window=3).corr(pd.DataFrame(v1001)).tail(1).as_matrix()[-1]
            v10[-i0] = stats.rankdata(v100)
        v1 = v10.sum(axis=0)
        out[:] = v0 * v1

# (-1 * rank(covariance(rank(high), rank(volume), 5)))
class Alpha16(CustomFactor):
    inputs = [USEquityPricing.high, USEquityPricing.volume]
    window_length = 5

    def compute(self, today, assets, out, high, volume):
        v0 = np.full(out.shape[0], -1.0)
        v100 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v1000 = high[-i0]
            v100[-i0] = stats.rankdata(v1000)
        v101 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v1010 = volume[-i0]
            v101[-i0] = stats.rankdata(v1010)
        v10 = pd.DataFrame(v100).rolling(window=5).cov(pd.DataFrame(v101)).tail(1).as_matrix()[-1]
        v1 = stats.rankdata(v10)
        out[:] = v0 * v1

# (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))
class Alpha17(CustomFactor):
    adv20_in = AverageDollarVolume(window_length=20)
    adv20_in.window_safe = True
    inputs = [USEquityPricing.volume, USEquityPricing.close, adv20_in]
    window_length = 10

    def compute(self, today, assets, out, volume, close, adv20):
        v000 = np.full(out.shape[0], -1.0)
        v00100 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v00100[-i0] = close[-i0]
        v0010 = pd.DataFrame(v00100).rank().tail(1).as_matrix()[-1]
        v001 = stats.rankdata(v0010)
        v00 = v000 * v001
        v0100 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v01000 = np.empty((2, out.shape[0]))
            for i1 in range(1, 3):
                v01000[-i1] = close[- i0 -i1]
            v0100[-i0] = v01000[-1] - v01000[-2]
        v010 = v0100[-1] - v0100[-2]
        v01 = stats.rankdata(v010)
        v0 = v00 * v01
        v100 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v1000 = volume[-i0]
            v1001 = adv20[-i0]
            v100[-i0] = v1000 / v1001
        v10 = pd.DataFrame(v100).rank().tail(1).as_matrix()[-1]
        v1 = stats.rankdata(v10)
        out[:] = v0 * v1

# (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))
class Alpha18(CustomFactor):
    inputs = [USEquityPricing.close, USEquityPricing.open]
    window_length = 10

    def compute(self, today, assets, out, close, open):
        v0 = np.full(out.shape[0], -1.0)
        v10000 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v1000000 = close[-i0]
            v1000001 = open[-i0]
            v100000 = v1000000 - v1000001
            v10000[-i0] = np.abs(v100000)
        v1000 = np.std(v10000, axis=0)
        v10010 = close[-1]
        v10011 = open[-1]
        v1001 = v10010 - v10011
        v100 = v1000 + v1001
        v1010 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v1010[-i0] = close[-i0]
        v1011 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v1011[-i0] = open[-i0]
        v101 = pd.DataFrame(v1010).rolling(window=10).corr(pd.DataFrame(v1011)).tail(1).as_matrix()[-1]
        v10 = v100 + v101
        v1 = stats.rankdata(v10)
        out[:] = v0 * v1

# ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))
class Alpha19(CustomFactor):
    inputs = [USEquityPricing.close, Returns(window_length=2)]
    window_length = 250

    def compute(self, today, assets, out, close, returns):
        v00 = np.full(out.shape[0], -1.0)
        v01000 = close[-1]
        v010010 = close[-8]
        v01001 = v010010 # delay
        v0100 = v01000 - v01001
        v01010 = np.empty((8, out.shape[0]))
        for i0 in range(1, 9):
            v01010[-i0] = close[-i0]
        v0101 = v01010[-1] - v01010[-8]
        v010 = v0100 + v0101
        v01 = np.sign(v010)
        v0 = v00 * v01
        v10 = np.full(out.shape[0], 1.0)
        v1100 = np.full(out.shape[0], 1.0)
        v11010 = np.empty((250, out.shape[0]))
        for i0 in range(1, 251):
            v11010[-i0] = returns[-i0]
        v1101 = v11010.sum(axis=0)
        v110 = v1100 + v1101
        v11 = stats.rankdata(v110)
        v1 = v10 + v11
        out[:] = v0 * v1

# (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))
class Alpha20(CustomFactor):
    inputs = [USEquityPricing.high, USEquityPricing.close, USEquityPricing.open, USEquityPricing.low]
    window_length = 2

    def compute(self, today, assets, out, high, close, open, low):
        v000 = np.full(out.shape[0], -1.0)
        v00100 = open[-1]
        v001010 = high[-2]
        v00101 = v001010 # delay
        v0010 = v00100 - v00101
        v001 = stats.rankdata(v0010)
        v00 = v000 * v001
        v0100 = open[-1]
        v01010 = close[-2]
        v0101 = v01010 # delay
        v010 = v0100 - v0101
        v01 = stats.rankdata(v010)
        v0 = v00 * v01
        v100 = open[-1]
        v1010 = low[-2]
        v101 = v1010 # delay
        v10 = v100 - v101
        v1 = stats.rankdata(v10)
        out[:] = v0 * v1

# ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1 * 1))))
class Alpha21(CustomFactor):
    adv20_in = AverageDollarVolume(window_length=20)
    adv20_in.window_safe = True
    inputs = [USEquityPricing.volume, USEquityPricing.close, adv20_in]
    window_length = 8

    def compute(self, today, assets, out, volume, close, adv20):
        v00000 = np.empty((8, out.shape[0]))
        for i0 in range(1, 9):
            v00000[-i0] = close[-i0]
        v0000 = v00000.sum(axis=0)
        v0001 = np.full(out.shape[0], 8.0)
        v000 = v0000 / v0001
        v0010 = np.empty((8, out.shape[0]))
        for i0 in range(1, 9):
            v0010[-i0] = close[-i0]
        v001 = np.std(v0010, axis=0)
        v00 = v000 + v001
        v0100 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v0100[-i0] = close[-i0]
        v010 = v0100.sum(axis=0)
        v011 = np.full(out.shape[0], 2.0)
        v01 = v010 / v011
        v0 = v00 < v01
        v10 = np.full(out.shape[0], -1.0)
        v11 = np.full(out.shape[0], 1.0)
        v1 = v10 * v11
        v20000 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v20000[-i0] = close[-i0]
        v2000 = v20000.sum(axis=0)
        v2001 = np.full(out.shape[0], 2.0)
        v200 = v2000 / v2001
        v201000 = np.empty((8, out.shape[0]))
        for i0 in range(1, 9):
            v201000[-i0] = close[-i0]
        v20100 = v201000.sum(axis=0)
        v20101 = np.full(out.shape[0], 8.0)
        v2010 = v20100 / v20101
        v20110 = np.empty((8, out.shape[0]))
        for i0 in range(1, 9):
            v20110[-i0] = close[-i0]
        v2011 = np.std(v20110, axis=0)
        v201 = v2010 - v2011
        v20 = v200 < v201
        v21 = np.full(out.shape[0], 1.0)
        v22000 = np.full(out.shape[0], 1.0)
        v220010 = volume[-1]
        v220011 = adv20[-1]
        v22001 = v220010 / v220011
        v2200 = v22000 < v22001
        v220100 = volume[-1]
        v220101 = adv20[-1]
        v22010 = v220100 / v220101
        v22011 = np.full(out.shape[0], 1.0)
        v2201 = v22010 == v22011
        v220 = v2200 | v2201
        v221 = np.full(out.shape[0], 1.0)
        v2220 = np.full(out.shape[0], -1.0)
        v2221 = np.full(out.shape[0], 1.0)
        v222 = v2220 * v2221
        v22lgcl = np.empty(out.shape[0])
        v22lgcl[v220] = 1
        v22lgcl[~v220] = v222[~v220]
        v22 = v22lgcl
        v2lgcl = np.empty(out.shape[0])
        v2lgcl[v20] = 1
        v2lgcl[~v20] = v22[~v20]
        v2 = v2lgcl
        vlgcl = np.empty(out.shape[0])
        vlgcl[v0] = v1[v0]
        vlgcl[~v0] = v2[~v0]
        out[:] = vlgcl

# (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
class Alpha22(CustomFactor):
    inputs = [USEquityPricing.high, USEquityPricing.volume, USEquityPricing.close]
    window_length = 20

    def compute(self, today, assets, out, high, volume, close):
        v0 = np.full(out.shape[0], -1.0)
        v100 = np.empty((6, out.shape[0]))
        for i0 in range(1, 7):
            v1000 = np.empty((5, out.shape[0]))
            for i1 in range(1, 6):
                v1000[-i1] = high[- i0 -i1]
            v1001 = np.empty((5, out.shape[0]))
            for i1 in range(1, 6):
                v1001[-i1] = volume[- i0 -i1]
            v100[-i0] = pd.DataFrame(v1000).rolling(window=5).corr(pd.DataFrame(v1001)).tail(1).as_matrix()[-1]
        v10 = v100[-1] - v100[-6]
        v1100 = np.empty((20, out.shape[0]))
        for i0 in range(1, 21):
            v1100[-i0] = close[-i0]
        v110 = np.std(v1100, axis=0)
        v11 = stats.rankdata(v110)
        v1 = v10 * v11
        out[:] = v0 * v1

# (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
class Alpha23(CustomFactor):
    inputs = [USEquityPricing.high]
    window_length = 20

    def compute(self, today, assets, out, high):
        v0000 = np.empty((20, out.shape[0]))
        for i0 in range(1, 21):
            v0000[-i0] = high[-i0]
        v000 = v0000.sum(axis=0)
        v001 = np.full(out.shape[0], 20.0)
        v00 = v000 / v001
        v01 = high[-1]
        v0 = v00 < v01
        v10 = np.full(out.shape[0], -1.0)
        v110 = np.empty((3, out.shape[0]))
        for i0 in range(1, 4):
            v110[-i0] = high[-i0]
        v11 = v110[-1] - v110[-3]
        v1 = v10 * v11
        v2 = np.full(out.shape[0], 0.0)
        vlgcl = np.empty(out.shape[0])
        vlgcl[v0] = v1[v0]
        vlgcl[~v0] = 0
        out[:] = vlgcl

# ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))
class Alpha24(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 201

    def compute(self, today, assets, out, close):
        v00000 = np.empty((101, out.shape[0]))
        for i0 in range(1, 102):
            v0000000 = np.empty((100, out.shape[0]))
            for i1 in range(1, 101):
                v0000000[-i1] = close[- i0 -i1]
            v000000 = v0000000.sum(axis=0)
            v000001 = np.full(out.shape[0], 100.0)
            v00000[-i0] = v000000 / v000001
        v0000 = v00000[-1] - v00000[-101]
        v00010 = close[-101]
        v0001 = v00010 # delay
        v000 = v0000 / v0001
        v001 = np.full(out.shape[0], 0.05)
        v00 = v000 < v001
        v01000 = np.empty((101, out.shape[0]))
        for i0 in range(1, 102):
            v0100000 = np.empty((100, out.shape[0]))
            for i1 in range(1, 101):
                v0100000[-i1] = close[- i0 -i1]
            v010000 = v0100000.sum(axis=0)
            v010001 = np.full(out.shape[0], 100.0)
            v01000[-i0] = v010000 / v010001
        v0100 = v01000[-1] - v01000[-101]
        v01010 = close[-101]
        v0101 = v01010 # delay
        v010 = v0100 / v0101
        v011 = np.full(out.shape[0], 0.05)
        v01 = v010 == v011
        v0 = v00 | v01
        v10 = np.full(out.shape[0], -1.0)
        v110 = close[-1]
        v1110 = np.empty((100, out.shape[0]))
        for i0 in range(1, 101):
            v1110[-i0] = close[-i0]
        v111 = np.min(v1110, axis=0)
        v11 = v110 - v111
        v1 = v10 * v11
        v20 = np.full(out.shape[0], -1.0)
        v210 = np.empty((4, out.shape[0]))
        for i0 in range(1, 5):
            v210[-i0] = close[-i0]
        v21 = v210[-1] - v210[-4]
        v2 = v20 * v21
        vlgcl = np.empty(out.shape[0])
        vlgcl[v0] = v1[v0]
        vlgcl[~v0] = v2[~v0]
        out[:] = vlgcl

# rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
class Alpha25(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    adv20_in = AverageDollarVolume(window_length=20)
    adv20_in.window_safe = True
    inputs = [USEquityPricing.high, USEquityPricing.close, Returns(window_length=2), adv20_in, vwap_in]
    window_length = 1

    def compute(self, today, assets, out, high, close, returns, adv20, vwap):
        v00000 = np.full(out.shape[0], -1.0)
        v00001 = returns[-1]
        v0000 = v00000 * v00001
        v0001 = adv20[-1]
        v000 = v0000 * v0001
        v001 = vwap[-1]
        v00 = v000 * v001
        v010 = high[-1]
        v011 = close[-1]
        v01 = v010 - v011
        v0 = v00 * v01
        out[:] = stats.rankdata(v0)

# (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
class Alpha26(CustomFactor):
    inputs = [USEquityPricing.volume, USEquityPricing.high]
    window_length = 13

    def compute(self, today, assets, out, volume, high):
        v0 = np.full(out.shape[0], -1.0)
        v10 = np.empty((3, out.shape[0]))
        for i0 in range(1, 4):
            v100 = np.empty((5, out.shape[0]))
            for i1 in range(1, 6):
                v1000 = np.empty((5, out.shape[0]))
                for i2 in range(1, 6):
                    v1000[-i2] = volume[- i0 - i1 -i2]
                v100[-i1] = pd.DataFrame(v1000).rank().tail(1).as_matrix()[-1]
            v101 = np.empty((5, out.shape[0]))
            for i1 in range(1, 6):
                v1010 = np.empty((5, out.shape[0]))
                for i2 in range(1, 6):
                    v1010[-i2] = high[- i0 - i1 -i2]
                v101[-i1] = pd.DataFrame(v1010).rank().tail(1).as_matrix()[-1]
            v10[-i0] = pd.DataFrame(v100).rolling(window=5).corr(pd.DataFrame(v101)).tail(1).as_matrix()[-1]
        v1 = np.max(v10, axis=0)
        out[:] = v0 * v1

#  ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
class Alpha27(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    inputs = [USEquityPricing.volume, vwap_in]
    window_length = 8

    def compute(self, today, assets, out, volume, vwap):
        v00 = np.full(out.shape[0], 0.5)
        v01000 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v010000 = np.empty((6, out.shape[0]))
            for i1 in range(1, 7):
                v0100000 = volume[-i0-i1]
                v010000[-i1] = stats.rankdata(v0100000)
            v010001 = np.empty((6, out.shape[0]))
            for i1 in range(1, 7):
                v0100010 = vwap[-i0-i1]
                v010001[-i1] = stats.rankdata(v0100010)
            v01000[-i0] = pd.DataFrame(v010000).rolling(window=6).corr(pd.DataFrame(v010001)).tail(1).as_matrix()[-1]
        v0100 = v01000.sum(axis=0)
        v0101 = np.full(out.shape[0], 2.0)
        v010 = v0100 / v0101
        v01 = stats.rankdata(v010)
        v0 = v00 < v01
        v10 = np.full(out.shape[0], -1.0)
        v11 = np.full(out.shape[0], 1.0)
        v1 = v10 * v11
        v2 = np.full(out.shape[0], 1.0)
        vlgcl = np.empty(out.shape[0])
        vlgcl[v0] = v1[v0]
        vlgcl[~v0] = 1
        out[:] = vlgcl

# scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
class Alpha28(CustomFactor):
    adv20_in = AverageDollarVolume(window_length=20)
    adv20_in.window_safe = True
    inputs = [USEquityPricing.high, USEquityPricing.close, adv20_in, USEquityPricing.low]
    window_length = 5

    def compute(self, today, assets, out, high, close, adv20, low):
        v0000 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v0000[-i0] = adv20[-i0]
        v0001 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v0001[-i0] = low[-i0]
        v000 = pd.DataFrame(v0000).rolling(window=5).corr(pd.DataFrame(v0001)).tail(1).as_matrix()[-1]
        v00100 = high[-1]
        v00101 = low[-1]
        v0010 = v00100 + v00101
        v0011 = np.full(out.shape[0], 2.0)
        v001 = v0010 / v0011
        v00 = v000 + v001
        v01 = close[-1]
        v0 = v00 - v01
        out[:] = v0/ np.abs(v0).sum()

# (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
class Alpha29(CustomFactor):
    inputs = [USEquityPricing.close, Returns(window_length=2)]
    window_length = 12

    def compute(self, today, assets, out, close, returns):
        v000 = np.empty((1, out.shape[0]))
        for i0 in range(1, 2):
            v00000000 = np.empty((1, out.shape[0]))
            for i1 in range(1, 2):
                v000000000 = np.empty((2, out.shape[0]))
                for i2 in range(1, 3):
                    v000000000000 = np.full(out.shape[0], -1.0)
                    v00000000000100 = np.empty((6, out.shape[0]))
                    for i3 in range(1, 7):
                        v000000000001000 = close[-i0 - i1 - i2 - i3]
                        v000000000001001 = np.full(out.shape[0], 1.0)
                        v00000000000100[-i3] = v000000000001000 - v000000000001001
                    v0000000000010 = v00000000000100[-1] - v00000000000100[-6]
                    v000000000001 = stats.rankdata(v0000000000010)
                    v00000000000 = v000000000000 * v000000000001
                    v0000000000 = stats.rankdata(v00000000000)
                    v000000000[-i2] = stats.rankdata(v0000000000)
                v00000000[-i1] = np.min(v000000000, axis=0)
            v0000000 = v00000000.sum(axis=0)
            v000000 = np.log(v0000000)
            v00000 = v000000 / np.abs(v000000).sum()
            v0000 = stats.rankdata(v00000)
            v000[-i0] = stats.rankdata(v0000)
        v00 = np.prod(v000, axis=0)
        v01 = np.full(out.shape[0], 5.0)
        v0 = np.minimum(v00, v01)
        v10 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v1000 = np.full(out.shape[0], -1.0)
            v1001 = returns[-7]
            v100 = v1000 * v1001
            v10[-i0] = v100  # delay
        v1 = pd.DataFrame(v10).rank().tail(1).as_matrix()[-1]
        out[:] = v0 + v1

# (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
class Alpha30(CustomFactor):
    inputs = [USEquityPricing.volume, USEquityPricing.close]
    window_length = 20

    def compute(self, today, assets, out, volume, close):
        v000 = np.full(out.shape[0], 1.0)
        v00100000 = close[-1]
        v001000010 = close[-2]
        v00100001 = v001000010  # delay
        v0010000 = v00100000 - v00100001
        v001000 = np.sign(v0010000)
        v001001000 = close[-2]
        v00100100 = v001001000  # delay
        v001001010 = close[-3]
        v00100101 = v001001010  # delay
        v0010010 = v00100100 - v00100101
        v001001 = np.sign(v0010010)
        v00100 = v001000 + v001001
        v00101000 = close[-3]
        v0010100 = v00101000  # delay
        v00101010 = close[-4]
        v0010101 = v00101010  # delay
        v001010 = v0010100 - v0010101
        v00101 = np.sign(v001010)
        v0010 = v00100 + v00101
        v001 = stats.rankdata(v0010)
        v00 = v000 - v001
        v010 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v010[-i0] = volume[-i0]
        v01 = v010.sum(axis=0)
        v0 = v00 * v01
        v10 = np.empty((20, out.shape[0]))
        for i0 in range(1, 21):
            v10[-i0] = volume[-i0]
        v1 = v10.sum(axis=0)
        out[:] = v0 / v1

# ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
class Alpha31(CustomFactor):
    adv20_in = AverageDollarVolume(window_length=20)
    adv20_in.window_safe = True
    inputs = [USEquityPricing.close, adv20_in, USEquityPricing.low]
    window_length = 21

    def compute(self, today, assets, out, close, adv20, low):
        v000000 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v0000000 = np.full(out.shape[0], -1.0)
            v0000001000 = np.empty((11, out.shape[0]))
            for i1 in range(1, 12):
                v0000001000[-i1] = close[-i0 - i1]
            v000000100 = v0000001000[-1] - v0000001000[-11]
            v00000010 = stats.rankdata(v000000100)
            v0000001 = stats.rankdata(v00000010)
            v000000[-i0] = v0000000 * v0000001
        v00000 = (v000000 * (np.arange(1.0, 11, 1.0) / 55)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v0000 = stats.rankdata(v00000)
        v000 = stats.rankdata(v0000)
        v00 = stats.rankdata(v000)
        v0100 = np.full(out.shape[0], -1.0)
        v01010 = np.empty((4, out.shape[0]))
        for i0 in range(1, 5):
            v01010[-i0] = close[-i0]
        v0101 = v01010[-1] - v01010[-4]
        v010 = v0100 * v0101
        v01 = stats.rankdata(v010)
        v0 = v00 + v01
        v1000 = np.empty((12, out.shape[0]))
        for i0 in range(1, 13):
            v1000[-i0] = adv20[-i0]
        v1001 = np.empty((12, out.shape[0]))
        for i0 in range(1, 13):
            v1001[-i0] = low[-i0]
        v100 = pd.DataFrame(v1000).rolling(window=12).corr(pd.DataFrame(v1001)).tail(1).as_matrix()[-1]
        v10 = v100 / np.abs(v100).sum()
        v1 = np.sign(v10)
        out[:] = v0 + v1

# (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))
class Alpha32(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    inputs = [USEquityPricing.close, vwap_in]
    window_length = 236

    def compute(self, today, assets, out, close, vwap):
        v00000 = np.empty((7, out.shape[0]))
        for i0 in range(1, 8):
            v00000[-i0] = close[-i0]
        v0000 = v00000.sum(axis=0)
        v0001 = np.full(out.shape[0], 7.0)
        v000 = v0000 / v0001
        v001 = close[-1]
        v00 = v000 - v001
        v0 = v00 / np.abs(v00).sum()
        v10 = np.full(out.shape[0], 20.0)
        v1100 = np.empty((230, out.shape[0]))
        for i0 in range(1, 231):
            v1100[-i0] = vwap[-i0]
        v1101 = np.empty((230, out.shape[0]))
        for i0 in range(1, 231):
            v11010 = close[-6]
            v1101[-i0] = v11010  # delay
        v110 = pd.DataFrame(v1100).rolling(window=230).corr(pd.DataFrame(v1101)).tail(1).as_matrix()[-1]
        v11 = v110 / np.abs(v110).sum()
        v1 = v10 * v11
        out[:] = v0 + v1

# rank((-1 * ((1 - (open / close))^1)))
class Alpha33(CustomFactor):
    inputs = [USEquityPricing.close, USEquityPricing.open]
    window_length = 1

    def compute(self, today, assets, out, close, open):
        v00 = np.full(out.shape[0], -1.0)
        v0100 = np.full(out.shape[0], 1.0)
        v01010 = open[-1]
        v01011 = close[-1]
        v0101 = v01010 / v01011
        v010 = v0100 - v0101
        v011 = np.full(out.shape[0], 1.0)
        v01 = np.power(v010, v011)
        v0 = v00 * v01
        out[:] = stats.rankdata(v0)

# rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
class Alpha34(CustomFactor):
    inputs = [USEquityPricing.close, Returns(window_length=2)]
    window_length = 5

    def compute(self, today, assets, out, close, returns):
        v000 = np.full(out.shape[0], 1.0)
        v001000 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v001000[-i0] = returns[-i0]
        v00100 = np.std(v001000, axis=0)
        v001010 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v001010[-i0] = returns[-i0]
        v00101 = np.std(v001010, axis=0)
        v0010 = v00100 / v00101
        v001 = stats.rankdata(v0010)
        v00 = v000 - v001
        v010 = np.full(out.shape[0], 1.0)
        v01100 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v01100[-i0] = close[-i0]
        v0110 = v01100[-1] - v01100[-2]
        v011 = stats.rankdata(v0110)
        v01 = v010 - v011
        v0 = v00 + v01
        out[:] = stats.rankdata(v0)

# ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))
class Alpha35(CustomFactor):
    inputs = [USEquityPricing.volume, USEquityPricing.close, USEquityPricing.high, Returns(window_length=2),
              USEquityPricing.low]
    window_length = 32

    def compute(self, today, assets, out, volume, close, high, returns, low):
        v000 = np.empty((32, out.shape[0]))
        for i0 in range(1, 33):
            v000[-i0] = volume[-i0]
        v00 = pd.DataFrame(v000).rank().tail(1).as_matrix()[-1]
        v010 = np.full(out.shape[0], 1.0)
        v0110 = np.empty((16, out.shape[0]))
        for i0 in range(1, 17):
            v011000 = close[-i0]
            v011001 = high[-i0]
            v01100 = v011000 + v011001
            v01101 = low[-i0]
            v0110[-i0] = v01100 - v01101
        v011 = pd.DataFrame(v0110).rank().tail(1).as_matrix()[-1]
        v01 = v010 - v011
        v0 = v00 * v01
        v10 = np.full(out.shape[0], 1.0)
        v110 = np.empty((32, out.shape[0]))
        for i0 in range(1, 33):
            v110[-i0] = returns[-i0]
        v11 = pd.DataFrame(v110).rank().tail(1).as_matrix()[-1]
        v1 = v10 - v11
        out[:] = v0 * v1

# (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
class Alpha36(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    adv20_in = AverageDollarVolume(window_length=20)
    adv20_in.window_safe = True
    inputs = [adv20_in, vwap_in, USEquityPricing.volume, Returns(window_length=2), USEquityPricing.close,
              USEquityPricing.open]
    window_length = 200

    def compute(self, today, assets, out, adv20, vwap, volume, returns, close, open):
        v00000 = np.full(out.shape[0], 2.21)
        v0000100 = np.empty((15, out.shape[0]))
        for i0 in range(1, 16):
            v00001000 = close[-i0]
            v00001001 = open[-i0]
            v0000100[-i0] = v00001000 - v00001001
        v0000101 = np.empty((15, out.shape[0]))
        for i0 in range(1, 16):
            v00001010 = volume[-2]
            v0000101[-i0] = v00001010  # delay
        v000010 = pd.DataFrame(v0000100).rolling(window=15).corr(pd.DataFrame(v0000101)).tail(1).as_matrix()[-1]
        v00001 = stats.rankdata(v000010)
        v0000 = v00000 * v00001
        v00010 = np.full(out.shape[0], 0.7)
        v0001100 = open[-1]
        v0001101 = close[-1]
        v000110 = v0001100 - v0001101
        v00011 = stats.rankdata(v000110)
        v0001 = v00010 * v00011
        v000 = v0000 + v0001
        v0010 = np.full(out.shape[0], 0.73)
        v001100 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v00110000 = np.full(out.shape[0], -1.0)
            v00110001 = returns[-7]
            v0011000 = v00110000 * v00110001
            v001100[-i0] = v0011000  # delay
        v00110 = pd.DataFrame(v001100).rank().tail(1).as_matrix()[-1]
        v0011 = stats.rankdata(v00110)
        v001 = v0010 * v0011
        v00 = v000 + v001
        v01000 = np.empty((6, out.shape[0]))
        for i0 in range(1, 7):
            v01000[-i0] = vwap[-i0]
        v01001 = np.empty((6, out.shape[0]))
        for i0 in range(1, 7):
            v01001[-i0] = adv20[-i0]
        v0100 = pd.DataFrame(v01000).rolling(window=6).corr(pd.DataFrame(v01001)).tail(1).as_matrix()[-1]
        v010 = np.abs(v0100)
        v01 = stats.rankdata(v010)
        v0 = v00 + v01
        v10 = np.full(out.shape[0], 0.6)
        v1100000 = np.empty((200, out.shape[0]))
        for i0 in range(1, 201):
            v1100000[-i0] = close[-i0]
        v110000 = v1100000.sum(axis=0)
        v110001 = np.full(out.shape[0], 200.0)
        v11000 = v110000 / v110001
        v11001 = open[-1]
        v1100 = v11000 - v11001
        v11010 = close[-1]
        v11011 = open[-1]
        v1101 = v11010 - v11011
        v110 = v1100 * v1101
        v11 = stats.rankdata(v110)
        v1 = v10 * v11
        out[:] = v0 + v1

# (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
class Alpha37(CustomFactor):
    inputs = [USEquityPricing.close, USEquityPricing.open]
    window_length = 202

    def compute(self, today, assets, out, close, open):
        v000 = np.empty((200, out.shape[0]))
        for i0 in range(1, 201):
            v00000 = open[-2]
            v00001 = close[-2]
            v0000 = v00000 - v00001
            v000[-i0] = v0000  # delay
        v001 = np.empty((200, out.shape[0]))
        for i0 in range(1, 201):
            v001[-i0] = close[-i0]
        v00 = pd.DataFrame(v000).rolling(window=200).corr(pd.DataFrame(v001)).tail(1).as_matrix()[-1]
        v0 = stats.rankdata(v00)
        v100 = open[-1]
        v101 = close[-1]
        v10 = v100 - v101
        v1 = stats.rankdata(v10)
        out[:] = v0 + v1

# ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
class Alpha38(CustomFactor):
    inputs = [USEquityPricing.close, USEquityPricing.open]
    window_length = 10

    def compute(self, today, assets, out, close, open):
        v00 = np.full(out.shape[0], -1.0)
        v0100 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v0100[-i0] = close[-i0]
        v010 = pd.DataFrame(v0100).rank().tail(1).as_matrix()[-1]
        v01 = stats.rankdata(v010)
        v0 = v00 * v01
        v100 = close[-1]
        v101 = open[-1]
        v10 = v100 / v101
        v1 = stats.rankdata(v10)
        out[:] = v0 * v1

# ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))
class Alpha39(CustomFactor):
    adv20_in = AverageDollarVolume(window_length=20)
    adv20_in.window_safe = True
    inputs = [USEquityPricing.volume, USEquityPricing.close, Returns(window_length=2), adv20_in]
    window_length = 250

    def compute(self, today, assets, out, volume, close, returns, adv20):
        v00 = np.full(out.shape[0], -1.0)
        v01000 = np.empty((8, out.shape[0]))
        for i0 in range(1, 9):
            v01000[-i0] = close[-i0]
        v0100 = v01000[-1] - v01000[-8]
        v01010 = np.full(out.shape[0], 1.0)
        v0101100 = np.empty((9, out.shape[0]))
        for i0 in range(1, 10):
            v01011000 = volume[-i0]
            v01011001 = adv20[-i0]
            v0101100[-i0] = v01011000 / v01011001
        v010110 = (v0101100 * (np.arange(1.0, 10, 1.0) / 45)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v01011 = stats.rankdata(v010110)
        v0101 = v01010 - v01011
        v010 = v0100 * v0101
        v01 = stats.rankdata(v010)
        v0 = v00 * v01
        v10 = np.full(out.shape[0], 1.0)
        v1100 = np.empty((250, out.shape[0]))
        for i0 in range(1, 251):
            v1100[-i0] = returns[-i0]
        v110 = v1100.sum(axis=0)
        v11 = stats.rankdata(v110)
        v1 = v10 + v11
        out[:] = v0 * v1

# ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
class Alpha40(CustomFactor):
    inputs = [USEquityPricing.high, USEquityPricing.volume]
    window_length = 10

    def compute(self, today, assets, out, high, volume):
        v00 = np.full(out.shape[0], -1.0)
        v0100 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v0100[-i0] = high[-i0]
        v010 = np.std(v0100, axis=0)
        v01 = stats.rankdata(v010)
        v0 = v00 * v01
        v10 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v10[-i0] = high[-i0]
        v11 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v11[-i0] = volume[-i0]
        v1 = pd.DataFrame(v10).rolling(window=10).corr(pd.DataFrame(v11)).tail(1).as_matrix()[-1]
        out[:] = v0 * v1

# (((high * low)^0.5) - vwap)
class Alpha41(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    inputs = [USEquityPricing.high, USEquityPricing.low, vwap_in]
    window_length = 1

    def compute(self, today, assets, out, high, low, vwap):
        v000 = high[-1]
        v001 = low[-1]
        v00 = v000 * v001
        v01 = np.full(out.shape[0], 0.5)
        v0 = np.power(v00, v01)
        v1 = vwap[-1]
        out[:] = v0 - v1

# (rank((vwap - close)) / rank((vwap + close)))
class Alpha42(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    inputs = [USEquityPricing.close, vwap_in]
    window_length = 1

    def compute(self, today, assets, out, close, vwap):
        v000 = vwap[-1]
        v001 = close[-1]
        v00 = v000 - v001
        v0 = stats.rankdata(v00)
        v100 = vwap[-1]
        v101 = close[-1]
        v10 = v100 + v101
        v1 = stats.rankdata(v10)
        out[:] = v0 / v1

# (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
class Alpha43(CustomFactor):
    adv20_in = AverageDollarVolume(window_length=20)
    adv20_in.window_safe = True
    inputs = [USEquityPricing.volume, USEquityPricing.close, adv20_in]
    window_length = 20

    def compute(self, today, assets, out, volume, close, adv20):
        v00 = np.empty((20, out.shape[0]))
        for i0 in range(1, 21):
            v000 = volume[-i0]
            v001 = adv20[-i0]
            v00[-i0] = v000 / v001
        v0 = pd.DataFrame(v00).rank().tail(1).as_matrix()[-1]
        v10 = np.empty((8, out.shape[0]))
        for i0 in range(1, 9):
            v100 = np.full(out.shape[0], -1.0)
            v1010 = np.empty((8, out.shape[0]))
            for i1 in range(1, 9):
                v1010[-i1] = close[-i0 - i1]
            v101 = v1010[-1] - v1010[-8]
            v10[-i0] = v100 * v101
        v1 = pd.DataFrame(v10).rank().tail(1).as_matrix()[-1]
        out[:] = v0 * v1

# (-1 * correlation(high, rank(volume), 5))
class Alpha44(CustomFactor):
    inputs = [USEquityPricing.high, USEquityPricing.volume]
    window_length = 5

    def compute(self, today, assets, out, high, volume):
        v0 = np.full(out.shape[0], -1.0)
        v10 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v10[-i0] = high[-i0]
        v11 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v110 = volume[-i0]
            v11[-i0] = stats.rankdata(v110)
        v1 = pd.DataFrame(v10).rolling(window=5).corr(pd.DataFrame(v11)).tail(1).as_matrix()[-1]
        out[:] = v0 * v1

# (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))
class Alpha45(CustomFactor):
    inputs = [USEquityPricing.volume, USEquityPricing.close]
    window_length = 26

    def compute(self, today, assets, out, volume, close):
        v0 = np.full(out.shape[0], -1.0)
        v100000 = np.empty((20, out.shape[0]))
        for i0 in range(1, 21):
            v1000000 = close[-6]
            v100000[-i0] = v1000000  # delay
        v10000 = v100000.sum(axis=0)
        v10001 = np.full(out.shape[0], 20.0)
        v1000 = v10000 / v10001
        v100 = stats.rankdata(v1000)
        v1010 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v1010[-i0] = close[-i0]
        v1011 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v1011[-i0] = volume[-i0]
        v101 = pd.DataFrame(v1010).rolling(window=2).corr(pd.DataFrame(v1011)).tail(1).as_matrix()[-1]
        v10 = v100 * v101
        v1100 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v11000 = np.empty((5, out.shape[0]))
            for i1 in range(1, 6):
                v11000[-i1] = close[-i0 - i1]
            v1100[-i0] = v11000.sum(axis=0)
        v1101 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v11010 = np.empty((20, out.shape[0]))
            for i1 in range(1, 21):
                v11010[-i1] = close[-i0 - i1]
            v1101[-i0] = v11010.sum(axis=0)
        v110 = pd.DataFrame(v1100).rolling(window=2).corr(pd.DataFrame(v1101)).tail(1).as_matrix()[-1]
        v11 = stats.rankdata(v110)
        v1 = v10 * v11
        out[:] = v0 * v1

# ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1)))))
class Alpha46(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 21

    def compute(self, today, assets, out, close):
        v00 = np.full(out.shape[0], 0.25)
        v010000 = close[-21]
        v01000 = v010000  # delay
        v010010 = close[-11]
        v01001 = v010010  # delay
        v0100 = v01000 - v01001
        v0101 = np.full(out.shape[0], 10.0)
        v010 = v0100 / v0101
        v011000 = close[-11]
        v01100 = v011000  # delay
        v01101 = close[-1]
        v0110 = v01100 - v01101
        v0111 = np.full(out.shape[0], 10.0)
        v011 = v0110 / v0111
        v01 = v010 - v011
        v0 = v00 < v01
        v10 = np.full(out.shape[0], -1.0)
        v11 = np.full(out.shape[0], 1.0)
        v1 = v10 * v11
        v2000000 = close[-21]
        v200000 = v2000000  # delay
        v2000010 = close[-11]
        v200001 = v2000010  # delay
        v20000 = v200000 - v200001
        v20001 = np.full(out.shape[0], 10.0)
        v2000 = v20000 / v20001
        v2001000 = close[-11]
        v200100 = v2001000  # delay
        v200101 = close[-1]
        v20010 = v200100 - v200101
        v20011 = np.full(out.shape[0], 10.0)
        v2001 = v20010 / v20011
        v200 = v2000 - v2001
        v201 = np.full(out.shape[0], 0.0)
        v20 = v200 < v201
        v21 = np.full(out.shape[0], 1.0)
        v2200 = np.full(out.shape[0], -1.0)
        v2201 = np.full(out.shape[0], 1.0)
        v220 = v2200 * v2201
        v2210 = close[-1]
        v22110 = close[-2]
        v2211 = v22110  # delay
        v221 = v2210 - v2211
        v22 = v220 * v221
        v2lgcl = np.empty(out.shape[0])
        v2lgcl[v20] = 1
        v2lgcl[~v20] = v22[~v20]
        v2 = v2lgcl
        vlgcl = np.empty(out.shape[0])
        vlgcl[v0] = v1[v0]
        vlgcl[~v0] = v2[~v0]
        out[:] = vlgcl

# ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))
class Alpha47(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    adv20_in = AverageDollarVolume(window_length=20)
    adv20_in.window_safe = True
    inputs = [USEquityPricing.volume, USEquityPricing.close, USEquityPricing.high, adv20_in, vwap_in]
    window_length = 6

    def compute(self, today, assets, out, volume, close, high, adv20, vwap):
        v000000 = np.full(out.shape[0], 1.0)
        v000001 = close[-1]
        v00000 = v000000 / v000001
        v0000 = stats.rankdata(v00000)
        v0001 = volume[-1]
        v000 = v0000 * v0001
        v001 = adv20[-1]
        v00 = v000 / v001
        v0100 = high[-1]
        v010100 = high[-1]
        v010101 = close[-1]
        v01010 = v010100 - v010101
        v0101 = stats.rankdata(v01010)
        v010 = v0100 * v0101
        v01100 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v01100[-i0] = high[-i0]
        v0110 = v01100.sum(axis=0)
        v0111 = np.full(out.shape[0], 5.0)
        v011 = v0110 / v0111
        v01 = v010 / v011
        v0 = v00 * v01
        v100 = vwap[-1]
        v1010 = vwap[-6]
        v101 = v1010  # delay
        v10 = v100 - v101
        v1 = stats.rankdata(v10)
        out[:] = v0 - v1

'''
# (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))
class Alpha48(CustomFactor):
    sub_industry_in = morningstar.asset_classification.morningstar_industry_group_code.latest
    sub_industry_in.window_safe = True
    inputs = [USEquityPricing.close, sub_industry_in]
    window_length = 254

    def compute(self, today, assets, out, close, sub_industry):
        v00000 = np.empty((250, out.shape[0]))
        for i0 in range(1, 251):
            v000000 = np.empty((2, out.shape[0]))
            for i1 in range(1, 3):
                v000000[-i1] = close[-i0 - i1]
            v00000[-i0] = v000000[-1] - v000000[-2]
        v00001 = np.empty((250, out.shape[0]))
        for i0 in range(1, 251):
            v000010 = np.empty((2, out.shape[0]))
            for i1 in range(1, 3):
                v0000100 = close[-2]
                v000010[-i1] = v0000100  # delay
            v00001[-i0] = v000010[-1] - v000010[-2]
        v0000 = pd.DataFrame(v00000).rolling(window=250).corr(pd.DataFrame(v00001)).tail(1).as_matrix()[-1]
        v00010 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v00010[-i0] = close[-i0]
        v0001 = v00010[-1] - v00010[-2]
        v000 = v0000 * v0001
        v001 = close[-1]
        v00 = v000 / v001
        v01 = sub_industry[-1]
        v0 = demean_by_group(v00, v01)
        v10 = np.empty((250, out.shape[0]))
        for i0 in range(1, 251):
            v10000 = np.empty((2, out.shape[0]))
            for i1 in range(1, 3):
                v10000[-i1] = close[-i0 - i1]
            v1000 = v10000[-1] - v10000[-2]
            v10010 = close[-2]
            v1001 = v10010  # delay
            v100 = v1000 / v1001
            v101 = np.full(out.shape[0], 2.0)
            v10[-i0] = np.power(v100, v101)
        v1 = v10.sum(axis=0)
        out[:] = v0 / v1
'''
# (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
class Alpha49(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 21

    def compute(self, today, assets, out, close):
        v000000 = close[-21]
        v00000 = v000000  # delay
        v000010 = close[-11]
        v00001 = v000010  # delay
        v0000 = v00000 - v00001
        v0001 = np.full(out.shape[0], 10.0)
        v000 = v0000 / v0001
        v001000 = close[-11]
        v00100 = v001000  # delay
        v00101 = close[-1]
        v0010 = v00100 - v00101
        v0011 = np.full(out.shape[0], 10.0)
        v001 = v0010 / v0011
        v00 = v000 - v001
        v010 = np.full(out.shape[0], -1.0)
        v011 = np.full(out.shape[0], 0.1)
        v01 = v010 * v011
        v0 = v00 < v01
        v1 = np.full(out.shape[0], 1.0)
        v200 = np.full(out.shape[0], -1.0)
        v201 = np.full(out.shape[0], 1.0)
        v20 = v200 * v201
        v210 = close[-1]
        v2110 = close[-2]
        v211 = v2110  # delay
        v21 = v210 - v211
        v2 = v20 * v21
        vlgcl = np.empty(out.shape[0])
        vlgcl[v0] = 1
        vlgcl[~v0] = v2[~v0]
        out[:] = vlgcl

# (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
class Alpha50(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    inputs = [USEquityPricing.volume, vwap_in]
    window_length = 10

    def compute(self, today, assets, out, volume, vwap):
        v0 = np.full(out.shape[0], -1.0)
        v10 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v1000 = np.empty((5, out.shape[0]))
            for i1 in range(1, 6):
                v10000 = volume[-i0 - i1]
                v1000[-i1] = stats.rankdata(v10000)
            v1001 = np.empty((5, out.shape[0]))
            for i1 in range(1, 6):
                v10010 = vwap[-i0 - i1]
                v1001[-i1] = stats.rankdata(v10010)
            v100 = pd.DataFrame(v1000).rolling(window=5).corr(pd.DataFrame(v1001)).tail(1).as_matrix()[-1]
            v10[-i0] = stats.rankdata(v100)
        v1 = np.max(v10, axis=0)
        out[:] = v0 * v1

# (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
class Alpha51(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 21

    def compute(self, today, assets, out, close):
        v000000 = close[-21]
        v00000 = v000000  # delay
        v000010 = close[-11]
        v00001 = v000010  # delay
        v0000 = v00000 - v00001
        v0001 = np.full(out.shape[0], 10.0)
        v000 = v0000 / v0001
        v001000 = close[-11]
        v00100 = v001000  # delay
        v00101 = close[-1]
        v0010 = v00100 - v00101
        v0011 = np.full(out.shape[0], 10.0)
        v001 = v0010 / v0011
        v00 = v000 - v001
        v010 = np.full(out.shape[0], -1.0)
        v011 = np.full(out.shape[0], 0.05)
        v01 = v010 * v011
        v0 = v00 < v01
        v1 = np.full(out.shape[0], 1.0)
        v200 = np.full(out.shape[0], -1.0)
        v201 = np.full(out.shape[0], 1.0)
        v20 = v200 * v201
        v210 = close[-1]
        v2110 = close[-2]
        v211 = v2110  # delay
        v21 = v210 - v211
        v2 = v20 * v21
        vlgcl = np.empty(out.shape[0])
        vlgcl[v0] = 1
        vlgcl[~v0] = v2[~v0]
        out[:] = vlgcl

# ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))
class Alpha52(CustomFactor):
    inputs = [USEquityPricing.volume, Returns(window_length=2), USEquityPricing.low]
    window_length = 240

    def compute(self, today, assets, out, volume, returns, low):
        v0000 = np.full(out.shape[0], -1.0)
        v00010 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v00010[-i0] = low[-i0]
        v0001 = np.min(v00010, axis=0)
        v000 = v0000 * v0001
        v00100 = np.empty((5, out.shape[0]))
        for i0 in range(6, 11):
            v00100[5 - i0] = low[-i0]
        v0010 = np.min(v00100, axis=0)
        v001 = v0010  # delay
        v00 = v000 + v001
        v010000 = np.empty((240, out.shape[0]))
        for i0 in range(1, 241):
            v010000[-i0] = returns[-i0]
        v01000 = v010000.sum(axis=0)
        v010010 = np.empty((20, out.shape[0]))
        for i0 in range(1, 21):
            v010010[-i0] = returns[-i0]
        v01001 = v010010.sum(axis=0)
        v0100 = v01000 - v01001
        v0101 = np.full(out.shape[0], 220.0)
        v010 = v0100 / v0101
        v01 = stats.rankdata(v010)
        v0 = v00 * v01
        v10 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v10[-i0] = volume[-i0]
        v1 = pd.DataFrame(v10).rank().tail(1).as_matrix()[-1]
        out[:] = v0 * v1

# (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
class Alpha53(CustomFactor):
    inputs = [USEquityPricing.high, USEquityPricing.close, USEquityPricing.low]
    window_length = 10

    def compute(self, today, assets, out, high, close, low):
        v0 = np.full(out.shape[0], -1.0)
        v10 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v10000 = close[-i0]
            v10001 = low[-i0]
            v1000 = v10000 - v10001
            v10010 = high[-i0]
            v10011 = close[-i0]
            v1001 = v10010 - v10011
            v100 = v1000 - v1001
            v1010 = close[-i0]
            v1011 = low[-i0]
            v101 = v1010 - v1011
            v10[-i0] = v100 / v101
        v1 = v10[-1] - v10[-10]
        out[:] = v0 * v1

# ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
class Alpha54(CustomFactor):
    inputs = [USEquityPricing.high, USEquityPricing.close, USEquityPricing.open, USEquityPricing.low]
    window_length = 1

    def compute(self, today, assets, out, high, close, open, low):
        v00 = np.full(out.shape[0], -1.0)
        v0100 = low[-1]
        v0101 = close[-1]
        v010 = v0100 - v0101
        v0110 = open[-1]
        v0111 = np.full(out.shape[0], 5.0)
        v011 = np.power(v0110, v0111)
        v01 = v010 * v011
        v0 = v00 * v01
        v100 = low[-1]
        v101 = high[-1]
        v10 = v100 - v101
        v110 = close[-1]
        v111 = np.full(out.shape[0], 5.0)
        v11 = np.power(v110, v111)
        v1 = v10 * v11
        out[:] = v0 / v1

# (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))
class Alpha55(CustomFactor):
    inputs = [USEquityPricing.high, USEquityPricing.close, USEquityPricing.low, USEquityPricing.volume]
    window_length = 18

    def compute(self, today, assets, out, high, close, low, volume):
        v0 = np.full(out.shape[0], -1.0)
        v10 = np.empty((6, out.shape[0]))
        for i0 in range(1, 7):
            v10000 = close[-i0]
            v100010 = np.empty((12, out.shape[0]))
            for i1 in range(1, 13):
                v100010[-i1] = low[-i0 - i1]
            v10001 = np.min(v100010, axis=0)
            v1000 = v10000 - v10001
            v100100 = np.empty((12, out.shape[0]))
            for i1 in range(1, 13):
                v100100[-i1] = high[-i0 - i1]
            v10010 = np.max(v100100, axis=0)
            v100110 = np.empty((12, out.shape[0]))
            for i1 in range(1, 13):
                v100110[-i1] = low[-i0 - i1]
            v10011 = np.min(v100110, axis=0)
            v1001 = v10010 - v10011
            v100 = v1000 / v1001
            v10[-i0] = stats.rankdata(v100)
        v11 = np.empty((6, out.shape[0]))
        for i0 in range(1, 7):
            v110 = volume[-i0]
            v11[-i0] = stats.rankdata(v110)
        v1 = pd.DataFrame(v10).rolling(window=6).corr(pd.DataFrame(v11)).tail(1).as_matrix()[-1]
        out[:] = v0 * v1

# (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
class Alpha56(CustomFactor):
    inputs = [Returns(window_length=2)]
    window_length = 10

    def compute(self, today, assets, out, returns):
        v0 = np.full(out.shape[0], 0.0)
        v10 = np.full(out.shape[0], 1.0)
        v110000 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v110000[-i0] = returns[-i0]
        v11000 = v110000.sum(axis=0)
        v110010 = np.empty((3, out.shape[0]))
        for i0 in range(1, 4):
            v1100100 = np.empty((2, out.shape[0]))
            for i1 in range(1, 3):
                v1100100[-i1] = returns[-i0 - i1]
            v110010[-i0] = v1100100.sum(axis=0)
        v11001 = v110010.sum(axis=0)
        v1100 = v11000 / v11001
        v110 = stats.rankdata(v1100)
        v11100 = returns[-1]
        v11101 = cap[-1]
        v1110 = v11100 * v11101
        v111 = stats.rankdata(v1110)
        v11 = v110 * v111
        v1 = v10 * v11
        out[:] = v0 - v1

# (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
class Alpha57(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    inputs = [USEquityPricing.close, vwap_in]
    window_length = 32

    def compute(self, today, assets, out, close, vwap):
        v0 = np.full(out.shape[0], 0.0)
        v10 = np.full(out.shape[0], 1.0)
        v1100 = close[-1]
        v1101 = vwap[-1]
        v110 = v1100 - v1101
        v1110 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v111000 = np.empty((30, out.shape[0]))
            for i1 in range(1, 31):
                v111000[-i1] = close[-i0 - i1]
            v11100 = np.argmax(v111000, axis=0)
            v1110[-i0] = stats.rankdata(v11100)
        v111 = (v1110 * (np.arange(1.0, 3, 1.0) / 3)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v11 = v110 / v111
        v1 = v10 * v11
        out[:] = v0 - v1

'''
# (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))
class Alpha58(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    sectors_in = morningstar.asset_classification.morningstar_sector_code.latest
    sectors_in.window_safe = True
    inputs = [USEquityPricing.volume, sectors_in, vwap_in]
    window_length = 17

    def compute(self, today, assets, out, volume, sectors, vwap):
        v0 = np.full(out.shape[0], -1.0)
        v10 = np.empty((6, out.shape[0]))
        for i0 in range(1, 7):
            v100 = np.empty((8, out.shape[0]))
            for i1 in range(1, 9):
                v1000 = np.empty((4, out.shape[0]))
                for i2 in range(1, 5):
                    v10000 = vwap[-i0 - i1 - i2]
                    v10001 = sectors[-i0 - i1 - i2]
                    v1000[-i2] = demean_by_group(v10000, v10001)
                v1001 = np.empty((4, out.shape[0]))
                for i2 in range(1, 5):
                    v1001[-i2] = volume[-i0 - i1 - i2]
                v100[-i1] = pd.DataFrame(v1000).rolling(window=4).corr(pd.DataFrame(v1001)).tail(1).as_matrix()[-1]
            v10[-i0] = (v100 * (np.arange(1.0, 9, 1.0) / 36)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v1 = pd.DataFrame(v10).rank().tail(1).as_matrix()[-1]
        out[:] = v0 * v1
'''
'''
# (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))
class Alpha59(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    industry_in = morningstar.asset_classification.morningstar_industry_code.latest
    industry_in.window_safe = True
    inputs = [USEquityPricing.volume, industry_in, vwap_in]
    window_length = 28

    def compute(self, today, assets, out, volume, industry, vwap):
        v0 = np.full(out.shape[0], -1.0)
        v10 = np.empty((8, out.shape[0]))
        for i0 in range(1, 9):
            v100 = np.empty((16, out.shape[0]))
            for i1 in range(1, 17):
                v1000 = np.empty((4, out.shape[0]))
                for i2 in range(1, 5):
                    v1000000 = vwap[-i0 - i1 - i2]
                    v1000001 = np.full(out.shape[0], 0.728317)
                    v100000 = v1000000 * v1000001
                    v1000010 = vwap[-i0 - i1 - i2]
                    v10000110 = np.full(out.shape[0], 1.0)
                    v10000111 = np.full(out.shape[0], 0.728317)
                    v1000011 = v10000110 - v10000111
                    v100001 = v1000010 * v1000011
                    v10000 = v100000 + v100001
                    v10001 = industry[-i0 - i1 - i2]
                    v1000[-i2] = demean_by_group(v10000, v10001)
                v1001 = np.empty((4, out.shape[0]))
                for i2 in range(1, 5):
                    v1001[-i2] = volume[-i0 - i1 - i2]
                v100[-i1] = pd.DataFrame(v1000).rolling(window=4).corr(pd.DataFrame(v1001)).tail(1).as_matrix()[-1]
            v10[-i0] = (v100 * (np.arange(1.0, 17, 1.0) / 136)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v1 = pd.DataFrame(v10).rank().tail(1).as_matrix()[-1]
        out[:] = v0 * v1
'''
# (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))
class Alpha60(CustomFactor):
    inputs = [USEquityPricing.high, USEquityPricing.close, USEquityPricing.low, USEquityPricing.volume]
    window_length = 10

    def compute(self, today, assets, out, high, close, low, volume):
        v0 = np.full(out.shape[0], 0.0)
        v10 = np.full(out.shape[0], 1.0)
        v1100 = np.full(out.shape[0], 2.0)
        v1101000000 = close[-1]
        v1101000001 = low[-1]
        v110100000 = v1101000000 - v1101000001
        v1101000010 = high[-1]
        v1101000011 = close[-1]
        v110100001 = v1101000010 - v1101000011
        v11010000 = v110100000 - v110100001
        v110100010 = high[-1]
        v110100011 = low[-1]
        v11010001 = v110100010 - v110100011
        v1101000 = v11010000 / v11010001
        v1101001 = volume[-1]
        v110100 = v1101000 * v1101001
        v11010 = stats.rankdata(v110100)
        v1101 = v11010 / np.abs(v11010).sum()
        v110 = v1100 * v1101
        v111000 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v111000[-i0] = close[-i0]
        v11100 = np.argmax(v111000, axis=0)
        v1110 = stats.rankdata(v11100)
        v111 = v1110 / np.abs(v1110).sum()
        v11 = v110 - v111
        v1 = v10 * v11
        out[:] = v0 - v1

# (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
class Alpha61(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    adv180_in = AverageDollarVolume(window_length=180)
    adv180_in.window_safe = True
    inputs = [adv180_in, vwap_in]
    window_length = 17

    def compute(self, today, assets, out, adv180, vwap):
        v000 = vwap[-1]
        v0010 = np.empty((16, out.shape[0]))
        for i0 in range(1, 17):
            v0010[-i0] = vwap[-i0]
        v001 = np.min(v0010, axis=0)
        v00 = v000 - v001
        v0 = stats.rankdata(v00)
        v100 = np.empty((18, out.shape[0]))
        for i0 in range(1, 19):
            v100[-i0] = vwap[-i0]
        v101 = np.empty((18, out.shape[0]))
        for i0 in range(1, 19):
            v101[-i0] = adv180[-i0]
        v10 = pd.DataFrame(v100).rolling(window=18).corr(pd.DataFrame(v101)).tail(1).as_matrix()[-1]
        v1 = stats.rankdata(v10)
        out[:] = v0 < v1

# ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
class Alpha62(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    adv20_in = AverageDollarVolume(window_length=20)
    adv20_in.window_safe = True
    inputs = [USEquityPricing.high, USEquityPricing.open, adv20_in, USEquityPricing.low, vwap_in]
    window_length = 32

    def compute(self, today, assets, out, high, open, adv20, low, vwap):
        v0000 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v0000[-i0] = vwap[-i0]
        v0001 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v00010 = np.empty((22, out.shape[0]))
            for i1 in range(1, 23):
                v00010[-i1] = adv20[-i0 - i1]
            v0001[-i0] = v00010.sum(axis=0)
        v000 = pd.DataFrame(v0000).rolling(window=10).corr(pd.DataFrame(v0001)).tail(1).as_matrix()[-1]
        v00 = stats.rankdata(v000)
        v010000 = open[-1]
        v01000 = stats.rankdata(v010000)
        v010010 = open[-1]
        v01001 = stats.rankdata(v010010)
        v0100 = v01000 + v01001
        v01010000 = high[-1]
        v01010001 = low[-1]
        v0101000 = v01010000 + v01010001
        v0101001 = np.full(out.shape[0], 2.0)
        v010100 = v0101000 / v0101001
        v01010 = stats.rankdata(v010100)
        v010110 = high[-1]
        v01011 = stats.rankdata(v010110)
        v0101 = v01010 + v01011
        v010 = v0100 < v0101
        v01 = stats.rankdata(v010)
        v0 = v00 < v01
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1

'''
# ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1)
class Alpha63(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    industry_in = morningstar.asset_classification.morningstar_industry_code.latest
    industry_in.window_safe = True
    adv180_in = AverageDollarVolume(window_length=180)
    adv180_in.window_safe = True
    inputs = [USEquityPricing.close, industry_in, USEquityPricing.open, adv180_in, vwap_in]
    window_length = 63

    def compute(self, today, assets, out, close, industry, open, adv180, vwap):
        v0000 = np.empty((8, out.shape[0]))
        for i0 in range(1, 9):
            v00000 = np.empty((3, out.shape[0]))
            for i1 in range(1, 4):
                v000000 = close[-i0 - i1]
                v000001 = industry[-i0 - i1]
                v00000[-i1] = demean_by_group(v000000, v000001)
            v0000[-i0] = v00000[-1] - v00000[-3]
        v000 = (v0000 * (np.arange(1.0, 9, 1.0) / 36)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v00 = stats.rankdata(v000)
        v0100 = np.empty((12, out.shape[0]))
        for i0 in range(1, 13):
            v01000 = np.empty((14, out.shape[0]))
            for i1 in range(1, 15):
                v0100000 = vwap[-i0 - i1]
                v0100001 = np.full(out.shape[0], 0.318108)
                v010000 = v0100000 * v0100001
                v0100010 = open[-i0 - i1]
                v01000110 = np.full(out.shape[0], 1.0)
                v01000111 = np.full(out.shape[0], 0.318108)
                v0100011 = v01000110 - v01000111
                v010001 = v0100010 * v0100011
                v01000[-i1] = v010000 + v010001
            v01001 = np.empty((14, out.shape[0]))
            for i1 in range(1, 15):
                v010010 = np.empty((37, out.shape[0]))
                for i2 in range(1, 38):
                    v010010[-i2] = adv180[-i0 - i1 - i2]
                v01001[-i1] = v010010.sum(axis=0)
            v0100[-i0] = pd.DataFrame(v01000).rolling(window=14).corr(pd.DataFrame(v01001)).tail(1).as_matrix()[-1]
        v010 = (v0100 * (np.arange(1.0, 13, 1.0) / 78)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v01 = stats.rankdata(v010)
        v0 = v00 - v01
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1
'''
# ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)
class Alpha64(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    adv120_in = AverageDollarVolume(window_length=120)
    adv120_in.window_safe = True
    inputs = [adv120_in, USEquityPricing.high, USEquityPricing.open, USEquityPricing.low, vwap_in]
    window_length = 29

    def compute(self, today, assets, out, adv120, high, open, low, vwap):
        v0000 = np.empty((17, out.shape[0]))
        for i0 in range(1, 18):
            v00000 = np.empty((13, out.shape[0]))
            for i1 in range(1, 14):
                v0000000 = open[-i0 - i1]
                v0000001 = np.full(out.shape[0], 0.178404)
                v000000 = v0000000 * v0000001
                v0000010 = low[-i0 - i1]
                v00000110 = np.full(out.shape[0], 1.0)
                v00000111 = np.full(out.shape[0], 0.178404)
                v0000011 = v00000110 - v00000111
                v000001 = v0000010 * v0000011
                v00000[-i1] = v000000 + v000001
            v0000[-i0] = v00000.sum(axis=0)
        v0001 = np.empty((17, out.shape[0]))
        for i0 in range(1, 18):
            v00010 = np.empty((13, out.shape[0]))
            for i1 in range(1, 14):
                v00010[-i1] = adv120[-i0 - i1]
            v0001[-i0] = v00010.sum(axis=0)
        v000 = pd.DataFrame(v0000).rolling(window=17).corr(pd.DataFrame(v0001)).tail(1).as_matrix()[-1]
        v00 = stats.rankdata(v000)
        v0100 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v01000000 = high[-i0]
            v01000001 = low[-i0]
            v0100000 = v01000000 + v01000001
            v0100001 = np.full(out.shape[0], 2.0)
            v010000 = v0100000 / v0100001
            v010001 = np.full(out.shape[0], 0.178404)
            v01000 = v010000 * v010001
            v010010 = vwap[-i0]
            v0100110 = np.full(out.shape[0], 1.0)
            v0100111 = np.full(out.shape[0], 0.178404)
            v010011 = v0100110 - v0100111
            v01001 = v010010 * v010011
            v0100[-i0] = v01000 + v01001
        v010 = v0100[-1] - v0100[-5]
        v01 = stats.rankdata(v010)
        v0 = v00 < v01
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1

# ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
class Alpha65(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    adv60_in = AverageDollarVolume(window_length=60)
    adv60_in.window_safe = True
    inputs = [adv60_in, USEquityPricing.open, vwap_in]
    window_length = 15

    def compute(self, today, assets, out, adv60, open, vwap):
        v0000 = np.empty((6, out.shape[0]))
        for i0 in range(1, 7):
            v000000 = open[-i0]
            v000001 = np.full(out.shape[0], 0.00817205)
            v00000 = v000000 * v000001
            v000010 = vwap[-i0]
            v0000110 = np.full(out.shape[0], 1.0)
            v0000111 = np.full(out.shape[0], 0.00817205)
            v000011 = v0000110 - v0000111
            v00001 = v000010 * v000011
            v0000[-i0] = v00000 + v00001
        v0001 = np.empty((6, out.shape[0]))
        for i0 in range(1, 7):
            v00010 = np.empty((9, out.shape[0]))
            for i1 in range(1, 10):
                v00010[-i1] = adv60[-i0 - i1]
            v0001[-i0] = v00010.sum(axis=0)
        v000 = pd.DataFrame(v0000).rolling(window=6).corr(pd.DataFrame(v0001)).tail(1).as_matrix()[-1]
        v00 = stats.rankdata(v000)
        v0100 = open[-1]
        v01010 = np.empty((14, out.shape[0]))
        for i0 in range(1, 15):
            v01010[-i0] = open[-i0]
        v0101 = np.min(v01010, axis=0)
        v010 = v0100 - v0101
        v01 = stats.rankdata(v010)
        v0 = v00 < v01
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1

# ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
class Alpha66(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    inputs = [USEquityPricing.high, USEquityPricing.open, USEquityPricing.low, vwap_in]
    window_length = 18

    def compute(self, today, assets, out, high, open, low, vwap):
        v0000 = np.empty((7, out.shape[0]))
        for i0 in range(1, 8):
            v00000 = np.empty((5, out.shape[0]))
            for i1 in range(1, 6):
                v00000[-i1] = vwap[-i0 - i1]
            v0000[-i0] = v00000[-1] - v00000[-5]
        v000 = (v0000 * (np.arange(1.0, 8, 1.0) / 28)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v00 = stats.rankdata(v000)
        v010 = np.empty((7, out.shape[0]))
        for i0 in range(1, 8):
            v0100 = np.empty((11, out.shape[0]))
            for i1 in range(1, 12):
                v01000000 = low[-i0 - i1]
                v01000001 = np.full(out.shape[0], 0.96633)
                v0100000 = v01000000 * v01000001
                v01000010 = low[-i0 - i1]
                v010000110 = np.full(out.shape[0], 1.0)
                v010000111 = np.full(out.shape[0], 0.96633)
                v01000011 = v010000110 - v010000111
                v0100001 = v01000010 * v01000011
                v010000 = v0100000 + v0100001
                v010001 = vwap[-i0 - i1]
                v01000 = v010000 - v010001
                v010010 = open[-i0 - i1]
                v01001100 = high[-i0 - i1]
                v01001101 = low[-i0 - i1]
                v0100110 = v01001100 + v01001101
                v0100111 = np.full(out.shape[0], 2.0)
                v010011 = v0100110 / v0100111
                v01001 = v010010 - v010011
                v0100[-i1] = v01000 / v01001
            v010[-i0] = (v0100 * (np.arange(1.0, 12, 1.0) / 66)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v01 = pd.DataFrame(v010).rank().tail(1).as_matrix()[-1]
        v0 = v00 + v01
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1

'''
# ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)
class Alpha67(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    sub_industry_in = morningstar.asset_classification.morningstar_industry_group_code.latest
    sub_industry_in.window_safe = True
    adv20_in = AverageDollarVolume(window_length=20)
    adv20_in.window_safe = True
    sectors_in = morningstar.asset_classification.morningstar_sector_code.latest
    sectors_in.window_safe = True
    inputs = [USEquityPricing.high, sub_industry_in, adv20_in, sectors_in, vwap_in]
    window_length = 6

    def compute(self, today, assets, out, high, sub_industry, adv20, sectors, vwap):
        v0000 = high[-1]
        v00010 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v00010[-i0] = high[-i0]
        v0001 = np.min(v00010, axis=0)
        v000 = v0000 - v0001
        v00 = stats.rankdata(v000)
        v0100 = np.empty((6, out.shape[0]))
        for i0 in range(1, 7):
            v01000 = vwap[-i0]
            v01001 = sectors[-i0]
            v0100[-i0] = demean_by_group(v01000, v01001)
        v0101 = np.empty((6, out.shape[0]))
        for i0 in range(1, 7):
            v01010 = adv20[-i0]
            v01011 = sub_industry[-i0]
            v0101[-i0] = demean_by_group(v01010, v01011)
        v010 = pd.DataFrame(v0100).rolling(window=6).corr(pd.DataFrame(v0101)).tail(1).as_matrix()[-1]
        v01 = stats.rankdata(v010)
        v0 = np.power(v00, v01)
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1
'''
# ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
class Alpha68(CustomFactor):
    adv15_in = AverageDollarVolume(window_length=15)
    adv15_in.window_safe = True
    inputs = [USEquityPricing.high, adv15_in, USEquityPricing.low, USEquityPricing.close]
    window_length = 22

    def compute(self, today, assets, out, high, adv15, low, close):
        v000 = np.empty((14, out.shape[0]))
        for i0 in range(1, 15):
            v0000 = np.empty((9, out.shape[0]))
            for i1 in range(1, 10):
                v00000 = high[-i0 - i1]
                v0000[-i1] = stats.rankdata(v00000)
            v0001 = np.empty((9, out.shape[0]))
            for i1 in range(1, 10):
                v00010 = adv15[-i0 - i1]
                v0001[-i1] = stats.rankdata(v00010)
            v000[-i0] = pd.DataFrame(v0000).rolling(window=9).corr(pd.DataFrame(v0001)).tail(1).as_matrix()[-1]
        v00 = pd.DataFrame(v000).rank().tail(1).as_matrix()[-1]
        v0100 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v010000 = close[-i0]
            v010001 = np.full(out.shape[0], 0.518371)
            v01000 = v010000 * v010001
            v010010 = low[-i0]
            v0100110 = np.full(out.shape[0], 1.0)
            v0100111 = np.full(out.shape[0], 0.518371)
            v010011 = v0100110 - v0100111
            v01001 = v010010 * v010011
            v0100[-i0] = v01000 + v01001
        v010 = v0100[-1] - v0100[-2]
        v01 = stats.rankdata(v010)
        v0 = v00 < v01
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1

'''
# ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1)
class Alpha69(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    industry_in = morningstar.asset_classification.morningstar_industry_code.latest
    industry_in.window_safe = True
    adv20_in = AverageDollarVolume(window_length=20)
    adv20_in.window_safe = True
    inputs = [USEquityPricing.close, industry_in, adv20_in, vwap_in]
    window_length = 13

    def compute(self, today, assets, out, close, industry, adv20, vwap):
        v0000 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v00000 = np.empty((4, out.shape[0]))
            for i1 in range(1, 5):
                v000000 = vwap[-i0 - i1]
                v000001 = industry[-i0 - i1]
                v00000[-i1] = demean_by_group(v000000, v000001)
            v0000[-i0] = v00000[-1] - v00000[-4]
        v000 = np.max(v0000, axis=0)
        v00 = stats.rankdata(v000)
        v010 = np.empty((9, out.shape[0]))
        for i0 in range(1, 10):
            v0100 = np.empty((5, out.shape[0]))
            for i1 in range(1, 6):
                v010000 = close[-i0 - i1]
                v010001 = np.full(out.shape[0], 0.490655)
                v01000 = v010000 * v010001
                v010010 = vwap[-i0 - i1]
                v0100110 = np.full(out.shape[0], 1.0)
                v0100111 = np.full(out.shape[0], 0.490655)
                v010011 = v0100110 - v0100111
                v01001 = v010010 * v010011
                v0100[-i1] = v01000 + v01001
            v0101 = np.empty((5, out.shape[0]))
            for i1 in range(1, 6):
                v0101[-i1] = adv20[-i0 - i1]
            v010[-i0] = pd.DataFrame(v0100).rolling(window=5).corr(pd.DataFrame(v0101)).tail(1).as_matrix()[-1]
        v01 = pd.DataFrame(v010).rank().tail(1).as_matrix()[-1]
        v0 = np.power(v00, v01)
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1
'''
'''
# ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1)
class Alpha70(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    industry_in = morningstar.asset_classification.morningstar_industry_code.latest
    industry_in.window_safe = True
    adv50_in = AverageDollarVolume(window_length=50)
    adv50_in.window_safe = True
    inputs = [USEquityPricing.close, industry_in, vwap_in, adv50_in]
    window_length = 35

    def compute(self, today, assets, out, close, industry, vwap, adv50):
        v0000 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v0000[-i0] = vwap[-i0]
        v000 = v0000[-1] - v0000[-2]
        v00 = stats.rankdata(v000)
        v010 = np.empty((18, out.shape[0]))
        for i0 in range(1, 19):
            v0100 = np.empty((18, out.shape[0]))
            for i1 in range(1, 19):
                v01000 = close[-i0 - i1]
                v01001 = industry[-i0 - i1]
                v0100[-i1] = demean_by_group(v01000, v01001)
            v0101 = np.empty((18, out.shape[0]))
            for i1 in range(1, 19):
                v0101[-i1] = adv50[-i0 - i1]
            v010[-i0] = pd.DataFrame(v0100).rolling(window=18).corr(pd.DataFrame(v0101)).tail(1).as_matrix()[-1]
        v01 = pd.DataFrame(v010).rank().tail(1).as_matrix()[-1]
        v0 = np.power(v00, v01)
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1
'''
# max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))
class Alpha71(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    adv180_in = AverageDollarVolume(window_length=180)
    adv180_in.window_safe = True
    inputs = [USEquityPricing.close, vwap_in, USEquityPricing.open, USEquityPricing.low, adv180_in]
    window_length = 49

    def compute(self, today, assets, out, close, vwap, open, low, adv180):
        v00 = np.empty((16, out.shape[0]))
        for i0 in range(1, 17):
            v000 = np.empty((4, out.shape[0]))
            for i1 in range(1, 5):
                v0000 = np.empty((18, out.shape[0]))
                for i2 in range(1, 19):
                    v00000 = np.empty((3, out.shape[0]))
                    for i3 in range(1, 4):
                        v00000[-i3] = close[-i0 - i1 - i2 - i3]
                    v0000[-i2] = pd.DataFrame(v00000).rank().tail(1).as_matrix()[-1]
                v0001 = np.empty((18, out.shape[0]))
                for i2 in range(1, 19):
                    v00010 = np.empty((12, out.shape[0]))
                    for i3 in range(1, 13):
                        v00010[-i3] = adv180[-i0 - i1 - i2 - i3]
                    v0001[-i2] = pd.DataFrame(v00010).rank().tail(1).as_matrix()[-1]
                v000[-i1] = pd.DataFrame(v0000).rolling(window=18).corr(pd.DataFrame(v0001)).tail(1).as_matrix()[-1]
            v00[-i0] = (v000 * (np.arange(1.0, 5, 1.0) / 10)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v0 = pd.DataFrame(v00).rank().tail(1).as_matrix()[-1]
        v10 = np.empty((4, out.shape[0]))
        for i0 in range(1, 5):
            v100 = np.empty((16, out.shape[0]))
            for i1 in range(1, 17):
                v1000000 = low[-i0 - i1]
                v1000001 = open[-i0 - i1]
                v100000 = v1000000 + v1000001
                v1000010 = vwap[-i0 - i1]
                v1000011 = vwap[-i0 - i1]
                v100001 = v1000010 + v1000011
                v10000 = v100000 - v100001
                v1000 = stats.rankdata(v10000)
                v1001 = np.full(out.shape[0], 2.0)
                v100[-i1] = np.power(v1000, v1001)
            v10[-i0] = (v100 * (np.arange(1.0, 17, 1.0) / 136)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v1 = pd.DataFrame(v10).rank().tail(1).as_matrix()[-1]
        out[:] = np.maximum(v0, v1)

# (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))
class Alpha72(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    adv40_in = AverageDollarVolume(window_length=40)
    adv40_in.window_safe = True
    inputs = [USEquityPricing.high, USEquityPricing.volume, adv40_in, USEquityPricing.low, vwap_in]
    window_length = 28

    def compute(self, today, assets, out, high, volume, adv40, low, vwap):
        v000 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v0000 = np.empty((9, out.shape[0]))
            for i1 in range(1, 10):
                v000000 = high[-i0 - i1]
                v000001 = low[-i0 - i1]
                v00000 = v000000 + v000001
                v00001 = np.full(out.shape[0], 2.0)
                v0000[-i1] = v00000 / v00001
            v0001 = np.empty((9, out.shape[0]))
            for i1 in range(1, 10):
                v0001[-i1] = adv40[-i0 - i1]
            v000[-i0] = pd.DataFrame(v0000).rolling(window=9).corr(pd.DataFrame(v0001)).tail(1).as_matrix()[-1]
        v00 = (v000 * (np.arange(1.0, 11, 1.0) / 55)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v0 = stats.rankdata(v00)
        v100 = np.empty((3, out.shape[0]))
        for i0 in range(1, 4):
            v1000 = np.empty((7, out.shape[0]))
            for i1 in range(1, 8):
                v10000 = np.empty((4, out.shape[0]))
                for i2 in range(1, 5):
                    v10000[-i2] = vwap[-i0 - i1 - i2]
                v1000[-i1] = pd.DataFrame(v10000).rank().tail(1).as_matrix()[-1]
            v1001 = np.empty((7, out.shape[0]))
            for i1 in range(1, 8):
                v10010 = np.empty((19, out.shape[0]))
                for i2 in range(1, 20):
                    v10010[-i2] = volume[-i0 - i1 - i2]
                v1001[-i1] = pd.DataFrame(v10010).rank().tail(1).as_matrix()[-1]
            v100[-i0] = pd.DataFrame(v1000).rolling(window=7).corr(pd.DataFrame(v1001)).tail(1).as_matrix()[-1]
        v10 = (v100 * (np.arange(1.0, 4, 1.0) / 6)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v1 = stats.rankdata(v10)
        out[:] = v0 / v1

# (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
class Alpha73(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    inputs = [USEquityPricing.open, USEquityPricing.low, vwap_in]
    window_length = 23

    def compute(self, today, assets, out, open, low, vwap):
        v0000 = np.empty((3, out.shape[0]))
        for i0 in range(1, 4):
            v00000 = np.empty((6, out.shape[0]))
            for i1 in range(1, 7):
                v00000[-i1] = vwap[-i0 - i1]
            v0000[-i0] = v00000[-1] - v00000[-6]
        v000 = (v0000 * (np.arange(1.0, 4, 1.0) / 6)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v00 = stats.rankdata(v000)
        v010 = np.empty((17, out.shape[0]))
        for i0 in range(1, 18):
            v0100 = np.empty((3, out.shape[0]))
            for i1 in range(1, 4):
                v0100000 = np.empty((3, out.shape[0]))
                for i2 in range(1, 4):
                    v010000000 = open[-i0 - i1 - i2]
                    v010000001 = np.full(out.shape[0], 0.147155)
                    v01000000 = v010000000 * v010000001
                    v010000010 = low[-i0 - i1 - i2]
                    v0100000110 = np.full(out.shape[0], 1.0)
                    v0100000111 = np.full(out.shape[0], 0.147155)
                    v010000011 = v0100000110 - v0100000111
                    v01000001 = v010000010 * v010000011
                    v0100000[-i2] = v01000000 + v01000001
                v010000 = v0100000[-1] - v0100000[-3]
                v01000100 = open[-i0 - i1]
                v01000101 = np.full(out.shape[0], 0.147155)
                v0100010 = v01000100 * v01000101
                v01000110 = low[-i0 - i1]
                v010001110 = np.full(out.shape[0], 1.0)
                v010001111 = np.full(out.shape[0], 0.147155)
                v01000111 = v010001110 - v010001111
                v0100011 = v01000110 * v01000111
                v010001 = v0100010 + v0100011
                v01000 = v010000 / v010001
                v01001 = np.full(out.shape[0], -1.0)
                v0100[-i1] = v01000 * v01001
            v010[-i0] = (v0100 * (np.arange(1.0, 4, 1.0) / 6)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v01 = pd.DataFrame(v010).rank().tail(1).as_matrix()[-1]
        v0 = np.maximum(v00, v01)
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1

# ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)
class Alpha74(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    adv30_in = AverageDollarVolume(window_length=30)
    adv30_in.window_safe = True
    inputs = [USEquityPricing.high, USEquityPricing.close, adv30_in, vwap_in, USEquityPricing.volume]
    window_length = 52

    def compute(self, today, assets, out, high, close, adv30, vwap, volume):
        v0000 = np.empty((15, out.shape[0]))
        for i0 in range(1, 16):
            v0000[-i0] = close[-i0]
        v0001 = np.empty((15, out.shape[0]))
        for i0 in range(1, 16):
            v00010 = np.empty((37, out.shape[0]))
            for i1 in range(1, 38):
                v00010[-i1] = adv30[-i0 - i1]
            v0001[-i0] = v00010.sum(axis=0)
        v000 = pd.DataFrame(v0000).rolling(window=15).corr(pd.DataFrame(v0001)).tail(1).as_matrix()[-1]
        v00 = stats.rankdata(v000)
        v0100 = np.empty((11, out.shape[0]))
        for i0 in range(1, 12):
            v0100000 = high[-i0]
            v0100001 = np.full(out.shape[0], 0.0261661)
            v010000 = v0100000 * v0100001
            v0100010 = vwap[-i0]
            v01000110 = np.full(out.shape[0], 1.0)
            v01000111 = np.full(out.shape[0], 0.0261661)
            v0100011 = v01000110 - v01000111
            v010001 = v0100010 * v0100011
            v01000 = v010000 + v010001
            v0100[-i0] = stats.rankdata(v01000)
        v0101 = np.empty((11, out.shape[0]))
        for i0 in range(1, 12):
            v01010 = volume[-i0]
            v0101[-i0] = stats.rankdata(v01010)
        v010 = pd.DataFrame(v0100).rolling(window=11).corr(pd.DataFrame(v0101)).tail(1).as_matrix()[-1]
        v01 = stats.rankdata(v010)
        v0 = v00 < v01
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1

# (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))
class Alpha75(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    adv50_in = AverageDollarVolume(window_length=50)
    adv50_in.window_safe = True
    inputs = [USEquityPricing.volume, adv50_in, USEquityPricing.low, vwap_in]
    window_length = 12

    def compute(self, today, assets, out, volume, adv50, low, vwap):
        v000 = np.empty((4, out.shape[0]))
        for i0 in range(1, 5):
            v000[-i0] = vwap[-i0]
        v001 = np.empty((4, out.shape[0]))
        for i0 in range(1, 5):
            v001[-i0] = volume[-i0]
        v00 = pd.DataFrame(v000).rolling(window=4).corr(pd.DataFrame(v001)).tail(1).as_matrix()[-1]
        v0 = stats.rankdata(v00)
        v100 = np.empty((12, out.shape[0]))
        for i0 in range(1, 13):
            v1000 = low[-i0]
            v100[-i0] = stats.rankdata(v1000)
        v101 = np.empty((12, out.shape[0]))
        for i0 in range(1, 13):
            v1010 = adv50[-i0]
            v101[-i0] = stats.rankdata(v1010)
        v10 = pd.DataFrame(v100).rolling(window=12).corr(pd.DataFrame(v101)).tail(1).as_matrix()[-1]
        v1 = stats.rankdata(v10)
        out[:] = v0 < v1

'''
# (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81, 8.14941), 19.569), 17.1543), 19.383)) * -1)
class Alpha76(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    sectors_in = morningstar.asset_classification.morningstar_sector_code.latest
    sectors_in.window_safe = True
    adv81_in = AverageDollarVolume(window_length=81)
    adv81_in.window_safe = True
    inputs = [sectors_in, USEquityPricing.low, vwap_in, adv81_in]
    window_length = 64

    def compute(self, today, assets, out, sectors, low, vwap, adv81):
        v0000 = np.empty((12, out.shape[0]))
        for i0 in range(1, 13):
            v00000 = np.empty((2, out.shape[0]))
            for i1 in range(1, 3):
                v00000[-i1] = vwap[-i0 - i1]
            v0000[-i0] = v00000[-1] - v00000[-2]
        v000 = (v0000 * (np.arange(1.0, 13, 1.0) / 78)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v00 = stats.rankdata(v000)
        v010 = np.empty((19, out.shape[0]))
        for i0 in range(1, 20):
            v0100 = np.empty((17, out.shape[0]))
            for i1 in range(1, 18):
                v01000 = np.empty((20, out.shape[0]))
                for i2 in range(1, 21):
                    v010000 = np.empty((8, out.shape[0]))
                    for i3 in range(1, 9):
                        v0100000 = low[-i0 - i1 - i2 - i3]
                        v0100001 = sectors[-i0 - i1 - i2 - i3]
                        v010000[-i3] = demean_by_group(v0100000, v0100001)
                    v010001 = np.empty((8, out.shape[0]))
                    for i3 in range(1, 9):
                        v010001[-i3] = adv81[-i0 - i1 - i2 - i3]
                    v01000[-i2] = \
                    pd.DataFrame(v010000).rolling(window=8).corr(pd.DataFrame(v010001)).tail(1).as_matrix()[-1]
                v0100[-i1] = pd.DataFrame(v01000).rank().tail(1).as_matrix()[-1]
            v010[-i0] = (v0100 * (np.arange(1.0, 18, 1.0) / 153)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v01 = pd.DataFrame(v010).rank().tail(1).as_matrix()[-1]
        v0 = np.maximum(v00, v01)
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1
'''

# min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
class Alpha77(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    adv40_in = AverageDollarVolume(window_length=40)
    adv40_in.window_safe = True
    inputs = [USEquityPricing.high, adv40_in, USEquityPricing.low, vwap_in]
    window_length = 20

    def compute(self, today, assets, out, high, adv40, low, vwap):
        v000 = np.empty((20, out.shape[0]))
        for i0 in range(1, 21):
            v0000000 = high[-i0]
            v0000001 = low[-i0]
            v000000 = v0000000 + v0000001
            v000001 = np.full(out.shape[0], 2.0)
            v00000 = v000000 / v000001
            v00001 = high[-i0]
            v0000 = v00000 + v00001
            v00010 = vwap[-i0]
            v00011 = high[-i0]
            v0001 = v00010 + v00011
            v000[-i0] = v0000 - v0001
        v00 = (v000 * (np.arange(1.0, 21, 1.0) / 210)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v0 = stats.rankdata(v00)
        v100 = np.empty((6, out.shape[0]))
        for i0 in range(1, 7):
            v1000 = np.empty((3, out.shape[0]))
            for i1 in range(1, 4):
                v100000 = high[-i0 - i1]
                v100001 = low[-i0 - i1]
                v10000 = v100000 + v100001
                v10001 = np.full(out.shape[0], 2.0)
                v1000[-i1] = v10000 / v10001
            v1001 = np.empty((3, out.shape[0]))
            for i1 in range(1, 4):
                v1001[-i1] = adv40[-i0 - i1]
            v100[-i0] = pd.DataFrame(v1000).rolling(window=3).corr(pd.DataFrame(v1001)).tail(1).as_matrix()[-1]
        v10 = (v100 * (np.arange(1.0, 7, 1.0) / 21)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v1 = stats.rankdata(v10)
        out[:] = np.minimum(v0, v1)

# (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
class Alpha78(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    adv40_in = AverageDollarVolume(window_length=40)
    adv40_in.window_safe = True
    inputs = [USEquityPricing.volume, adv40_in, USEquityPricing.low, vwap_in]
    window_length = 26

    def compute(self, today, assets, out, volume, adv40, low, vwap):
        v000 = np.empty((7, out.shape[0]))
        for i0 in range(1, 8):
            v0000 = np.empty((20, out.shape[0]))
            for i1 in range(1, 21):
                v000000 = low[-i0 - i1]
                v000001 = np.full(out.shape[0], 0.352233)
                v00000 = v000000 * v000001
                v000010 = vwap[-i0 - i1]
                v0000110 = np.full(out.shape[0], 1.0)
                v0000111 = np.full(out.shape[0], 0.352233)
                v000011 = v0000110 - v0000111
                v00001 = v000010 * v000011
                v0000[-i1] = v00000 + v00001
            v000[-i0] = v0000.sum(axis=0)
        v001 = np.empty((7, out.shape[0]))
        for i0 in range(1, 8):
            v0010 = np.empty((20, out.shape[0]))
            for i1 in range(1, 21):
                v0010[-i1] = adv40[-i0 - i1]
            v001[-i0] = v0010.sum(axis=0)
        v00 = pd.DataFrame(v000).rolling(window=7).corr(pd.DataFrame(v001)).tail(1).as_matrix()[-1]
        v0 = stats.rankdata(v00)
        v100 = np.empty((6, out.shape[0]))
        for i0 in range(1, 7):
            v1000 = vwap[-i0]
            v100[-i0] = stats.rankdata(v1000)
        v101 = np.empty((6, out.shape[0]))
        for i0 in range(1, 7):
            v1010 = volume[-i0]
            v101[-i0] = stats.rankdata(v1010)
        v10 = pd.DataFrame(v100).rolling(window=6).corr(pd.DataFrame(v101)).tail(1).as_matrix()[-1]
        v1 = stats.rankdata(v10)
        out[:] = np.power(v0, v1)

'''
# (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))), IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644)))
class Alpha79(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    sectors_in = morningstar.asset_classification.morningstar_sector_code.latest
    sectors_in.window_safe = True
    adv150_in = AverageDollarVolume(window_length=150)
    adv150_in.window_safe = True
    inputs = [USEquityPricing.close, USEquityPricing.open, vwap_in, sectors_in, adv150_in]
    window_length = 23

    def compute(self, today, assets, out, close, open, vwap, sectors, adv150):
        v000 = np.empty((2, out.shape[0]))
        for i0 in range(1, 3):
            v000000 = close[-i0]
            v000001 = np.full(out.shape[0], 0.60733)
            v00000 = v000000 * v000001
            v000010 = open[-i0]
            v0000110 = np.full(out.shape[0], 1.0)
            v0000111 = np.full(out.shape[0], 0.60733)
            v000011 = v0000110 - v0000111
            v00001 = v000010 * v000011
            v0000 = v00000 + v00001
            v0001 = sectors[-i0]
            v000[-i0] = demean_by_group(v0000, v0001)
        v00 = v000[-1] - v000[-2]
        v0 = stats.rankdata(v00)
        v100 = np.empty((15, out.shape[0]))
        for i0 in range(1, 16):
            v1000 = np.empty((4, out.shape[0]))
            for i1 in range(1, 5):
                v1000[-i1] = vwap[-i0 - i1]
            v100[-i0] = pd.DataFrame(v1000).rank().tail(1).as_matrix()[-1]
        v101 = np.empty((15, out.shape[0]))
        for i0 in range(1, 16):
            v1010 = np.empty((9, out.shape[0]))
            for i1 in range(1, 10):
                v1010[-i1] = adv150[-i0 - i1]
            v101[-i0] = pd.DataFrame(v1010).rank().tail(1).as_matrix()[-1]
        v10 = pd.DataFrame(v100).rolling(window=15).corr(pd.DataFrame(v101)).tail(1).as_matrix()[-1]
        v1 = stats.rankdata(v10)
        out[:] = v0 < v1
'''
'''
# ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))), IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)
class Alpha80(CustomFactor):
    industry_in = morningstar.asset_classification.morningstar_industry_code.latest
    industry_in.window_safe = True
    adv10_in = AverageDollarVolume(window_length=10)
    adv10_in.window_safe = True
    inputs = [USEquityPricing.high, industry_in, USEquityPricing.open, adv10_in]
    window_length = 10

    def compute(self, today, assets, out, high, industry, open, adv10):
        v00000 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v00000000 = open[-i0]
            v00000001 = np.full(out.shape[0], 0.868128)
            v0000000 = v00000000 * v00000001
            v00000010 = high[-i0]
            v000000110 = np.full(out.shape[0], 1.0)
            v000000111 = np.full(out.shape[0], 0.868128)
            v00000011 = v000000110 - v000000111
            v0000001 = v00000010 * v00000011
            v000000 = v0000000 + v0000001
            v000001 = industry[-i0]
            v00000[-i0] = demean_by_group(v000000, v000001)
        v0000 = v00000[-1] - v00000[-5]
        v000 = np.sign(v0000)
        v00 = stats.rankdata(v000)
        v010 = np.empty((6, out.shape[0]))
        for i0 in range(1, 7):
            v0100 = np.empty((5, out.shape[0]))
            for i1 in range(1, 6):
                v0100[-i1] = high[-i0 - i1]
            v0101 = np.empty((5, out.shape[0]))
            for i1 in range(1, 6):
                v0101[-i1] = adv10[-i0 - i1]
            v010[-i0] = pd.DataFrame(v0100).rolling(window=5).corr(pd.DataFrame(v0101)).tail(1).as_matrix()[-1]
        v01 = pd.DataFrame(v010).rank().tail(1).as_matrix()[-1]
        v0 = np.power(v00, v01)
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1
'''
# ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
class Alpha81(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    adv10_in = AverageDollarVolume(window_length=10)
    adv10_in.window_safe = True
    inputs = [USEquityPricing.volume, adv10_in, vwap_in]
    window_length = 73

    def compute(self, today, assets, out, volume, adv10, vwap):
        v00000 = np.empty((15, out.shape[0]))
        for i0 in range(1, 16):
            v000000000 = np.empty((8, out.shape[0]))
            for i1 in range(1, 9):
                v000000000[-i1] = vwap[-i0 - i1]
            v000000001 = np.empty((8, out.shape[0]))
            for i1 in range(1, 9):
                v0000000010 = np.empty((50, out.shape[0]))
                for i2 in range(1, 51):
                    v0000000010[-i2] = adv10[-i0 - i1 - i2]
                v000000001[-i1] = v0000000010.sum(axis=0)
            v00000000 = \
            pd.DataFrame(v000000000).rolling(window=8).corr(pd.DataFrame(v000000001)).tail(1).as_matrix()[-1]
            v0000000 = stats.rankdata(v00000000)
            v0000001 = np.full(out.shape[0], 4.0)
            v000000 = np.power(v0000000, v0000001)
            v00000[-i0] = stats.rankdata(v000000)
        v0000 = np.prod(v00000, axis=0)
        v000 = np.log(v0000)
        v00 = stats.rankdata(v000)
        v0100 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v01000 = vwap[-i0]
            v0100[-i0] = stats.rankdata(v01000)
        v0101 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v01010 = volume[-i0]
            v0101[-i0] = stats.rankdata(v01010)
        v010 = pd.DataFrame(v0100).rolling(window=5).corr(pd.DataFrame(v0101)).tail(1).as_matrix()[-1]
        v01 = stats.rankdata(v010)
        v0 = v00 < v01
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1

'''
# (min(rank(decay_linear(delta(open, 1.46063), 14.8717)), Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)
class Alpha82(CustomFactor):
    sectors_in = morningstar.asset_classification.morningstar_sector_code.latest
    sectors_in.window_safe = True
    inputs = [USEquityPricing.volume, USEquityPricing.open, sectors_in]
    window_length = 37

    def compute(self, today, assets, out, volume, open, sectors):
        v0000 = np.empty((15, out.shape[0]))
        for i0 in range(1, 16):
            v00000 = np.empty((2, out.shape[0]))
            for i1 in range(1, 3):
                v00000[-i1] = open[-i0 - i1]
            v0000[-i0] = v00000[-1] - v00000[-2]
        v000 = (v0000 * (np.arange(1.0, 16, 1.0) / 120)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v00 = stats.rankdata(v000)
        v010 = np.empty((13, out.shape[0]))
        for i0 in range(1, 14):
            v0100 = np.empty((7, out.shape[0]))
            for i1 in range(1, 8):
                v01000 = np.empty((17, out.shape[0]))
                for i2 in range(1, 18):
                    v010000 = volume[-i0 - i1 - i2]
                    v010001 = sectors[-i0 - i1 - i2]
                    v01000[-i2] = demean_by_group(v010000, v010001)
                v01001 = np.empty((17, out.shape[0]))
                for i2 in range(1, 18):
                    v0100100 = open[-i0 - i1 - i2]
                    v0100101 = np.full(out.shape[0], 0.634196)
                    v010010 = v0100100 * v0100101
                    v0100110 = open[-i0 - i1 - i2]
                    v01001110 = np.full(out.shape[0], 1.0)
                    v01001111 = np.full(out.shape[0], 0.634196)
                    v0100111 = v01001110 - v01001111
                    v010011 = v0100110 * v0100111
                    v01001[-i2] = v010010 + v010011
                v0100[-i1] = pd.DataFrame(v01000).rolling(window=17).corr(pd.DataFrame(v01001)).tail(1).as_matrix()[
                    -1]
            v010[-i0] = (v0100 * (np.arange(1.0, 8, 1.0) / 28)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v01 = pd.DataFrame(v010).rank().tail(1).as_matrix()[-1]
        v0 = np.minimum(v00, v01)
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1
'''
# ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))
class Alpha83(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    inputs = [USEquityPricing.high, USEquityPricing.close, USEquityPricing.low, vwap_in, USEquityPricing.volume]
    window_length = 8

    def compute(self, today, assets, out, high, close, low, vwap, volume):
        v000000 = high[-3]
        v000001 = low[-3]
        v00000 = v000000 - v000001
        v0000100 = np.empty((5, out.shape[0]))
        for i0 in range(3, 8):
            v0000100[2 - i0] = close[-i0]
        v000010 = v0000100.sum(axis=0)
        v000011 = np.full(out.shape[0], 5.0)
        v00001 = v000010 / v000011
        v0000 = v00000 / v00001
        v000 = v0000  # delay
        v00 = stats.rankdata(v000)
        v0100 = volume[-1]
        v010 = stats.rankdata(v0100)
        v01 = stats.rankdata(v010)
        v0 = v00 * v01
        v1000 = high[-1]
        v1001 = low[-1]
        v100 = v1000 - v1001
        v10100 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v10100[-i0] = close[-i0]
        v1010 = v10100.sum(axis=0)
        v1011 = np.full(out.shape[0], 5.0)
        v101 = v1010 / v1011
        v10 = v100 / v101
        v110 = vwap[-1]
        v111 = close[-1]
        v11 = v110 - v111
        v1 = v10 / v11
        out[:] = v0 / v1

# SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))
class Alpha84(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    inputs = [USEquityPricing.close, vwap_in]
    window_length = 36

    def compute(self, today, assets, out, close, vwap):
        v00 = np.empty((21, out.shape[0]))
        for i0 in range(1, 22):
            v000 = vwap[-i0]
            v0010 = np.empty((15, out.shape[0]))
            for i1 in range(1, 16):
                v0010[-i1] = vwap[-i0 - i1]
            v001 = np.max(v0010, axis=0)
            v00[-i0] = v000 - v001
        v0 = pd.DataFrame(v00).rank().tail(1).as_matrix()[-1]
        v10 = np.empty((6, out.shape[0]))
        for i0 in range(1, 7):
            v10[-i0] = close[-i0]
        v1 = v10[-1] - v10[-6]
        out[:] = np.power(v0, v1)

# (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))
class Alpha85(CustomFactor):
    adv30_in = AverageDollarVolume(window_length=30)
    adv30_in.window_safe = True
    inputs = [USEquityPricing.high, USEquityPricing.close, adv30_in, USEquityPricing.low, USEquityPricing.volume]
    window_length = 17

    def compute(self, today, assets, out, high, close, adv30, low, volume):
        v000 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v00000 = high[-i0]
            v00001 = np.full(out.shape[0], 0.876703)
            v0000 = v00000 * v00001
            v00010 = close[-i0]
            v000110 = np.full(out.shape[0], 1.0)
            v000111 = np.full(out.shape[0], 0.876703)
            v00011 = v000110 - v000111
            v0001 = v00010 * v00011
            v000[-i0] = v0000 + v0001
        v001 = np.empty((10, out.shape[0]))
        for i0 in range(1, 11):
            v001[-i0] = adv30[-i0]
        v00 = pd.DataFrame(v000).rolling(window=10).corr(pd.DataFrame(v001)).tail(1).as_matrix()[-1]
        v0 = stats.rankdata(v00)
        v100 = np.empty((7, out.shape[0]))
        for i0 in range(1, 8):
            v1000 = np.empty((4, out.shape[0]))
            for i1 in range(1, 5):
                v100000 = high[-i0 - i1]
                v100001 = low[-i0 - i1]
                v10000 = v100000 + v100001
                v10001 = np.full(out.shape[0], 2.0)
                v1000[-i1] = v10000 / v10001
            v100[-i0] = pd.DataFrame(v1000).rank().tail(1).as_matrix()[-1]
        v101 = np.empty((7, out.shape[0]))
        for i0 in range(1, 8):
            v1010 = np.empty((10, out.shape[0]))
            for i1 in range(1, 11):
                v1010[-i1] = volume[-i0 - i1]
            v101[-i0] = pd.DataFrame(v1010).rank().tail(1).as_matrix()[-1]
        v10 = pd.DataFrame(v100).rolling(window=7).corr(pd.DataFrame(v101)).tail(1).as_matrix()[-1]
        v1 = stats.rankdata(v10)
        out[:] = np.power(v0, v1)

# ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open + close) - (vwap + open)))) * -1)
class Alpha86(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    adv20_in = AverageDollarVolume(window_length=20)
    adv20_in.window_safe = True
    inputs = [USEquityPricing.close, adv20_in, vwap_in, USEquityPricing.open]
    window_length = 41

    def compute(self, today, assets, out, close, adv20, vwap, open):
        v000 = np.empty((20, out.shape[0]))
        for i0 in range(1, 21):
            v0000 = np.empty((6, out.shape[0]))
            for i1 in range(1, 7):
                v0000[-i1] = close[-i0 - i1]
            v0001 = np.empty((6, out.shape[0]))
            for i1 in range(1, 7):
                v00010 = np.empty((15, out.shape[0]))
                for i2 in range(1, 16):
                    v00010[-i2] = adv20[-i0 - i1 - i2]
                v0001[-i1] = v00010.sum(axis=0)
            v000[-i0] = pd.DataFrame(v0000).rolling(window=6).corr(pd.DataFrame(v0001)).tail(1).as_matrix()[-1]
        v00 = pd.DataFrame(v000).rank().tail(1).as_matrix()[-1]
        v01000 = open[-1]
        v01001 = close[-1]
        v0100 = v01000 + v01001
        v01010 = vwap[-1]
        v01011 = open[-1]
        v0101 = v01010 + v01011
        v010 = v0100 - v0101
        v01 = stats.rankdata(v010)
        v0 = v00 < v01
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1

'''
# (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81, IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)
class Alpha87(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    industry_in = morningstar.asset_classification.morningstar_industry_code.latest
    industry_in.window_safe = True
    adv81_in = AverageDollarVolume(window_length=81)
    adv81_in.window_safe = True
    inputs = [USEquityPricing.close, industry_in, vwap_in, adv81_in]
    window_length = 32

    def compute(self, today, assets, out, close, industry, vwap, adv81):
        v0000 = np.empty((3, out.shape[0]))
        for i0 in range(1, 4):
            v00000 = np.empty((3, out.shape[0]))
            for i1 in range(1, 4):
                v0000000 = close[-i0 - i1]
                v0000001 = np.full(out.shape[0], 0.369701)
                v000000 = v0000000 * v0000001
                v0000010 = vwap[-i0 - i1]
                v00000110 = np.full(out.shape[0], 1.0)
                v00000111 = np.full(out.shape[0], 0.369701)
                v0000011 = v00000110 - v00000111
                v000001 = v0000010 * v0000011
                v00000[-i1] = v000000 + v000001
            v0000[-i0] = v00000[-1] - v00000[-3]
        v000 = (v0000 * (np.arange(1.0, 4, 1.0) / 6)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v00 = stats.rankdata(v000)
        v010 = np.empty((14, out.shape[0]))
        for i0 in range(1, 15):
            v0100 = np.empty((5, out.shape[0]))
            for i1 in range(1, 6):
                v010000 = np.empty((13, out.shape[0]))
                for i2 in range(1, 14):
                    v0100000 = adv81[-i0 - i1 - i2]
                    v0100001 = industry[-i0 - i1 - i2]
                    v010000[-i2] = demean_by_group(v0100000, v0100001)
                v010001 = np.empty((13, out.shape[0]))
                for i2 in range(1, 14):
                    v010001[-i2] = close[-i0 - i1 - i2]
                v01000 = pd.DataFrame(v010000).rolling(window=13).corr(pd.DataFrame(v010001)).tail(1).as_matrix()[
                    -1]
                v0100[-i1] = np.abs(v01000)
            v010[-i0] = (v0100 * (np.arange(1.0, 6, 1.0) / 15)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v01 = pd.DataFrame(v010).rank().tail(1).as_matrix()[-1]
        v0 = np.maximum(v00, v01)
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1

'''
# min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))
class Alpha88(CustomFactor):
    adv60_in = AverageDollarVolume(window_length=60)
    adv60_in.window_safe = True
    inputs = [USEquityPricing.high, USEquityPricing.close, USEquityPricing.open, USEquityPricing.low, adv60_in]
    window_length = 37

    def compute(self, today, assets, out, high, close, open, low, adv60):
        v000 = np.empty((8, out.shape[0]))
        for i0 in range(1, 9):
            v000000 = open[-i0]
            v00000 = stats.rankdata(v000000)
            v000010 = low[-i0]
            v00001 = stats.rankdata(v000010)
            v0000 = v00000 + v00001
            v000100 = high[-i0]
            v00010 = stats.rankdata(v000100)
            v000110 = close[-i0]
            v00011 = stats.rankdata(v000110)
            v0001 = v00010 + v00011
            v000[-i0] = v0000 - v0001
        v00 = (v000 * (np.arange(1.0, 9, 1.0) / 36)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v0 = stats.rankdata(v00)
        v10 = np.empty((3, out.shape[0]))
        for i0 in range(1, 4):
            v100 = np.empty((7, out.shape[0]))
            for i1 in range(1, 8):
                v1000 = np.empty((8, out.shape[0]))
                for i2 in range(1, 9):
                    v10000 = np.empty((8, out.shape[0]))
                    for i3 in range(1, 9):
                        v10000[-i3] = close[-i0 - i1 - i2 - i3]
                    v1000[-i2] = pd.DataFrame(v10000).rank().tail(1).as_matrix()[-1]
                v1001 = np.empty((8, out.shape[0]))
                for i2 in range(1, 9):
                    v10010 = np.empty((21, out.shape[0]))
                    for i3 in range(1, 22):
                        v10010[-i3] = adv60[-i0 - i1 - i2 - i3]
                    v1001[-i2] = pd.DataFrame(v10010).rank().tail(1).as_matrix()[-1]
                v100[-i1] = pd.DataFrame(v1000).rolling(window=8).corr(pd.DataFrame(v1001)).tail(1).as_matrix()[-1]
            v10[-i0] = (v100 * (np.arange(1.0, 8, 1.0) / 28)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v1 = pd.DataFrame(v10).rank().tail(1).as_matrix()[-1]
        out[:] = np.minimum(v0, v1)

'''
# (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry), 3.48158), 10.1466), 15.3012))
class Alpha89(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    industry_in = morningstar.asset_classification.morningstar_industry_code.latest
    industry_in.window_safe = True
    adv10_in = AverageDollarVolume(window_length=10)
    adv10_in.window_safe = True
    inputs = [industry_in, USEquityPricing.low, vwap_in, adv10_in]
    window_length = 29

    def compute(self, today, assets, out, industry, low, vwap, adv10):
        v00 = np.empty((4, out.shape[0]))
        for i0 in range(1, 5):
            v000 = np.empty((6, out.shape[0]))
            for i1 in range(1, 7):
                v0000 = np.empty((7, out.shape[0]))
                for i2 in range(1, 8):
                    v000000 = low[-i0 - i1 - i2]
                    v000001 = np.full(out.shape[0], 0.967285)
                    v00000 = v000000 * v000001
                    v000010 = low[-i0 - i1 - i2]
                    v0000110 = np.full(out.shape[0], 1.0)
                    v0000111 = np.full(out.shape[0], 0.967285)
                    v000011 = v0000110 - v0000111
                    v00001 = v000010 * v000011
                    v0000[-i2] = v00000 + v00001
                v0001 = np.empty((7, out.shape[0]))
                for i2 in range(1, 8):
                    v0001[-i2] = adv10[-i0 - i1 - i2]
                v000[-i1] = pd.DataFrame(v0000).rolling(window=7).corr(pd.DataFrame(v0001)).tail(1).as_matrix()[-1]
            v00[-i0] = (v000 * (np.arange(1.0, 7, 1.0) / 21)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v0 = pd.DataFrame(v00).rank().tail(1).as_matrix()[-1]
        v10 = np.empty((15, out.shape[0]))
        for i0 in range(1, 16):
            v100 = np.empty((10, out.shape[0]))
            for i1 in range(1, 11):
                v1000 = np.empty((4, out.shape[0]))
                for i2 in range(1, 5):
                    v10000 = vwap[-i0 - i1 - i2]
                    v10001 = industry[-i0 - i1 - i2]
                    v1000[-i2] = demean_by_group(v10000, v10001)
                v100[-i1] = v1000[-1] - v1000[-4]
            v10[-i0] = (v100 * (np.arange(1.0, 11, 1.0) / 55)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v1 = pd.DataFrame(v10).rank().tail(1).as_matrix()[-1]
        out[:] = v0 - v1

'''
'''
# ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry), low, 5.38375), 3.21856)) * -1)
class Alpha90(CustomFactor):
    adv40_in = AverageDollarVolume(window_length=40)
    adv40_in.window_safe = True
    sub_industry_in = morningstar.asset_classification.morningstar_industry_group_code.latest
    sub_industry_in.window_safe = True
    inputs = [USEquityPricing.close, adv40_in, sub_industry_in, USEquityPricing.low]
    window_length = 8

    def compute(self, today, assets, out, close, adv40, sub_industry, low):
        v0000 = close[-1]
        v00010 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v00010[-i0] = close[-i0]
        v0001 = np.max(v00010, axis=0)
        v000 = v0000 - v0001
        v00 = stats.rankdata(v000)
        v010 = np.empty((3, out.shape[0]))
        for i0 in range(1, 4):
            v0100 = np.empty((5, out.shape[0]))
            for i1 in range(1, 6):
                v01000 = adv40[-i0 - i1]
                v01001 = sub_industry[-i0 - i1]
                v0100[-i1] = demean_by_group(v01000, v01001)
            v0101 = np.empty((5, out.shape[0]))
            for i1 in range(1, 6):
                v0101[-i1] = low[-i0 - i1]
            v010[-i0] = pd.DataFrame(v0100).rolling(window=5).corr(pd.DataFrame(v0101)).tail(1).as_matrix()[-1]
        v01 = pd.DataFrame(v010).rank().tail(1).as_matrix()[-1]
        v0 = np.power(v00, v01)
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1
'''
'''
# ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close, IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)
class Alpha91(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    industry_in = morningstar.asset_classification.morningstar_industry_code.latest
    industry_in.window_safe = True
    adv30_in = AverageDollarVolume(window_length=30)
    adv30_in.window_safe = True
    inputs = [USEquityPricing.volume, USEquityPricing.close, industry_in, adv30_in, vwap_in]
    window_length = 34

    def compute(self, today, assets, out, volume, close, industry, adv30, vwap):
        v000 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v0000 = np.empty((4, out.shape[0]))
            for i1 in range(1, 5):
                v00000 = np.empty((16, out.shape[0]))
                for i2 in range(1, 17):
                    v000000 = np.empty((10, out.shape[0]))
                    for i3 in range(1, 11):
                        v0000000 = close[-i0 - i1 - i2 - i3]
                        v0000001 = industry[-i0 - i1 - i2 - i3]
                        v000000[-i3] = demean_by_group(v0000000, v0000001)
                    v000001 = np.empty((10, out.shape[0]))
                    for i3 in range(1, 11):
                        v000001[-i3] = volume[-i0 - i1 - i2 - i3]
                    v00000[-i2] = \
                    pd.DataFrame(v000000).rolling(window=10).corr(pd.DataFrame(v000001)).tail(1).as_matrix()[-1]
                v0000[-i1] = (v00000 * (np.arange(1.0, 17, 1.0) / 136)[:, np.newaxis]).sum(axis=0)  # decay_linear
            v000[-i0] = (v0000 * (np.arange(1.0, 5, 1.0) / 10)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v00 = pd.DataFrame(v000).rank().tail(1).as_matrix()[-1]
        v0100 = np.empty((3, out.shape[0]))
        for i0 in range(1, 4):
            v01000 = np.empty((4, out.shape[0]))
            for i1 in range(1, 5):
                v01000[-i1] = vwap[-i0 - i1]
            v01001 = np.empty((4, out.shape[0]))
            for i1 in range(1, 5):
                v01001[-i1] = adv30[-i0 - i1]
            v0100[-i0] = pd.DataFrame(v01000).rolling(window=4).corr(pd.DataFrame(v01001)).tail(1).as_matrix()[-1]
        v010 = (v0100 * (np.arange(1.0, 4, 1.0) / 6)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v01 = stats.rankdata(v010)
        v0 = v00 - v01
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1
'''
# min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))
class Alpha92(CustomFactor):
    adv30_in = AverageDollarVolume(window_length=30)
    adv30_in.window_safe = True
    inputs = [USEquityPricing.high, USEquityPricing.close, adv30_in, USEquityPricing.open, USEquityPricing.low]
    window_length = 33

    def compute(self, today, assets, out, high, close, adv30, open, low):
        v00 = np.empty((19, out.shape[0]))
        for i0 in range(1, 20):
            v000 = np.empty((15, out.shape[0]))
            for i1 in range(1, 16):
                v0000000 = high[-i0 - i1]
                v0000001 = low[-i0 - i1]
                v000000 = v0000000 + v0000001
                v000001 = np.full(out.shape[0], 2.0)
                v00000 = v000000 / v000001
                v00001 = close[-i0 - i1]
                v0000 = v00000 + v00001
                v00010 = low[-i0 - i1]
                v00011 = open[-i0 - i1]
                v0001 = v00010 + v00011
                v000[-i1] = v0000 < v0001
            v00[-i0] = (v000 * (np.arange(1.0, 16, 1.0) / 120)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v0 = pd.DataFrame(v00).rank().tail(1).as_matrix()[-1]
        v10 = np.empty((7, out.shape[0]))
        for i0 in range(1, 8):
            v100 = np.empty((7, out.shape[0]))
            for i1 in range(1, 8):
                v1000 = np.empty((8, out.shape[0]))
                for i2 in range(1, 9):
                    v10000 = low[-i0 - i1 - i2]
                    v1000[-i2] = stats.rankdata(v10000)
                v1001 = np.empty((8, out.shape[0]))
                for i2 in range(1, 9):
                    v10010 = adv30[-i0 - i1 - i2]
                    v1001[-i2] = stats.rankdata(v10010)
                v100[-i1] = pd.DataFrame(v1000).rolling(window=8).corr(pd.DataFrame(v1001)).tail(1).as_matrix()[-1]
            v10[-i0] = (v100 * (np.arange(1.0, 8, 1.0) / 28)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v1 = pd.DataFrame(v10).rank().tail(1).as_matrix()[-1]
        out[:] = np.minimum(v0, v1)

'''
# (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664)))
class Alpha93(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    industry_in = morningstar.asset_classification.morningstar_industry_code.latest
    industry_in.window_safe = True
    adv81_in = AverageDollarVolume(window_length=81)
    adv81_in.window_safe = True
    inputs = [USEquityPricing.close, industry_in, vwap_in, adv81_in]
    window_length = 44

    def compute(self, today, assets, out, close, industry, vwap, adv81):
        v00 = np.empty((8, out.shape[0]))
        for i0 in range(1, 9):
            v000 = np.empty((20, out.shape[0]))
            for i1 in range(1, 21):
                v0000 = np.empty((17, out.shape[0]))
                for i2 in range(1, 18):
                    v00000 = vwap[-i0 - i1 - i2]
                    v00001 = industry[-i0 - i1 - i2]
                    v0000[-i2] = demean_by_group(v00000, v00001)
                v0001 = np.empty((17, out.shape[0]))
                for i2 in range(1, 18):
                    v0001[-i2] = adv81[-i0 - i1 - i2]
                v000[-i1] = pd.DataFrame(v0000).rolling(window=17).corr(pd.DataFrame(v0001)).tail(1).as_matrix()[-1]
            v00[-i0] = (v000 * (np.arange(1.0, 21, 1.0) / 210)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v0 = pd.DataFrame(v00).rank().tail(1).as_matrix()[-1]
        v100 = np.empty((16, out.shape[0]))
        for i0 in range(1, 17):
            v1000 = np.empty((4, out.shape[0]))
            for i1 in range(1, 5):
                v100000 = close[-i0 - i1]
                v100001 = np.full(out.shape[0], 0.524434)
                v10000 = v100000 * v100001
                v100010 = vwap[-i0 - i1]
                v1000110 = np.full(out.shape[0], 1.0)
                v1000111 = np.full(out.shape[0], 0.524434)
                v100011 = v1000110 - v1000111
                v10001 = v100010 * v100011
                v1000[-i1] = v10000 + v10001
            v100[-i0] = v1000[-1] - v1000[-4]
        v10 = (v100 * (np.arange(1.0, 17, 1.0) / 136)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v1 = stats.rankdata(v10)
        out[:] = v0 / v1
'''
# ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
class Alpha94(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    adv60_in = AverageDollarVolume(window_length=60)
    adv60_in.window_safe = True
    inputs = [adv60_in, vwap_in]
    window_length = 40

    def compute(self, today, assets, out, adv60, vwap):
        v0000 = vwap[-1]
        v00010 = np.empty((12, out.shape[0]))
        for i0 in range(1, 13):
            v00010[-i0] = vwap[-i0]
        v0001 = np.min(v00010, axis=0)
        v000 = v0000 - v0001
        v00 = stats.rankdata(v000)
        v010 = np.empty((3, out.shape[0]))
        for i0 in range(1, 4):
            v0100 = np.empty((18, out.shape[0]))
            for i1 in range(1, 19):
                v01000 = np.empty((20, out.shape[0]))
                for i2 in range(1, 21):
                    v01000[-i2] = vwap[-i0 - i1 - i2]
                v0100[-i1] = pd.DataFrame(v01000).rank().tail(1).as_matrix()[-1]
            v0101 = np.empty((18, out.shape[0]))
            for i1 in range(1, 19):
                v01010 = np.empty((4, out.shape[0]))
                for i2 in range(1, 5):
                    v01010[-i2] = adv60[-i0 - i1 - i2]
                v0101[-i1] = pd.DataFrame(v01010).rank().tail(1).as_matrix()[-1]
            v010[-i0] = pd.DataFrame(v0100).rolling(window=18).corr(pd.DataFrame(v0101)).tail(1).as_matrix()[-1]
        v01 = pd.DataFrame(v010).rank().tail(1).as_matrix()[-1]
        v0 = np.power(v00, v01)
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1

# (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
class Alpha95(CustomFactor):
    adv40_in = AverageDollarVolume(window_length=40)
    adv40_in.window_safe = True
    inputs = [USEquityPricing.high, adv40_in, USEquityPricing.open, USEquityPricing.low]
    window_length = 43

    def compute(self, today, assets, out, high, adv40, open, low):
        v000 = open[-1]
        v0010 = np.empty((12, out.shape[0]))
        for i0 in range(1, 13):
            v0010[-i0] = open[-i0]
        v001 = np.min(v0010, axis=0)
        v00 = v000 - v001
        v0 = stats.rankdata(v00)
        v10 = np.empty((12, out.shape[0]))
        for i0 in range(1, 13):
            v10000 = np.empty((13, out.shape[0]))
            for i1 in range(1, 14):
                v100000 = np.empty((19, out.shape[0]))
                for i2 in range(1, 20):
                    v10000000 = high[-i0 - i1 - i2]
                    v10000001 = low[-i0 - i1 - i2]
                    v1000000 = v10000000 + v10000001
                    v1000001 = np.full(out.shape[0], 2.0)
                    v100000[-i2] = v1000000 / v1000001
                v10000[-i1] = v100000.sum(axis=0)
            v10001 = np.empty((13, out.shape[0]))
            for i1 in range(1, 14):
                v100010 = np.empty((19, out.shape[0]))
                for i2 in range(1, 20):
                    v100010[-i2] = adv40[-i0 - i1 - i2]
                v10001[-i1] = v100010.sum(axis=0)
            v1000 = pd.DataFrame(v10000).rolling(window=13).corr(pd.DataFrame(v10001)).tail(1).as_matrix()[-1]
            v100 = stats.rankdata(v1000)
            v101 = np.full(out.shape[0], 5.0)
            v10[-i0] = np.power(v100, v101)
        v1 = pd.DataFrame(v10).rank().tail(1).as_matrix()[-1]
        out[:] = v0 < v1

# (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
class Alpha96(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    adv60_in = AverageDollarVolume(window_length=60)
    adv60_in.window_safe = True
    inputs = [USEquityPricing.volume, USEquityPricing.close, vwap_in, adv60_in]
    window_length = 51

    def compute(self, today, assets, out, volume, close, vwap, adv60):
        v000 = np.empty((8, out.shape[0]))
        for i0 in range(1, 9):
            v0000 = np.empty((4, out.shape[0]))
            for i1 in range(1, 5):
                v00000 = np.empty((4, out.shape[0]))
                for i2 in range(1, 5):
                    v000000 = vwap[-i0 - i1 - i2]
                    v00000[-i2] = stats.rankdata(v000000)
                v00001 = np.empty((4, out.shape[0]))
                for i2 in range(1, 5):
                    v000010 = volume[-i0 - i1 - i2]
                    v00001[-i2] = stats.rankdata(v000010)
                v0000[-i1] = pd.DataFrame(v00000).rolling(window=4).corr(pd.DataFrame(v00001)).tail(1).as_matrix()[
                    -1]
            v000[-i0] = (v0000 * (np.arange(1.0, 5, 1.0) / 10)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v00 = pd.DataFrame(v000).rank().tail(1).as_matrix()[-1]
        v010 = np.empty((13, out.shape[0]))
        for i0 in range(1, 14):
            v0100 = np.empty((14, out.shape[0]))
            for i1 in range(1, 15):
                v01000 = np.empty((13, out.shape[0]))
                for i2 in range(1, 14):
                    v010000 = np.empty((4, out.shape[0]))
                    for i3 in range(1, 5):
                        v0100000 = np.empty((7, out.shape[0]))
                        for i4 in range(1, 8):
                            v0100000[-i4] = close[-i0 - i1 - i2 - i3 - i4]
                        v010000[-i3] = pd.DataFrame(v0100000).rank().tail(1).as_matrix()[-1]
                    v010001 = np.empty((4, out.shape[0]))
                    for i3 in range(1, 5):
                        v0100010 = np.empty((4, out.shape[0]))
                        for i4 in range(1, 5):
                            v0100010[-i4] = adv60[-i0 - i1 - i2 - i3 - i4]
                        v010001[-i3] = pd.DataFrame(v0100010).rank().tail(1).as_matrix()[-1]
                    v01000[-i2] = \
                    pd.DataFrame(v010000).rolling(window=4).corr(pd.DataFrame(v010001)).tail(1).as_matrix()[-1]
                v0100[-i1] = np.argmax(v01000, axis=0)
            v010[-i0] = (v0100 * (np.arange(1.0, 15, 1.0) / 105)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v01 = pd.DataFrame(v010).rank().tail(1).as_matrix()[-1]
        v0 = np.maximum(v00, v01)
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1

'''
# ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)
class Alpha97(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    adv60_in = AverageDollarVolume(window_length=60)
    adv60_in.window_safe = True
    industry_in = morningstar.asset_classification.morningstar_industry_code.latest
    industry_in.window_safe = True
    inputs = [adv60_in, industry_in, USEquityPricing.low, vwap_in]
    window_length = 63

    def compute(self, today, assets, out, adv60, industry, low, vwap):
        v0000 = np.empty((20, out.shape[0]))
        for i0 in range(1, 21):
            v00000 = np.empty((4, out.shape[0]))
            for i1 in range(1, 5):
                v00000000 = low[-i0 - i1]
                v00000001 = np.full(out.shape[0], 0.721001)
                v0000000 = v00000000 * v00000001
                v00000010 = vwap[-i0 - i1]
                v000000110 = np.full(out.shape[0], 1.0)
                v000000111 = np.full(out.shape[0], 0.721001)
                v00000011 = v000000110 - v000000111
                v0000001 = v00000010 * v00000011
                v000000 = v0000000 + v0000001
                v000001 = industry[-i0 - i1]
                v00000[-i1] = demean_by_group(v000000, v000001)
            v0000[-i0] = v00000[-1] - v00000[-4]
        v000 = (v0000 * (np.arange(1.0, 21, 1.0) / 210)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v00 = stats.rankdata(v000)
        v010 = np.empty((7, out.shape[0]))
        for i0 in range(1, 8):
            v0100 = np.empty((16, out.shape[0]))
            for i1 in range(1, 17):
                v01000 = np.empty((19, out.shape[0]))
                for i2 in range(1, 20):
                    v010000 = np.empty((5, out.shape[0]))
                    for i3 in range(1, 6):
                        v0100000 = np.empty((8, out.shape[0]))
                        for i4 in range(1, 9):
                            v0100000[-i4] = low[-i0 - i1 - i2 - i3 - i4]
                        v010000[-i3] = pd.DataFrame(v0100000).rank().tail(1).as_matrix()[-1]
                    v010001 = np.empty((5, out.shape[0]))
                    for i3 in range(1, 6):
                        v0100010 = np.empty((17, out.shape[0]))
                        for i4 in range(1, 18):
                            v0100010[-i4] = adv60[-i0 - i1 - i2 - i3 - i4]
                        v010001[-i3] = pd.DataFrame(v0100010).rank().tail(1).as_matrix()[-1]
                    v01000[-i2] = \
                    pd.DataFrame(v010000).rolling(window=5).corr(pd.DataFrame(v010001)).tail(1).as_matrix()[-1]
                v0100[-i1] = pd.DataFrame(v01000).rank().tail(1).as_matrix()[-1]
            v010[-i0] = (v0100 * (np.arange(1.0, 17, 1.0) / 136)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v01 = pd.DataFrame(v010).rank().tail(1).as_matrix()[-1]
        v0 = v00 - v01
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1
'''

# (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))
class Alpha98(CustomFactor):
    vwap_in = VWAP(window_length=2)
    vwap_in.window_safe = True
    adv5_in = AverageDollarVolume(window_length=5)
    adv5_in.window_safe = True
    adv15_in = AverageDollarVolume(window_length=15)
    adv15_in.window_safe = True
    inputs = [adv5_in, adv15_in, USEquityPricing.open, vwap_in]
    window_length = 44

    def compute(self, today, assets, out, adv5, adv15, open, vwap):
        v000 = np.empty((7, out.shape[0]))
        for i0 in range(1, 8):
            v0000 = np.empty((5, out.shape[0]))
            for i1 in range(1, 6):
                v0000[-i1] = vwap[-i0 - i1]
            v0001 = np.empty((5, out.shape[0]))
            for i1 in range(1, 6):
                v00010 = np.empty((26, out.shape[0]))
                for i2 in range(1, 27):
                    v00010[-i2] = adv5[-i0 - i1 - i2]
                v0001[-i1] = v00010.sum(axis=0)
            v000[-i0] = pd.DataFrame(v0000).rolling(window=5).corr(pd.DataFrame(v0001)).tail(1).as_matrix()[-1]
        v00 = (v000 * (np.arange(1.0, 8, 1.0) / 28)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v0 = stats.rankdata(v00)
        v100 = np.empty((8, out.shape[0]))
        for i0 in range(1, 9):
            v1000 = np.empty((7, out.shape[0]))
            for i1 in range(1, 8):
                v10000 = np.empty((9, out.shape[0]))
                for i2 in range(1, 10):
                    v100000 = np.empty((21, out.shape[0]))
                    for i3 in range(1, 22):
                        v1000000 = open[-i0 - i1 - i2 - i3]
                        v100000[-i3] = stats.rankdata(v1000000)
                    v100001 = np.empty((21, out.shape[0]))
                    for i3 in range(1, 22):
                        v1000010 = adv15[-i0 - i1 - i2 - i3]
                        v100001[-i3] = stats.rankdata(v1000010)
                    v10000[-i2] = \
                    pd.DataFrame(v100000).rolling(window=21).corr(pd.DataFrame(v100001)).tail(1).as_matrix()[-1]
                v1000[-i1] = np.argmin(v10000, axis=0)
            v100[-i0] = pd.DataFrame(v1000).rank().tail(1).as_matrix()[-1]
        v10 = (v100 * (np.arange(1.0, 9, 1.0) / 36)[:, np.newaxis]).sum(axis=0)  # decay_linear
        v1 = stats.rankdata(v10)
        out[:] = v0 - v1

# ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1)
class Alpha99(CustomFactor):
    adv60_in = AverageDollarVolume(window_length=60)
    adv60_in.window_safe = True
    inputs = [USEquityPricing.high, adv60_in, USEquityPricing.low, USEquityPricing.volume]
    window_length = 28

    def compute(self, today, assets, out, high, adv60, low, volume):
        v0000 = np.empty((9, out.shape[0]))
        for i0 in range(1, 10):
            v00000 = np.empty((20, out.shape[0]))
            for i1 in range(1, 21):
                v0000000 = high[-i0 - i1]
                v0000001 = low[-i0 - i1]
                v000000 = v0000000 + v0000001
                v000001 = np.full(out.shape[0], 2.0)
                v00000[-i1] = v000000 / v000001
            v0000[-i0] = v00000.sum(axis=0)
        v0001 = np.empty((9, out.shape[0]))
        for i0 in range(1, 10):
            v00010 = np.empty((20, out.shape[0]))
            for i1 in range(1, 21):
                v00010[-i1] = adv60[-i0 - i1]
            v0001[-i0] = v00010.sum(axis=0)
        v000 = pd.DataFrame(v0000).rolling(window=9).corr(pd.DataFrame(v0001)).tail(1).as_matrix()[-1]
        v00 = stats.rankdata(v000)
        v0100 = np.empty((6, out.shape[0]))
        for i0 in range(1, 7):
            v0100[-i0] = low[-i0]
        v0101 = np.empty((6, out.shape[0]))
        for i0 in range(1, 7):
            v0101[-i0] = volume[-i0]
        v010 = pd.DataFrame(v0100).rolling(window=6).corr(pd.DataFrame(v0101)).tail(1).as_matrix()[-1]
        v01 = stats.rankdata(v010)
        v0 = v00 < v01
        v1 = np.full(out.shape[0], -1.0)
        out[:] = v0 * v1

'''
# (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) - scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry))) * (volume / adv20))))
class Alpha100(CustomFactor):
    adv20_in = AverageDollarVolume(window_length=20)
    adv20_in.window_safe = True
    sub_industry_in = morningstar.asset_classification.morningstar_industry_group_code.latest
    sub_industry_in.window_safe = True
    inputs = [USEquityPricing.volume, adv20_in, sub_industry_in, USEquityPricing.high, USEquityPricing.low,
              USEquityPricing.close]
    window_length = 30

    def compute(self, today, assets, out, volume, adv20, sub_industry, high, low, close):
        v0 = np.full(out.shape[0], 0.0)
        v10 = np.full(out.shape[0], 1.0)
        v11000 = np.full(out.shape[0], 1.5)
        v1100100000000 = close[-1]
        v1100100000001 = low[-1]
        v110010000000 = v1100100000000 - v1100100000001
        v1100100000010 = high[-1]
        v1100100000011 = close[-1]
        v110010000001 = v1100100000010 - v1100100000011
        v11001000000 = v110010000000 - v110010000001
        v110010000010 = high[-1]
        v110010000011 = low[-1]
        v11001000001 = v110010000010 - v110010000011
        v1100100000 = v11001000000 / v11001000001
        v1100100001 = volume[-1]
        v110010000 = v1100100000 * v1100100001
        v11001000 = stats.rankdata(v110010000)
        v11001001 = sub_industry[-1]
        v1100100 = demean_by_group(v11001000, v11001001)
        v1100101 = sub_industry[-1]
        v110010 = demean_by_group(v1100100, v1100101)
        v11001 = v110010 / np.abs(v110010).sum()
        v1100 = v11000 * v11001
        v11010000 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v11010000[-i0] = close[-i0]
        v11010001 = np.empty((5, out.shape[0]))
        for i0 in range(1, 6):
            v110100010 = adv20[-i0]
            v11010001[-i0] = stats.rankdata(v110100010)
        v1101000 = pd.DataFrame(v11010000).rolling(window=5).corr(pd.DataFrame(v11010001)).tail(1).as_matrix()[-1]
        v110100100 = np.empty((30, out.shape[0]))
        for i0 in range(1, 31):
            v110100100[-i0] = close[-i0]
        v11010010 = np.argmin(v110100100, axis=0)
        v1101001 = stats.rankdata(v11010010)
        v110100 = v1101000 - v1101001
        v110101 = sub_industry[-1]
        v11010 = demean_by_group(v110100, v110101)
        v1101 = v11010 / np.abs(v11010).sum()
        v110 = v1100 - v1101
        v1110 = volume[-1]
        v1111 = adv20[-1]
        v111 = v1110 / v1111
        v11 = v110 * v111
        v1 = v10 * v11
        out[:] = v0 - v1
'''
# ((close - open) / ((high - low) + 0.001))
class Alpha101(CustomFactor):
    inputs = [USEquityPricing.high, USEquityPricing.close, USEquityPricing.open, USEquityPricing.low]
    window_length = 1

    def compute(self, today, assets, out, high, close, open, low):
        v00 = close[-1]
        v01 = open[-1]
        v0 = v00 - v01
        v100 = high[-1]
        v101 = low[-1]
        v10 = v100 - v101
        v11 = np.full(out.shape[0], 0.001)
        v1 = v10 + v11
        out[:] = v0 / v1
