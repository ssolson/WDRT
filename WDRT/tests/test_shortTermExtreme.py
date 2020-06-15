import unittest
import numpy as np
import matplotlib.pyplot as plt
import WDRT.shortTermExtreme as ste
import os
from os.path import abspath, dirname, join, isfile

testdir = dirname(abspath(__file__))
datadir = join(testdir, join('..','..','examples','data'))

class TestShortTermExtreme(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        pass
    @classmethod
    def tearDownClass(self):
        pass

    def test_shortTermExtreme(self):

        # load response time series
        responseData = os.path.join(datadir,'data.csv')
        data = ste.loadtxt(responseData, delimiter=',')
        t = data['t']
        response = data['data']

        # find global peaks
        t_peaks, peaks = ste.globalPeaks(t, response)

        # get the 1-hour extreme distribution using the Weibull tail fit method
        x_e = np.linspace(0, 2 * np.max(peaks), 10000)
        t_x = (t[-1] - t[0])
        t_st = 1. * 60. * 60.
        stextreme_dist, peaks_dist, _, _, _ = ste.extremeDistribution_WeibullTailFit(x=peaks, x_e=x_e, t_x=t_x, t_st=t_st)


## plot
#plt.figure()
#plt.plot(t, response, 'k-')
#plt.plot(t_peaks, peaks, 'go')
#plt.plot([0, t[-1]], [0, 0], 'k--')
#plt.xlabel('Time, $t$ [s]')
#plt.ylabel('Response, $x$')
#plt.grid(True)
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
## plot
#plt.figure()
#ax = plt.subplot(2, 1, 1)
#plt.plot(x_e, peaks_dist.pdf(x_e), 'g-', label='Peak distribution')
#plt.plot(x_e, stextreme_dist.pdf(x_e), 'r-', label='Extreme distribution')
#xlim = ax.get_xlim()
#ylim = ax.get_ylim()
#plt.ylim([0, ylim[1]])
#plt.xlim([0, xlim[1]])
#plt.ylabel('$PDF(x)$')
#plt.ylabel('Response, $x$')
#plt.grid(True)
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#plt.legend()
#
#ax = plt.subplot(2, 1, 2)
#plt.plot(x_e, peaks_dist.cdf(x_e), 'g-')
#plt.plot(x_e, stextreme_dist.cdf(x_e), 'r-')
#xlim = ax.get_xlim()
#ylim = ax.get_ylim()
#plt.ylim([0, ylim[1]])
#plt.xlim([0, xlim[1]])
#plt.xlabel('Response, $x$')
#plt.ylabel('$CDF(x)$')
#plt.grid(True)
#plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
#
## goodness of fit plots
#gof_plots = ste.goodnessOfFitPlots(data=peaks, prob_func=peaks_dist, np_return=1000001, x_pdf=x_e, bins_pdf=20)

#plt.show()

    def test_shortTermExtreme2(self):


        #    method=1
        methods=[1, 2, 4, 5]
        for method in methods:
                # 1 - All peaks Weibull
                # 2 - Weibull tail fit
                # 3 - Peaks over threshold
                # 4 - Block maxima GEV
                # 5 - Block maxima Gumbel

            # load global peaks
            t_peaks_file = os.path.join(datadir, 't.dat')
            peaks_file   = os.path.join(datadir, 'peaks.dat')
            t_peaks = np.loadtxt(t_peaks_file)
            peaks = np.loadtxt(peaks_file)/1000.
    
            # get the 1-hour extreme distribution using the method selected above
            x_e = np.linspace(0, 2 * np.max(peaks), 10000)
            t_x = (t_peaks[-1]-t_peaks[0]) + ((t_peaks[-1]-t_peaks[0])/(1.*len(peaks)))
            t_st = 1. * 60. * 60.
            if method==1:
                    stextreme_dist, peaks_dist, _ = ste.extremeDistribution_Weibull(x=peaks, x_e=x_e, t_x=t_x, t_st=t_st)
            elif method==2:
                    stextreme_dist, peaks_dist, _, _, _ = ste.extremeDistribution_WeibullTailFit(x=peaks, x_e=x_e, t_x=t_x, t_st=t_st)
            elif method==3:
                    thresh = np.mean(peaks) + 1.4*np.std(peaks)
                    thresh_x = np.min(x_e[x_e>thresh])
                    stextreme_dist, peaks_dist, pot_dist, _ = ste.extremeDistribution_peaksOverThreshold(x=peaks, x_e=x_e, t_x=t_x, t_st=t_st, u=thresh)
            elif method==4:
                    stextreme_dist,_,bm = ste.extremeDistribution_blockMaximaGEV(x=peaks, t=t_peaks, t_st=t_st)
            elif method == 5:
                    stextreme_dist,_,bm = ste.extremeDistribution_blockMaximaGumb(x=peaks, t=t_peaks, t_st=t_st)

#            # goodness of fit plots
#            if method==1 or method==2:
#                    bm = ecm.blockMaxima(x=peaks, t=t_peaks, t_st=t_st)
#                    _ = ecm.goodnessOfFitPlots(data=peaks, prob_func=peaks_dist, np_return=1000001, x_pdf=x_e, bins_pdf=20, response_name='PTO Force', response_name_2='Peaks',response_units='kN')
#            if not method==3:
#                    fig_gof = ecm.goodnessOfFitPlots(data=bm, prob_func=stextreme_dist, np_return=10001, x_pdf=x_e, bins_pdf=20, response_name='PTO Force', response_name_2='1-hr Extreme',response_units='kN')
#            if method==3:
#                    bm = ecm.blockMaxima(x=peaks, t=t_peaks, t_st=t_st)
#                    _ = ecm.goodnessOfFitPlots(data=peaks[peaks>thresh_x], prob_func=peaks_dist, np_return=100001, x_pdf=x_e[x_e>thresh_x], bins_pdf=20,m_prob=1.*len(peaks[peaks<thresh_x]), response_name='PTO Force', response_name_2='Peaks',response_units='kN')
#                    _ = ecm.goodnessOfFitPlots(data=peaks[peaks>thresh]-thresh, prob_func=pot_dist, np_return=100001, x_pdf=x_e[x_e>thresh]-thresh, bins_pdf=20, response_name='PTO Force', response_name_2='Peaks Over Threshold',response_units='kN')
#                    fig_gof = ecm.goodnessOfFitPlots(data=bm, prob_func=stextreme_dist, np_return=10001, x_pdf=x_e[x_e>thresh_x], bins_pdf=20, response_name='PTO Force', response_name_2='1-hr Extreme',response_units='kN')

## plot
#plt.figure()
#if method==3:
#        plt.plot(t_peaks[peaks<thresh], peaks[peaks<thresh], 'ko', alpha=0.2)
#        plt.plot(t_peaks[peaks>thresh], peaks[peaks>thresh], 'go')
#        plt.plot([0, t_peaks[-1]], [thresh, thresh], 'r--')
#else:
#        plt.plot(t_peaks, peaks, 'go')
#plt.plot([0, t_peaks[-1]], [0, 0], 'k--')
#plt.xlabel('Time, $t$ [s]')
#plt.ylabel('Response, $x$')
#plt.xlim([0,3600*2])
#plt.grid(True)
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
#
#plt.figure()


    def test_shortTermExtreme3(self):
        t_peaks_file = os.path.join(datadir, 't.dat')
        peaks_file   = os.path.join(datadir, 'peaks.dat')
        t_peaks = np.loadtxt(t_peaks_file)
        peaks = np.loadtxt(peaks_file)/1000.

        t_st = 1. * 60. * 60

        f1, f2, ev = ste.compare_methods(peaks, t_peaks, t_st, 
                                 methods=[1, 2, 4, 5],
                                 colors=['g', 'b', 'r', 'k', 'k'],
                                 lines=['-', '-', '-', '-', '--'])
        #plt.show()
