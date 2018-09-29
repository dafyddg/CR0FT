#!/usr/bin/python
#-*- coding: UTF8 -*-

#!/usr/pkgsrc/20140707/bin/python2.7
#!/usr/bin/python

# bandpass-cgi.py
# D. Gibbon, 2018-09-11

opsys = 'linux'
# opsys = 'solaris'

import warnings
warnings.filterwarnings("ignore")

from scipy.signal import butter, lfilter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cgi, cgitb; cgitb.enable()
from cgi import escape

if opsys == 'linux':
        localdirectory = '/var/www/Webtmp/'
        webdirectory = '/Webtmp/'
elif opsys == 'solaris':
        localdirectory = '../../docs/Webtmp/'
        webdirectory = '/gibbon/Webtmp/'
else:
        print 'Unknown operating system.'; exit()

#=========================================================================
# Initialise HTML

def inithtml():
    print "Content-type: text/html\n\n"
    print '<html><head><title>Bandpass filter</title>'
    print """
    <style type="text/css">
        tdp {font-size:14}
        td {font-size:14}
        p {font-size:14}
        li {font-size:14}
        small {font-size:14}
        big {font-size:18;font-weight:bold}
        verybig {font-size:24;font-weight:bold}
    </style>
    """
    print '</head><body>'
    return

#=========================================================================

def terminatehtml():
    print "</body></html>"
    return

#=========================================================================

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 5000.0
    lowcut = 500.0
    highcut = 1250.0

    figwidth = 8
    figheight = 2
    fontsize = 8

    # Plot the frequency response for a few different orders.
    plt.figure(1,(figwidth,figheight))
    plt.clf()
    for order in [3, 6, 9]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)',fontsize=fontsize)
    plt.ylabel('Gain',fontsize=fontsize)
    plt.grid(True)
    plt.legend(loc='best',fontsize=fontsize)
    plt.tight_layout(pad=0.1,w_pad=0.5,h_pad=0.1)
    plt.tick_params(axis='both',labelsize=fontsize)
    plt.tight_layout()
    plt.savefig(localdirectory+'bandpass01.png')


    # Filter a noisy signal.
    T = 0.05
    nsamples = T * fs
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.02
    f0 = 600.0
    x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
    x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
    x += a * np.cos(2 * np.pi * f0 * t + .11)
    x += 0.03 * np.cos(2 * np.pi * 2000 * t)

    plt.figure(2,(figwidth,figheight))
    plt.clf()
    plt.plot(t, x, label='Noisy signal')

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
    plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
    plt.xlabel('time (seconds)',fontsize=fontsize)
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left',fontsize=fontsize)
    plt.tick_params(axis='both',labelsize=fontsize)
    plt.tight_layout(pad=0.1,w_pad=0.5,h_pad=0.1)
    plt.savefig(localdirectory+'bandpass02.png')

inithtml()

print '<img src=\"'+webdirectory+'bandpass01.png\" width=\"100%\">'
print '<img src=\"'+webdirectory+'bandpass02.png\" width=\"100%\">'

terminatehtml()
