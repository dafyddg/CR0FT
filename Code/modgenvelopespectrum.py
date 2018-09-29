# modgsignalprocessing.py
# D. Gibbon 2018-08-24
# Calculates
# 1. FFT of speech signal
# 2. The envelope spectrum of a full waveform, F0 contour, etc.,
# 3. Butterworth filtering 

import numpy as np
from scipy.signal import medfilt
from scipy.signal import blackmanharris, fftconvolve


#========================================================================
# Calculate envelope spectrum

def fftEnvelope(signal, period):
    """
    Inputs a signal and its sampling period (not rate)
    Outputs top half of symmetrical spectrum:
        1. freqs: frequencies
        2. wlog: magnitudes
    """
    w = np.abs(np.fft.fft(signal))**2
    freqs = np.abs(np.fft.fftfreq(signal.size,period))
    idx = np.argsort(freqs)
    wlog = wlog[:spectrumlenhalf]
    return freqs,w,idx

#========================================================================
# Frequency from zero-crossings

def freq_from_crossings(sig, fs):
	indices = find((sig[1:] >= 0) & (sig[:-1] < 0))
	crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in indices]
	diffc = diff(crossings)

	# BUG: np.average does not like empty lists
	if len(diffc) > 0:
		return 1.0 * fs / average(diff(crossings))
	else:
		return 0

#=====================================
# Frequency from peak picking
# Strategy: zero crossings of derivative
# Note: Windowing is not used as it degrades the result

def freq_from_peaks(sig,fs):
	sig = diff(sig**2)    # Square to get more prominent peaks
	return freq_from_crossings(sig,fs)/2.0
        
#========================================================================
# Calculate envelope from waveform

def envelopecalc(waveform,peakwindowfactor):
    """
    The input waveform is an np.array, window is in seconds, e.g. 0.01
    windowsamples = fs/peakwindow
    """


    peaksrange = np.arange(len(y)-peakwindow)
    signalpeaks = []
    for i in peaksrange:
        signalpeaks += [np.max(y[i:i+peakwindow])]
    peakwindowhalf = int(round(peakwindow/2.0))
    buffer = [0]*peakwindowhalf
    envelope = np.array(buffer + signalpeaks + buffer)
    return envelope

#========================================================================
# Calculate AEMS
# Maybe apply peak-picking to spectrum, cepstrum-like?

def envelopespectrumcalc(samprate,envelope,hertzmin,hertzmax,medianwindow):
    """
    Calculates the symmetrical spectral slice for the whole envelope:
    1. Calculate FFT with frequencies, amplitudes (and indices, not needed)
    2. Return the frequencies and amplitudes of the second half of the spectrum.
    3. Note that for F0 envelope, samprate is the F0 estimation framerate.
    """

    period = 1.0/samprate
    freqs,w,idx = fft1(envelope,period)
    spectrumlen = len(freqs)

#    spectrumlenhalf = int(round(spectrumlen/2.0))
#    freqs = freqs[:spectrumlenhalf]
#    freqs[0] = 0    # Just in case the middle of the symmetrical spectrum is not 0

    minhertz = freqs[0]; maxhertz = freqs[-1]
    rangehertz = maxhertz - minhertz
    samplesperhertz = int(round(1.0*spectrumlenhalf / maxhertz))

# median filter over magnitudes (1 means switch off) 
    wlog[0] = np.median(wlog)
    wlog = medfilt(wlog,medianwindow)

    return freqs,wlog,spectrumlenhalf,minhertz,maxhertz,rangehertz

#========================================================================
