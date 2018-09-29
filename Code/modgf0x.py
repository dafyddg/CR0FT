# modgf0x.py

# Math computing - required module, install math

import numpy as np
from numpy.fft import rfft, irfft       # irfft not used
from numpy import argmax, sqrt, mean, absolute, linspace, log10, logical_and, average, diff, correlate

# Maybe replace this?

from matplotlib.mlab import find

# Scientific computing - required module, install scipy

from scipy.signal import medfilt
from scipy.signal import blackmanharris, fftconvolve

from scipy.signal import butter, lfilter, freqz, filtfilt, hann, hamming
from scipy.interpolate import interp1d

#==============================================================
# Regression line

def polyregline(x,y,d):
    x = range(len(y))
    fit, res, _, _, _ = np.polyfit(x, y, d, full=True)
    yfit = np.polyval(fit,x)
    return yfit
# fit with np.polyfit; linear (degree 1) regression only

def polyregline2(x,y,d):
    m, b, _, _, _ = stats.linregress(x, y)
    poly = [ m * xx + b for xx in x]
    return poly

#==============================================================
# Butterworth filters
# Source: adapted from various internet Q&A sites

def butter_lowpass(cutoff, fs, order):
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='low', analog=False)
	return b, a

def butter_lowpass_filter(data, cutoff, fs, order):
	b, a = butter_lowpass(cutoff, fs, order=order)
	y = lfilter(b, a, data)
	return y

def butter_highpass(cutoff, fs, order):
	nyq = 0.5 * fs
	normal_cutoff = cutoff / nyq
	b, a = butter(order, normal_cutoff, btype='high', analog=False)
	return b, a

def butter_highpass_filter(data, cutoff, fs, order):
	b, a = butter_highpass(cutoff, fs, order=order)
	y = lfilter(b, a, data)
	return y

def butterworth(signal,fs,cutoffhi,orderhi,cutofflo,orderlo):
	buttery = signal
	buttery = butter_highpass_filter(buttery, cutoffhi, fs, orderhi)
	buttery = butter_lowpass_filter(buttery, cutofflo, fs, orderlo)
	return buttery

#==============================================================================
# Calculate FFT from signal using sampling period

def fft1(e,period):
        w = np.abs(np.fft.fft(e))**2
        freqs = np.abs(np.fft.fftfreq(e.size,period))
        idx = np.argsort(freqs)
        return freqs,w,idx

#==============================================================================
# Calculation of signal envelope from rectified signal, with peak detection window length.
def envelopecalc(rectifiedsignal,amwindowlen):
    peaksrange = np.arange(len(rectifiedsignal)-amwindowlen)
    signalpeaks = []
    for i in peaksrange:
        signalpeaks += [np.max(rectifiedsignal[i:i+amwindowlen])]
    amwindowhalf = int(round(amwindowlen/2.0))
#    fillvalue = np.median(signalpeaks)
    fillvalue = 0
    filler = [fillvalue]*amwindowhalf
    envelope = np.array(filler + signalpeaks + filler)
    return envelope
#==============================================================================

#==============================================================
# F0 detection algorithms
# Source: adapted from various internet Q&A sites

#=====================================
# FFT

def parabolic(f, x):
    xv = 1.0/2.0 * (f[x-1] - f[x+1]) / float(f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1.0/4.0 * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

#=================

def freq_from_fft(sig, fs):
    windowed = sig * blackmanharris(len(sig))
    f = rfft(windowed)
    i = argmax(abs(f)) # Just use this for less-accurate, naive version
    true_i = parabolic(abs(f), i)[0]
    return fs * true_i / len(windowed)

#=====================================
# Zero crossings
# Note: Windowing is not used as it degrades the result

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
# Voice detection - this is not used; degrades the result
# Accepted if average absolute difference in voicewin < voicediff 

def voicedetector(f0vector,voicewin,voicediff):
    f0new = []
    for i in range(len(f0vector[:-voicewin])):
        dispwin = f0xvector[i:i+voicewin]
        try:
            if np.median(np.abs(np.diff(dispwin))) < voicediff:
                f0new += [f0xvector[i]]
        except:
            if np.mean(np.abs(np.diff(dispwin))) < voicediff:
                f0new += [f0xvector[i]]
            else:
                f0new += [0]
    return f0new

#=====================================
# Peak picking
# Strategy: zero crossings of derivative
# Note: Windowing is not used as it degrades the result

def freq_from_peaks(sig,fs):
	sig = diff(sig**2)
	return freq_from_crossings(sig,fs)/2.0	

#=====================================
# S0FT hybrid
# Strategy: combine (by multiplying) up to 3 algorithms
# Outliers are reduced to zero, so multiplying zeroes disagreements

def f0x(signal, fs, fft, zerox, peaks, polydegree, f0framerate, f0stepfactor, f0median, f0min, f0max, centreclip, f0clipmin, f0clipmax, cutoffhi, orderhi, cutofflo, orderlo):


        f0winsamples = int(round(fs * f0framerate))
        f0step = int(round(f0winsamples/f0stepfactor))

        y = butterworth(signal, fs, cutoffhi, orderhi, cutofflo, orderlo)

        yabs = np.abs(y)
        ymax = np.max(yabs)
        ymin = np.min(yabs)
        ydiff = ymax-ymin

        clip = ymin + int(round(ydiff * centreclip))
#       perc = np.percentile(y,clipfactor)
        y = np.array([ yy if abs(yy)>clip else 0 for yy in y ])

        hannwin = hann(f0winsamples)    # Zero ends
        hammingwin = hamming(f0winsamples,sym=True)    # Non-zero ends
        hammingzwin = hamming(f0winsamples,sym=False)   # Zero ends

        f0xvector = []
        for i in range(0,len(y[:])-f0winsamples,f0step):
                sigwin = y[i:i+f0winsamples]
##               sigwin = hannwin * sigwin
##               sigwin = hammingwin * sigwin
##               sigwin = hammingzwin * sigwin
                f1 = freq_from_fft(sigwin,fs)
                f2 = freq_from_crossings(sigwin,fs)
                f3 = freq_from_peaks(sigwin,fs)

                if fft == "on" and zerox == "on" and peaks == "on":
                        f = np.array((f1 * f2 * f3)**0.333)
                elif fft == "on" and peaks == "on":
                        f = np.array((f1 * f3)**0.5)
                elif fft == "on" and zerox == "on":
                        f = np.array((f1 * f2)**0.5)
                elif zerox == "on" and peaks == "on":
                        f = np.array((f2 * f3)**0.5)
                elif fft == "on":
                    f = f1
                elif zerox == "on":
                    f = f2
                elif peaks == "on":
                    f = f3
                try:
                        fint = int(round(f))
                except:
#                        print "Indeterminate (nan exception)"
                        fint = np.median(sigwin)
#                       fint =f0min
#                       fint = 0
                f0xvector += [fint]

# Polynomial model of f0xvector via f0xmedfill
        f0squashvec = np.array([ x for x in f0xvector if x<f0clipmax and x>f0clipmin ])
        f0xmed = np.median(f0squashvec)
        f0xmedfill = np.array([ x if x<f0clipmax and x>f0clipmin else f0xmed  for x in f0xvector ])
        f0xpoly = np.array(polyregline(range(len(f0xmedfill)),f0xmedfill,polydegree))

# F0 median filter
        f0xvector = medfilt(f0xvector,f0median)
# F0 low and high clipping
        f0xvector = np.array([ yy if yy>f0clipmin else 0 for yy in f0xvector])
        f0xvector = np.array([ yy if yy<f0clipmax else 0 for yy in f0xvector])

#======================
# Noise / voice detection from F0 and removal 
# Worsens result, but only slighly in optimal setting

#    f0xvector = np.array(voicedetector(f0xvector,voicewin,voicediff))

        return f0xvector, f0xpoly

