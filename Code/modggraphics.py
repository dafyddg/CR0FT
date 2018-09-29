# modggraphics.py

import numpy as np

import matplotlib
matplotlib.use('Agg')   # WWW and saved graphics files
from matplotlib.mlab import find
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.signal import medfilt, butter, lfilter, freqz, filtfilt, hann, hamming

# Stream to web (resolution of spectrogram is very poor)
# if opsys == "linux":
#    import mpld3       # Stream to web

#==============================================================================
#==============================================================================
# Figure
# Should be working with a dictionary here.
#==============================================================================

fontsize = 12

def graphics(fs, signal, f0vectorvals, polydegree, windows, frequency, modelstuff, periphery, howmanyplotz):

# Note ugly technique for making local variables global.

    global howmanyplots

    howmanyplots = [howmanyplotz][0]

    if howmanyplots == 'multipleplots':
        global fontsize
        global f0min, f0max, f0clipmin, f0clipmax

    vectornames, vectorarray, vectortiming = f0vectorvals
    f0framerate, f0stepfactor = windows
    f0mini,f0maxi, f0clipmini, f0clipmaxi = frequency
    comparison, envelope, spectrogram = modelstuff
    opsys,filebase, figwidth, figheight = periphery

    f0min = [f0mini][0]
    f0max = [f0maxi][0]
    f0clipmin = [f0clipmini][0]
    f0clipmax = [f0clipmaxi][0]

    lensignal = len(signal)
    duration = float(lensignal)/float(fs)

    lenvectornames = len(vectornames)

    veclen = np.min([ len(x) for x in vectorarray if len(x) > 0 ] )
    if veclen > 0:
        vectorarray = np.array( [ x[:veclen] for x in vectorarray ] )

    s0ftvector = s0ftpolyvector = f0polyvector = f0raptvector = f0pyraptvector = f0praatvector = np.array([])
    s0ftflag = s0ftpolyflag = raptflag = pyraptflag = praatflag = spectrogramflag = envelopeflag = comparisonflag = False

    for name,vector,time in zip(vectornames, vectorarray, vectortiming):
        if name == 'S0FT':
            s0ftflag = True
            s0ftvector = vector
            s0fttime = "%.3f"%time
        if name == 'S0FTPOLY':
            s0ftpolyflag = True
            s0ftpolyvector = vector
            s0ftpolytime = "%.3f"%time
        if name == 'RAPT':
            raptflag = True
            f0raptvector = vector
            f0rapttime = "%.3f"%time
        if name == 'PyRapt':
            pyraptflag = True
            f0pyraptvector = vector
            f0pyrapttime = "%.3f"%time
        if name == 'Praat':
            praatflag = True
            f0praatvector = vector
            f0praattime = "%.3f"%time

    if spectrogram == 'on': spectrogramflag = True
    if envelope == 'on': envelopeflag = True
    if comparison == 'on': comparisonflag = True

# Peak height adjustment:
#		more filtering
#   take f0 coherence (std?) into consideration (maybe use RAPT voicing for this)

#==============================================================================
# Define figure

    fontsize = 12

    rowspan = 2; colspan = 3
    rowheight = 1.5	# inches

    validplotcount = 1	# Waveform

    if lenvectornames > 0:
        validplotcount += lenvectornames

    if spectrogramflag:
        validplotcount += 1

    figrows =  validplotcount * rowspan

    spectrogramrows = 3
    figrows += spectrogramrows-rowspan	# 

    figcols = colspan
    figheight = rowheight * figrows

    if howmanyplots == 'singleplot':
        print "The figure consists of a single plot.<br>"
        fig = plt.figure(1,figsize=(figwidth,figheight))

# Initial row and column count and span
    rownum = 0; colnum = 0

#=====================================================================
# Embedded table structure

    if howmanyplots == 'multipleplots':
        print '<table align=\"center\" border=\"1\"><tr><td>'
        print '<table bgcolor=\"lightblue\" align=\"center\" border=\"0\" cellspacing=\"10\" cellpadding=\"10\" width=\"100%\">'

    ax1_title = "S0FT: F0 estimator ("+filebase+")"
    if envelopeflag:
        legend = ['RecSig','Env']
        xlabel = 'Signal waveform (rectified) and amplitude envelope'
    else:
        legend = ['Rect Wave']
        xlabel = 'Signal waveform (full-wave rectified, i.e. abs(waveform))'
    rownum,colnum = waveformplot(duration,fs, signal, figrows, figcols, rownum, colnum, rowspan, colspan,xlabel,ax1_title, legend, envelopeflag)

# There is always a s0ftpolyvector model for s0ftvector.
    if s0ftflag:
        ax2_title = "S0FT - Dafydd Gibbon. Simple F0 Tracker. \nSee Documentation link on input form."
        s0ftcolor = 'b'
        if s0ftpolyflag and polydegree > 0:
            s0ftie = 's0ftpolyflag'
            xlabel = 'Red line: polynomial model, degree '+str(polydegree)
            legend = ['S0FT','Polynomial']
        else:
            s0ftie = 's0ftflag'
            xlabel = 'FFT (spectral peak, zero-crossing, peak-picking)'
            legend = ['S0FT']
        rownum,colnum = f0plot(duration, s0ftvector, s0ftpolyvector, s0fttime, figrows, figcols, rownum, colnum, rowspan, colspan, comparisonflag, ax2_title, xlabel, legend, s0ftcolor,s0ftie)

    if raptflag:
        ax3_title= 'RAPT - David Talkin. A Robust Algorithm for Pitch Tracking (RAPT). In: W. B. Kleijn and K. K. Palatal\n(eds), Speech Coding and Synthesis, pp. 497-518, Elsevier Science B.V., 1995'
        raptcolor = 'g'
        if comparisonflag and s0ftflag:
            comparevector = s0ftvector
            legend = ['RAPT','RAPT-S0FT']
            xlabel = 'Red line: comparison with S0FT, zero raised for visibility'
        else:
            comparevector = []
            legend = ['RAPT']
            xlabel = 'RAPT (enhanced cross-correlation)'
        rownum,colnum = f0plot(duration, f0raptvector, comparevector, f0rapttime, figrows, figcols, rownum, colnum, rowspan, colspan, comparisonflag, ax3_title, xlabel, legend, raptcolor, 'raptflag')

    if pyraptflag:
        ax4_title = "PyRapt - Daniel Gaspari\nMandarin Tone Trainer. Master's thesis, Harvard Extension School, 2016"
        legend = 'PyRapt','RAPT-PyRapt'
        pyraptcolor = 'c'
        if comparisonflag and raptflag:
            comparevector = f0raptvector
            xlabel = 'Red line: comparison with RAPT, zero raised for visibility'
        else:
            comparevector = []
            xlabel = 'PyRapt (modified port of RAPT to Python NumPy & SciPy)'
        rownum,colnum = f0plot(duration, f0pyraptvector, f0raptvector, f0pyrapttime, figrows, figcols, rownum, colnum, rowspan, colspan, comparisonflag, ax4_title, xlabel, legend, pyraptcolor,'pyraptflag')

    if praatflag:
        ax5_title = "Praat 6.0.04 - Boersma, Paul. Praat, a system for doing phonetics by computer.\nGlot International 5:9/10, 341-345. 2001"
        legend = 'Praat','RAPT-Praat'
        pyraptcolor = 'violet'
        if comparisonflag and raptflag:
            comparevector = f0raptvector
            xlabel = 'Red line: comparison with RAPT, zero raised for visibility'
        else:
            comparevector = []
            xlabel = 'Praat (default autocorrelation setting)'
        rownum,colnum = f0plot(duration, f0praatvector, f0raptvector, f0praattime, figrows, figcols, rownum, colnum, rowspan, colspan, comparisonflag, ax5_title, xlabel, legend, pyraptcolor, 'praatflag')

    if spectrogramflag:
        ax6_title = "Spectrogram"
        rownum,colnum = spectrogramplot(duration, fs, signal, figrows, figcols, rownum, colnum, spectrogramrows, colspan, ax6_title)

    if howmanyplots == 'multipleplots':
        print "</table>"						# Inner table with colour and no border between figures.
        print "</td></tr></table>"	# Outer table with border

    return

#==============================================================================
# Waveform plot

def waveformplot(duration,fs,signal,fr,fc,rn,cn,rs,cs,title,xlabel,legend, envelopeflag):

    if howmanyplots == 'multipleplots':
        fig = plt.figure(1,figsize=(10,3))

    ax1 = plt.subplot2grid((1,1), (0,0),rowspan=2,colspan=1)
#    ax1 = plt.subplot2grid((fr,fc), (rn,cn),rowspan=rs,colspan=cs)
    ax1.set_title(title,fontsize=fontsize)
    plt.grid(linewidth=2,which='both', axis='x')

    ax1.set_ylabel('Rectified Amplitude')
    y = signal
    y = np.array(y)
    y = abs(y)

    x = [ 1.0*x/fs for x in range(len(y)) ]

    plt.plot(x, y, color='lightblue', linewidth=5)
    ax1.set_yticks([])

    if envelopeflag:
        k = int(round(fs * 0.02))	# peak search window length in samples
        envelope = envelopecalc(y,k)
        ax1.set_ylim(0, max(max(y),max(envelope))*1.05)
        plt.plot(x,envelope,linewidth=6,color='pink')
        plt.plot(x,envelope,linewidth=1,color='r')
    else:
        ax1.set_ylim(-0.1, max(y)*1.05)

    xmin = -0.01
    xmax = np.ceil(x[-1])+0.01

    plt.hlines(0,xmin,xmax)
    ax1.set_xlim(xmin,xmax)

    plt.legend(legend,fontsize=fontsize-3)

    if howmanyplots == 'multipleplots':
        plt.tight_layout()
        webfilename = '/Webtmp/s0ftwaveform.png'
        localfilename = '/var/www/Webtmp/s0ftwaveform.png'
        plt.savefig(localfilename)
        print '<tr align=\"center\"><td><img src='+webfilename+' width=\"100%\"></td></tr>'
        plt.close('all')

    return rn+rs, cn
#==============================================================================

def fft1(e,period):
        w = np.abs(np.fft.fft(e))**2
        freqs = np.abs(np.fft.fftfreq(e.size,period))
        idx = np.argsort(freqs)
        return freqs,w,idx
 
#==============================================================================

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
# RAPT, PyRapt and Praat plots

def f0plotwrapper(duration,f0vector,z,fr,fc,rn,cn,rs,cs,comparisonflag,title,xlabel, legend, color, flag,f0mini,f0maxi,f0clipmini,f0clipmaxi):

    global f0min, f0max
    f0min = [f0mini][0]
    f0max = [f0maxi][0]

    global f0clipmin, f0clipmax
    f0clipmin = [f0clipmini][0]
    f0clipmax = [f0clipmaxi][0]

    f0plot(duration,f0vector,z,fr,fc,rn,cn,rs,cs,comparisonflag,title,xlabel, legend, color, flag)

    return rn+rs, cn


def f0plot(duration,f0vector,z, proctime, fr,fc,rn,cn,rs,cs,comparisonflag,title,xlabel, legend, color, flag):

    if howmanyplots == 'multipleplots':
        fig = plt.figure(1,figsize=(10,3))

    ax3 = plt.subplot2grid((1,1), (0,0),rowspan=2,colspan=1)
#    ax3 = plt.subplot2grid((fr,fc), (rn,cn),rowspan=rs,colspan=cs)
    ax3.set_title(title,fontsize=fontsize)
    ax3.set_xlabel(xlabel + " (processing time: "+proctime+"s)")

    plt.ylabel('Freq [Hz]',fontsize=fontsize)
    plt.grid(color='b', linestyle='--', linewidth=1)

    y = f0vector
    leny = len(y)
    lenz = len(z)

    xaxis = np.linspace(0,np.ceil(duration),leny,endpoint=True,dtype=np.float16)

    ax3.set_xlim(-0.01,np.ceil(duration))
    ax3.set_ylim(f0min,f0max)

    plt.scatter(xaxis,y,s=10,color=color)

    if lenz > 0 and not flag == 's0ftflag':

        if comparisonflag and flag in ['raptflag','pyraptflag','praatflag']:
            z = np.mean(f0max-f0min)+np.array(y - z)

        if comparisonflag or flag == 's0ftpolyflag':
            for xx,yy,pp in zip(xaxis,y,z):
                    if yy > f0clipmin and yy<f0clipmax:
                        plt.scatter(xx,pp,s=10,color='r')

    plt.legend(legend,fontsize=fontsize-3)

    if howmanyplots == 'multipleplots':
        plt.tight_layout()
        webfilename = '/Webtmp/s0ft'+flag+'.png'
        localfilename = '/var/www/Webtmp/s0ft'+flag+'.png'
        plt.savefig(localfilename)
        print '<tr align=\"center\"><td><img src='+webfilename+' width=\"100%\"></td></tr>'
        plt.close('all')

    return rn+rs, cn

#==============================================================================
# Spectrogram plot

def spectrogramplot(duration,fs,y,fr,fc,rn,cn,rs,cs,title):

    if howmanyplots == 'multipleplots':
        fig = plt.figure(1,figsize=(10,3))

    ax5 = plt.subplot2grid((1,1), (0,0),rowspan=2,colspan=1)
#    ax5 = plt.subplot2grid((fr,fc), (rn,cn),rowspan=rs,colspan=cs)
    ax5.set_title("Spectrogram", fontsize=fontsize)
    plt.ylabel('Freq [Hz]',fontsize=fontsize)
    plt.grid(color='b', linestyle='--', linewidth=1)

    specmin = 0
    specmax = 1000
    nykvistspect = int(round(fs/2.0))
    if nykvistspect < 3000:
        specmax = nykvistspect
    specwinfactor = 0.01
    specwin = int(float(65536*specwinfactor))

    NFFT = specwin
    ax5.specgram(y, NFFT=NFFT, Fs=fs)
    plt.axis(ymin=specmin, ymax=specmax)
    ax5.set_xlim(0.0,np.ceil(duration))

    if howmanyplots == 'multipleplots':
        plt.tight_layout()
        webfilename = '/Webtmp/s0ftspectrogram.png'
        localfilename = '/var/www/Webtmp/s0ftspectrogram.png'
        print '<tr align=\"center\"><td><img src='+webfilename+' width=\"100%\"></td></tr>'
        plt.savefig(localfilename)
        plt.close('all')

    return rn+rs, cn

#==================================================================

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

# EOF =============================================================
