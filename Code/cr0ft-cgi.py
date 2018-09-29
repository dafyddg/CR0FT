#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division

print "Content-type: text/html\n\n"

##!/usr/pkgsrc/20140707/bin/python2.7
##!/usr/bin/python
# s0ft-cgi.py
# D. Gibbon
# S0FT - Simple F0 Tracker
# CLI created: 2018-08-08
# CGI created: 2018-08-14
# PyRapt integrated: 2018-08-19f

#==============================================================
# Hard-wired stuff

opsys = "linux"
# opsys = "solaris"

pyraptswitch = "False"

#==============================================================

# Import modules (libraries)

# System and CGI builtins

import os, sys, re, time
import cgi, cgitb; cgitb.enable()
from cgi import escape

# Math - required module, install numpy

import numpy as np

# Graphics computing - required module, install matplotlib

import matplotlib
matplotlib.use('Agg')   # WWW and saved graphics files

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Stream to web (resolution of spectrogram is very poor)
# if opsys == "linux":
#    import mpld3       # Stream to web

import scipy.io.wavfile as wave
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import euclidean, cityblock

# S0FT modules

import modgf0x
import modgrapt
import modgpyrapt
import modgpraat
import modggraphics

#==============================================================

# GLOBAL STUFF:

# CGI HTML input, main caller, HTML figure output

#==============================================================
# CGI handling


cgifields = [
    # External user and admin information
	  'metadata', 'filebase',
    # F0 algorithms
    'fft', 'zerox', 'peaks', 'polymodel',
    'comparisonflag','envelopeflag',
    's0ftflag','raptflag', 'pyraptflag', 'praatflag','spectrogramflag',
    # Waveform handling:  downsampling, centre-clipping, Butterworth filters
   'downsample','cutofflo','orderlo','cutoffhi','orderhi','centreclip',
    # F0 sample window and step
    'f0framerate','f0stepfactor',
    # Outlier handling, downsampling
    'f0clipmin','f0clipmax','f0median',
    # Noise filter (negative effect on current examples)
    'voicewin','voicediff',
    # Display dimensions
    'figwidth', 'figheight', 'f0min','f0max','howmanyplots',
    # Shortcut for higher pitch and lower pitch voices
    'voicetype'
    ]

#======================
# Fetch CGI parameters as dictionary

def cgitransferlines(cgifields):
    fs = cgi.FieldStorage()
    fieldvalues = []
    for field in cgifields:
        if fs.has_key(field):
            fieldname = fs[field].value
        else:
            fieldname = '1066'
        fieldvalues = fieldvalues + [(field,fieldname)]
    return dict(fieldvalues)

#======================
# Decode CGI parameters

cgiparams = cgitransferlines(cgifields)

paramlist = map(str,[ cgiparams[x] for x in cgifields[1:] ])
paramzipped = zip(cgifields[1:],paramlist)
paramjoined = [ ':'.join(x) for x in paramzipped ]


#==============================================================
# Define file addresses

filebase = cgiparams['filebase']

wavfilename = "Data/" + filebase + ".wav"

if opsys == "linux":
    figlocaladdress = '/var/www/Webtmp/'+filebase+'.png'
    figwebaddress = '/Webtmp/'+filebase+'.png'
    audioaddress = wavfilename

    f0filename = "/var/www/Webtmp/"+filebase + ".f0"
    f0csvfilename = "/var/www/Webtmp/"+filebase + ".csv"
    f0logfilename = "/var/www/Webtmp/"+filebase + ".f0log"
    paramfilename = "/var/www/Webtmp/"+"params"

else:
    figlocaladdress = '../../docs/Webtmp/'+filebase+'.png'
    figwebaddress = '/gibbon/Webtmp/'+filebase+'.png'
    audioaddress = '/gibbon/S0FT/'+wavfilename

    f0filename = "../../docs/Webtmp/"+filebase + ".f0"
    f0csvfilename = "../../docs/Webtmp/"+filebase + ".csv"
    f0logfilename = "../../docs/Webtmp/"+filebase + ".f0log"

#==============================================================

def inithtml():
    print '<html><head><title>CR0FT - Comparison of Robust F0 Trackers. D. Gibbon</title>'
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

def terminatehtml():
    print "</body></html>"
    return

#==============================================================
# F0 vector calculation for f0x, rapt, pyrapt, praat

def calculatef0vectors(f0args,displayflags):

    signal, wavfilename, filebase,fs, fft, zerox, peaks, f0framerate, f0stepfactor, f0median, f0min, f0max, centreclip, f0clipmin, f0clipmax, cutoffhi, orderhi, cutofflo, orderlo = f0args

    vectorlist = []
    timelist = []
    vectorflaglist = []

#======================
# S0FT F0 estimation with zero-crossing and peak-picking
    try:
        if s0ftflag == 'on':
            begin = time.time()
            f0xvector, f0xpoly = modgf0x.f0x(signal, fs, fft, zerox, peaks, polydegree, f0framerate, f0stepfactor, f0median, f0min, f0max, centreclip, f0clipmin, f0clipmax, cutoffhi, orderhi, cutofflo, orderlo)
            end = time.time()
            timef0x = end - begin
            vectorflaglist += ['S0FT','S0FTPOLY']
            vectorlist += [f0xvector,f0xpoly]
            timelist += [timef0x,timef0x]
    except:
        print "Error calculating S0FT with this example. One possible cause is that the F0 sample window may be too short or too long for the F0 range of the data or the filter settings. Try the default value and different parameters, and check with other data."
        exit()

#======================
# RAPT
    try:
        if raptflag == 'on':
            begin = time.time()
            raptfilenames = f0filename, f0csvfilename, f0logfilename, paramfilename
            f0raptvector = modgrapt.f0rapt(wavfilename, f0min, f0max, 0.01, 0.02,raptfilenames)[0]
            end = time.time()
            timef0rapt = end - begin
            vectorflaglist += ['RAPT']
            vectorlist += [f0raptvector]
            timelist += [timef0rapt]
    except:
        print "RAPT is not available on this server."
        exit()

#======================
# PyRapt

    if pyraptflag == 'on':
#        print "Note: PyRapt is slow (&asymp;50s on this server).<b>"
        begin = time.time()
        f0pyraptvector = modgpyrapt.rapt(wavfilename)

        end = time.time()
        timef0pyrapt = end - begin
        vectorflaglist += ['PyRapt']
        vectorlist += [f0pyraptvector]
        timelist += [timef0pyrapt]

#======================
# Praat
# type: ac, cc, default
# The settings 'ac', 'cc' cause a weird vector length effect,
# which also affects the other pitch extractors. Must investigate.
    if True:
        if praatflag == 'on':

            begin = time.time()
            type = 'default'
            x,y = modgpraat.praatf0estimate(filebase,f0min,f0max,0.01,type)
            f0praatvector = y
            end = time.time()
            timef0praat = end - begin
            vectorflaglist += ['Praat']
            vectorlist += [f0praatvector]
            timelist += [timef0praat]
    if False:
        print "Praat is not available on this server."
        exit()

#==============================================================
# Length equalised to length of shortest vector

    f0length = 0
    f0lengths = map(len, vectorlist)
    if len(f0lengths) > 0:
        f0length = np.min(f0lengths)
        vectorlist = [ v[:f0length] for v in vectorlist ]
    else:
        print "Error: no algorithms selected."
        exit()

# Praat has a quite different vector length for params ac, cc

    return vectorflaglist, np.array(vectorlist), timelist

#==============================================================================
#==============================================================================
# Calculate f0 vector distances

#f0xmedfill,f0raptmedfill,f0pyraptmedfill,f0xpyraptpearson,evaltable = vectordistances(f0xvector,f0raptvector,f0pyraptvector,f0clipmin,f0clipmax)

def vectordistances(f0vectorvals):
    vectornames, vectors, vectortiming = f0vectorvals

    pearsoncorrelations = []
    diffsequences = []
    absnormdiffs = []
    euclideandist = []
    cityblockdist = []
    i = -1
    for name01,vector01,time01 in zip(vectornames,vectors,vectortiming)[:-1]:
        i += 1
        for name02,vector02,time02 in zip(vectornames,vectors,vectortiming)[i:]:
            if name01 != name02:
                pearson_r,significance = pearsonr(vector01,vector02)
#                pearson_r = np.corrcoef(vector01,vector02)[0][1]
# Trap for borderline cases of Pearson correlation failure,
# adding a small amount of noise which does not affect the rounded output.
# This NaN (not a number) failure occurs, for example,
# when an F0 vector happens to contain identical numbers.
# This occurs with PyRapt, which generates F0 vectors with identical values of 200
# with the sinusoidal calibration data. Strangely, the other software does not.
# This issue is not captured in the pearsonr and corrcoef library functions.
                if np.isnan(pearson_r):
                    vector01 = vector01 + np.random.random(len(vector01))*0.00001-0.5
                    vector02 = vector02 + np.random.random(len(vector02))*0.00001-0.5
                    pearson_r,significance = pearsonr(vector01,vector02)
                pearsoncorrelations += [ [name01,name02,pearson_r,significance] ]

                diffsequences += [ vector01-vector02 ]

                vabsdiff = np.abs(vector01 - vector02)
                vsum = vector01 + vector02 + 0.0000001
                vnormabsdiff = np.mean(vabsdiff / vsum)
                absnormdiffs += [vnormabsdiff]

                eucl = euclidean(vector01,vector02)
                euclideandist += [ eucl ]

                taxicab = cityblock(vector01,vector02)
                cityblockdist += [ taxicab ]

    return pearsoncorrelations,diffsequences,absnormdiffs,euclideandist, cityblockdist

#==============================================================
# Export vectors

def exportvectors(f0xector,f0raptvector,f0pyraptvector):

# jiayan,f
# liangjj,f
# liupeng,m
# liuqp,f
# shjj,m
# tangzd,m
# wangwei,f
# wuxi,f
# wn,m
    export = True
    if export:

        print "<b>Writing S0FT and RAPT vectors to file.</b>"

        expolen = np.min([len(f0xvector),len(f0raptvector)])

        f0xstring = ','.join(map(str,f0xvector[:expolen]))
        f0raptstring = ','.join(map(str,f0raptvector[:expolen]))

        xstring = ','.join(map(str,[ x * 0.01 for x in range(expolen) ]))

        handle = open("/home/gibbon/Desktop/S0FTexport-"+filebase+".csv", "w")
        handle.write("S0FT,"+filebase+","+f0xstring+"\n")
        handle.write("RAPT,"+filebase+","+f0raptstring+"\n")
        handle.write("Time,"+filebase+","+xstring+"\n")

#        handle.write("# "+filebase+" S0FT:PyRapt distances: Pearson: "+"%.3f"%(1.0-f0xpyraptpearson)+", Normalised Manhattan: "+"%.3f"%(1.0-rsratiomean)+", Normalised Euclid: "+"%.3f"%(1.0-rseuclid))

        handle.close()
    return

#==============================================================
# This needs to be generalised.

def valuetable(fs,signallength,signalduration,f0winsamples,f0step,voicetype,audioaddress):

    print '<table align="center" valign="top">'

    print '<tr valign="top" align="center"><td colspan="3"><b>'+metadata,'</b></td></tr>'

    print '<tr><td>'

    print '<table align="center" valign="top">'
    print '<tr><td>Sampling rate:</td><td>',fs,'Hz</td></tr>'
    print '<tr><td>Signal length:</td><td>',signallength,'samples</td></tr>'
    print '<tr><td>Signal duration:</td><td>',signalduration,'s</td></tr>'
    print '</table>'

    print '</td><td>'

    print '<table align="center" valign="top">'
    print '<tr><td>F0 sample window:</td><td>',f0framerate,'(s),',f0winsamples,'samples</td></tr>'
    print '<tr><td>F0 window step:</td><td>factor',f0stepfactor,",", f0step,'samples</td></tr>'
    print '<tr valign="top"><td>Voice selection:</td><td>"'+voicetype+'"</td></tr>'
    print '</table>'

    print '</td><td>'
    print '<table>'
    print '<tr><td>'
    print '<audio controls style="width: 350px;">'
    print '<source src="'+audioaddress+'" type="audio/wav" length="100%">'
    print 'Your browser does not support the audio element.</audio>'
    print '</td></tr></table>'

    print '</td></tr></table>'

    return

#==============================================================
# Format distance table

def formatdistancetable(distanceindices):

    pearsoncorrelations,diffsequences,absnormdiffs,euclideandist,manhattandist = distanceindices

    distancerows = ""
    for pc,absnormdiff,euclid,manhattan in zip(pearsoncorrelations,absnormdiffs,euclideandist,manhattandist):
        absnormdiffstring = "%.3f"%absnormdiff
        euclidstring = "%.3f"%euclid
        manhattanstring = "%.3f"%manhattan
        name01,name02,r,p = pc
        rstring = "%.3f"%r
        if r != 0.0: dstring = "%.3f"%(1-r)	# Why this condition?
        else: dstring = '<i>huge</i>'
        if p < 0.001: p = '<i>p</i> << 0.01'
        elif p < 0.01: p = '<i>p</i> < 0.01'
        elif p < 0.05: p = '<i>p</i> < 0.05'
        else:
           p = 'no'
        text = ['<i>'+name01+'</i>:<i>'+name02+'</i>', rstring, dstring, p, absnormdiffstring, euclidstring, manhattanstring]

        distancerows += "<tr valign=\"top\"><td align=center>"+"</td><td align=center>".join(text)+"</td></tr>\n"

    pearsoncomment = """<font size =-2><b>Distance measures:</b><ol>
<li>Pearson\'s <i>r</i> reflects the <i>shape</i> of the vector, and is invariant with respect to height and magnitude.
<li>Mean normalised absolute difference (||v<sub>1</sub>-v<sub>2</sub>||, average of absolute differences normalised by dividing by the sums) reflects the <i>combined difference</i> in height, magnitude and shape.
<li>Euclidean distance (square root of the sum of squared differences) reflects the <i>combined difference</i> but as absolute, not normalised value.
<li>Manhattan distance (the non-averaged sum of the absolute distances) reflects the <i>combined difference</i>, reducing the effect of outliers; basically the non-normalised version of the normalised absolute difference.
</ol>
These measures are the most common of a wide range of distance measures.</font>"""

    header01 = '<b>Relations between F0 extractor results and RAPT as benchmark</b>'
    header02 = '<tr valign=\"top\" align=\"center\"><td><b>'+ '</b></td><td><b>'.join(['Estimators','Pearson\'s <i>r</i>','Correlation Distance','Significance','||v<sub>1</sub>-v<sub>2</sub>||','Euclidean','Manhattan'])+'</td></tr>'

    distancetable = """
<p><table>
<tr valign=\"top\">
    <td>
        <table cellspacing=5 cell cellpadding=5>
            <tr valign=\"top\" align=\"center\">
                <td colspan="7">\n"""+header01+"""</td>
            </tr>
            <tr valign=\",top\">
                <td>"""+header02+'\n'+distancerows+"""</td>
            </tr>
        </table>
    </td>
    <td width=\"40%\" valign=\"middle\">"""+pearsoncomment+"""</td>
</tr>
</table></p>\n"""

    return distancetable

#===================================================================
# Output figure table

def figuretable(figwebaddress):
    print '<br><table align="center" width="100%">'
    print '<tr align="center"></td><td>'
    print '<img src='+figwebaddress+' width="90%">'
    print '</td></tr></table>'

#    print '<a href=\"' + figwebaddress + '\">Downloadable figure</a>'
# mpld3 has very poor resolution of the spectrogram
#    htmlstr = mpld3.fig_to_html(fig)
#    print htmlstr

    return

#==============================================================================
#==============================================================================
# Main caller

#==============================================================
# Fetch waveform and sample rate
# Maybe apply downsample after low pass filtering? (cf. RAPT, PyRapt)

if True:

#    print filebase,wavfilename
#    exit()

    fs,signal = wave.read(wavfilename)

    nykvistfrequency = 2000
    downsamplemax = int(round(1.0*fs/nykvistfrequency))

    try:
        downsample = int(cgiparams['downsample'])
    except:
        print "Warning: downsampling factor is "+str(downsample)+", must be a positive integer.<br>"

    if downsample < 1:
        print "Warning: downsampling factor is "+str(downsample)+", must be > 1<br>."

    if downsample > downsamplemax:
        print "Warning: Downsampling factor ",downsample,"infringes a relatively safe Nykvist condition of",nykvistfrequency,"Hz for sampling speech F0.<br>Infringement of the Nykvist condition (F<sub>sample</sub> > 2&#8901;F<sub>max</sub>) will distort the F0 track and narrow the spectrogram range, and may trigger a processing error.<br>"

#==============================================================
# Decode CGI parameters

    voicetype = cgiparams['voicetype']

    if voicetype == 'low':
        f0min = 50
        f0max = 250
        cutoffhi =  95
        orderhi = 3
        cutofflo = 135
        orderlo = 5
        f0clipmin = 60
        f0clipmax = 250
        centreclip = 0.07

    elif voicetype == 'high':
        f0min = 120
        f0max = 350
        cutoffhi = 150
        orderhi = 3
        cutofflo = 250
        orderlo = 5
        f0clipmin = 120
        f0clipmax = 350
        centreclip = 0.15

    else:
        f0min = int(cgiparams['f0min'])
        f0max = int(cgiparams['f0max'])
        cutoffhi = int(cgiparams['cutoffhi'])
        orderhi = int(cgiparams['orderhi'])
        cutofflo = int(cgiparams['cutofflo'])
        orderlo = int(cgiparams['orderlo'])
        f0clipmin = int(cgiparams['f0clipmin'])
        f0clipmax = int(cgiparams['f0clipmax'])
        centreclip = float(cgiparams['centreclip'])

    metadata = cgiparams['metadata']

    fft = cgiparams['fft']
    zerox = cgiparams['zerox']
    peaks = cgiparams['peaks']

    comparisonflag = cgiparams['comparisonflag']
    envelopeflag = cgiparams['envelopeflag']
    s0ftflag = cgiparams['s0ftflag']
    raptflag = cgiparams['raptflag']
    pyraptflag = cgiparams['pyraptflag']
    praatflag = cgiparams['praatflag']
    spectrogramflag = cgiparams['spectrogramflag']

    displayflags = [
        comparisonflag, envelopeflag,
        s0ftflag, raptflag, pyraptflag, praatflag, spectrogramflag
        ]

    polydegree = int(cgiparams['polymodel'])
    if polydegree > 40:
        print "Polydegree ",polydegree," reset to 40."
        polydegree = 40
    if polydegree < 0:
        print "Polydegree ",polydegree," reset to 0."
        polydegree = 0

    f0framerate = float(cgiparams['f0framerate'])
    f0stepfactor = int(cgiparams['f0stepfactor'])

    f0median = int(cgiparams['f0median'])

    voicewin = int(cgiparams['voicewin'])
    voicediff = int(cgiparams['voicediff'])

    figwidth = int(cgiparams['figwidth'])
    figheight = int(cgiparams['figheight'])
    howmanyplots = cgiparams['howmanyplots']

if False:
    print "Parameter value error."
    exit()

#==============================================================================
#==============================================================================

# Downsample signal

try:
    signal = np.array(signal[:])
    signal = signal[::downsample] 
    fs = int(round(1.0*fs/downsample))

    signallength = len(signal)
    signalduration = float(signallength) / float(fs)

    f0winsamples = int(round(fs * f0framerate))
    f0step = int(round(f0winsamples/f0stepfactor))

except:
    print "Error fetching signal."
    exit()

#==============================================================================
#==============================================================================
# Calculate f0vectors

print "<b>F0 framerate:",f0framerate,"Step Factor:",f0stepfactor,"Downsample:",downsample,"Signal length:",len(signal),"</b><br>"

if True:

    f0vectorargs = signal, wavfilename, filebase,fs, fft, zerox, peaks, f0framerate, f0stepfactor, f0median, f0min, f0max, centreclip, f0clipmin, f0clipmax, cutoffhi, orderhi, cutofflo, orderlo

    f0vectorvals = calculatef0vectors(f0vectorargs,displayflags)

#    vectornames, vectorarray, vectortiming = f0vectorvals


if False:

    print "Frequency vectors calculation error."
    exit()

#==============================================================================
# Calculate comparisons between F0 vectors

if comparisonflag == 'on':

    if True:

        distanceindices = vectordistances(f0vectorvals)

        distancetable = formatdistancetable(distanceindices)

    if False:
        print "Error in calculating distances."
        exit()

#==============================================================================
"""
Comment from the web:
Pearson Correlation
Pearson Correlation measures the similarity in shape between two profiles. The formula for the Pearson Correlation distance is: d = 1 - r
where r = Z(x)·Z(y)/n
is the dot product of the z-scores of the vectors x and y. The z-score of x is constructed by subtracting from x its mean and dividing by its standard deviation.

Also note that "Pearson distance" is a weighted type of Euclidean distance, and not the "correlation distance" using Pearson correlation coefficient.

Try Euclidean distance, because Pearson scales with the same outcome for multiplication and addition.
"""
#==============================================================================
#==============================================================================
# Output caller

inithtml()

# Make and print basic values of the signal used by s0ft:
valuetable(fs, signallength, signalduration, f0winsamples, f0step, voicetype, audioaddress)

if comparisonflag == 'on':
   print distancetable

#print "<p align=center>System time: S0FT %.2fs, RAPT %.2fs, PyRapt %.2fs, PyRapt/S0FT %.2f:1</p>"%(timef0x,timef0rapt,timef0pyrapt,timef0pyrapt/timef0x)
#print "</p>"

#==============================================================================
#==============================================================================
# Figure from module modggraphics - MOVE FUNCTION!

windows = f0framerate, f0stepfactor
frequency = f0min,f0max, f0clipmin, f0clipmax
periphery = opsys,filebase, figwidth, figheight
modelstuff = comparisonflag, envelopeflag, spectrogramflag

fig = modggraphics.graphics(fs, signal, f0vectorvals, polydegree, windows, frequency, modelstuff, periphery,howmanyplots)

# Only applies to single figure version:

if howmanyplots == 'singleplot':
    plt.savefig(figlocaladdress)
    figuretable(figwebaddress)

plt.close()

#==============================================================================
#==============================================================================

terminatehtml()

#==================================================================
# EOF =============================================================
