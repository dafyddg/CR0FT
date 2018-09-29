#!/usr/bin/python
# praatf0.py
# D. Gibbon
# Modified from U Reichel: extract_f0.praat
# 2018-08-09

import sys, os, re
import numpy as np
import matplotlib.pyplot as plt

praatscriptname = "f0new.praat"

def writetext(text,filename):
	handle = open(filename, "w")
	handle.write(text)
	handle.close()
	return

def readtext(filename):
	handle = open(filename,"r")
	text = handle.readlines()
	handle.close()
	return text

praatscriptA = """
################################################
## f0 extraction ###############################
################################################

# f0 extraction from audio file
# output: 2 column table <frame timestamp> <f0 value>
# Modified, DG: no timestamp.
# call:
# > extract_f0.praat framelength minfrequ maxfrequ inputDir outputDir \
#   audioExtension f0FileExtension
# example:
# > extract_f0.praat 0.01 50 400 /my/Audio/Dir/ /my/F0/Dir/ wav f0

form input
     word fileName
     real framelength
     real minfrequ
     real maxfrequ
     word dir
     word diro
     word aud_ext
     word f0_ext
endform

#framelength=0.01
#minfrequ=120
#maxfrequ=400
#dir$="."
#diro$="."
#aud_ext$="wav"
#f0_ext$="csv"

#fileName$="wuxi-8s.wav"

## precision of time output
timeprec=round(abs(log10(framelength)))

## main ###############################################################

# generate output file name: e.g. dir$/a.wav --> fo$/a.f0
call get_of
# delete already existing output files
filedelete 'fo$'
# echo 'fileName$' --> 'fo$'
# read audio into sound object
Read from file... 'dir$'/'fileName$'
call get_f0

## gets name of output file (exchanges extension to ext$) ###########

procedure get_of
  fstem$=left$(fileName$,rindex(fileName$,"."))
  fo$="'diro$'/'fstem$''f0_ext$'"
endproc

## extracts and outputs f0 contour ##################################

procedure get_f0
  # from sound object to pitch object pobj
#  pobj = To Pitch... framelength minfrequ maxfrequ
"""

praatscriptB = """
  # frame count
  nof = Get number of frames
  for iof to nof
    # time of current frame
    time = Get time from frame... iof
    # time value to string
    times$ = fixed$('time', 'timeprec')
    # get f0 value of current frame
    pitch = Get value in frame... iof Hertz
    # if no f0 value (pause, supposed measurement error)
    if pitch = undefined
      # append row <time> 0 to fo$
      fileappend 'fo$' 'times$' 0'newline$'
    else
      # append row <time> <pitch> to fo$
      fileappend 'fo$' 'times$' 'pitch''newline$'
    endif
  endfor
endproc
"""

#=================================================================

def praatf0estimate(filebase,f0min,f0max,framerate,type):

	wavfilename = 'Data/'+filebase + ".wav"
	f0filename = 'Data/'+filebase + ".praat.csv"

	if type == "cc":
		praattype = 'pobj = To Pitch (cc): 0, ' + str(f0min) + ', 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, ' + str(f0max)
	elif type == "ac":
		praattype = 'pobj = To Pitch (ac): 0, ' + str(f0min) + ', 15, "no", 0.03, 0.45, 0.01, 0.35, 0.14, ' + str(f0max)
	else:
		praattype = 'pobj = To Pitch... framelength minfrequ maxfrequ'

	praatscript = praatscriptA + praattype + praatscriptB
	writetext(praatscript,praatscriptname)

	paramstring = ' '.join([wavfilename,str(framerate),str(f0min),str(f0max),".",".","wav","praat.csv"])
	praatcommand = "praat --run " + praatscriptname + " " + paramstring

	os.system(praatcommand)
	f0strings = readtext(f0filename)
	f0strings = [ x.rstrip() for x in f0strings ]
	f0table = [ x.split(' ') for x in f0strings ]
	f0table = [ map(float,x) for x in f0table ]
	f0array = np.array(zip(*f0table))
	f0 = f0array[0]
	voice = f0array[1]
	return f0,voice

#=================================================================

