# modgrapt.py

import os
import numpy as np

"""
Daniel Hirst's min max f0 estimation procedure
We have found that a good approach is to do pitch detection in two 
steps. In the first step you use standard parameters and then from 
the distribution of pitch values, you get the 1st and 3rd quartile 
which we have found are quite well correlated with the minimum and 
maximum pitch, and finally use the estimated min and max for a second 
pitch detection. This avoids a considerable number of octave errors 
which are frequently found when using the standard arguments.
"""
#================================================================
#======== RAPT

def f0rapt(wavfilename, f0min, f0max, framerate, freqweight, (f0filename, f0csvfilename, f0logfilename, paramfilename)):

    """
    wavfilename = 'Data/'+filebase + ".wav"

    f0filename = filebase + ".f0"
    f0csvfilename = filebase + ".csv"
    f0logfilename = filebase + ".f0log"
    paramfilename = "params"
    """

    paramstring = "float min_f0 = "+str(f0min)+";\nfloat max_f0 = " + str(f0max)+";\nfloat frame_step = " + str(framerate) + ";\nfloat freq_weight = " + str(freqweight) + ";\n"

    handle = open(paramfilename, "w")
    handle.write(paramstring)
    handle.close()

    getf0filename = "/opt/esps/bin/get_f0"
    getpplainfilename = "/opt/esps/bin/pplain"
    getf0command = getf0filename+" -P "+paramfilename+" "+wavfilename + " " + f0filename
    getcsvcommand = getpplainfilename+" "+f0filename + " > " + f0csvfilename
    os.system(getf0command + " 2> " + f0logfilename)
    os.system(getcsvcommand + " 2>> " + f0logfilename)
    cleanf0filescommand = "rm -f *.f0 *.csv *log"
#    os.system(cleanf0filescommand)

    handle = open(f0csvfilename,"r")
    csvlines = handle.readlines()
    handle.close()
    csvtable = [ line.split(" ")[:-1] for line in csvlines if line != '' ]
    if len(csvtable) < 1:
        print "No f0 output detected by RAPT."
        exit()
    csvarray = np.array([ map(float,x) for x in csvtable])
    f0list = csvarray[:,0]
    voicelist = csvarray[:,1]

    return f0list,voicelist

