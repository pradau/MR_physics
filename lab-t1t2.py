#!/usr/bin/env python
# Calculate the T1 from T1w images that have several TR's, and constant TE.
# Calculate the T2 from T2w images that have several TE's, and constant TR.
# All are from spin echo sequences.
# This script was used with a conda environment called myopencv described in lab.py.
# Use with conda environment: myopencv


import pydicom as dicom
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2
from scipy import stats
import string
import pandas as pd

#Show a pyplot image with simple defaults.
def my_imshow(img, title=""):
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.title(title)
    plt.show()


def plot_histogram(name, hist, xmin = 0, xmax=255, xhair1=None, xhair1col="", xhair2=None, xhair2col=""):
    ''' standard matplotlib method to plot a histogram ''' 
    plt.title("Histogram from " + name)
    plt.xlabel("Bins")
    plt.ylabel("Number of Pixels")
    plt.plot(hist)
    plt.xlim([xmin, xmax])
    if xhair1 != None:
        plt.axvline(x=xhair1[0], color=xhair1col)
        plt.axhline(y=xhair1[1], color=xhair1col)
    if xhair2 != None:
        plt.axvline(x=xhair2[0], color=xhair2col)
        plt.axhline(y=xhair2[1], color=xhair2col)

    plt.show()



def write_scatterplot( x, y1, x2, y2, y1err, y2err, fit, r2, SEm, fit_fn, thfit_fn, 
    title, labels, annot_loc, plotfilename ):
    ''' create and save a figure showing several scatter plots of the 2 series (old, new) vs. ROI'''
    fig = plt.figure()
    #left subplot
    ax = fig.add_subplot(121)
    #ax.plot(x, y1, 'o')
    ax.set_title(labels[0])
    ax.set_ylabel(labels[2])
    ax.set_xlabel(labels[1])
    ax.errorbar(x, y1, yerr=y1err, fmt='o')    
    ax.plot(x, thfit_fn, 'r--', label="theory")
    #right subplot
    ax2 = fig.add_subplot(122)
    ax2.plot(x2, fit_fn(x2), 'b-', label="fit")
#     ax2.plot(x2, thfit_fn(x2), 'r--', label="theory")
    ax2.set_title(labels[3])
    ax2.set_ylabel(labels[5])
    ax2.set_xlabel(labels[4])
    ax2.errorbar(x2, y2, fmt='o', yerr=y2err)  
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
    fig.suptitle(title, fontsize=16)
#     plt.legend(loc="upper left", bbox_to_anchor=[0, 1],
#            ncol=1, shadow=True, fancybox=True)   
           
    # Annotate the right subplot with the line fit coefficients and equation 
    #  location of the first annotation is at (a,b) coordinates with vertical 'step' separation.
    (a,b,step) = annot_loc
    fitstr = 'Slope={:.3}'.format(fit[0])   
    statstr = 'r^2={:.5g}'.format(r2)   
    ax2.annotate(labels[6], xy=(a,b), xytext=(a,b))
    ax2.annotate(fitstr, xy=(a,b+step), xytext=(a,b+step))
    ax2.annotate(statstr, xy=(a,b+2*step), xytext=(a,b+2*step))

    #dpi=300 makes the whole figure large.
    fig.savefig(plotfilename, dpi=300)
    #remove the figure and dealloc the memory.
    plt.close(fig)
    print("Done writing scatterplot figure to file. ")
    
    

# read the dicom file indicated and calculate Signal (assuming a phantom image).        
def get_signal( filenameDCM, df, index ):
    ds = dicom.read_file(filenameDCM)
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(ds.Rows), int(ds.Columns))
#     (Nx,Ny) = ConstPixelDims
    # Load spacing values (in mm)
#     ConstPixelSpacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]), float(ds.SliceThickness))
#     (dx,dy,dz) = ConstPixelSpacing
    #pixel size * matrix size / unit conversion (mm to cm)
#     FOVx = ConstPixelSpacing[0]* ConstPixelDims[0]/10
#     FOVy = ConstPixelSpacing[1]* ConstPixelDims[1]/10
    for data_element in ds:
        if "Number of Averages" in data_element.name:
            NSA = data_element.value
    
    SeriesNumber = ds.SeriesNumber
    TR = ds.RepetitionTime
    TE = ds.EchoTime

    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=ds.pixel_array.dtype)
    # store the raw image data
    ArrayDicom[:, :] = ds.pixel_array  

    img_title = str(ds.SeriesDescription).replace(" ","_")
    img = ArrayDicom

    # construct a grayscale histogram
    mymax = np.max(img)
    xmax = 2**10
    if xmax < mymax:
        print("The histogram will not encompass the full range of intensities in the image.")

    hist = cv2.calcHist([img.astype(np.float32)], [0], None, [xmax], [0, xmax])

    print("mode pixel's count of the histogram", max(hist))
    # mode_result = stats.mode(img, axis=None)
    #stats.mode returns a 1-element list for each, and this is to change it into a regular tuple
    mode_result = ( float(stats.mode(img, axis=None)[0]), float(stats.mode(img, axis=None)[1]) )
    print("mode pixel intensity ", mode_result[0])
    print("mode pixel's count (peak in histogram) ", mode_result[1])

    print("For the higher intensity peak:")
    #Find max in histogram using only intensities starting at xmin, to obtain high intensity peak.
    xmin = int(mode_result[0]) + 200
    hist2 = hist[xmin:]

    mode_result_hi = (np.argmax(hist2)+xmin, np.max(hist2))
    print(" mode pixel intensity ", mode_result_hi[0])
    print(" mode pixel's count (peak in histogram) ", mode_result_hi[1])

    #The following suggests that pixel intensity 0 is artifactual since it has much higher frequency 
    #  than pixels with >0 intensity.
    # for px in range(10):
    #     print("For pixel with intensity %d has count: %d" %( px, hist[px] ))
    xhair1 = mode_result
    xhair2 = (1,hist[1])
#     plot_histogram(img_title, hist, xmax=50, xhair1=xhair1, xhair1col="red", xhair2=xhair2, xhair2col = "violet")

    xhair1 = mode_result_hi
    xhair2 = None
    newimg = img.copy()

    # Use a threshold halfway between the noise and signal peaks in histogram.
    thr_noise = 0.5*(mode_result[0]+mode_result_hi[0])
    print('thr_noise', thr_noise)
    ret, mask = cv2.threshold(newimg, thr_noise, 255, cv2.THRESH_BINARY)

    img_sig = np.copy(newimg)
    img_for_calc = np.copy(newimg).astype(np.float64)
    noisemask = np.copy(newimg)
    noise_for_calc = np.copy(newimg).astype(np.float64)

    #There should be a way to do this with OpenCV functions but it was not trivial.
    for x in range(newimg.shape[0]):
        for y in range(newimg.shape[1]):
            #if this is not from the object
            if mask[x,y] == 0:
                img_sig[x,y] = 0
                img_for_calc[x,y] = np.nan
                # if this is from the artifactual 0 pixels (near borders)
                if newimg[x,y] == 0:
                    noisemask[x,y] = 255
                    noise_for_calc[x,y] = np.nan                
            #this is from the object
            else:
                noisemask[x,y] = 255
                noise_for_calc[x,y] = np.nan
            
    #I'ved masked out the non-contributing pixels with NaN's so I can calculate with Numpy's special
    #  functions for handling arrays with NaN values.
#     my_imshow(img_sig, title="Signal")

    cv2.imwrite('out/' + img_title + '_Signal.png', img_sig)
    mean = np.nanmean(img_for_calc)
    # Using ddof=1 so that we get sample std.dev. rather than population std.dev.
    stdev = np.nanstd(img_for_calc, ddof=1)
    #Calc Standard Error by dividing SD by sqrt(number of pixels in sample)
    # determine the actual N in the calculation by counting all pixels that were not masked (i.e. were not NaNs)
    N = np.count_nonzero(~np.isnan(img_for_calc))
    print("N",N)
    stdev = np.divide(stdev, np.sqrt(N))
    print('Signal Mean ', mean)
    print('Signal SE ', stdev)

    df.loc[index] = [filenameDCM, img_title, SeriesNumber, mean, stdev, TR, TE]


#Calc the std.dev. of a f(X,Y)  from formula: Variance(Z) = Variance(X) + Variance(Y)
#  where Variance = (sd/x)^2
#  and evaluated f(X,Y) is 'mean' that is found from variables in 'x_list' with std.dev. 'sd_list'
def calc_sd_from_list(mean, x_list, sd_list):
    sum = 0
    for i in range(len(sd_list)):
        sum += (sd_list[i]/x_list[i])**2
    return np.abs(mean) * np.sqrt(sum)

# calculate the SD for the ratio list, where the numerator is signal_list (errors sigSD_ref),
#  denominator is a scalar sig_ref (errors sigSD_ref).
def calc_sd_from_ratiolist(ratio, signal_list, sig_ref, signalSD_list, sigSD_ref):
    ratioSD=[0]*len(ratio)
    for j in range(len(ratioSD)):
        x_list = [signal_list[j], sig_ref]
        SD_list = [signalSD_list[j], sigSD_ref]
        ratioSD[j] = calc_sd_from_list(ratio[j], x_list, SD_list)
    return ratioSD


#Calculate natural log(x), and approximate error given argument x and its std.dev. sd.
# by the min-max method. Arguments and output can be single values or lists.
def calc_log_err(x, sd):
    lg = np.log(x)
    print('lg',lg)
    #height of the lower (lo) and upper (hi) error bars in log units.
    lo_bar = np.abs(np.log(np.subtract(x,sd)) - lg)
    hi_bar = np.abs(np.log(np.add(x,sd)) - lg)
    # the error bars are not symmetrical, but approximate by using the average of the two error bars.
    bar = 0.5* np.add(lo_bar, hi_bar)
    return lg, bar


# Calc reciprocal of x element wise, and approximate error given argument x and its std.dev. sd.
# by the min-max method. Arguments and output can be single values or lists. 
def calc_reciprocal_err(x, sd):
    r = np.reciprocal(x)
    #height of the lower (lo) and upper (hi) error bars in reciprocal units.
    hi_bar = np.abs(np.reciprocal(np.subtract(x,sd)) - r)
    lo_bar = np.abs(np.reciprocal(np.add(x,sd)) - r)    
    # the error bars are not symmetrical, but approximate by using the average of the two error bars.
    bar = 0.5* np.add(lo_bar, hi_bar)
    return r, bar

    

def signal_t1(k, T1, time):
    return k*(1-np.exp(-1*time/T1))

def signal_t2(k, T2, time):
    return k*np.exp(-1*time/T2)




### MAIN ###
PathDicom = "./Lab_Phantom/T1T2"
df_outfile = './dfT1T2out.csv'
scatter_outdir = './scatterT1T2'
if not os.path.isfile(df_outfile):
    lstFilesDCM = []
    lstDirName = []
    for dirName, subdirList, fileList in os.walk(PathDicom):
        basedirName = os.path.basename(dirName)
        if basedirName.startswith("SE_T"):  
            for filename in fileList:
                # check whether the file is DICOM
                if filename.lower().endswith(".dcm"):
                    lstFilesDCM.append(os.path.join(dirName,filename))
                    lstDirName.append(basedirName)
    
    columns = ["Filename", "Title", "SeriesNumber", "Signal", "SignalSD", "TR", "TE"]
    df = pd.DataFrame(index=range(0,len(lstFilesDCM)), columns=columns)
    index = 0 # using an index to row in dataframe so that I can pre-allocate the df memory.
    
    #read the metadata from the DICOM files and calculate the phantom Signal for each image.
    for filename in lstFilesDCM:
        print("=== FILENAME %s ===" % filename)
        get_signal(filename, df, index)
        index += 1
#     print("filename dataframe")
#     print(df.head(1))
    df.to_csv(df_outfile, index=False)
else:
    df = pd.read_csv(df_outfile)

### Calculate T2 ###

#Create dataframe with only T2w images' parameters
dfT2 = df.loc[df["Title"] == "SE_T2w"]
#Start the indexing at 0
dfT2.reset_index(inplace=True)

echotime = [x for x in range(20,100,20)]
dfecho = {x:pd.DataFrame() for x in echotime}
signal_list = []
signalSD_list = []
for time in echotime:
    dfecho[time] = dfT2.loc[dfT2["TE"] == time]
    #average signal from images having same TE=time
    signal = dfecho[time]["Signal"].mean()
    signal_list.append(signal)
    #std.dev of signal from images having same TE=time
    signalSD = calc_sd_from_list(signal, list(dfecho[time]["Signal"]), list(dfecho[time]["SignalSD"]))
    signalSD_list.append( signalSD )
    print("Signal avg for echo with TE %d: %f +/- %f" % (time, signal, signalSD))
    
#line fit of data
fit = [0,0]

#normalize the signal by the reference signal (of first time point TE=20ms).
sig_ref = signal_list[0]
sigSD_ref = signalSD_list[0]
time_ref = echotime[0]

ratio = np.divide(signal_list, sig_ref)

#propagate the errors to the ratios
ratioSD = calc_sd_from_ratiolist(ratio, signal_list, sig_ref, signalSD_list, sigSD_ref)

log_signal, log_signal_sd = calc_log_err(ratio, ratioSD)

shift_echotime = echotime
print('shift_echotime', shift_echotime)
fit[0], fit[1], r_value, _, SEm = stats.linregress( shift_echotime, log_signal )
r2 = r_value**2

print('slope = ', fit[0])
print('intercept = ', fit[1])
print("Standard error of slope: ", SEm)

T2value, T2err = calc_reciprocal_err(-1*fit[0], SEm)
T2str = 'T2 = %2.2f +/- %2.2f ms'%(T2value, T2err)
print(T2str)

#  returns the function that can be used to calculate any intermediate points from the fit.
fit_fn = np.poly1d(fit)
print('r^2= ', r2)

#theoretical relationship
# Calculate the scaling constant k in the equation Signal = k * exp(-t/T2)
k = sig_ref / signal_t2(1, T2value, time_ref)
thfit_fn=[]
for time in echotime:
    thfit_fn.append(signal_t2(k, T2value, time))
print ('thfit_fn',thfit_fn)

print ('fit',fit)
print('fit_fn', fit_fn)
title = "Relationship of T2w Signal to Echo Time"
labels = ["Signal", "Echo Time (TE) [ms]", "Average Signal", "Log(Signal/ReferenceSignal)", 
    "Echo Time (TE) [ms]", "Log(Signal/ReferenceSignal)", T2str]
step = (np.max(log_signal)-np.min(log_signal))/10
annot_loc = (time_ref,min(log_signal),step)

out_pathname = os.path.join(scatter_outdir, "T2.png")
write_scatterplot( echotime, signal_list, echotime, log_signal, signalSD_list, log_signal_sd, fit, r2, SEm, 
    fit_fn, thfit_fn, title, labels, annot_loc, out_pathname )


### Calculate T1 ###
#Create dataframe with only T1w images' parameters.
dfT1 = df.loc[df["Title"] != "SE_T2w"]
TE = dfT1["TE"].mean()
print("TE",TE)
dfT1 = dfT1.drop(["Filename","Title","SeriesNumber","TE"], axis=1)
#Start the indexing at 0
dfT1 = dfT1.reset_index()
print(dfT1)
reptime = sorted(set(dfT1["TR"]))
dfrep = {x:pd.DataFrame() for x in reptime}
signal_list = []
signalSD_list = []
for time in reptime:    
    dfrep[time] = dfT1.loc[dfT1["TR"] == time]
    #average signal from images having same TR=time
    signal = dfrep[time]["Signal"].mean()
    signal_list.append(signal)
    #std.dev of signal from images having same TR=time
    signalSD = calc_sd_from_list(signal, list(dfrep[time]["Signal"]), list(dfrep[time]["SignalSD"]))
    signalSD_list.append( signalSD )
    print("Signal avg for images with TR %d: %f +/- %f" % (time, signal, signalSD))

#line fit of data
fit = [0,0]

#normalize the signal by the reference signal (of last time point TR=3000ms).
# print("signal ref", signal_list[-1])
#Change coordinates so this is an exponential decay i.e. (1-exp(-TR/T1)) => exp(-TR/T1)
sig_ref = signal_list[-1]
sigSD_ref = signalSD_list[-1]
time_ref = reptime[-1]
ratio = 1 - np.divide(signal_list, sig_ref ) 
#remove the reference data point from the lists
ratio = ratio[:-1]
shift_reptime = reptime[:-1]

#propagate the errors to the ratios
ratioSD = calc_sd_from_ratiolist(ratio, signal_list, sig_ref, signalSD_list, sigSD_ref)

print('ratioSD',ratioSD)
print('ratio', ratio)
# print('shift_reptime', shift_reptime)

#calculate the log signals and propagate the errors
log_signal, log_signal_sd = calc_log_err(ratio, ratioSD)
print('ratioSD')
print(ratioSD)
print('log_signal_sd')
print(log_signal_sd)

fit[0], fit[1], r_value, _, SEm = stats.linregress( shift_reptime, log_signal )
r2 = r_value**2

print('slope = ', fit[0])
print('intercept = ', fit[1])
print("Standard error of slope: ", SEm)

T1value, T1err = calc_reciprocal_err(-1*fit[0], SEm)
T1str = 'T1 = %2.2f +/- %2.2f ms'%(T1value, T1err)
print(T1str)

#  returns the function that can be used to calculate any intermediate points from the fit.
fit_fn = np.poly1d(fit)
print('r^2= ', r2)
# print('SE of m is', SEm)

#theoretical function of Signal vs. time given this T1.
# Calculate the scaling constant k in the equation Signal = k * (1-exp(-t/T1))
k = sig_ref / signal_t1(1, T1value, time_ref)
thfit_fn=[]
for time in reptime:
    thfit_fn.append(signal_t1(k, T1value, time))
print ('thfit_fn',thfit_fn)

print('fit_fn', fit_fn)
title = "Relationship of T1w Signal to Repetition Time"
labels = ["Signal", "Rep Time (TR) [ms]", "Average Signal", "Log(1 - Signal/ReferenceSignal)", 
    "Rep Time (TR) [ms]", "Log(1 - Signal/ReferenceSignal)", T1str]
step = (np.max(log_signal)-np.min(log_signal))/10
annot_loc = (reptime[0],min(log_signal),step)

out_pathname = os.path.join(scatter_outdir, "T1.png")
write_scatterplot( reptime, signal_list, shift_reptime, log_signal, signalSD_list, log_signal_sd, fit, r2, SEm, 
    fit_fn, thfit_fn, title, labels, annot_loc, out_pathname )


print("DONE!")
sys.exit(0)
