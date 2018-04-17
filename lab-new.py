#!/usr/bin/env python
# Calculate SNR from a set of images where there is one independent variable in the MR sequence parameters
#  such as bandwidth, matrix size, FOV, NSA etc.
# Environment:
# I have setup a conda environment called myopencv in order to do this
# Environment note. Use a conda environment to ensure python can uniquely determine library that should be called.
# Example Error: objc[1093]: Class TKApplication is implemented in both /System/Library/Frameworks/Tk.framework/Versions/8.5/Tk (0x1095bc1f8) and /Users/pradau/anaconda/lib/libtk8.5.dylib (0x10974de40). One of the two will be used. Which one is undefined.

# Run these lines in the terminal to create and activate the environment 'myopencv' where I'll install all required packages.
# Note that I'm specifying version 3.5 since I know that there is a OpenCV3 build for it currently. (Not python 3.6)
# conda create --name myopencv python=3.5 matplotlib
# source activate myopencv

# Ensure that all other packages we require are installed. Run these lines in the terminal
# To install OpenCV3 so that we can use 'import cv2', we can use the Anaconda package installer. Run this line at the terminal
# conda install -c menpo opencv3 
# Install pydicom by:
# conda install -c conda-forge pydicom
# Install pillow by:
# conda install pillow
# Install scipy by
# conda install scipy

import pydicom as dicom
import os
import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image
# from PIL import ImageChops
import sys
import cv2
from scipy import stats
import string
import pandas as pd

#Calc the std.dev. of a product or quotient of two variables.
def calc_combined_sd(mean, x, x_stdev, y, y_stdev):
    return np.abs(mean) * np.sqrt((x_stdev/x)**2 + (y_stdev/y)**2)

# Calc min max error from a ratio of two variables p/q with  std.dev sd_p and sd_q
def calc_ratio_err(p, sd_p, q, sd_q):
    r = np.divide(p, q)
    #height of the lower (lo) and upper (hi) error bars.
    hi_val = np.divide(np.add(p, sd_p), np.subtract(q,sd_q))
    lo_val = np.divide(np.subtract(p, sd_p), np.add(q,sd_q))
    print("lo, hi %f %f" % (lo_val, hi_val))
    hi_bar = np.abs(hi_val - r)
    lo_bar = np.abs(lo_val - r)
    print("lobar, hibar %f %f" % (lo_bar, hi_bar))
    # the error bars are not symmetrical, but approximate by using the average of the two error bars.
    bar = 0.5* np.add(lo_bar, hi_bar)
    return r, bar


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


#Show a pyplot image with simple defaults.
def my_imshow(img, title=""):
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.title(title)
    plt.show()


# Linear regression from two 1-D vectors (x,y)
# Returns: slope, intercept, R^2 (coeff. of determination).
# NOTE: this is used instead of linregress from scipy because this method gives me the option to force intercept to 0, and will calc R^2 correctly for that case.
# NOTE2: unlike scipy linregress, this will return R^2 (not R) since this is more convenient.
def linregress_choose_intercept( x, y, bFindIntercept=True ):
    if bFindIntercept:
        #create a matrix with x in first row and 1's in 2nd row, then take transpose so that x is a column vector.
        A = np.vstack([x, np.ones(len(x))]).T    
        #residuals, rank and singular val matrix (s) are not used (could be replaced by "_" in this line.)
        ( m, b ), res, rank, s = np.linalg.lstsq(A, y,rcond=None)
        #R^2 (coefficient of determination) is identical to the square of Pearson's r (in the case where m and b are used in regression).
        r_value, pval = stats.pearsonr( y, x)
        rsqr = r_value * r_value
    else:
        #linear regression for equation y = mx (intercept b is forced to 0).
        # The difference compared with above is that there is only one column in A matrix (for the slope) since we are determining only 1 parameter.
        A = np.vstack([x]).T
        solution, res, rank, s = np.linalg.lstsq(A, y,rcond=None)
        m = solution[0]
        b = 0
        y_avg = np.mean( y )
        #Sum of squared residuals for the estimation.
        SSres = res[0]
        SStot = np.sum( np.square( y - y_avg ) ) 
        #This is a very general formula for coeff. of determination.
        rsqr = 1 - SSres/SStot
    return m, b, rsqr


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



def write_scatterplot( x, y1, x2, y2, y1err, y2err, fit, r2, SEm, fit_fn, thfit_fn, title, labels, annot_loc, plotfilename ):
    ''' create and save a figure showing several scatter plots of the 2 series (old, new) vs. ROI'''
    fig = plt.figure()

    #left subplot
    ax = fig.add_subplot(121)
    #ax.plot(x, y1, 'o')
    ax.set_title(labels[0])
    ax.set_ylabel(labels[2])
    ax.set_xlabel(labels[1])
    ax.errorbar(x, y1, yerr=y1err, fmt='o')  
  
    #right subplot
    ax2 = fig.add_subplot(122)
    ax2.plot(x2, fit_fn(x2), 'b-', label="fit")
    ax2.plot(x2, thfit_fn(x2), 'r--', label="theory")
    ax2.set_title(labels[3])
    ax2.set_ylabel(labels[5])
    ax2.set_xlabel(labels[4])
    ax2.errorbar(x2, y2, fmt='o', yerr=y2err)  
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
    fig.suptitle(title, fontsize=16)
    plt.legend(loc="upper left", bbox_to_anchor=[0, 1],
           ncol=1, shadow=True, fancybox=True)   
           
    # Annotate the right subplot with the line fit coefficients and equation 
    #  location of the first annotation is at (a,b) coordinates with vertical 'step' separation.
    (a,b,step) = annot_loc
#     fitstr = 'm={:.3}  b={:.3g}'.format(fit[0],fit[1])   
#     statstr = 'r^2={:.5g}   SE(m)={:.3g}'.format(r2, SEm)   
#     ax2.annotate('y = mx + b', xy=(a,b), xytext=(a,b))
    fitstr = 'm={:.3} +/- {:.3}'.format(fit[0], SEm)   
    intstr = 'b={:.3}'.format(fit[1])
    statstr = 'r^2={:.5g}'.format(r2)   
    ax2.annotate('y = mx + b', xy=(a,b), xytext=(a,b))
    ax2.annotate(fitstr, xy=(a,b+step), xytext=(a,b+2*step))
    ax2.annotate(intstr, xy=(a,b+step), xytext=(a,b+step))
    ax2.annotate(statstr, xy=(a,b+2*step), xytext=(a,b+3*step))
#     fig.show()

    #dpi=300 makes the whole figure large.
    fig.savefig(plotfilename, dpi=300)
    #remove the figure and dealloc the memory.
    plt.close(fig)
    print("Done writing scatterplot figure to file. ")
    
    

# read the dicom file indicated and calculate SNR (assuming a phantom image).        
def get_snr( filenameDCM, df, index ):
    # Get ref file
#     RefDs = dicom.read_file(filename)
    ds = dicom.read_file(filenameDCM)
    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    #  Assuming a single image per exam
    ConstPixelDims = (int(ds.Rows), int(ds.Columns))
    (Nx,Ny) = ConstPixelDims
    
#     print('ConstPixelDims',ConstPixelDims)
    # Load spacing values (in mm)
    ConstPixelSpacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]), float(ds.SliceThickness))
    (dx,dy,dz) = ConstPixelSpacing
#     print('ConstPixelSpacing',ConstPixelSpacing)
    #pixel size * matrix size / unit conversion (mm to cm)
    FOVx = ConstPixelSpacing[0]* ConstPixelDims[0]/10
    FOVy = ConstPixelSpacing[1]* ConstPixelDims[1]/10
    for data_element in ds:
        if "Number of Averages" in data_element.name:
            NSA = data_element.value
#             print(data_element.name)
#             print("NSA", data_element.value)
        if "Pixel Bandwidth" in data_element.name:
            PBW = data_element.value
#             print(data_element.name)
#             print("PBW", data_element.value)        
            
    x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
    y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])

    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims, dtype=ds.pixel_array.dtype)
#     filenameDCM = lstFilesDCM[0]
#     ds = dicom.read_file(filenameDCM)
    # store the raw image data
    ArrayDicom[:, :] = ds.pixel_array  

    # img_title = str(ds.PatientName) + " " + str(ds.SeriesDescription)
    img_title = str(ds.SeriesDescription).replace(" ","_")
    # print(ArrayDicom)

    img = ArrayDicom
    # print('img.dtype',ds.pixel_array.dtype)

    # Show the grayscale image with matplotlib
    # plt.imshow(img, cmap='gray', interpolation='bicubic')
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()

    # construct a grayscale histogram
    # We use np.max because img is a numpy ndarray.
#     print("max pixel intensity ", np.max(img))

    mymax = np.max(img)
    xmax = 2**10
    if xmax < mymax:
        print("The histogram will not encompass the full range of intensities in the image.")

    hist = cv2.calcHist([img.astype(np.float32)], [0], None, [xmax], [0, xmax])

    # Can use these lines to show that bin 908 has the largest pixel intensity (which happens to be 908 here).
    # for num in range(xmax-900):
    #     if hist[900+num] > 0:
    #         print('count in bin '+str(num+900)+':'+ str(hist[900+num]))
    

#     print("mode pixel's count of the histogram", max(hist))
    # mode_result = stats.mode(img, axis=None)
    #stats.mode returns a 1-element list for each, and this is to change it into a regular tuple
    mode_result = ( float(stats.mode(img, axis=None)[0]), float(stats.mode(img, axis=None)[1]) )
#     print("mode pixel intensity ", mode_result[0])
#     print("mode pixel's count (peak in histogram) ", mode_result[1])

    print("For the higher intensity peak:")
    #Find max in histogram using only intensities starting at xmin, to obtain high intensity peak.
    xmin = 600
    hist2 = hist[xmin:]

    mode_result_hi = (np.argmax(hist2)+xmin, np.max(hist2))
#     print(" mode pixel intensity ", mode_result_hi[0])
#     print(" mode pixel's count (peak in histogram) ", mode_result_hi[1])

    #The following suggests that pixel intensity 0 is artifactual since it has much higher frequency 
    #  than pixels with >0 intensity.
    # for px in range(10):
    #     print("For pixel with intensity %d has count: %d" %( px, hist[px] ))

    xhair1 = mode_result
    xhair2 = (1,hist[1])
    # plot_histogram(img_title, hist, xmax=50, xhair1=xhair1, xhair1col="red", xhair2=xhair2, xhair2col = "violet")

    xhair1 = mode_result_hi
    xhair2 = None
    # plot_histogram(img_title, hist, xmin = xmin, xmax = xmin+200, xhair1=xhair1, xhair1col="red")
#     print('img.dtype', img.dtype)

    # newimg = img.astype(np.uint8)
    newimg = img.copy()
#     print('newimg.dtype', newimg.dtype)

    # Use a threshold halfway between the noise and signal peaks in histogram.
    thr_noise = 0.5*(mode_result[0]+mode_result_hi[0])
    #thr_noise = 0.5*np.max(newimg)
    print('thr_noise', thr_noise)
    ret, mask = cv2.threshold(newimg, thr_noise, 255, cv2.THRESH_BINARY)

    img_sig = np.copy(newimg)
    img_for_calc = np.copy(newimg).astype(np.float64)
    noisemask = np.copy(newimg)
    noise_for_calc = np.copy(newimg).astype(np.float64)

    #There should be a way to do this with OpenCV functions but it took far to long to figure out.
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

#     my_imshow(noisemask, title="Noise")
    cv2.imwrite('out/' + img_title + '_Noise.png', noisemask)

    noise_mean = np.nanmean(noise_for_calc)
    noise_stdev = np.nanstd(noise_for_calc, ddof=1)
    #Calc Standard Error by dividing SD by sqrt(number of pixels in sample)
    # determine the actual N in the calculation by counting all pixels that were not masked (i.e. were not NaNs)
    N2 = np.count_nonzero(~np.isnan(noise_for_calc))
    print("N2",N2)
    noise_stdev = np.divide(noise_stdev, np.sqrt(N2))
    print('Noise Mean ', noise_mean)
    print('Noise SE ', noise_stdev)
    #snr = mean / noise_mean
    #The error for the variance of SNR is found by min-max method.
    snr, snr_stdev = calc_ratio_err( mean, stdev, noise_mean, noise_stdev)
    print('snr, snr_stdev')
    print(snr, snr_stdev)
    
    df.loc[index] = [filenameDCM, img_title, snr, snr_stdev, FOVx, FOVy, NSA, PBW, dx, dy, dz, Nx, Ny]
    


### MAIN ###
PathDicom = "./Lab_Phantom/SNR"
df_outfile = './df_out.csv'
if not os.path.isfile(df_outfile):
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if filename.lower().endswith(".dcm"):  # check whether the file is DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))

    columns = ["Filename", "Title", "SNR", "SNR_SD", "FOVx", "FOVy", "NSA", "PBW", "dx", "dy", "dz", "Nx", "Ny"]
    df = pd.DataFrame(index=range(0,len(lstFilesDCM)), columns=columns)
    index = 0 # using an index to row in dataframe so that I can pre-allocate the df memory.
    #read the metadata from the DICOM files and calculate the SNR for each image.
    for filename in lstFilesDCM:
        print("=== FILENAME %s ===" % filename)
        get_snr(filename, df, index)
        index += 1
    print(df)
    df.to_csv(df_outfile, index=False)
else:
    df = pd.read_csv(df_outfile)


#Create dataframe with the standard (reference) file's parameters
dfstd = df.loc[df["Title"] == "SPGR_STANDARD"]
#Start the indexing at 0
dfstd.reset_index(inplace=True)
#convert dfstd to a dictionary for easier access.
dstd = {key: dfstd.at[0,key] for key in dfstd.columns}

##### FOV #####
#Create dataframe with only info related to FOV
dffov = df.drop(["Filename","NSA", "PBW", "dx", "dy", "dz", "Nx", "Ny"], axis=1)
# Select only the study rows where the FOV was modified.
bool_list = [ x.endswith("cm") for x in list(df["Title"]) ]
dffov = dffov.loc[bool_list]
fov = list(dffov["FOVx"] * dffov["FOVy"])
snr = list(dffov["SNR"])
snr_sd = list(dffov["SNR_SD"])

#Divide the SNRs by the reference SNR
snr_norm = [ x / dstd["SNR"] for x in snr ]
#Divide the FOVs by the reference FOV
fov_norm = [ x / (dstd["FOVx"] * dstd["FOVy"]) for x in fov ]
snr_sd_norm = snr_sd.copy()
#Append the values from the reference volume. Will only use these in the non-relative left subplot.
fov.append((dstd["FOVx"] * dstd["FOVy"]))
snr.append( dstd["SNR"] )
snr_sd.append( dstd["SNR_SD"] )

# Revise x-axis list so that first (left) subplot is SNR vs FOV rather than SNR vs FOV^2.
fov = np.sqrt(fov)

#line fit of data
#  returns the line coefficients from the linear fit
# fit = np.polyfit(fov_norm,fov_snr_norm, 1)
fit = [0,0]
#Can use this to find regression line with intercept
fit[0], fit[1], r_value, _, SEm = stats.linregress( fov_norm, snr_norm )
r2 = r_value**2
# fit[0], fit[1], r2 = linregress_choose_intercept( fov_norm, snr_norm, bFindIntercept=False )

#  returns the function that can be used to calculate any intermediate points from the fit.
fit_fn = np.poly1d(fit)
print('r^2= ', r2)
# print('SE of m is', SEm)

#theoretical linear relationship
thfit = np.polyfit(fov_norm,fov_norm, 1)
thfit_fn = np.poly1d(thfit)

print ('fit',fit)
print('fit_fn', fit_fn)
title = "Relationship of SNR to Field-of-View"
labels = ["SNR", "Field-of-View (FOV=FOVx=FOVy) [cm]", "Signal to Noise Ratio (SNR)", "Relative SNR", "FOV^2 / Reference(FOV^2)", "SNR / Reference(SNR)"]
step = (np.max(fov_norm)-np.min(fov_norm))/15
annot_loc = (min(fov_norm)+0.5,min(snr_norm),step)
write_scatterplot( fov, snr, fov_norm, snr_norm, snr_sd, snr_sd_norm, fit, r2, SEm, 
    fit_fn, thfit_fn, title, labels, annot_loc, "./scatter/FOV.png" )

##### BW #####
#Create dataframe with only info related to BW
dfbw = df.drop(["Filename","NSA", "PBW", "dx", "dy", "dz", "Nx", "Ny", "FOVx", "FOVy"], axis=1)
# Select only the study rows where BW was modified.
bool_list = [ x.endswith("kHz") for x in list(df["Title"]) ]
dfbw = dfbw.loc[bool_list]
print(dfbw)
#Get the SNRs and their std.dev. 
snr = list(dfbw["SNR"])
snr_sd = list(dfbw["SNR_SD"])
#Divide the SNRs by the reference SNR
snr_norm = [ x / dstd["SNR"] for x in snr ]
snr_sd_norm = snr_sd.copy()
# Obtain bandwidth from exam series descriptions.
bw = [ int(x.strip("kHz").strip("SPGR_BW")) for x in dfbw["Title"] ]
key = "BW"
#Reference receiver bandwidth from lab instructions
dstd[key] = 16
#Normalized to obtain a linear relationship
bw_norm = [ np.sqrt(dstd[key] / x)  for x in bw ]
print("bw_norm", bw_norm)
#Append the value from the reference volume. Will only use in non-relative left subplot.
bw.append(dstd[key])
snr.append( dstd["SNR"] )
snr_sd.append( dstd["SNR_SD"] )
#line fit of data
fit = [0,0]
fit[0], fit[1], r_value, _, SEm = stats.linregress( bw_norm, snr_norm )
r2 = r_value**2
#fit[0], fit[1], r2 = linregress_choose_intercept( bw_norm, snr_norm, bFindIntercept=False )
#  returns the function that can be used to calculate any intermediate points from the fit.
fit_fn = np.poly1d(fit)
#theoretical linear relationship
thfit = np.polyfit(bw_norm, bw_norm, 1)
thfit_fn = np.poly1d(thfit)
title = "Relationship of SNR to Receiver Bandwidth"
labels = ["SNR", "Receiver Bandwith (BW) [kHz]", "Signal to Noise Ratio (SNR)", 
    "Relative SNR", "sqrt(Reference(BW) / BW)","SNR / Reference(SNR)"]
outpathname = os.path.join("./scatter", key + ".png" )
step = (np.max(bw_norm)-np.min(bw_norm))/15
annot_loc = (min(bw_norm)+0.4,min(snr_norm),step)
write_scatterplot( bw, snr, bw_norm, snr_norm, snr_sd, snr_sd_norm, fit, r2, SEm, fit_fn, 
    thfit_fn, title, labels, annot_loc, outpathname )

##### NSA #####
#Create dataframe with only info related to NSA ("param")
dfnsa = df.drop(["Filename", "PBW", "dx", "dy", "dz", "Nx", "Ny", "FOVx", "FOVy"], axis=1)
# Select only the study rows where the param was modified.
bool_list = [ x.startswith("SPGR_NEX_") for x in list(df["Title"]) ]
dfnsa = dfnsa.loc[bool_list]
print(dfnsa)
#Get the SNRs and their std.dev. 
snr = list(dfnsa["SNR"])
snr_sd = list(dfnsa["SNR_SD"])
#Divide the SNRs by the reference SNR
snr_norm = [ x / dstd["SNR"] for x in snr ]
snr_sd_norm = snr_sd.copy()
key = "NSA"
nsa = [ x for x in dfnsa[key] ]
#Normalized to obtain a linear relationship
nsa_norm = [ np.sqrt( x / dstd[key] )  for x in nsa ]
print("nsa_norm", nsa_norm)
#Append the value from the reference volume. Will only use in non-relative left subplot.
nsa.append(dstd[key])
snr.append( dstd["SNR"] )
snr_sd.append( dstd["SNR_SD"] )
#line fit of data
fit = [0,0]
fit[0], fit[1], r_value, _, SEm = stats.linregress( nsa_norm, snr_norm )
r2 = r_value**2
#fit[0], fit[1], r2 = linregress_choose_intercept( nsa_norm, snr_norm, bFindIntercept=False )
#  returns the function that can be used to calculate any intermediate points from the fit.
fit_fn = np.poly1d(fit)
#theoretical linear relationship
thfit = np.polyfit(nsa_norm, nsa_norm, 1)
thfit_fn = np.poly1d(thfit)
title = "Relationship of SNR to Number of Signal Averages"
labels = ["SNR", "No. of Signal Averages (NSA)", "Signal to Noise Ratio (SNR)", 
    "Relative SNR", "sqrt(NSA / Reference(NSA))","SNR / Reference(SNR)"]
outpathname = os.path.join("./scatter", key + ".png" )
step = (np.max(nsa_norm)-np.min(nsa_norm))/15
annot_loc = (min(nsa_norm)+0.2,min(snr_norm),step)
write_scatterplot( nsa, snr, nsa_norm, snr_norm, snr_sd, snr_sd_norm, fit, r2, SEm, fit_fn, 
    thfit_fn, title, labels, annot_loc, outpathname )

##### Matrix size #####
#Create dataframe with only info related to N (matrix size).
print('df',df)
dfmat = df.drop(["Filename","NSA", "PBW", "dx", "dy", "dz", "FOVx", "FOVy"], axis=1)
# Select only the study rows where the  was modified.
bool_list = [ (x.startswith("SPGR_512") or x.startswith("SPGR_256")) for x in list(df["Title"]) ]
dfmat = dfmat.loc[bool_list]
print('dfmat', dfmat)

# create the list of matrix sizes, in units of the reference size (256x256)
loc=5
units=1000
# the matrix sizes are extracted from the series description (since the dicom header doesn't match).
mat_size = [ float(x[loc:loc+3])*float(x[loc+4:loc+7])/units for x in dfmat["Title"] ]
mat_size_ref = dstd["Nx"] * dstd["Ny"] / units
snr = list(dfmat["SNR"])
snr_sd = list(dfmat["SNR_SD"])

#Divide the SNRs by the reference SNR
snr_norm = [ x / dstd["SNR"] for x in snr ]
#Divide the matrix size by the reference 
mat_size_norm = [np.sqrt(mat_size_ref / x ) for x in mat_size ]
snr_sd_norm = snr_sd.copy()
# print('snr_sd_norm', snr_sd_norm)
# for j in range(len(snr_sd_norm)):
#     snr_sd_norm[j] = calc_combined_sd(snr_norm[j], snr[j], snr_sd[j], dstd["SNR"], dstd["SNR_SD"])
#     print('dstd["SNR"], dstd["SNR_SD"]')
#     print("%f, %f" %( dstd["SNR"], dstd["SNR_SD"]))
#     #= calc_sd_from_ratiolist(snr_norm, snr, dstd["SNR"], snr_sd, dstd["SNR_SD"])
# print('snr_sd_norm', snr_sd_norm)
# sys.exit(0)

#Append the values from the reference volume. Will only use these in the non-relative left subplot.
mat_size.append(mat_size_ref)
snr.append( dstd["SNR"] )
snr_sd.append( dstd["SNR_SD"] )

#line fit of data
#  returns the line coefficients from the linear fit
fit = [0,0]
fit[0], fit[1], r_value, _, SEm = stats.linregress( mat_size_norm, snr_norm )
r2 = r_value**2
#fit[0], fit[1], r2 = linregress_choose_intercept( mat_size_norm, snr_norm, bFindIntercept=False )

#  returns the function that can be used to calculate any intermediate points from the fit.
fit_fn = np.poly1d(fit)
print('r^2= ', r2)

#theoretical linear relationship
thfit = np.polyfit(mat_size_norm, mat_size_norm, 1)
thfit_fn = np.poly1d(thfit)

print ('fit',fit)
print('fit_fn', fit_fn)
title = "Relationship of SNR to Matrix Size"
labels = ["SNR", "Matrix Size (Nx*Ny) [10^3]", "Signal to Noise Ratio (SNR)", "Relative SNR", 
    "Reference(sqrt(Nx * Ny))/sqrt(Nx * Ny)", "SNR / Reference(SNR)"]
step = (np.max(mat_size_norm)-np.min(mat_size_norm))/15
annot_loc = (min(mat_size_norm)+0.2,min(snr_norm),step)
outpathname = os.path.join("./scatter", "mat_size.png" )
write_scatterplot( mat_size, snr, mat_size_norm, snr_norm, snr_sd, snr_sd_norm, fit, r2, SEm, 
    fit_fn, thfit_fn, title, labels, annot_loc, outpathname )
       
print("DONE!")
sys.exit(0)
