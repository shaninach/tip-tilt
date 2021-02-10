import numpy as np 
from astropy.modeling import models, fitting
from astropy.io import fits
import os
import pandas as pd 
from matplotlib import pyplot as plt

# functions
def boxes (data,N,max_brightness,background):
    """ divides the image to (N*N/1.5) cells. N is any multiple of 3 larger than 6 (9,12,15,18...)
    in each cell, chooses a star by finding the maximum flux higher than SNR = 6, lower than 90% of saturated pixel
    output: box = dictionary of values of each cell. 
    cents = dictionary with values of centroid location in each box"""
    data = pd.DataFrame(data)
    datacut = data.loc[30:(len(data)-30),30:(len(data[1])-30)]
    y = int(len(data)/N)
    x = int(len(data.columns)/(N/1.5))
    inty = list(range(int(N)))
    intx = list(range(int(N/1.5)))
    box = {}  
    name = 0
    for j in inty:
        row = datacut.iloc[j*y:((j+1)*y)] # slice to rows
        for i in intx:
            box[name] = row.loc[:,(i*x):((i+1)*x)] # each row slice to N/1.5 cells
            name += 1     
    
    cents = {}
    for n in list(range(len(box))): 
        maxlist = box[n].max()
        maxval = max(maxlist)
        SNR = maxval/background # S/N ratio
        if SNR <10: #
            cents[n] = np.nan
        elif SNR > 10 and maxval <= 0.8*max_brightness:
            cents[n] = np.where(box[n] == maxval) # cents will be the reseted indices of centroid's location in box [row, column]
        elif maxval > 0.8*max_brightness:
            saty = np.where(maxlist == maxval) # saturated pixel row serial number inside maxlist
            if len(saty[0]) > 1: # if there's more than one of this value
                newlist = maxlist.drop(maxlist.index[int(saty[0][0])]) # drop the value with first location in saty
                for v in list(range(len(newlist))):
                    if abs(int(saty[0][0]-v)) >= 50: # if distance between value in newlist bigger than 50 pixels from saturated pixel
                        if newlist.iloc[v]/background >10 and newlist.iloc[v] <= 0.8*max_brightness:
                            cents[n] = np.where(box[n] == newlist.iloc[v])
                        else: 
                            cents[n] = np.nan
            else: # same process as previous loop in case there's only one saturated pixel in maxlist
                newlist = maxlist.drop(maxlist.index[int(saty[0])])
                for v in list(range(len(newlist))):
                    if abs(int(saty[0]-v)) >= 50:
                        if newlist.iloc[v] <= 0.9*max_brightness and newlist.iloc[v]/background > 6:
                            cents[n] = np.where(box[n] == newlist.iloc[v])
                        else: 
                            cents[n] = np.nan

    return box, cents

def seeing(data,box,cents,pscale): 
    """ box = dictionary with cells of image 
        cents = dictionary with locations of centroids of each cell 
        pscale = plate scale of telescope (arcsec/pixel).
        output: list of seeing values - FWHM of 2D gaussian fit of each cell"""
    see = list(range(len(cents)))
    for n in list(range(len(cents))):
        if type(cents[n]) == float: # meaning it's a nan
            see[n] = np.nan
        else:  # type should be tuple
            hb = 30   # half box size of fit to gausian (around brightest pixel)
            yc = int(box[n].index[int(cents[n][0])]) # the real loction of centroid row in original data
            xc = int(box[n].columns[int(cents[n][1])]) # the real location of centroid column in original data
            fitbox = data[(yc-hb):(yc+hb), (xc-hb):(xc+hb)] # box limits: data[yc+-hb, xc+-hb]
            yp, xp = fitbox.shape
            y, x, = np.mgrid[:yp, :xp]  # Generate grid of same size like box to put the fit on
            f_init = models.Gaussian2D()  # Declare what function you want to fit to your data
            fit_f = fitting.LevMarLSQFitter()  # Declare what fitting function you want to use
            f = fit_f(f_init, x, y, np.array(fitbox))  # Fit the model to your data (box)
            std = [f.x_stddev[0],f.y_stddev[0]]
            #see[n] = float((np.mean(std) * 2.355 * pscale)) # multiply of std mean by 2.355 and by conversion coefficient to get seeing
            see[n] = np.mean([f.x_fwhm, f.y_fwhm])* pscale
            if see[n] >20:
                see[n] = np.nan
    
    return see

# main loop - 
max_brightness = 65000 # saturation value (white pixel) 
N = 12 # any multipple of 3 from 6. each image will be divided to N*(N/1.5)
path = r"D:\Master\LAST"  # image folder path
os.chdir(path)  # define the path as current directory
pscale = 1.25  # conversion coefficient of LAST!
hdul = fits.open("LAST.0.1_20200822.145204.403_clear__sci_proc_im_000.fits")  # Open image
data = hdul[0].data  # data = pixel brightness
background = np.median(data)
#data = data-background # reduce background noise from each pixel
box, cents = boxes(data,N,max_brightness,background)
see = seeing(data,box,cents,pscale)

# heatmap size = full image size 
heat = pd.DataFrame(data.copy())
datacut = heat.loc[30:(len(data)-30),30:(len(data[1])-30)]
for n in list(range(len(box))):
    ri = box[n].index[0]
    rf = box[n].index[int(len(box[n].index))-1]
    ci = box[n].columns[0]
    cf = box[n].columns[int(len(box[n].columns))-1]
    datacut.loc[ri:rf, ci:cf] = see[n]
    
#plt.figure(n)
plt.imshow(heat)
plt.clim(0,np.nanmax(see))
plt.colorbar()


# if the error "The fit may be unsuccessful" comes up, check the fit_info, 'message': 
#fit_f.fit_info
#fit_f.fit_info['message']
"""
# check individualfit - 
n = 83 # depends on the box you want to check
hb = 30   # half box size of fit to gausian (around brightest pixel)
yc = int(box[n].index[int(cents[n][0])]) # the real loction of centroid row in original data
xc = int(box[n].columns[int(cents[n][1])]) # the real location of centroid column in original data
fitbox = data[(yc-hb):(yc+hb), (xc-hb):(xc+hb)] # box limits: data[yc+-hb, xc+-hb]
yp, xp = fitbox.shape
y, x, = np.mgrid[:yp, :xp]  # Generate grid of same size like box to put the fit on
f_init = models.Gaussian2D()  # Declare what function you want to fit to your data
fit_f = fitting.LevMarLSQFitter()  # Declare what fitting function you want to use
f = fit_f(f_init, x, y, np.array(fitbox))  # Fit the model to your data (box)
std = [f.x_stddev[0],f.y_stddev[0]]
seecheck = float((np.mean(std) * 2.355 * pscale))

plt.figure(figsize=(8, 2))
plt.subplot(1, 3, 1)
plt.imshow(fitbox) # raw data
plt.title("Data")
plt.subplot(1, 3, 2)
plt.imshow(f(x, y))   # fit to normal dist
plt.title("Model")
plt.subplot(1, 3, 3)
plt.imshow(fitbox - f(x, y))  # residuals
plt.title("Residual")
plt.show() 

"""

""" 
xdis = list(range(len(cents)))
ydis = list(range(len(cents)))
radial = list(range(len(cents)))
for n in list(range(len(cents))):   # changing back the indices of cents to the original indices of image
    if type(cents[n]) == tuple:
        xdis[n] = abs(int(round(len(box[n].index)/2) - (cents[n][0][0] + box[n].index[0])))  # index of central column - cents column index + first index of box
        ydis[n] = abs(int(round(len(box[n].columns)/2) - cents[n][1][0] + box[n].columns[0]))
        radial[n] = ((xdis[n]**2)+(ydis[n]**2))**0.5
    else:
        xdis[n] = np.nan
        ydis[n] = np.nan
        radial[n] = np.nan   
plt.scatter(xdis,see)
plt.scatter(ydis,see)
plt.scatter(radial,see)
plt.xlabel('distance from central pixel (pixels)')
plt.ylabel('seeing value (arcsec)')
plt.legend(['x axis','y axis','radial'])

s = list(see[i] for i in [3,20,34,38,42,46,50,60] )
x = list(xdis[i] for i in [3,20,34,38,42,46,50,60] )
y = list(ydis[i] for i in [3,20,34,38,42,46,50,60] )
rad = list(radial[i] for i in [3,20,34,38,42,46,50,60] )
plt.scatter(x,s)
plt.scatter(y,s)
plt.scatter(rad,s)
plt.xlabel('distance from central pixel (pixels)')
plt.ylabel('seeing value (arcsec)')
plt.legend(['x axis','y axis','radial'])
 """







