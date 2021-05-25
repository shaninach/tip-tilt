# this code take astronomical images and creates a heatmap showing the seeing in different areas.
# it first divides the image to N*N/1.5 "boxes", and then it divides each of those to n mini boxes.
# then, in each minibox it finds a star and fits  a 2D gausssian to find the fwhm of it. for each box the 
# fwhm in averaged, and that's the final value in the heatmap. 

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
    datacut = data.loc[30:(len(data)-30),30:(len(data.columns)-30)]
    y = int(len(data)/N)
    x = int(len(data.columns)/(N/1.5))
    inty = list(range(int(N)))
    intx = list(range(int(N/1.5)))
    box = {}  
    name = 0
    for j in inty:
        row = datacut.iloc[j*y:((j+1)*y)] # slice to rows
        for i in intx:
            box[name] = row.iloc[:,(i*x):((i+1)*x)] # each row slice to N/1.5 cells
            name += 1     

    return box

def miniboxes (data,N, max_brightness, background):
    """ divides the image to (N^2) cells. 
    in each cell, chooses a star by finding the maximum flux higher than SNR = 10, lower than 90% of saturated pixel
    output: box = dictionary of values of each cell. 
    cents = dictionary with values of centroid location in each box"""
    datacut = data.iloc[30:int(len(data.index)-1)-30,30:int(len(data.columns)-1)-30]
    y = int(len(datacut)/N)
    x = int(len(datacut.columns)/N)
    inty = list(range(int(N)))
    intx = list(range(int(N)))
    minibox = {}  
    name = 0
    for j in inty:
        row = datacut.iloc[j*y:((j+1)*y)] # slice to rows
        for i in intx:
            minibox[name] = row.iloc[:,(i*x):((i+1)*x)] # each row slice to N cells
            name += 1  
            
    minicents = {}
    for n in list(range(len(minibox))): 
        maxlist = minibox[n].max()
        maxval = max(maxlist)
        SNR = maxval/background # S/N ratio
        if SNR <10: #
            minicents[n] = np.nan
        elif SNR > 10 and maxval <= 0.8*max_brightness:
            minicents[n] = np.where(minibox[n] == maxval) # cents will be the reseted indices of centroid's location in box [row, column]
        elif maxval > 0.8*max_brightness:
            saty = np.where(maxlist == maxval) # saturated pixel row serial number inside maxlist
            if len(saty[0]) > 1: # if there's more than one of this value
                newlist = maxlist.drop(maxlist.index[int(saty[0][0])]) # drop the value with first location in saty
                for v in list(range(len(newlist))):
                    if abs(int(saty[0][0]-v)) >= 50: # if distance between value in newlist bigger than 50 pixels from saturated pixel
                        if newlist.iloc[v]/background >10 and newlist.iloc[v] <= 0.8*max_brightness:
                            minicents[n] = np.where(minibox[n] == newlist.iloc[v])
                        else: 
                            minicents[n] = np.nan
            else: # same process as previous loop in case there's only one saturated pixel in maxlist
                newlist = maxlist.drop(maxlist.index[int(saty[0])])
                for v in list(range(len(newlist))):
                    if abs(int(saty[0]-v)) >= 50:
                        if newlist.iloc[v] <= 0.9*max_brightness and newlist.iloc[v]/background > 6:
                            minicents[n] = np.where(minibox[n] == newlist.iloc[v])
                        else: 
                            minicents[n] = np.nan
    return minibox, minicents

def miniseeing (box, minibox, minicents, pscale):
    box = pd.DataFrame(box)
    see = list(range(len(minicents)))
    for n in list(range(len(minicents))):
        if type(minicents[n]) == float: # meaning it's a nan
            see[n] = np.nan
        else:  # type should be tuple
            hb = 30   # half box size of fit to gausian (around brightest pixel)
            yc = int(minibox[n].index[int(minicents[n][0])]) # the real loction of centroid row in original data
            xc = int(minibox[n].columns[int(minicents[n][1])]) # the real location of centroid column in original data
            fitbox = box.loc[(yc-hb):(yc+hb), (xc-hb):(xc+hb)] # box limits: data[yc+-hb, xc+-hb]
            yp, xp = fitbox.shape
            y, x, = np.mgrid[:yp, :xp]  # Generate grid of same size like box to put the fit on
            f_init = models.Gaussian2D()  # Declare what function you want to fit to your data
            fit_f = fitting.LevMarLSQFitter()  # Declare what fitting function you want to use
            f = fit_f(f_init, x, y, np.array(fitbox))  # Fit the model to your data (box)
            see[n] = np.mean([f.x_fwhm, f.y_fwhm])*pscale
            if see[n] >20:
                see[n] = np.nan  
    avsee = np.nanmean(see)
    return avsee

# main loop - 
max_brightness = 65000 # saturation value (white pixel) 
N = 9 # multipple of 3 hihger than 6 on. each image will be divided to N*(N/1.5)
path = r"D:\Master\LAST_ERAN"  # image folder path
os.chdir(path)  # define the path as current directory
pscale = 0.604 # plate scale of telescope (arcsec/pixel)
figname = "LAST.0.1_20200822.145102.527_clear__sci_proc_im_000.fits"
hdul = fits.open(figname)  # Open image
data = hdul[0].data  # data = pixel brightness
background = np.median(data)
#data = data-background # reduce background noise from each pixel
box = boxes(data,N,max_brightness,background)
#see = seeing(data,box,cents,pscale)

avsee = list(range(len(box)))
for n in list(range(len(box))):
    minibox,minicents = miniboxes(box[n],2,max_brightness,background)
    avsee[n] = miniseeing(box[n], minibox, minicents, pscale)

heat = pd.DataFrame(data.copy()) # heatmap size = full image size 
datacut = heat.loc[30:(len(data)-30),30:(len(data[1])-30)]
for n in list(range(len(box))):
    ri = box[n].index[0] # first row of box index 
    rf = box[n].index[int(len(box[n].index))-1] # last row index
    ci = box[n].columns[0] # first column index 
    cf = box[n].columns[int(len(box[n].columns))-1] # last column index 
    datacut.loc[ri:rf, ci:cf] = avsee[n] # for each box[n] change the value to the value of the box seeing of that area
#plt.figure(n)
plt.imshow(heat)
plt.clim(0,np.nanmax(avsee))
plt.colorbar()
plt.xlabel('vertical position (pixel)', fontsize = 11)
plt.ylabel('horizontal position (pixel)',fontsize = 11)    
#plt.savefig('tip-tilt-'+figname+'.pdf') 
# if the error "The fit may be unsuccessful" comes up, check the fit_info, 'message': 
#fit_f.fit_info
#fit_f.fit_info['message']


#%%
# check individual fit - 
plt.imshow(box[27])

b = 27 # index of box to check
n = 2 # index of minibox to check
minibox,minicents = miniboxes(box[b],2,max_brightness,background)
hb = 30   # half box size of fit to gausian (around brightest pixel)
yc = int(minibox[n].index[int(minicents[n][0])]) # the real loction of centroid row in original data
xc = int(minibox[n].columns[int(minicents[n][1])]) # the real location of centroid column in original data
fitbox = box[b].loc[(yc-hb):(yc+hb), (xc-hb):(xc+hb)] # box limits: data[yc+-hb, xc+-hb]
yp, xp = fitbox.shape
y, x, = np.mgrid[:yp, :xp]  # Generate grid of same size like box to put the fit on
f_init = models.Gaussian2D()  # Declare what function you want to fit to your data
fit_f = fitting.LevMarLSQFitter()  # Declare what fitting function you want to use
f = fit_f(f_init, x, y, np.array(fitbox))  # Fit the model to your data (box)
seecheck = np.mean([f.x_fwhm, f.y_fwhm])*pscale

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

plt.imshow(minibox[2])
plt.colobar()

