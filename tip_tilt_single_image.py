# this code takes astronomical images and creates a heatmap showing the seeing in different areas of the image.
# it first divides the image to N*N/1.5 "boxes", and then it divides each of those to n mini boxes.
# then, in each minibox it finds a star and fits a 2D gausssian to find the fwhm of it. for each box the 
# fwhm is averaged, and that's the final value in the heatmap.


import os
pathscript = input("please write the path of the script \"tip_tilt_new.py\": \n\n")
os.chdir(pathscript)
from astropy.io import fits
import pandas as pd 
from matplotlib import pyplot as plt
from tip_tilt import boxes,cent, see 
import time
pathimages = input("please write the images path: \n\n")
os.chdir(pathscript)

#%%      ######################## input ##############################
image = "LAST.1.01.4_20210721.190108.650_clear__sci_raw_im_001.fits" 
max_brightness = 65000 # saturation value (white pixel) 
N = 9 # multipple of 3 hihger than 6 on. each image will be divided to N*(N/1.5)
fwhm = 10
hb = 20
start_time = time.time()
figname = image
SNR = 10
pscale = 1.25 # plate scale of LAST (arcsec/pixel)

####################### main ##########################
data = fits.getdata(image)
box = boxes(data,N,max_brightness)
cents,cents_final = cent(data,box,fwhm,pscale,hb,SNR)
# seeing = dictionary with fwhm calculated for each of the stars in "cents"
# avsee = mean of fwhm for every box
seeing, avsee = see(data,box,cents,hb,fwhm,cents_final,pscale)
heat = pd.DataFrame(data.copy()) # heatmap size = full image size 
heat = heat.iloc[53:(len(heat)-30),30:(len(heat.columns)-61)]
for n in list(range(len(box))):
    ri = box[n].index[0] # first row of box index 
    rf = box[n].index[int(len(box[n].index))-1] # last row index
    ci = box[n].columns[0] # first column index 
    cf = box[n].columns[int(len(box[n].columns))-1] # last column index 
   # for each box[n] change the value to the value of the box seeing of that area
    heat.loc[ri:rf, ci:cf] = avsee[n]
   
 #plt.figure(n)
plt.ioff()
plt.imshow(heat)
plt.clim(0,12)
plt.colorbar()
plt.xlabel('position (pixel)', fontsize = 11)
plt.ylabel('position (pixel)',fontsize = 11)    
#plt.savefig('tip-tilt-'+figname+'.pdf') 
plt.title(figname[:31], fontsize = 10)
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()

cs = plt.contourf(heat, levels = [3,5,7,9], cmap="bone", origin="lower")
CSl = plt.contour(heat, levels = [3,5,7,9], colors='tab:pink')
plt.colorbar(cs)
plt.show()

