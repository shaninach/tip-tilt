# this code takes astronomical images and creates a heatmap showing the seeing in different areas of the image.
# it first divides the image to N*N/1.5 "boxes", and then it divides each of those to n mini boxes.
# then, in each minibox it finds a star and fits a 2D gausssian to find the fwhm of it. for each box the 
# fwhm is averaged, and that's the final value in the heatmap. 

import numpy as np 
from astropy.io import fits
import os
import pandas as pd 
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.style.use('default')

# functions
def slice_when(function, iterable):
    """
    this function slices a list by the condition given in function. 
    iterable = the list you want to slice
    function = a function to work on iterable's elements. 
    returns = the list sliced (if not sliced at all this will return the original list (iterable).
    """
    i, x, size = 0, 0, len(iterable)
    iterable = np.sort(iterable)
    while i < size-1:
        if function(iterable[i], iterable[i+1]):
            yield iterable[x:i+1]
            x = i + 1
        i += 1
    yield iterable[x:size]
      
def find_seeing(profile, pscale):
    from scipy.interpolate import UnivariateSpline
    x = np.linspace(0,len(profile),len(profile))
    # plt.scatter(x,profile) # for visualization when debuging
    half_max = np.array(profile)-max(profile)/2 # Y minus half maximum (lower)
    spl = UnivariateSpline(x, half_max)
    xs =  np.linspace(0, len(profile), 1000)
    roots = spl.roots()
    see = np.nan
    # only if the maximum value of profile is >60, and there's at list one root, carry on:
    if (max(profile)-min(profile)> 2):
        for f in np.arange(0,100,5):
            spl.set_smoothing_factor(f)
            spl(xs)
            """# for visualization when debuging
            plt.scatter(x,profile)
            plt.plot(xs, spl(xs), 'g', lw=3) 
            plt.plot(xs, np.zeros(1000)) 
            """
            roots = spl.roots()
            if len(roots)>=2 and len(roots)<=4:
                see=(roots[-1]-roots[0])*pscale
                break
            elif len(roots)<=1:
                see = np.nan
                
    else:
        see = np.nan
    return see 
            
def COM(data,yc,xc,fwhm,hb):
    window = pd.DataFrame(data).loc[(yc-hb):(yc+hb),(xc-hb):(xc+hb)]
    xw = []
    X = [] 
    yw = [] 
    Y = []     
    for y in window.index:
        for x in window.columns:
            if ((x-xc)**2 + (y-yc)**2 <= fwhm**2):
                #window[y,x] = 1600
                xw.append(window.loc[y,x]) 
                yw.append(window.loc[y,x])
                X.append(x)
                Y.append(y) 
                """
                window.loc[y,x] = 2000   # to visualize the location of area that is calculated 
                plt.imshow(window)
                plt.colorbar()   
                """                
    wmean= pd.DataFrame({'xpos': X, 'xw': xw, 'ypos': Y, 'yw': yw })
    x = int((wmean['xpos']*wmean['xw']).sum()/wmean['xw'].sum())  # center of mass position x axis  
    y = int((wmean['ypos']*wmean['yw']).sum()/wmean['yw'].sum())  # center of mass position y axis  
    return y,x

def boxes (data,N,max_brightness):
    """ divides the image to (N*N/1.5) cells. N is any multiple of 3 larger than 6 (9,12,15,18...)
    in each cell, chooses a star by finding the maximum flux higher than SNR = 6, lower than 90% of saturated pixel
    output: box = dictionary of values of each cell. 
    cents = dictionary with values of centroid location in each box"""
    import pandas as pd 
    data = pd.DataFrame(data)
    datacut = data.iloc[53:(len(data)-30),30:(len(data.columns)-61)]
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

def cent (data,box,fwhm,pscale,hb,SNR):
    """ box = dictionary containing boxes. 
        fwhm = estimated fwhm of stars in image. 
        pscale = plate scale of the telescope. 
        returns: 
        cents = dictionary with centroids of stars for each box. 
        seeing = dictionary with the list of seeing values for each box (lenght of list depends on number of stars found in that box)"""
    cents = {}
    for boxi in range(len(box)): # = number of boxes
        box[boxi] = (box[boxi]-np.median(box[boxi]))/np.std(box[boxi]) # change data into SNR 
        maxi = data.max()
        cents[boxi] ={}
        cent = np.where(np.logical_and(box[boxi] >= SNR, box[boxi] <= maxi)) # brightest pixels in box
        dfcents = pd.DataFrame({'y': cent[0], 'x': cent[1]}) # new table containing columns of x, y indices of each pixel 
        centsdif = dfcents.diff()
    ## group pixels to different stars by the proximity to one another 
        i = 0
        d = 0
        dividers = centsdif[(np.abs(centsdif['y'])>=8) | (np.abs(centsdif['x']) >=8)]
        for row in dividers.index:
            cents[boxi][i] = dfcents.iloc[d:row]
            i+=1 
            d = row
    
    ## keep only stars of minimum length (of pixels) = fwhm
        for key in range(len(cents[boxi])): # must stay with range because keys change during loop 
            if len(cents[boxi][key]) > 0 and len(cents[boxi][key]) < fwhm:
                cents[boxi].pop(key) 
        # sigma clipping - anything higher or lower than 3sigma 
        for key in cents[boxi].keys():
            lowy = cents[boxi][key]['y'].mean()-3*cents[boxi][key]['y'].std()
            upy = cents[boxi][key]['y'].mean()+3*cents[boxi][key]['y'].std()
            lowx = cents[boxi][key]['x'].mean()-3*cents[boxi][key]['x'].std()
            upx = cents[boxi][key]['x'].mean()+3*cents[boxi][key]['x'].std()
            for i in cents[boxi][key].index:
                if cents[boxi][key]['y'].loc[i] < lowy or cents[boxi][key]['y'].loc[i] > upy:
                    cents[boxi][key].drop(i,0) # delete the row
                elif cents[boxi][key]['x'].loc[i] < lowx or cents[boxi][key]['x'].loc[i] > upx:
                    cents[boxi][key].drop(i,0) # delete the row
    
    cents_final = {}
    for boxi in range(len(box)):
        cents_final[boxi] = {}
        for key in cents[boxi].keys():
            cents_final[boxi][key] = []
            if len(cents[boxi]) == 0:
                continue
            else:
                yc = int(box[boxi].index[0] + cents[boxi][key].mean()['y']) # index of centroid row (in original data)
                xc = int(box[boxi].columns[0] + cents[boxi][key].mean()['x']) # index of centroid column 
                yc, xc = COM(data, yc, xc, fwhm,hb)
                cents_final[boxi][key] = (yc, xc)
    return cents, cents_final

def see (data,box,cents,hb,fwhm,cents_final,pscale):
    # fit the data to Gaussian 
    seeing ={}
    for boxi in range(len(box)):
        seeing[boxi] = []
        for key in cents[boxi].keys():
            see = [] 
            if len(cents[boxi][key]) == 0:
                seeing[boxi].append(np.nan)
            else:
                yc,xc = cents_final[boxi][key]
                #plt.imshow(data[yc-hb:yc+hb,xc-hb:xc+hb])
                vertical = list(data[yc-hb:yc+hb,xc]) 
                vertical -= np.median(vertical)
                see.append(find_seeing(vertical,pscale))
                #plt.scatter(np.linspace(0,len(vertical),len(vertical)),vertical)
                
                horizontal = list(data[yc,xc-hb:xc+hb]) 
                horizontal -= np.median(horizontal)
                see.append(find_seeing(horizontal,pscale))
                #plt.scatter(np.linspace(0,len(horizontal),len(horizontal)),horizontal)
                
                hd = int(np.sqrt(2)*hb)
                Ldiagonal = [data[yc+i,xc+i] for i in range(-hd,hd)]
                Ldiagonal -= np.median(Ldiagonal) # reduce background to be around 0
                #plt.scatter(np.linspace(0,len(Ldiagonal),len(Ldiagonal)),Ldiagonal)
                see.append(find_seeing(Ldiagonal,pscale))
                    
                Rdiagonal = [data[yc+i,xc-i] for i in range(-hd,hd)]
                Rdiagonal -= np.median(Rdiagonal) # reduce background to be around 0
                #plt.scatter(np.linspace(0,len(Rdiagonal),len(Rdiagonal)),Rdiagonal)
                see.append(find_seeing(Rdiagonal,pscale))
            
            # if the std between the different profiles ishigher than 2 arcsec, drop this result:
            if np.std(see) <=fwhm: 
                seeing[boxi].append(np.nanmean(see))
            else:
                seeing[boxi].append(np.nan)       
                
        s = seeing.copy()
        i = 0
        while i < len(s[boxi])-1:
            if s[boxi][i] > 3*np.std(s[boxi]):
                s[boxi].pop(i)
            else:
                i+=1
                
    avsee = [0]*len(seeing)
    for i in range(len(seeing)):
        if np.count_nonzero(~np.isnan(s[i])) == 0:
            avsee[i] = np.nan
        else:
            avsee[i] = np.nanmean(s[i])

    return seeing, avsee


#%%  main loop 
if __name__ == '__main__':
    import time
    
    max_brightness = 65000 # saturation value (white pixel) 
    N = 9 # multipple of 3 hihger than 6 on. each image will be divided to N*(N/1.5)
    path = r"D:\Master\tiptilt21.7\0.0005mm0"  # image folder path
    os.chdir(path)  # define the path as current directory
    fwhm = 10
    hb = 20
    pscale = 1.25 # plate scale of LAST (arcsec/pixel)
    
    for image in os.listdir(path):
        start_time = time.time()
        figname = image
        image = path+'\\'+image
        data = fits.getdata(image)
        box = boxes(data,N,max_brightness)
        cents,cents_final = cent(data,box,fwhm,pscale,hb,SNR=10)
        # seeing = dictionary with fwhm calculated for each of the stars in "cents"
        # avsee = mean of fwhm for every box
        seeing, avsee = see(data,box,cents,hb,fwhm,cents_final,pscale)
        heat = pd.DataFrame(data.copy()) # heatmap size = full image size 
        heat = heat.loc[30:(len(data)-30),30:(len(data[1])-30)]
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
        plt.savefig('tip-tilt-'+figname+'.pdf') 
        plt.title(figname[:31], fontsize = 10)
        print("--- %s seconds ---" % (time.time() - start_time))
        plt.show()

#%%
    import time
    image = r"D:\Master\tiptilt21.7\0.0005mm0\LAST.1.01.4_20210721.190108.650_clear__sci_raw_im_001.fits" 
    max_brightness = 65000 # saturation value (white pixel) 
    N = 9 # multipple of 3 hihger than 6 on. each image will be divided to N*(N/1.5)
    fwhm = 10
    hb = 20
    start_time = time.time()
    figname = image
    SNR = 10
    pscale = 1.25 # plate scale of LAST (arcsec/pixel)
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
        
        
        #%%    
    cs = plt.contourf(heat, levels = [3,5,7,9], cmap="bone", origin="lower")
    CSl = plt.contour(heat, levels = [3,5,7,9], colors='tab:pink')
    plt.colorbar(cs)
    plt.show()