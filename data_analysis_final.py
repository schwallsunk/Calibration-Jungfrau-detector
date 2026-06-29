import requests
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
from scipy import special 
from matplotlib.colors import LogNorm



# Define constants, camel case is neglected
gainz = ["LowGain","HighGain"]

properties_phyiscal = {"materials":["Ag","Cr","Cu","Fe","In","Mo","Se","Ti"],"energy":[22.2,5.4,8,6.4,24.2,17.5,11.2,4.5]}
initial_guesses_mu={"LowGain":[210,50,100,100,220,200,100,50],"HighGain":[500,150,200,150,600,400,250,100]}
initial_guesses_sigma={"LowGain":[3,0.5,0.5,1,3,3,1,1],"HighGain":[5,3,3,3,5,5,5,3]}
initial_guesses_plateau={"LowGain":[100,1000,1000,1000,100,300,1000,1000],"HighGain":[100,500,800,800,80,100,500,800]}
ADUs_bimodal = {"LowGain":[],"HighGain":[]}
n_bins =  16383 # Dut to 14 bit ADC
path_to_storage = r'E:\\detecting_the_unknown\\results_plateau\\'


# This is the header with various meta data 
# mostly used for debugging
# taken from the provided jupyter notebook as is
header_dt = np.dtype(
    [
        ("Frame Number", "u8"),
        ("SubFrame Number/ExpLength", "u4"),
        ("Packet Number", "u4"),
        ("Bunch ID", "u8"),
        ("Timestamp", "u8"),
        ("Module Id", "u2"),
        ("Row", "u2"),
        ("Column", "u2"),
        ("Reserved", "u2"),
        ("Debug", "u4"),
        ("Round Robin Number", "u2"),
        ("Detector Type", "u1"),
        ("Header Version", "u1"),
        ("Packets caught mask", "8u8")
    ]
)

# The data type describing the file is created by taking the header and then 
# adding a 512x1024 field of unsigned 16 bit integers
# taken from the provided jupyter notebook as is
# Important note: Data is effectively 16 bit long. 
# This means there are two bits added to every pixel describing the gain setting of the pixel.
# We need to get rid of them by right shifting the data
raw_dt = np.dtype([('header', header_dt), ('images', np.uint16, (512,1024))])

# Then we can package up the reading and decoding in a function
# normally you would read from disk but it is equally easy to fetch
# data from the internet
# taken from the provided jupyter notebook as is
def download(fname, n_frames = 100):
    t0 = time.perf_counter()
    #replace with group 2 for second group
    base_url = 'https://drive.switch.ch/index.php/s/???????/download?path=%2Fgroup2&files={}' #change according to your url scheme to fetch data online
    url = base_url.format(fname)
    print(f'fetching: {url} - ', end = '')
    response = requests.get(url, stream=True)


    data = np.zeros(n_frames, dtype = raw_dt)
    for i,chunk in enumerate(response.iter_content(chunk_size=raw_dt.itemsize)):
        data[i] = np.frombuffer(chunk, dtype = raw_dt)
        if i==n_frames - 1:
            break
        
    
    t = time.perf_counter()-t0
    print(f'{t:.2f}s')
    return data

# Function describing fitted gauss peak
def gauss(x, mu, sigma, A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def plateau_gauss(x, mu, sigma, A, plateau):
    return A*np.exp(-(x-mu)**2/2/sigma**2)+plateau*(1-special.erf((x-mu)/(sigma*np.sqrt(2))))/2
 

#In case single fitting wants to be done and splitting of every single histo
def enc_per_pixel(N_pixel,inital_guess_mu, initial_guess_sigma,gain_global):
    y,x,_=plt.hist(N_pixel, bins=n_bins, alpha=.3, label='data')
    x=(x[1:]+x[:-1])/2
    expected = (inital_guess_mu, initial_guess_sigma, 1) # First guesses for gaussian fit
    result = np.zeros_like(y, dtype=float) # create array with zeros for hist
    np.log10(y, where=(y > 0), out=result) # take logarithm of all values larger than 0 to prevent inf runaway
    params, cov = curve_fit(gauss, x, result, expected) # fit on lograithmed data => simplifies the peak detection for the algorithm massivel  
    sig_noise=params[1]
    return (1000*sig_noise/(gain_global)/3.62)

# Main loop for the fitting of the bimodals on all given raw data
for gain in gainz:
    for i, material in enumerate(properties_phyiscal["materials"]):
        try:
            data = download('T-{}-{}_d0_f0_0.raw'.format(material,gain)) # download corresponding data set
            current_settings = "Gain: {} Material:{}".format(gain,material) # create label for plts
            data['images'] = [np.right_shift(image,2) for image in data['images'][:]] # getting rid of gain setting bits which are not used
            mean_image = data['images'].mean(axis = 0)
            data['images'] = [np.absolute(image-mean_image.astype(int)) for image in data['images'][:]] # get rid of background by subtracting the mean value of each single pixel across the aqusition series, important to take abs val
            real_frames=data['images'].flatten() # convert 2D arrays into 1D arrays for histogram production
            plt.close() # Make plt canvas pristine 
            y,x,_=plt.hist(real_frames, bins=np.arange(n_bins), alpha=.3, label='data') # Plot histogram
            y=y[20:]
            x=x[20:]
            plt.ylim(1, 1e6) # y axis limits
            plt.yscale('log') # log scale
            plt.xlim(0, 1000)
            x=(x[1:]+x[:-1])/2 # make it centered
            #x, y inputs can be lists or 1D numpy arrays
            expected = (initial_guesses_mu[gain][i], initial_guesses_sigma[gain][i], initial_guesses_plateau[gain][i]*2, initial_guesses_plateau[gain][i]) # First guesses for gaussian plateau fit
            try:
                result = np.zeros_like(y, dtype=float) # create array with zeros for hist
                np.log10(y, where=(y > 0), out=result) # take logarithm of all values larger than 0 to prevent inf runaway
                x_centers = x[:-1] + np.diff(x) / 2
                params, cov = curve_fit(plateau_gauss, x_centers, result, expected,bounds=(0, np.inf)) # fit on lograithmed data => simplifies the peak detection for the algorithm massively   
                sigma=np.sqrt(np.diag(cov)) # get sigma of each gaussian from covariance
                x_fit = np.linspace(x.min(), x.max(), 2000) # create x coords for plotting fit
                plt.plot(x_fit, 10**plateau_gauss(x_fit, *params), color='red', lw=3, label='model') # plot full bimodal
            except: # take care in case of non convergence in log space to switch back into normal space for second fitting try in case
                params, cov = curve_fit(plateau_gauss, x, y, expected)
                sigma=np.sqrt(np.diag(cov)) # get sigma of each gaussian from covariance
                x_fit = np.linspace(x.min(), x.max(), 500)
                #plot combined...
                plt.plot(x_fit, plateau_gauss(x_fit, *params), color='red', lw=3, label='model')
            plt.legend()
            plt.ylabel("Counts")
            plt.xlabel("[ADU]")
            plt.title(current_settings)
            print(pd.DataFrame(data={'params': params, 'sigma': sigma}, index=plateau_gauss.__code__.co_varnames[1:])) 
            results=pd.DataFrame(data={'params': params, 'sigma': sigma}, index=plateau_gauss.__code__.co_varnames[1:])
            results_rounded=results.round(decimals=3)
            plt.text(600, 100, results_rounded, style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
            path_fig=os.path.join(path_to_storage,"results_bimodal_{}_{}.png".format(material,gain))
            plt.savefig(path_fig)
            ADUs_bimodal[gain].append(float(results["params"].iloc[0]))
    
        except Exception as e:
          print("An exception occurred: {}".format(e))
print(ADUs_bimodal)


for gain in gainz:
    plt.close()
    x_points =[]
    y_points =[]
    for i, material in enumerate(properties_phyiscal["materials"]):
        x_points.append(ADUs_bimodal[gain][i])
        y_points.append(properties_phyiscal["energy"][i])
    plt.title("Resulting fit at gain: {}".format(gain))
    plt.xlabel("Photon energy [keV]")
    plt.ylabel("ADC channel [ADU]")
    plt.xlim(4,25)

    
    res = stats.linregress(y_points, x_points)
    plt.plot(y_points, x_points, 'o', label='experimental data')
    plt.plot(y_points, [(res.intercept + res.slope*y) for y in y_points], 'r', label='fitted line')
    df_regr = pd.DataFrame(data={"params":[res.slope, res.intercept, res.rvalue**2],"sigma":[res.stderr, res.intercept_stderr, res.pvalue**2]}, index=["slope", "intercept", "r²,p²"])
    if gain =="HighGain":
        plt.ylim(0,650)
        plt.text(5, 500, df_regr.round(decimals=3), style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    else:
        plt.ylim(0,260)
        plt.text(5, 200, df_regr.round(decimals=3), style='italic', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
        
    
    path_fig=os.path.join(path_to_storage,"results_fitting_{}.png".format(gain))
    plt.savefig(path_fig)
    # Do rms analysis of dark frames
    data = download('noise_floor-{}_d0_f0_0.raw'.format(gain),n_frames=1000) # fetch dark data
    data['images'] = [np.right_shift(image,2) for image in data['images'][:]]
    pd_rms=data['images'].std(axis = 0)
    factor = 1000/3.62/res.slope
    enc = pd_rms*factor
    #covert measured ADU in energy first and then into electron noise using the pair creation energy 
    plt.close() # Make plt canvas pristine 
    fig, ax = plt.subplots()
    ax.set_title('Pixel-wise <e- r.m.s.> at {} \n avg value: {} e- r.m.s.'.format(gain, int(np.mean(enc))))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    im = ax.imshow(enc,norm=LogNorm(vmin=np.median(enc)*0.8, vmax=1.5*np.median(enc)))
    fig.colorbar(im)
    path_fig=os.path.join(path_to_storage,"results_enc_{}.png".format(gain))
    print(gain)
    print("standard deviation of enc")
    print(int(np.std(enc)))
    plt.savefig(path_fig)