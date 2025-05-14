from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import os 
from scipy.optimize import curve_fit
from scipy import stats

def LoadPineExport(rootFolder):
    excitationFolder =  os.path.join(rootFolder, 'other plots', 'Excitation Waveform')
    currentFolder = os.path.join(rootFolder, 'other plots', 'Current')
    
    #get csv files
    
    excitationFiles = [f for f in os.listdir(excitationFolder) if f.endswith('.csv') and 'measured' in f.lower()]
    currentFiles = [f for f in os.listdir(currentFolder) if f.endswith('.csv') and 'measured' in f.lower()]
    
    #load csv files
    excitations={}
    currents ={}
    for file in excitationFiles:
        filePath = os.path.join(excitationFolder, file)
        df = pd.read_csv(filePath )
        excitations[file] = df 
    for file in currentFiles:
        filePath = os.path.join(currentFolder, file)
        df = pd.read_csv(filePath )
        currents[file] = df 
        
    datas={}
    for exitation in excitations.keys():
        #file will be in the form of 'Measured Current (I1).csv'
        #extract the part in the brackets
        if '(' in exitation:
            potentialInside = exitation.split('(')[1].split(')')[0]
            potential = excitations[exitation]
            
            currentInside = '(' + potentialInside.replace('E', 'I') + ')'
            current = currents[ [x for x in currents.keys() if currentInside in x][0] ]
        else:
            potentialInside = 'CV'
            potential = excitations[exitation]
            current = currents[ 'Measured Current.csv' ]
        
        time = potential['Time (s)'].values
        potential = potential['Potential (V)'].values
        
        timeC = current['Time (s)'].values
        current = current['Current (A)'].values
        
        alignedCurrent = np.interp(time, timeC, current)
        datas[potentialInside] = pd.DataFrame({'Time (s)': time, 'Potential (V)': potential, 'Current (A)': alignedCurrent})
    return datas


def LoadAllDatasets(rootFolder):
    voltammagrams={}
 
    for root, dirs, _ in os.walk(rootFolder):
        for dir_name in dirs:
            if dir_name == 'Voltammogram':
                datas= LoadPineExport( root )    
                key = os.path.basename(root)
                
                parts = {}
                maxPotentials = []
                minPotentials = []
                speeds = []
                for data in datas.keys():
                    times = np.array( datas[data]['Time (s)'])
                    potential = np.array( datas[data]['Potential (V)'])
                    maxPotential = np.max(potential)
                    minPotential = np.min(potential)
                    
                    maxPotentials .append( maxPotential)
                    minPotentials .append( minPotential)
                        
                    peaks, _ = find_peaks(potential,maxPotential*.1)
                    mins, _ = find_peaks(-potential,minPotential*-.1)
                    
                    combined = np.concatenate((peaks, mins))
                    combined.sort()
                    if len(combined) < 2:
                        speed = 0
                    else:
                        speed = (combined[1] - combined[0]) / (times[combined[1]] - times[combined[0]])
                    
                    speeds.append(speed)
                    parts[data] = {
                        'peaks_index': peaks,
                        'mins_index': mins,
                        'data': datas[data]
                    }
                
                cycles = np.min( [len(peaks), len(mins)] )  
                
                if len(maxPotentials) == 0:
                    continue 
                if len(maxPotentials) >1:
                    offset = maxPotentials[1] - maxPotentials[0]
                else :
                    offset = 0
                maxPotentials = np.max(maxPotentials)
                minPotentials = np.min(minPotentials)
                speeds = np.mean(speeds)                
                
                
                
                voltammagrams[key]={
                    'maxPotential_V': maxPotentials,
                    'minPotential_V': minPotentials,
                    'speed_Vs': speeds,
                    'parts': parts,
                    'offset_V': offset,
                    'cycles': cycles,
                }
                
    return voltammagrams


def RisingFalling( part):
    times =np.array( part ['data']['Time (s)'])
    potential = np.array( part ['data']['Potential (V)'])
    current = np.array( part ['data']['Current (A)'])
    peaks = np.array( part ['peaks_index'])
    mins = np.array( part ['mins_index'])
    combined = np.concatenate((peaks, mins))
    combined.sort()
    
    rising_segments = []
    falling_segments = []
   
    for i in range(len(combined)-1):
        start = combined[i]
        end = combined[i+1]
        if (potential[start] > potential[end]):
            falling_segments.append((start, end))
        else:
            rising_segments.append((start, end))
    start = combined[-1]
    end = len(times)-1
    if (potential[start] > potential[end]):
        falling_segments.append((start, end))
    else:
        rising_segments.append((start, end))

    rising_segments = [ {'times': times[start:end], 'potential': potential[start:end], 'current':  current[start:end]} for start, end in rising_segments]
    falling_segments = [ {'times': times[start:end], 'potential': potential[start:end], 'current':  current[start:end]} for start, end in falling_segments]
    return rising_segments, falling_segments

def SegmentSlopes(segments):
    # Function to determine the slopes of segments
    slopes = []
    intercepts = []
    
    for segment in segments:
        potential = segment['potential']
        current = segment['current']
        
        # Create segments of random lengths over multiple scales
        min_points_for_fit = 3  # Minimum points needed for a reliable fit

        # Try different segment sizes (small, medium, large)
        segment_sizes = [
            max(min_points_for_fit, int(len(potential) * 0.05)),  # Small segments (5% of data)
            max(min_points_for_fit, int(len(potential) * 0.1)),   # Medium segments (10% of data)
            max(min_points_for_fit, int(len(potential) * 0.2))    # Large segments (20% of data)
        ]

        for _ in range(250):  # Generate about 30 random segments
            # Pick a random segment size
            segment_size = np.random.choice(segment_sizes)
            
            # Pick a random starting point
            start_idx = np.random.randint(0, len(potential) - segment_size)
            end_idx = start_idx + segment_size
            
            # Fit a line to get slope and intercept
            x = potential[start_idx:end_idx]
            y = current[start_idx:end_idx]
            
            if len(np.unique(x)) < 2:  # Skip if all x values are the same
                continue
                
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            
            slopes.append(slope)
            intercepts.append(intercept)
    
    return slopes, intercepts

def DetermineElectrical(voltammagrams, key, electrode ='E1' , plot=False):
    voltammagram = voltammagrams[key]
    risingSegments,fallingSegments = RisingFalling(voltammagram['parts'][electrode])
    
    risingValueAtZero = []
    fallingValueAtZero = []
    for seg in risingSegments:
        risingValueAtZero.append( np.mean( seg['current'][ np.abs(seg['potential'])<.20]))
    for seg in fallingSegments:
        fallingValueAtZero.append( np.mean( seg['current'][ np.abs(seg['potential'])<.20]))
    risingValueAtZero = np.mean(risingValueAtZero)
    fallingValueAtZero = np.mean(fallingValueAtZero)
    
    # Function to determine the most common slope of potential vs current curve
    slopes_rising = []
    slopes_falling = []
    
    slopes_rising, intercepts_rising = SegmentSlopes(risingSegments)
    slopes_falling, intercepts_falling = SegmentSlopes(fallingSegments)
            
    slopes_combined = slopes_rising + slopes_falling
    
    # Create histogram and get the most frequent bin
    hist, bin_edges = np.histogram(slopes_combined, bins='auto')
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    most_common_idx = np.argmax(hist)
    most_common_slope = bin_centers[most_common_idx]
    
    std=np.sqrt( np.sum( [ (slopes_combined[i]-most_common_slope)**2 for i in range(len(slopes_combined))]))/ len(slopes_combined) 
    
    most_likely_rinsing_intercept = []
    most_likely_falling_intercept = []
    for i in range(len(slopes_rising)):    
        if abs(slopes_rising[i]-most_common_slope) < std:
            most_likely_rinsing_intercept.append(intercepts_rising[i])
    for i in range(len(slopes_falling)):
        if abs(slopes_falling[i]-most_common_slope) < std:
            most_likely_falling_intercept.append(intercepts_falling[i])
    # Calculate the average intercept for the most common slope
    intercepts_rising = np.mean(most_likely_rinsing_intercept)
    intercepts_falling = np.mean(most_likely_falling_intercept)    
    
    if plot:
        for seg in risingSegments:
            plt.plot(seg['potential'], seg['current'], label='Rising Segment', alpha=0.5)
            plt.plot(seg['potential'], most_common_slope * seg['potential'] + intercepts_rising, 'r--', label='Most Common Slope')
        for seg in fallingSegments:
            plt.plot(seg['potential'], seg['current'], label='Falling Segment', alpha=0.5)
            plt.plot(seg['potential'], most_common_slope * seg['potential'] + intercepts_falling, 'g--', label='Most Common Slope')
        plt.xlabel('Potential (V)')
        plt.ylabel('Current (A)')        
        plt.show()
    leakage = most_common_slope #current/potential (A/V)
    capacitance =  np.abs (intercepts_rising - intercepts_falling) / voltammagram['speed_Vs'] #F 
    capacitance_gap = np.abs(risingValueAtZero - fallingValueAtZero) / voltammagram['speed_Vs'] #F
    return leakage, capacitance,capacitance_gap,  {'slope': most_common_slope, 'intercept_rising': intercepts_rising, 'intercept_falling': intercepts_falling}



class PlotMode(Enum):
    ByTime = "Time"
    ByEachPotential = "Each Potential"
    ByE1Potential = "E1 Potential"
    ByE2Potential = "E2 Potential"
    
def SetPrefix(y):
    y=np.array(y)
    # Determine best unit prefix for current values
    y_abs_max = max(abs(y.min()), abs(y.max()))
    if y_abs_max < 1e-12:
        scale = 1e15
        prefix = 'f'
    if y_abs_max < 1e-9:
        scale = 1e12
        prefix = 'p'
    elif y_abs_max < 1e-6:
        scale = 1e9
        prefix = 'n'
    elif y_abs_max < 1e-3:
        scale = 1e6
        prefix = 'μ'
    elif y_abs_max < 1:
        scale = 1e3
        prefix = 'm'
    else:
        scale = 1
        prefix = ''    
    return scale, prefix




def AverageCycles( part):
    times =np.array( part ['data']['Time (s)'])
    potential = np.array( part ['data']['Potential (V)'])
    current = np.array( part ['data']['Current (A)'])
    peaks = np.array( part ['peaks_index'])
    
    if len(peaks) > 2:
        peaks = peaks[1:]

    times = times[peaks[0]:]
    potential = potential[peaks[0]:]
    current = current[peaks[0]:]
    peaks = peaks-peaks[0]
    for i in range(len(peaks)-1):
        start = peaks[i]
        end = peaks[i+1]
        if (end>len(times)):
            print('End is too long')
        times[start:end] = times[start:end] - times[start] 

    times[peaks[-1]:] = times[peaks[-1]:]-times[peaks[-1]]
    
    idx = np.argsort(times)
    times = times[idx]
    potential = potential[idx]
    current = current[idx]  
    
    smooth_current = savgol_filter(current, 11, 2)
        
    #loop back to the start of the array to close the loop
    potential2 =np.concatenate((potential, [potential[0]]))
    smooth_current =np.concatenate((smooth_current, [smooth_current[0]]))
    return times, potential2,  smooth_current
    

def PlotCV(voltammagrams,dataset, showTitle=True,showAverage=True, mode = PlotMode.ByE1Potential,  figsize=(6, 4)):
    voltammagram = voltammagrams[dataset]
    if len(voltammagram['parts'].keys())==2:
        subplots = True
    else:
        subplots = False
        
    eachPotential = False
    if mode == PlotMode.ByTime:
        xKey = 'Time (s)'
        yKey = 'Current (A)'
        eachPotential = True
        scaleV,prefixV = 1, ''
    elif mode == PlotMode.ByEachPotential:
        xKey = 'Potential (V)'
        yKey = 'Current (A)'
        eachPotential = True
        electrodes = list(voltammagram['parts'].keys())
        scaleV,prefixV = SetPrefix(   voltammagram['parts'][electrodes[0]]['data'][xKey])
    elif mode == PlotMode.ByE1Potential:
        dataKey = 'E1'
        xKey = 'Potential (V)'
        yKey = 'Current (A)'
        scaleV,prefixV = SetPrefix( voltammagram['parts'][dataKey]['data'][xKey])
    elif mode == PlotMode.ByE2Potential:
        dataKey = 'E2'
        xKey = 'Potential (V)'
        yKey = 'Current (A)'
        scaleV,prefixV = SetPrefix(  voltammagram['parts'][dataKey]['data'][xKey])
        
    plt.figure(figsize=figsize)        
    
    yAll = []
    for key in voltammagram['parts']:
        data = voltammagram['parts'][key]['data']
        y = data[yKey]
        yAll.extend(y)
        
    scale, prefix = SetPrefix( yAll)
    
    del yAll
    cc=0
    for key in voltammagram ['parts']:
        if showAverage and voltammagram['cycles']>1 and xKey!='Time (s)':
            data = voltammagram['parts'][key]['data']
            _, potential2,smooth_current = AverageCycles(voltammagram['parts'][key])
            if eachPotential==False:
                x = voltammagram['parts'][dataKey]['data'][xKey]
            else:
                x= data[xKey]
            if cc==1 and eachPotential==False:
                offset = voltammagram['offset_V']
            else:
                offset = 0
            plt.scatter(x*scaleV,data[yKey]*scale, s=0.5, alpha =.2)
            plt.plot((potential2-offset)*scaleV,smooth_current*scale, label=f'{key} average')
        else :
            data = voltammagram['parts'][key]['data']
            if eachPotential==False:
                x = voltammagram['parts'][dataKey]['data'][xKey]
            else:
                x= data[xKey]
            plt.plot(x*scaleV,data[yKey]*scale, label=key)
        cc+=1
    
    
    if mode == PlotMode.ByTime:
        plt.xlabel(f'Time ({prefixV}s)')
    elif mode == PlotMode.ByEachPotential:
        plt.xlabel(f'Potential ({prefixV}V)')
    elif mode == PlotMode.ByE1Potential:
        plt.xlabel(f'E1 Potential ({prefixV}V)')
    elif mode == PlotMode.ByE2Potential:
        plt.xlabel(f'E2 Potential ({prefixV}V)')
    
    plt.ylabel(f'Current ({prefix}A)')
    
    
    if showTitle:
        scaleVs, prefixVs = SetPrefix(voltammagram['speed_Vs'])
        title = f'{dataset} {np.round( scaleVs*voltammagram["speed_Vs"]):.0f}{prefixVs}V/s'
        if subplots:
            scale, prefix = SetPrefix(voltammagram['offset_V'])
            title += f' {np.round(scale*voltammagram["offset_V"]):.0f}{prefix}V'
        plt.title(title)
    if subplots:
        plt.legend()
    plt.show()
    


def gaussian(x, amp, cen, wid):
    """Gaussian function"""
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))

 

def multi_gaussian_with_exp(x, *params):
    """Multiple Gaussians with exponential background"""
    # First two parameters are for exponential background
    a, b = params[0:2]
    m,intercept = params[2:4]
    # The rest of parameters are for Gaussians (3 params per Gaussian)
    y_exp = a * np.exp(-b * x)
    
    y_line = m*x + intercept
    
    y_gauss = np.zeros_like(x)
    for i in range(4, len(params), 3):
        amp = params[i]
        cen = params[i+1]
        wid = params[i+2]
        y_gauss += gaussian(x, amp, cen, wid)
    
    return y_exp + y_gauss + y_line

def fit_multiple_gaussians_falling(falling_segment, raw_elect,  peak_locations ):
    """
    Fit multiple Gaussians to a falling segment
    
    Parameters:
    -----------
    falling_segment : dict
        Dictionary containing 'potential' and 'current' data for a falling segment
    num_gaussians : int
        Number of Gaussians to fit
    peak_locations : list, optional
        Initial guesses for the center of each Gaussian
        
    Returns:
    --------
    params : array
        Fitted parameters
    perr : array
        Standard deviations of the parameters
    """
    direction = -1
    x = falling_segment['potential']
    y = falling_segment['current']*direction
    idx= np.argsort(x)
    x = x[idx]
    y = y[idx]
    
    x=x[:int(len(x)*.9)]
    y=y[:int(len(y)*.9)]
    
    
    num_gaussians=len(peak_locations)
    # Initial guess for exponential background
    
    
    slope = raw_elect['slope']*-1
    intercept = raw_elect['intercept_falling']*-.8
    y_corr = y - slope*x + raw_elect['intercept_rising']
   
    x_10 = x[0:10]
    y_10=  np.log(y_corr[0:10])
    exp_fit = np.polyfit(x_10, y_10, 1)
    a0 =0# np.exp(exp_fit[1])
    b0 = -exp_fit[0]*.5
   
    
    # Initial parameters
    popt = [a0, b0 ,slope, intercept]
    
    # Set bounds
    lower_bounds = [0,           0,       slope*5, -6*intercept]
    upper_bounds = [np.inf, np.inf, 0, 2*intercept]
    
    # Generate initial guesses for Gaussian parameters if not provided
    if peak_locations is None:
        # Evenly space the peaks across the potential range
        peak_locations = np.linspace(x.min(), x.max(), num_gaussians+2)[1:-1]
    
    # Add initial guesses for each Gaussian
    for i in range(num_gaussians):
        # amplitude, center, width
        
        amp = np.mean( y_corr[np.abs(x-peak_locations[i])<.01])/2  # Initial amplitude guess
        cen = peak_locations[i]
        wid = 0.1  # Initial width guess
        
        popt.extend([amp, cen, wid])
        lower_bounds.extend([-1*np.inf, x.min(), 0.01])
        upper_bounds.extend([np.inf, x.max(), 0.5])
    
    try:
        # Perform the curve fitting
        
        popt, pcov = curve_fit(
            multi_gaussian_with_exp, 
            x, y, 
            p0=popt,
            bounds=(lower_bounds, upper_bounds),
            maxfev=100000
        )
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except RuntimeError as e:
        print(f"Error during curve fitting: {e}")
        return params, np.zeros_like(params)

def fit_multiple_gaussians_rising(rising_segment, raw_elect, peak_locations, falling_params=None):
    """
    Fit multiple Gaussians to a rising segment
    
    Parameters:
    -----------
    rising_segment : dict
        Dictionary containing 'potential' and 'current' data for a rising segment
    direction : float
        Direction multiplier for the current
    raw_elect : dict
        Dictionary with electrode parameters
    peak_locations : list
        Initial guesses for the center of each Gaussian
    falling_params : array, optional
        Parameters from the falling segment fit to use as initial guesses
        
    Returns:
    --------
    params : array
        Fitted parameters
    perr : array
        Standard deviations of the parameters
    """
    x = rising_segment['potential']
    y = rising_segment['current']
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    
    # Use a subset of data for better fitting
    x=x[10:-10]
    y=y[10:-10]
    
    num_gaussians = len(peak_locations)
    
    # Initial guess for background
    slope = raw_elect['slope']
    intercept = raw_elect['intercept_rising']*.8
    
     
    # Initial exponential background parameters
    y_corr = y - slope*x - raw_elect['intercept_rising']
     
    x_10 = x[:10]  # Use last 10 points for rising
    y_10 = np.log(y_corr[:10]*-1)
    exp_fit = np.polyfit(x_10, y_10, 1)
    a0 =-1* np.exp(exp_fit[1])/8
    b0 =-1* exp_fit[0]*0.5
    
    # Initial parameters
    popt = [a0, b0, slope, intercept]
    
    # Set bounds
    lower_bounds = [-1*np.inf, -np.inf, 0, -np.inf]
    upper_bounds = [0, np.inf, slope*5, intercept*5]
    
    # Add initial guesses for each Gaussian
    for i in range(num_gaussians):
        # amplitude, center, width
        
        amp = np.mean( y_corr[np.abs(x-peak_locations[i])<.05])  # Initial amplitude guess
        cen = peak_locations[i]
        wid = 0.1  # Initial width guess
        
        popt.extend([amp, cen, wid])
        lower_bounds.extend([-1*np.inf, x.min(), 0.01])
        upper_bounds.extend([np.inf, x.max(), 0.5])
    
    try:
       
        # Perform the curve fitting
        popt, pcov = curve_fit(
            multi_gaussian_with_exp, 
            x, y, 
            p0=popt,
            bounds=(lower_bounds, upper_bounds),
            maxfev=100000
        )
        perr = np.sqrt(np.diag(pcov))
        return popt, perr
    except RuntimeError as e:
        print(f"Error during curve fitting: {e}")
        return popt, np.zeros_like(popt)
def plot_gaussian_fits(segment, params, direction, title=None,figsize=(6, 4)):
    """Plot the data with the fitted Gaussians"""
    x = segment['potential']
    y = segment['current']*direction
    
    plt.figure(figsize=figsize)
    plt.scatter(x, y, label='Data', alpha=0.5, s=10)
    
    # Plot the overall fit
    y_fit = multi_gaussian_with_exp(x, *params)
    plt.plot(x, y_fit, 'r-', linewidth=2, label='Overall Fit')
    
    # Plot the exponential background
    a, b, m,I = params[0:4]
    y_exp = a * np.exp(-b * x)   
    plt.plot(x, y_exp, 'g--', linewidth=1.5, label='Exponential Background')
    plt.plot(x, m*x + I, 'b--', linewidth=1.5, label='Linear Background')
    
    # Plot individual Gaussians
    x_fine = np.linspace(x.min(), x.max(), 1000)
    for i in range(4, len(params), 3):
        amp = params[i]
        cen = params[i+1]
        wid = params[i+2]
        y_gauss = gaussian(x_fine, amp, cen, wid)
        plt.plot(x_fine, y_gauss   , '--', linewidth=1, 
                 label=f'Gaussian {i//3}: center={cen:.3f}V')
    
    plt.xlabel('Potential (V)')
    plt.ylabel('Current (A)')
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Return the information about the fitted peaks
    peak_info = []
    for i in range(4, len(params), 3):
        amp = params[i]
        cen = params[i+1]
        wid = params[i+2]
        area = amp * wid * np.sqrt(2 * np.pi)
        peak_info.append({
            'amplitude': amp,
            'center': cen,
            'width': wid,
            'area': area
        })
    
    return peak_info

def plot_combined_fits_with_table(voltammagrams,key,falling_segment, rising_segment, 
                                    params_falling, params_rising, 
                                      title=None,figsize=(6, 4)):
    """
    Plot the falling and rising segments, their fits, and Gaussian components.
    Include a table showing peak information.
    
    Parameters:
    -----------
    falling_segment : dict
        Dictionary containing 'potential' and 'current' data for falling segment
    rising_segment : dict
        Dictionary containing 'potential' and 'current' data for rising segment
    params_falling : array
        Fitted parameters for falling segment
    params_rising : array
        Fitted parameters for rising segment
     
    title : str, optional
        Plot title
    """
    # Create figure with two subplots (main plot and Gaussian components)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
    ax1 = fig.add_subplot(gs[0])  # Main plot
    ax2 = fig.add_subplot(gs[1])  # Gaussian components
    ax3 = fig.add_subplot(gs[2])  # Table
    
    # Get data for each segment
    x_falling = falling_segment['potential']
    y_falling = falling_segment['current']  
    x_rising = rising_segment['potential']
    y_rising = rising_segment['current']  
    
    scale,prefix= SetPrefix(y_falling)
    
    # Main plot - Scatter data
    ax1.scatter(x_falling, y_falling*scale, color='green', alpha=0.5, s=10, label='Falling Data')
    ax1.scatter(x_rising, y_rising*scale, color='blue', alpha=0.5, s=10, label='Rising Data')
    
    # Main plot - Overall fits
    y_fit_falling = multi_gaussian_with_exp(x_falling, *params_falling)*-1
    y_fit_rising = multi_gaussian_with_exp(x_rising, *params_rising)
    ax1.plot(x_falling, y_fit_falling*scale, 'g-', linewidth=2, label='Falling Fit')
    ax1.plot(x_rising, y_fit_rising*scale, 'b-', linewidth=2, label='Rising Fit')
    
    # Main plot - Background
    a_f, b_f, m_f, I_f = params_falling[0:4]
    a_r, b_r, m_r, I_r = params_rising[0:4]
    
    y_exp_falling = a_f * np.exp(-b_f * x_falling)
    y_line_falling = m_f * x_falling + I_f
    y_exp_rising = a_r * np.exp(-b_r * x_rising)
    y_line_rising = m_r * x_rising + I_r
    
    ax1.plot(x_falling, (y_exp_falling + y_line_falling)*-1*scale, 'g--', linewidth=1, label='Falling Background')
    ax1.plot(x_rising, (y_exp_rising + y_line_rising)*scale, 'b--', linewidth=1, label='Rising Background')
    
    # Set labels and grid for main plot
    ax1.set_xlabel('Potential (V)')
    ax1.set_ylabel(f'Current ({prefix}A)')
    if title:
        ax1.set_title(title)
    ax1.legend( )
    ax1.grid(True, alpha=0.3)
    
    # Create x range for Gaussian components
    x_range = np.linspace(
        min(x_falling.min(), x_rising.min()), 
        max(x_falling.max(), x_rising.max()), 
        1000
    )
    
    # Plot individual Gaussians in second subplot
    for i in range(4, len(params_falling), 3):
        amp = params_falling[i]
        cen = params_falling[i+1]
        wid = params_falling[i+2]
        y_gauss = gaussian(x_range, amp, cen, wid)
        ax2.plot(x_range, y_gauss*scale, 'g-', linewidth=1.5)
    
    for i in range(4, len(params_rising), 3):
        amp = params_rising[i]
        cen = params_rising[i+1]
        wid = params_rising[i+2]
        y_gauss = gaussian(x_range, amp, cen, wid)
        ax2.plot(x_range, y_gauss*scale, 'b-', linewidth=1.5)
    
    ax2.set_xlabel('Potential (V)')
    ax2.set_ylabel(f'Peak ({prefix}A)')
    ax2.grid(True, alpha=0.3)
    
    # Create peak info tables
    peak_info_falling = []
    peak_info_rising = []
    
    for i in range(4, len(params_falling), 3):
        amp = params_falling[i]
        cen = params_falling[i+1]
        wid = params_falling[i+2]
        area = amp * wid * np.sqrt(2 * np.pi)
        peak_info_falling.append({
            'amplitude': amp,
            'center': cen,
            'width': wid,
            'area': area
        })
    
    for i in range(4, len(params_rising), 3):
        amp = params_rising[i]
        cen = params_rising[i+1]
        wid = params_rising[i+2]
        area = amp * wid * np.sqrt(2 * np.pi)
        peak_info_rising.append({
            'amplitude': amp,
            'center': cen,
            'width': wid,
            'area': area
        })
    
    # Create table data
    # Create table data with one row per peak
    table_data = []
    
    # Constants for calculating electron count
    elementary_charge = 1.602176634e-19  # Coulombs per electron
    
    # Calculate scan speed
    scan_speed = voltammagrams[key]['speed_Vs']  # V/s
    
    # Calculate leakage and capacitance from model parameters
    leakage_falling = params_falling[2]  # A/V (slope from linear term)
    leakage_rising = params_rising[2]   # A/V (slope from linear term)
    
    # Calculate capacitance from intercepts
    intercept_falling = params_falling[3]
    intercept_rising = params_rising[3]
    capacitance_calc = abs(intercept_falling - intercept_rising) / scan_speed  # C = I/(dV/dt)
    
    # Add falling peaks
    for i, peak in enumerate(peak_info_falling):
        # Calculate charge in coulombs
        charge_coulombs = peak['area']
        # Calculate number of electrons
        electrons = abs(charge_coulombs / elementary_charge)
        
        row = [
            f'Peak {i+1} (Falling)',
            f"{peak['center']:.4f}",
            f"{scale*peak['amplitude']:.2e}",
            f"{charge_coulombs:.2e}",
            f"{electrons:.2e}"
        ]
        table_data.append(row)
    
    # Add rising peaks
    for i, peak in enumerate(peak_info_rising):
        # Calculate charge in coulombs
        charge_coulombs = peak['area']
        # Calculate number of electrons
        electrons = abs(charge_coulombs / elementary_charge)
        
        row = [
            f'Peak {i+1} (Rising)',
            f"{peak['center']:.4f}",
            f"{scale*peak['amplitude']:.2e}",
            f"{charge_coulombs:.2e}",
            f"{electrons:.2e}"
        ]
        table_data.append(row)
    
    # Add leakage and capacitance information
    table_data.append([
        "",
        "",
        "",
        "",
        ""
    ])
    table_data.append([
        f"Leakage {prefix}A/V",
        "",
        f"{scale*leakage_rising:.2e}",
        f"{-1*scale*leakage_falling:.2e}",
        ""
    ])
    table_data.append([
        "Capacitance (F)",
        "",
        f"{capacitance_calc:.2e}",
        "",
        ""
    ])
    
    # Customize the table
    ax3.axis('tight')
    ax3.axis('off')
    
    # Define column headers
    column_labels = ['Peak', 'Potential (V)', f'Amplitude ({prefix}A)', 'Charge (C)', 'Electrons']
    
    # Create the table
    table = ax3.table(
        cellText=table_data,
        colLabels=column_labels,
        loc='center',
        cellLoc='center',
        colColours=['lightgray'] * len(column_labels)
    )
    
    # Adjust table properties
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    plt.tight_layout()
    plt.show()
    
    return peak_info_falling, peak_info_rising

def analyze_voltammogram_with_combined_plots(voltammagrams, key, raw_elect, peak_locations_falling,peak_locations_rising,  electrode='E1',figsize=(6, 4)):
    """
    Analyze a voltammogram by fitting multiple Gaussians to both falling and rising segments
    and showing a combined visualization
    
    Parameters:
    -----------
    voltammagrams : dict
        Dictionary of voltammograms
    key : str
        Key of the voltammogram to analyze
    raw_elect : dict
        Dictionary with electrode parameters
    peak_locations : list
        Initial guesses for the center of each Gaussian
    electrode : str, optional
        Electrode to analyze
    """
    # Get the rising and falling segments
    rising_segments, falling_segments = RisingFalling(voltammagrams[key]['parts'][electrode])
    
    # Get the first segments
    falling_segment = falling_segments[0]
    rising_segment = rising_segments[0]
    
    # Fit multiple Gaussians to both segments
    params_falling, perr_falling = fit_multiple_gaussians_falling(falling_segment,   raw_elect, peak_locations_falling)
    params_rising, perr_rising = fit_multiple_gaussians_rising(rising_segment, raw_elect, peak_locations_rising, params_falling)
    
    # Plot combined results with table
    peak_info_falling, peak_info_rising = plot_combined_fits_with_table(
        voltammagrams,key,
        falling_segment, rising_segment, 
        params_falling, params_rising,  
        title=f"Gaussian Fits for {key}, {electrode} - Falling and Rising Segments",figsize=figsize
    )
    
    return params_falling, params_rising, peak_info_falling, peak_info_rising

def analyze_voltammogram_gaussians(voltammagrams, key,raw_elect, peak_locations,electrode='E1'):
    """
    Analyze a voltammogram by fitting multiple Gaussians to the falling segments
    
    Parameters:
    -----------
    voltammagrams : dict
        Dictionary of voltammograms
    key : str
        Key of the voltammogram to analyze
    electrode : str, optional
        Electrode to analyze
    num_gaussians : int, optional
        Number of Gaussians to fit
    peak_locations : list, optional
        Initial guesses for the center of each Gaussian
    """
    # Get the rising and falling segments
    rising_segments, falling_segments = RisingFalling(voltammagrams[key]['parts'][electrode])
    
    # We'll analyze the first falling segment
    falling_segment = falling_segments[0]
    
    # Fit multiple Gaussians
    params, perr = fit_multiple_gaussians_falling(falling_segment,-1, raw_elect,    peak_locations)
    params_rising, perr_rising = fit_multiple_gaussians_rising(rising_segments[0], 1, raw_elect, peak_locations, params)
    
    # Plot the results
    peak_info = plot_gaussian_fits(falling_segment, params, -1, 
                               f"Gaussian Fits for {key}, {electrode} - Falling Segment")
    
    # Print information about the peaks
    print(f"\nGaussian Peak Information for {key}, {electrode}:")
    print("-" * 50)
    for i, peak in enumerate(peak_info):
        print(f"Peak {i+1}:")
        print(f"  Center: {peak['center']:.4f} V")
        print(f"  Amplitude: {peak['amplitude']:.4e} A")
        print(f"  Width: {peak['width']:.4f} V")
        print(f"  Area: {peak['area']:.4e} A·V")
        print("-" * 50)
    
    # Print exponential background parameters
    print("\nExponential Background Parameters:")
    print(f"  a: {params[0]:.4e}")
    print(f"  b: {params[1]:.4f}")
    
    return params, peak_info
    