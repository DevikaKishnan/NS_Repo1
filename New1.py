
#readFileAndPlot.py]
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt

#######

# Use numpy to load the data contained in the file
data = np.loadtxt('Data.txt')

# plot the first column as x, and second column as y
u = data[:,1]
t = data[:,0]
plt.figure(1)
plt.plot(t, u)
plt.title("Unfiltered PPG data")
plt.xlabel('Time[sec]')
plt.ylabel('PPG data')
plt.show()


#print size of u
print "Size of data", u.size

#### Bandpassfilter to filter out required frequencies for Heart Rate from PPG data
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y




 # Sample rate and desired cutoff frequencies (in Hz).
fs = 200
lowcut = 2 
highcut = 30


# Filter the noisy signal.
y = butter_bandpass_filter(u, lowcut, highcut, fs, order=5)

plt.figure(2)
plt.subplot(2, 1, 2)
plt.plot(t, u, color ='crimson', label='data')
plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.ylabel('PPG data')
plt.title("Bandpass Filtered data for obtaining Heart Rate")
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()


#periodogram
N = u.size
FFT2 = abs(sc.fft(y,N))
f2 = 20*sc.log10(FFT2)
f2 = f2[range(N/2)] #remove mirrored part of FFT
freqs2 = sc.fftpack.fftfreq(N, t[1]-t[0])
freqs2 = freqs2[range(N/2)] #remove mirrored part of FFT


#Find highest peak in Periodogram
m = max(f2)
#Find corresponding frequency associated with highest peak
d =[i for i, j in enumerate(f2) if j == m]


#Plotting periodogram
x1 = freqs2[d]
y1 = max(f2)
plt.figure(3)
plt.subplot(2,1,1)
plt.plot(freqs2, f2,color='darkmagenta')
plt.ylabel("PSD")
plt.title('Periodogram for Heart Rate detection')
plt.grid()
plt.subplot(2,1,2)
plt.plot(freqs2,f2,color='turquoise')
plt.xlim((0,10))
plt.ylim((0,y1+20))
plt.text(x1,y1,'Peak corresponding to Maximum PSD')
plt.xlabel('Frequency(Hz)')
plt.ylabel('PSD')
plt.grid()
plt.show()

##Print PSD and frequency
print 'Maximum PSD:' , max(f2)
print "The frequency associated with maximum PSD is", freqs2[d], "Hz"

#Calculate Heart Rate
HeartRate = freqs2[d]*60
print "Heart Rate  =", HeartRate, "Beats per minute"

##RESPIRATION RATE

#Use  low-pass filter to get low DC signal
def butter_lowpass(cutoff3, fs3, order3=5):
    nyq = 0.5 * fs3
    normal_cutoff = cutoff3 / nyq
    b3, a3 = butter(order3, normal_cutoff, btype='low', analog=False)
    return b3, a3

def butter_lowpass_filter(data, cutoff3, fs3, order3=5):
    b3, a3 = butter_lowpass(cutoff3, fs3, order3=order3)
    y = lfilter(b3, a3, data)
    return y


# Filter requirements.
order3 = 4
fs3 = 200      # sample rate, Hz
cutoff3 = 0.3  # desired cutoff frequency of the filter, Hz

# Filter the data, and plot both the original and filtered signals.
rr = butter_lowpass_filter(u, cutoff3, fs3, order3)
plt.figure(4)
plt.subplot(2, 1, 2)
plt.plot(t, u, color ='crimson', label='data')
plt.plot(t, rr, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.ylabel('PPG data')
plt.title('Lowpass Filtered Data for Respiration rate detection')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()

##Periodogram 2
FFT3 = abs(sc.fft(rr,N))
f3 = 20*sc.log10(FFT3)
f3 = f3[range(N/2)]   #remove mirrored part of FFT
freqs3 = sc.fftpack.fftfreq(N, t[1]-t[0])
freqs3 = freqs3[range(N/2)]   #remove mirrored part of FFT



##calculate respiration rate

## Remove maximum PSD because it is at 0 Hz.
## Find best frequency
new = np.array(f3).tolist()
new.remove(max(new))
m3 = max(new)
d3 =[i for i, j in enumerate(new) if j == m3] ## the sample number associated to maximum PSD


##Plotting Periodogram 2
x2 = freqs3[d3]
y2 = m3
plt.figure(5)
plt.subplot(2,1,1)
plt.plot(freqs3, f3,linewidth = 2.5,color='firebrick')
plt.ylabel("PSD")
plt.title('Periodogram for Respiration Rate detection')
plt.grid()
plt.subplot(2,1,2)
plt.plot(freqs3, f3,linewidth = 2.5,color='darkolivegreen')
plt.xlim((0,2))
plt.ylim((0,y2+20))
plt.text(x2,y2,'Peak correspondig to Best Frequency')
plt.grid()
plt.show()

## Print PSD and Frequency
print 'Maximum PSD for best frequency' , m3
print "Frequency corresponding to maximum PSD", freqs3[d3], "Hz"

#Calculate Respiration Rate
RespRate = freqs3[d3]*60

#print "Respiration Rate =", RespRate
print "Respiration Rate  =", RespRate, "Breaths per minute"