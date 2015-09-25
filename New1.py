# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 15:08:14 2015

@author: Devi

"""
#readFileAndPlot.py]
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt

# Use numpy to load the data contained in the file
# ’fakedata.txt’ into a 2-D array called data
data = np.loadtxt('Data.txt')
print data.size
# plot the first column as x, and second column as y
plt.figure(1)
plt.plot(data[:,0], data[:,1])
plt.xlabel('Time')
plt.ylabel('PPG data')
plt.show()
# find peaks

#print u
u = data[:,1]
print u.size

#### Bandpassfilter
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


u = data[:,1]
t = data[:,0]

 # Sample rate and desired cutoff frequencies (in Hz).
fs = 200
lowcut = 2
highcut = 30


# Filter the noisy signal.
f0 = 200
T = 0.102
plt.figure(2)

y = butter_bandpass_filter(u, lowcut, highcut, fs, order=6)


plt.subplot(2, 1, 2)
plt.plot(t, u, 'b-', label='data')
plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()
#periodogram
t = data[:,0]
N = u.size

print t
FFT2 = abs(sc.fft(y,N))
print FFT2.size
f2 = 20*sc.log10(FFT2)
f2 = f2[range(N/2)]
print f2.size
freqs2 = sc.fftpack.fftfreq(N, t[1]-t[0])
freqs2 = freqs2[range(N/2)]
plt.plot(freqs2, f2)
#plt.axis([0.006,1.450, 0, 150])
plt.grid()
plt.show()

print 'maximum:' , max(f2)
m = max(f2)
d =[i for i, j in enumerate(f2) if j == m]
t[d]

print freqs2[d]
HeartRate = freqs2[d]*60

print "heart Rate  =", HeartRate


#Use bandpass filter to get low DC signal

import numpy as np
from scipy.signal import freqz
import matplotlib.pyplot as plt


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

# Get the filter coefficients so we can check its frequency response.
b3, a3 = butter_lowpass(cutoff3, fs3, order3)

# Plot the frequency response.
w, h = freqz(b3, a3, worN=8000)
plt.subplot(2, 1, 1)
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
plt.plot(cutoff3, 0.5*np.sqrt(2), 'ko')
plt.axvline(cutoff3, color='k')
plt.xlim(0, 0.5*fs3)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Frequency [Hz]')
plt.grid()


# Filter the data, and plot both the original and filtered signals.


rr = butter_lowpass_filter(u, cutoff3, fs3, order3)

plt.subplot(2, 1, 2)
plt.plot(t, u, 'b-', label='data')
plt.plot(t, rr, 'g-', linewidth=2, label='filtered data')
plt.xlabel('Time [sec]')
plt.grid()
plt.legend()

plt.subplots_adjust(hspace=0.35)
plt.show()


#periodogram 2
t = data[:,0]
N = u.size

print t
FFT3 = abs(sc.fft(rr,N))
print FFT3.size
f3 = 20*sc.log10(FFT3)
f3 = f3[range(N/2)]
print f3.size
freqs3 = sc.fftpack.fftfreq(N, t[1]-t[0])
freqs3 = freqs3[range(N/2)]
plt.plot(freqs3, f3)
plt.axis([0.006,1.450, 0, 150])
plt.grid()
plt.show()

##calculate respiration rate
print 'maximum:', max(f3)
new = np.array(f3).tolist()
new.remove(max(new))
m3 = max(new)

d3 =[i for i, j in enumerate(new) if j == m3]

print freqs3[d3]
RespRate = freqs3[d3]*60

print "Resp Rate  =", RespRate






#print "Respiration Rate =", RespRate




