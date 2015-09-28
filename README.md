# NS_Repo1
Extracting Heart Rate and Respiration Rate from a PPG signal

Gittrial1.py
This was a file I initiated to understand how Git Repositiry works. 

New1.py [Extraction of Heart Rate and Respiration Rate from PPG data] :

The first step was to visualize the data. I extracted the two columns seperately in order to plot the data and to be able to manipulate the data. 
I took an initial periodogram of the data to see if any viable information was available from the raw data. 

## Heart Rate:
On realizing that the data needed to be filtered, I sought to initiate a bandpass filter that would filter the frequencies which would give us frequencies in the range of possible heart rate values. I chose the cut-offs to be : low cut off = 2 Hz and high cut off = 8 Hz, keeping in mind the bandpass characteristics and the required frequency range (60-200 Hz). I applied the filter to the PPG data and plotted the original and filtered signal. I plotted the Periodogram of the filtered data to obtain the frequency corresponding to the maximum Power Spectral Density[PSD]. I calculated the frequency corresponding to the maximum PSD and displayed it. I calculated the Heart Rate using the particular frequency and displayed it. 

## Respiration Rate:
The Respiration Rate can be obained from the low DC signal that offsets the PPG data. For obtaining the frequency of this low DC signal, I initiated a band pass filter with specifications that allow us to obtain the low frequency signals that correspond to the respiration rate(12-16 breaths per minute). I used a band pass filter to remove the high frequency signals from the PPG data and plotted the original and filtered data. I used a periodogram to obtain the maximum PSD value. I calculated the frequency associated with the maximum PSD value and used it to calculate the Respiration Rate. 
