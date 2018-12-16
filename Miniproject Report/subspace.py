import numpy as np
from scipy.io import wavfile

#This function returns the denoised signal using subspace method as an array
#   wav_noisy is the path to noisy signal
#   K is the lenght of the windows used in the process
#   T is the number of windows we will consider for computing the covariance matrix
#   mu is the parameter that tune the enhancement. Higher values of mu gives better enhancement but also more distortion
#   threshold is the value of the threshold on the energy used for voice activity detection and computate of the noise covariance matrix
def subspace_enhance(wav_noisy,K,T,mu,threshold):
    
    #Open the noisy file and normalize it
    fs_noisy, noisy_signal = wavfile.read(wav_noisy)
    noisy_signal = np.divide(noisy_signal, 32768)
    
    #Noise co-variance
    Energy = noisy_signal*noisy_signal
    n_max_noise = int(2*(len(noisy_signal)-K)/K)
    Somme = np.zeros((K,K))
    p = 0
    for n in range (0,n_max_noise):
        a1 = int(n*K/2)
        b1 = int((n*K/2)+K)
        EnergyFrame = np.mean(Energy[a1:b1])
        if EnergyFrame < threshold :
            Zn = noisy_signal[a1:b1];
            Somme = np.add(Somme,np.outer(Zn,Zn))
            p += 1
    Noise_covariance_Matrix = np.divide(Somme, p)
    
    #Enhancement
    lenght = len(noisy_signal)
    n_max = int(2*(lenght-K)/K);
    Enhance_signal = np.zeros((lenght,1))
    offset = int(T*K/2)
    
    for n in range (T,n_max-T):
        a = int(n*K/2)
        b = int((n*K/2)+K)
        sample = noisy_signal[a:b]
        Somme = np.zeros((K,K))
        comp = 0
        for o in range (a-offset,a+offset):
            Zn = noisy_signal[o:o+K]
            Somme = np.add(Somme,np.outer(Zn,Zn))
            comp = comp+1
        Noisy_Covariance = np.divide(Somme, comp)
        sigma = np.matmul(np.linalg.pinv(Noise_covariance_Matrix),Noisy_Covariance)-np.eye(K)
        Eig,V1 = np.linalg.eig(sigma)
        Number_positive_Eig = 0
        for z in Eig :
            if z>0 :
                Number_positive_Eig+=1
        order = np.argsort(Eig, axis = -1, kind='quicksort')
        order = order[::-1]
        V = V1[:,order]
        V = -V
        Val = Eig[order]
        Val = Val[0:Number_positive_Eig]
        Q1 = np.zeros((K, K))
        for w in range (0,Number_positive_Eig):
            Q1[w,w] = Val[w]/(Val[w]+mu)
        Hopt = np.matmul(np.linalg.pinv(np.transpose(V)),Q1)
        Hopt = np.matmul(Hopt,np.transpose(V))
        Enhance = np.matmul(Hopt,sample.reshape(-1, 1))
        Hamming = np.hamming(K)
        Enhance = np.multiply(Enhance,Hamming.reshape(-1, 1))
        Enhance_signal[a:b] = np.add(Enhance_signal[a:b],Enhance);
    
    return fs_noisy, Enhance_signal