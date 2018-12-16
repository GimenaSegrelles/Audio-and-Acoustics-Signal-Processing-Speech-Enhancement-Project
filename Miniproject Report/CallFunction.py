import subspace
from scipy.io import wavfile

fs, enhanced_signal = subspace.subspace_enhance('Audio/noisy_SNR_10.wav', 80, 10, 10, 0.1)
wavfile.write('Audio/denoised_SNR_10.wav',fs,enhanced_signal)
