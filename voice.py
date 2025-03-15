import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks, butter, filtfilt, iirnotch

class AudioFrequencyDetector:
    def __init__(self, 
                 sample_rate=44100, 
                 chunk_size=4096, 
                 display_range=(20, 2000),
                 refresh_interval=30,
                 smoothing_factor=0.3):
        """
        Initialize the audio frequency detector.
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_freq, self.max_freq = display_range
        self.refresh_interval = refresh_interval
        self.smoothing_factor = smoothing_factor
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        # Frequency bins for FFT
        self.freq_bins = np.fft.rfftfreq(self.chunk_size, 1.0/self.sample_rate)
        self.min_idx = np.argmax(self.freq_bins >= self.min_freq)
        self.max_idx = np.argmax(self.freq_bins >= self.max_freq)
        if self.max_idx == 0:
            self.max_idx = len(self.freq_bins) - 1
            
        # Prepare smoothing buffer
        self.spectrum_buffer = None
        
        # Initialize plot
        self.setup_plot()
        
    def _apply_window(self, data):
        """Apply Hann window to reduce spectral leakage."""
        return data * np.hanning(len(data))
    
    def _noise_filter(self, data, cutoff=50.0, order=6, notch_freq=50.0, Q=30.0):
        """Apply a high-pass filter and a notch filter to remove background noise."""
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        filtered_data = filtfilt(b, a, data)

        # Notch filter to remove power line interference (50Hz or 60Hz)
        w0 = notch_freq / nyquist
        b_notch, a_notch = iirnotch(w0, Q)
        return filtfilt(b_notch, a_notch, filtered_data)
    
    def _normalize(self, data):
        """Normalize data to 0-1 range."""
        if np.max(data) - np.min(data) > 0:
            return (data - np.min(data)) / (np.max(data) - np.min(data))
        return data
    
    def _detect_peaks(self, spectrum, height_thresh=0.3, distance=10):
        """Find peaks in the spectrum."""
        norm_spectrum = self._normalize(spectrum)
        peaks, _ = find_peaks(norm_spectrum, height=height_thresh, distance=distance)
        peaks = [p for p in peaks if self.min_idx <= p <= self.max_idx]
        return peaks, norm_spectrum
    
    def update_plot(self, frame):
        """Update function for the matplotlib animation."""
        try:
            # Read audio data
            data = np.frombuffer(self.stream.read(self.chunk_size, exception_on_overflow=False), dtype=np.float32)
            
            # Apply preprocessing
            data = self._apply_window(data)
            data = self._noise_filter(data)  # Apply noise removal

            # Compute FFT
            fft_data = np.abs(np.fft.rfft(data))
            
            # Apply temporal smoothing
            if self.spectrum_buffer is None:
                self.spectrum_buffer = fft_data
            else:
                self.spectrum_buffer = (self.smoothing_factor * fft_data + 
                                      (1 - self.smoothing_factor) * self.spectrum_buffer)
            
            # Focus on frequency range
            spectrum = self.spectrum_buffer[self.min_idx:self.max_idx+1]
            freqs = self.freq_bins[self.min_idx:self.max_idx+1]
            
            # Find peaks
            peak_indices, norm_spectrum = self._detect_peaks(spectrum)
            
            # Determine dominant frequency
            if len(peak_indices) == 0:
                dominant_freq = 0
                info_text = "No significant frequency detected"
            else:
                dominant_freq = freqs[peak_indices[np.argmax(norm_spectrum[peak_indices])]]
                info_text = f"Dominant Frequency: {dominant_freq:.1f} Hz"
            
            # Update plot
            self.spectrum_line.set_data(freqs, norm_spectrum)
            self.peak_markers.set_data(freqs[peak_indices], norm_spectrum[peak_indices])
            self.text_box.set_text(info_text)
            
            return self.spectrum_line, self.peak_markers, self.text_box
            
        except Exception as e:
            print(f"Error in update: {e}")
            return self.spectrum_line, self.peak_markers, self.text_box
    
    def setup_plot(self):
        """Set up the visualization plot."""
        plt.style.use('dark_background')
        self.fig, self.ax1 = plt.subplots(figsize=(12, 6))
        
        # Spectrum plot
        self.spectrum_line, = self.ax1.plot([], [], '-', lw=2, color='cyan')
        self.peak_markers, = self.ax1.plot([], [], 'ro', ms=5)
        self.ax1.set_xlim(self.min_freq, self.max_freq)
        self.ax1.set_ylim(0, 1)
        self.ax1.set_title('Audio Frequency Spectrum', fontsize=15)
        self.ax1.set_xlabel('Frequency (Hz)')
        self.ax1.set_ylabel('Magnitude')
        self.ax1.grid(True, alpha=0.3)
        
        self.text_box = self.ax1.text(0.7, 0.9, '', transform=self.ax1.transAxes, 
                                      fontsize=14, bbox=dict(facecolor='black', alpha=0.7))
        
        self.fig.tight_layout()
    
    def run(self):
        """Run the audio analyzer with live visualization."""
        print("Starting audio frequency detection...")
        print("Press Ctrl+C to stop")
        
        try:
            self.ani = FuncAnimation(self.fig, self.update_plot, interval=self.refresh_interval, blit=False)
            plt.show()
            
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            self.close()
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()
        plt.close('all')


if __name__ == "__main__":
    detector = AudioFrequencyDetector(sample_rate=44100, chunk_size=4096, display_range=(20, 2000))
    detector.run()
