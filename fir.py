import numpy as np
from scipy.signal import firwin, lfilter
import os

def generate_and_save_fir_data(output_dir, num_taps, n_samples):
    """
    Generates data for a given number of taps and a fixed number of samples,
    then saves to text files.
    """
    print(f"Generating data for {num_taps} taps with {n_samples} samples...")
    
    # --- Generate ---
    np.random.seed(0) # fixed seed for reproducibility
    t = np.arange(n_samples)
    signal = np.sin(2 * np.pi * 0.05 * t) + 0.1 * np.random.randn(n_samples)
    
    cutoff = 0.1
    if num_taps <= 0:
        print(f"Skipping invalid num_taps: {num_taps}")
        return
        
    fir_coeffs = firwin(num_taps, cutoff)
    filtered_signal = lfilter(fir_coeffs, 1.0, signal)
    
    # --- Save ---
    precision_format = "%.18e"
    
    signal_path = os.path.join(output_dir, f"data_taps_{num_taps}.txt")
    coeffs_path = os.path.join(output_dir, f"taps_{num_taps}.txt")
    expected_path = os.path.join(output_dir, f"expected_taps_{num_taps}.txt")
    
    np.savetxt(signal_path, signal, fmt=precision_format)
    np.savetxt(coeffs_path, fir_coeffs, fmt=precision_format)
    np.savetxt(expected_path, filtered_signal, fmt=precision_format)
    
    print(f" -> Saved files for taps={num_taps}")

def main():
    """Main function to generate all required test data sets."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(script_dir, "data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Fixed number of samples for all tests
    NUM_SAMPLES = 100000
    
    tap_counts = [4, 8, 15, 16, 32, 63, 128, 255, 512, 1337, 2047, 4095, 8191, 16383, 32000, 32768]

    for taps in tap_counts:
        generate_and_save_fir_data(OUTPUT_DIR, taps, n_samples=NUM_SAMPLES)
        
    print("\nData generation complete.")

if __name__ == "__main__":
    main()