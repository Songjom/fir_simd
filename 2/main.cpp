#include <iostream>
#include <array>
#include <algorithm>
#include <iterator>
#include <vector>
#include <iomanip>
#include <numeric>
#include <cmath>
#include <fstream>
#include <chrono>
#include <string>

#include <immintrin.h>

std::vector<double> load_vector_from_file(
	const std::string& filepath
) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }
    std::vector<double> data;
    data.reserve(100000);
    double value;
    while (file >> value) {
        data.push_back(value);
    }
    return data;
}

bool verify_results(
    const std::vector<double>& expected, 
    const std::vector<double>& actual,
    const std::string& label,
    double tolerance = 1e-9
) {
    if (expected.size() != actual.size()) {
        std::cout << "Size mismatch for: " << label << "\n"
            << "Expected: " << expected.size() << "\n "
            << "Got: " << actual.size() << "\n" << std::endl;

        return false;
    }

    for (size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(expected[i] - actual[i]) > tolerance) {
            std::cout << "Verification FAILED for:" << label << "\n"
                << "At index: " << i << ".\n"
                << "Expected: " << expected[i] << "\n"
                << "Got: " << actual[i] << "\n"
                << "Difference: " << std::abs(actual[i] - expected[i])
                << "\n" << std::endl;

            return false;
        }
    }
    return true;
}

/*
 * Implements a naive Finite Impulse Response (FIR) filter using direct convolution.
 * This is a "naive" implementation because its time complexity is O(N * M),
 * where N is the number of samples and M is the number of taps.
 */
std::vector<double> fir_naive_vec(
    const std::vector<double>& x, 
    const std::vector<double>& h
) {
    size_t sample_n = x.size();
    size_t taps_n = h.size();

    std::vector<double> y(sample_n, 0.0);

    if (sample_n == 0) return {};
    if (taps_n == 0) return y;

    for (auto i = 0u; i < sample_n; ++i) {
        for (auto j = 0u; j < taps_n; ++j) {
            if (i >= j) {
                y[i] += h[j] * x[i - j];
            }
        }
    }

    return y;
}

/*
 * This is SIMD optimized version of Finite Impulse Response (FIR) filter using AVX2 instructions.
 *
 * Explanation of optimization:
 * SIMD allows to perform single operation on multiple data (hence the name).
 * 
 * One AVX register stores 4 double values.
 * This code attempts to explicitly make use of 15 registers out of 16, but reality is up to compiler.
 * 7 Accumulators are allocated, each capable of storing 4 double values.
 * In each inner iteration 7 input are laoded from vector per 1 tap value.
 * Tap value is broadcasted to all registers.
 * Then FMA (_mm256_fmadd) is used to multiply and add data as a * b + c in a single instruction.
 * This way this function computes 28 values per interation.
 *
 * Manual unrolling here created 7 explicit independent dependency chains in the inner loop.
 * It is also important to take into account FMA latency which this code hides by instruction level parallelism.
 */

std::vector<double> fir_avx2d_vec( //d stands for double
    const std::vector<double>& x, 
    const std::vector<double>& h
) {
	size_t sample_n = x.size();
	size_t taps_n = h.size();

    std::vector<double> y(sample_n, 0.0);

    if (sample_n == 0) return {};
    if (taps_n == 0) return y;
    
    alignas(32) std::vector<double> x_padded(taps_n - 1 + sample_n, 0.0);
    std::copy(x.begin(), x.end(), x_padded.begin() + taps_n - 1);

    const size_t VEC_SIZE = 4;      // 256 / 8 / sizeof(double) = 4
    size_t i = 0;

    for (; i + VEC_SIZE * 7 <= sample_n; i += VEC_SIZE * 7) {

        __m256d y_vec0 = _mm256_setzero_pd(); // For y[i...i+3]
        __m256d y_vec1 = _mm256_setzero_pd(); // For y[i+4...i+7]
        __m256d y_vec2 = _mm256_setzero_pd(); // For y[i+8...i+11]
        __m256d y_vec3 = _mm256_setzero_pd(); // For y[i+12...i+15]
        __m256d y_vec4 = _mm256_setzero_pd(); // For y[i+16...i+19]
        __m256d y_vec5 = _mm256_setzero_pd(); // For y[i+20...i+23]
        __m256d y_vec6 = _mm256_setzero_pd(); // For y[i+24...i+27]

        for (size_t j = 0; j < taps_n; ++j) {
            // Broadcast coefficent
            const __m256d h_vec = _mm256_set1_pd(h[j]);

            // The pointer to the start of the padded data for this iteration
            const double* x_ptr = &x_padded[i + taps_n - 1 - j];

            /*
            * Since Haswell (2013) alignment penalty is reduced drastically 
            * given that data is loaded in L1 cache.
            * 
			* This code has 1 cycle penalty for unaligned loads.
            */

            const __m256d x_vec0 = _mm256_loadu_pd(x_ptr + VEC_SIZE * 0); // Loads x for y[i..i+3]
			const __m256d x_vec1 = _mm256_loadu_pd(x_ptr + VEC_SIZE * 1); // y[i+4..i+7]
			const __m256d x_vec2 = _mm256_loadu_pd(x_ptr + VEC_SIZE * 2); // y[i+8..i+11]
            const __m256d x_vec3 = _mm256_loadu_pd(x_ptr + VEC_SIZE * 3); // ...
            const __m256d x_vec4 = _mm256_loadu_pd(x_ptr + VEC_SIZE * 4);
            const __m256d x_vec5 = _mm256_loadu_pd(x_ptr + VEC_SIZE * 5);
			const __m256d x_vec6 = _mm256_loadu_pd(x_ptr + VEC_SIZE * 6); // y[i+24..i+27]

            y_vec0 = _mm256_fmadd_pd(h_vec, x_vec0, y_vec0);
            y_vec1 = _mm256_fmadd_pd(h_vec, x_vec1, y_vec1);
            y_vec2 = _mm256_fmadd_pd(h_vec, x_vec2, y_vec2);
            y_vec3 = _mm256_fmadd_pd(h_vec, x_vec3, y_vec3);
            y_vec4 = _mm256_fmadd_pd(h_vec, x_vec4, y_vec4);
            y_vec5 = _mm256_fmadd_pd(h_vec, x_vec5, y_vec5);
            y_vec6 = _mm256_fmadd_pd(h_vec, x_vec6, y_vec6);
        }

        _mm256_storeu_pd(&y[i + VEC_SIZE * 0], y_vec0);
        _mm256_storeu_pd(&y[i + VEC_SIZE * 1], y_vec1);
        _mm256_storeu_pd(&y[i + VEC_SIZE * 2], y_vec2);
        _mm256_storeu_pd(&y[i + VEC_SIZE * 3], y_vec3);
        _mm256_storeu_pd(&y[i + VEC_SIZE * 4], y_vec4);
        _mm256_storeu_pd(&y[i + VEC_SIZE * 5], y_vec5);
        _mm256_storeu_pd(&y[i + VEC_SIZE * 6], y_vec6);
    }

    // Compute rest with scalar method
    for (; i < sample_n; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < taps_n; ++j) {
            sum += h[j] * x_padded[i + taps_n - 1 - j];
        }
        y[i] = sum;
    }

    return y;
}

int main(void) {
    auto tap_counts = { 4, 8, 15, 16, 32, 63, 128, 255, 512, 1337, 2047, 4095, 8191, 16383, 32000, 32768 };

    std::cout << std::left << std::setw(15) << "Num Taps"
        << std::left << std::setw(15) << "Correct?"
        << std::left << std::setw(20) << "Naive Time (ms)"
        << std::left << std::setw(20) << "AVX Time (ms)"
        << std::left << std::setw(15) << "Speedup" << std::endl;
    std::cout << std::string(85, '-') << std::endl;

    for (int n_taps : tap_counts) {
        try {
            std::string taps_str = std::to_string(n_taps);
            auto signal = load_vector_from_file("./data/data_taps_" + taps_str + ".txt");
            auto coeffs = load_vector_from_file("./data/taps_" + taps_str + ".txt");
            auto expected = load_vector_from_file("./data/expected_taps_" + taps_str + ".txt");

            // --- Time Naive ---
            auto start_naive = std::chrono::high_resolution_clock::now();
            auto y_naive = fir_naive_vec(signal, coeffs);
            auto end_naive = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> naive_ms = end_naive - start_naive;

            // --- Time AVX ---
            auto start_avx = std::chrono::high_resolution_clock::now();
            auto y_avx = fir_avx2d_vec(signal, coeffs);
            auto end_avx = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> avx_ms = end_avx - start_avx;

            bool is_correct = verify_results(expected, y_avx, std::to_string(n_taps), 1e-6);

            std::cout << std::left << std::setw(15) << n_taps
                << std::left << std::setw(15) << (is_correct ? "OK" : "FAIL")
                << std::left << std::setw(20) << std::fixed << std::setprecision(4) << naive_ms.count()
                << std::left << std::setw(20) << avx_ms.count()
                << std::left << std::setw(15) << std::fixed << std::setprecision(2) << (naive_ms.count() / avx_ms.count()) << "x"
                << std::endl;

        }
        catch (const std::runtime_error& e) {
            std::cerr << "Error processing for " << n_taps << " taps: " << e.what() << std::endl;
        }
    }

    return 0;
}

