#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_with_absmax.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass/arch/mma_sm89.h>  // SM80 for FP16 Tensor Cores
#include <cuda_fp16.h>
#include <iostream>

// Define the epilogue operation
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t,  // Output type
    8,                // Elements per access (vectorization)
    cutlass::half_t,  // Accumulator type
    cutlass::half_t   // Compute type
>;

// Define the threadblock swizzle
using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle<5>;

// Define the GEMM kernel
using GemmKernel = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,  // Matrix A: FP16, row-major
    cutlass::half_t, cutlass::layout::ColumnMajor,  // Matrix B: FP16, row-major
    cutlass::half_t, cutlass::layout::RowMajor,  // Matrix C: FP16, row-major
    cutlass::half_t,                             // Accumulator: FP16
    cutlass::arch::OpClassTensorOp,              // Use Tensor Cores
    cutlass::arch::Sm80,                         // SM80 (compatible with SM89)
    cutlass::gemm::GemmShape<128, 128, 32>,     // Threadblock tile size
    cutlass::gemm::GemmShape<64, 64, 32>,       // Warp tile size
    cutlass::gemm::GemmShape<16, 8, 16>,        // Instruction shape for FP16
    EpilogueOp,                                  // Epilogue operation
    ThreadblockSwizzle,                          // Threadblock swizzle
    2,                                           // Number of pipeline stages
    8,                                           // Alignment for A (in elements)
    8,                                           // Alignment for B (in elements)
    false                                        // Split-K serial disabled
>;

int main() {
    // Define problem size (M x N x K)
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    // Allocate device memory
    cutlass::half_t *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(cutlass::half_t));
    cudaMalloc(&d_B, K * N * sizeof(cutlass::half_t));
    cudaMalloc(&d_C, M * N * sizeof(cutlass::half_t));

    // Initialize matrices (fill with ones for testing)
    cudaMemset(d_A, 0x3c00, M * K * sizeof(cutlass::half_t)); // 1.0 in FP16
    cudaMemset(d_B, 0x3c00, K * N * sizeof(cutlass::half_t)); // 1.0 in FP16
    cudaMemset(d_C, 0, M * N * sizeof(cutlass::half_t));      // 0.0 in FP16

    // Define GEMM parameters
    cutlass::half_t alpha = cutlass::half_t(1.0f);  // Scaling factor for A*B
    cutlass::half_t beta = cutlass::half_t(0.0f);   // Scaling factor for C
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    // Create GEMM operation instance
    GemmKernel gemm_op;

    // Set up arguments for the GEMM kernel
    typename GemmKernel::Arguments args(
        problem_size,
	{d_A, M},  // A, leading dimension (lda)
	{d_B, K},  // B, leading dimension (ldb)
	{d_C, M},  // C, leading dimension (ldc)
	{d_C, M},  // D (output), leading dimension (ldd)
        {alpha, beta} // Epilogue scalar parameters
    );

    // Initialize the GEMM operation
    cutlass::Status status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM initialization failed" << std::endl;
        return -1;
    }

    // Launch the GEMM kernel
    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM kernel launch failed" << std::endl;
        return -1;
    }

    // Measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;

    // Warm-up run
    gemm_op.run();

    // Timed run (average over 10 iterations)
    cudaEventRecord(start);
    for (int i = 0; i < 10; ++i) {
        gemm_op.run();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    float gflop = 2.0f * float(problem_size.product()) / float(1.0e9) / ((milliseconds / 10) / 1000.0f);
    std::cout << "CUTLASS GEMM time: " << milliseconds / 10 << " ms" << " GFLOP " << gflop << " gflops" << std::endl;

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
