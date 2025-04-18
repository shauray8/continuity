#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/threadblock/default_mma.h>
#include <cutlass/arch/mma_sm89.h>
#include <cuda_fp16.h>

// Define the epilogue operation
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t,  // Output type
    8,                // Elements per access (vectorization)
    cutlass::half_t,  // Accumulator type
    cutlass::half_t   // Compute type
>;

// Define the threadblock swizzle
using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

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

// Extern "C" function to call from Python
extern "C" int cutlass_gemm_fp16(
    cutlass::half_t *d_A,  // Input matrix A
    cutlass::half_t* d_B,  // Input matrix B
    cutlass::half_t* d_C,  // Output matrix C
    int M, int N, int K,   // Dimensions
    float alpha, float beta // Scalars
) {
    // Define problem size
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    // Create GEMM operation instance
    GemmKernel gemm_op;

    // Set up arguments
    typename GemmKernel::Arguments args(
        problem_size,
	{d_A, M},  // A, lda
	{d_B, K},  // B, ldb
	{d_C, M},  // C, ldc
	{d_C, M},  // D (output), ldd
	{cutlass::half_t(alpha), cutlass::half_t(beta)} // Epilogue scalar parameters
    );

    // Initialize the GEMM
    cutlass::Status status = gemm_op.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        return -1; // Initialization failed
    }

    // Run the GEMM
    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) {
        return -2; // Kernel launch failed
    }

    // Synchronize to ensure completion
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return -3; // Synchronization failed
    }

    return 0; // Success
}

