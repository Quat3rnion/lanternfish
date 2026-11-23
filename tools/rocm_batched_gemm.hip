// Standalone reproducer for the gfx1036 ROCclr fence-scope failure. See docs/locate-edit.md.
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <cstdio>

#define CHECK(op) do { const int s = (op); if (s) { std::fprintf(stderr, "%s: %d\n", #op, s); return 1; } } while (0)

static __global__ void setup(float * a, float * b, float * c, float ** ptrs) {
    const int i = threadIdx.y;
    if (threadIdx.x || i >= 4) {
        return;
    }
    ptrs[i] = a + (i / 2) * 40;
    ptrs[4 + i] = b + i * 25;
    ptrs[8 + i] = c + i * 40;
}

int main() {
    float * a, * b, * c, ** ptrs;
    CHECK(hipMalloc(&a, 80 * sizeof(float)));
    CHECK(hipMalloc(&b, 100 * sizeof(float)));
    CHECK(hipMalloc(&c, 160 * sizeof(float)));
    CHECK(hipMalloc(&ptrs, 12 * sizeof(*ptrs)));
    float host[160];
    for (float & x : host) {
        x = 2;
    }
    CHECK(hipMemcpy(a, host, 80 * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(b, host, 100 * sizeof(float), hipMemcpyHostToDevice));
    hipStream_t stream;
    hipblasHandle_t handle;
    CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    CHECK(hipblasCreate(&handle));
    CHECK(hipblasSetStream(handle, stream));
    setup<<<1, dim3(16, 16), 0, stream>>>(a, b, c, ptrs);
    CHECK(hipGetLastError());
    const float alpha = 1, beta = 0;
    CHECK(hipblasGemmBatchedEx(handle, HIPBLAS_OP_T, HIPBLAS_OP_N, 8, 5, 5, &alpha,
          (const void **) ptrs, HIPBLAS_R_32F, 5, (const void **) (ptrs + 4), HIPBLAS_R_32F, 5,
          &beta, (void **) (ptrs + 8), HIPBLAS_R_32F, 8, 4, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    CHECK(hipStreamSynchronize(stream));
    CHECK(hipMemcpy(host, c, sizeof(host), hipMemcpyDeviceToHost));
    for (float x : host) {
        if (x != 20) {
            std::fprintf(stderr, "expected 20, got %g\n", x);
            return 1;
        }
    }
    std::puts("160/160 outputs exactly 20");
    CHECK(hipblasDestroy(handle));
    CHECK(hipStreamDestroy(stream));
    CHECK(hipFree(ptrs));
    CHECK(hipFree(c));
    CHECK(hipFree(b));
    CHECK(hipFree(a));
}
