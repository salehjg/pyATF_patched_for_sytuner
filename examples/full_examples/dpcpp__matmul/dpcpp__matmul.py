import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / 'src'))
sys.path.insert(0, str(_REPO_ROOT / 'examples' / 'full_examples'))
import _bench_common as bench

from pyatf import TP, Set, Tuner
from pyatf.cost_functions import dpcpp
from pyatf.search_techniques import AUCBandit
from pyatf.abort_conditions import Evaluations

WORKLOAD = 'matmul'
COMPILER = 'dpcpp'

parser = argparse.ArgumentParser(description='Tile/register-blocked matmul, ported from SyTuner')
bench.add_common_args(parser, COMPILER, default_output_dir=str(Path(__file__).resolve().parent / 'dir_dumps'))
args = parser.parse_args()

# Tiled, shared-memory, register-blocked matmul, ported from SyTuner's
# benchmarks/matmul/sycl-no-spec-const (same kernel, same 6 tunable parameters
# and restrictions as SyTuner's MatmulTunerBase -- this is its "VarDef-only"
# variant, which is exactly what dpcpp.py/acpp.py's -D-macro-per-TP model is).
#
# SyTuner's two "block_size_x"/"block_size_y" tune_params aren't included here:
# they're not independent tunables, just KernelTuner grid-computation
# bookkeeping equal to TILE_N//REG_TILE_N and TILE_M//REG_TILE_M, which this
# program already derives directly from TILE_*/REG_TILE_* like the original
# kernel does.
#
# WARMUP_RUNS/MEASUREMENT_RUNS and the cost/device file paths are also -D
# macros/placeholders, set per run below -- same mechanism as the tuning
# parameters themselves, just constant across one run instead of varying
# per evaluation.
MATMUL_SOURCE_TEMPLATE = '''
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>

#ifndef TILE_M
#define TILE_M 64
#endif
#ifndef TILE_N
#define TILE_N 64
#endif
#ifndef TILE_K
#define TILE_K 16
#endif
#ifndef REG_TILE_M
#define REG_TILE_M 4
#endif
#ifndef REG_TILE_N
#define REG_TILE_N 4
#endif
#ifndef UNROLL_K
#define UNROLL_K 1
#endif
#ifndef WARMUP_RUNS
#define WARMUP_RUNS 2
#endif
#ifndef MEASUREMENT_RUNS
#define MEASUREMENT_RUNS 5
#endif

static constexpr int kTileM    = TILE_M;
static constexpr int kTileN    = TILE_N;
static constexpr int kTileK    = TILE_K;
static constexpr int kRegTileM = REG_TILE_M;
static constexpr int kRegTileN = REG_TILE_N;
static constexpr int kBlockX   = kTileN / kRegTileN;
static constexpr int kBlockY   = kTileM / kRegTileM;

// Same problem size as SyTuner's own standalone driver default.
static constexpr int M = 8192;
static constexpr int N = 8192;
static constexpr int K = 8192;

class TiledMatMulKernel {
public:
    TiledMatMulKernel(const float* a, const float* b, float* c,
                      sycl::local_accessor<float, 1> localA,
                      sycl::local_accessor<float, 1> localB)
        : a_(a), b_(b), c_(c), localA_(localA), localB_(localB) {}

    void operator()(sycl::nd_item<2> item) const {
        constexpr int kThreadsPerBlk = kBlockX * kBlockY;
        const int blockRow = (int)item.get_group(0) * kTileM;
        const int blockCol = (int)item.get_group(1) * kTileN;
        const int tx = (int)item.get_local_id(1);
        const int ty = (int)item.get_local_id(0);
        const int rowBase   = ty * kRegTileM;
        const int colBase   = tx * kRegTileN;
        const int linearTid = ty * kBlockX + tx;

        float acc[kRegTileM][kRegTileN];
        #pragma unroll
        for (int i = 0; i < kRegTileM; ++i)
            #pragma unroll
            for (int j = 0; j < kRegTileN; ++j) acc[i][j] = 0.0f;

        float* sA = localA_.template get_multi_ptr<sycl::access::decorated::no>().get();
        float* sB = localB_.template get_multi_ptr<sycl::access::decorated::no>().get();

        for (int k0 = 0; k0 < K; k0 += kTileK) {
            constexpr int elementsA = kTileM * kTileK;
            for (int idx = linearTid; idx < elementsA; idx += kThreadsPerBlk) {
                int r = idx / kTileK, c = idx % kTileK;
                int gRow = blockRow + r, gCol = k0 + c;
                sA[idx] = (gRow < M && gCol < K) ? a_[(size_t)gRow * K + gCol] : 0.0f;
            }
            constexpr int elementsB = kTileK * kTileN;
            for (int idx = linearTid; idx < elementsB; idx += kThreadsPerBlk) {
                int r = idx / kTileN, c = idx % kTileN;
                int gRow = k0 + r, gCol = blockCol + c;
                sB[idx] = (gRow < K && gCol < N) ? b_[(size_t)gRow * N + gCol] : 0.0f;
            }
            sycl::group_barrier(item.get_group());

            #pragma unroll UNROLL_K
            for (int kk = 0; kk < kTileK; ++kk) {
                float aReg[kRegTileM];
                for (int i = 0; i < kRegTileM; ++i) aReg[i] = sA[(rowBase + i) * kTileK + kk];
                float bReg[kRegTileN];
                for (int j = 0; j < kRegTileN; ++j) bReg[j] = sB[kk * kTileN + colBase + j];
                for (int i = 0; i < kRegTileM; ++i)
                    for (int j = 0; j < kRegTileN; ++j)
                        acc[i][j] += aReg[i] * bReg[j];
            }
            sycl::group_barrier(item.get_group());
        }

        #pragma unroll
        for (int i = 0; i < kRegTileM; ++i) {
            int gRow = blockRow + rowBase + i;
            if (gRow >= M) continue;
            #pragma unroll
            for (int j = 0; j < kRegTileN; ++j) {
                int gCol = blockCol + colBase + j;
                if (gCol >= N) continue;
                c_[(size_t)gRow * N + gCol] = acc[i][j];
            }
        }
    }

private:
    const float* a_;
    const float* b_;
    float* c_;
    sycl::local_accessor<float, 1> localA_;
    sycl::local_accessor<float, 1> localB_;
};

int main() {
    sycl::queue q{ sycl::gpu_selector_v, sycl::property::queue::enable_profiling{} };

    std::vector<float> hostA((size_t)M * K), hostB((size_t)K * N), hostC((size_t)M * N, 0.0f);
    for (size_t i = 0; i < hostA.size(); ++i) hostA[i] = (float)((i % 13) * 0.1f - 0.5f);
    for (size_t i = 0; i < hostB.size(); ++i) hostB[i] = (float)(((i * 7) % 17) * 0.05f - 0.4f);

    float* devA = sycl::malloc_device<float>(hostA.size(), q);
    float* devB = sycl::malloc_device<float>(hostB.size(), q);
    float* devC = sycl::malloc_device<float>(hostC.size(), q);
    q.memcpy(devA, hostA.data(), hostA.size() * sizeof(float));
    q.memcpy(devB, hostB.data(), hostB.size() * sizeof(float));
    q.wait();

    size_t numTilesM = (M + kTileM - 1) / kTileM;
    size_t numTilesN = (N + kTileN - 1) / kTileN;
    sycl::nd_range<2> ndRange(sycl::range<2>(numTilesM * kBlockY, numTilesN * kBlockX),
                              sycl::range<2>(kBlockY, kBlockX));
    constexpr size_t shmemA = (size_t)kTileM * kTileK;
    constexpr size_t shmemB = (size_t)kTileK * kTileN;

    auto launch = [&]() {
        return q.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> localA(shmemA, h);
            sycl::local_accessor<float, 1> localB(shmemB, h);
            h.parallel_for(ndRange, TiledMatMulKernel(devA, devB, devC, localA, localB));
        });
    };

    for (int w = 0; w < WARMUP_RUNS; ++w) launch().wait();

    std::vector<double> times_ns;
    for (int run = 0; run < MEASUREMENT_RUNS; ++run) {
        auto ev = launch();
        ev.wait();
        times_ns.push_back((double)(ev.get_profiling_info<sycl::info::event_profiling::command_end>()
                                   - ev.get_profiling_info<sycl::info::event_profiling::command_start>()));
    }
    std::sort(times_ns.begin(), times_ns.end());
    double median_ns = times_ns[times_ns.size() / 2];

    q.memcpy(hostC.data(), devC, hostC.size() * sizeof(float)).wait();
    sycl::free(devA, q);
    sycl::free(devB, q);
    sycl::free(devC, q);

    // Sampled correctness check: a full M*N*K CPU reference would dominate
    // per-evaluation time at any non-tiny problem size, since this whole
    // program (including the check) reruns from scratch every pyATF
    // evaluation. Check a fixed set of pseudo-random output positions against
    // a direct dot-product instead.
    unsigned seed = 12345u;
    for (int s = 0; s < 256; ++s) {
        seed = seed * 1103515245u + 12345u;
        int i = (int)(seed % M);
        seed = seed * 1103515245u + 12345u;
        int j = (int)(seed % N);
        double sum = 0.0;
        for (int k = 0; k < K; ++k)
            sum += (double)hostA[(size_t)i * K + k] * (double)hostB[(size_t)k * N + j];
        double got = hostC[(size_t)i * N + j];
        if (std::fabs(sum - got) > 1e-2 * std::max(1.0, std::fabs(sum))) {
            fprintf(stderr, "result check failed at (%d,%d): got %f, expected %f\\n", i, j, got, sum);
            return 1;
        }
    }

    std::ofstream cost_out("__COST_FILE_PATH__");
    cost_out << median_ns;
    std::ofstream device_out("__DEVICE_FILE_PATH__");
    device_out << q.get_device().get_info<sycl::info::device::name>();
    return 0;
}
'''

PROBLEM_SIZE = {'problem_M': 8192, 'problem_N': 8192, 'problem_K': 8192}

# device limits for this machine (RTX 2000 Ada) -- SyTuner queries these live
# off the SYCL device (sytuner.query_device_limits()); pyATF has no built-in
# equivalent for the SYCL backends yet, so hardcoding what was actually
# queried here. Swap these for your own device's numbers.
MAX_WORKGROUP_SIZE = 1024
MAX_SHARED_MEMORY  = 49152  # bytes

# Step 1: Generate the Search Space (same 6 tunables + restrictions as
# SyTuner's MatmulTunerBase.tune_params()/restrictions())
TILE_M     = TP('TILE_M',     Set(16, 32, 64, 128, 256))
TILE_N     = TP('TILE_N',     Set(16, 32, 64, 128, 256))
TILE_K     = TP('TILE_K',     Set(4, 8, 16, 32, 64))
REG_TILE_M = TP('REG_TILE_M', Set(1, 2, 4, 8, 16),
                lambda TILE_M, REG_TILE_M: TILE_M % REG_TILE_M == 0 and 4 <= TILE_M // REG_TILE_M <= 32)
REG_TILE_N = TP('REG_TILE_N', Set(1, 2, 4, 8, 16),
                lambda TILE_N, REG_TILE_N: TILE_N % REG_TILE_N == 0 and 4 <= TILE_N // REG_TILE_N <= 32)
UNROLL_K   = TP('UNROLL_K',   Set(1, 2, 4),
                lambda TILE_M, TILE_N, TILE_K, REG_TILE_M, REG_TILE_N, UNROLL_K:
                    UNROLL_K <= TILE_K
                    and (TILE_M // REG_TILE_M) * (TILE_N // REG_TILE_N) <= MAX_WORKGROUP_SIZE
                    and (TILE_M * TILE_K + TILE_K * TILE_N) * 4 <= MAX_SHARED_MEMORY)

# Step 2 & 3: repeat the whole autotuning procedure from scratch --runs times
# (matching SyTuner's KernelTuner runner default, AUTOTUNE_RUNS=5) -- each run
# gets its own fresh search technique (AUCBandit carries state across a run),
# its own isolated log/result/cost/device files, and its own freshly compiled
# source (the cost/device file paths are baked into the source as literals).
session = bench.new_session_id()
for run_idx in range(args.runs):
    paths = bench.run_paths(args.output_dir, COMPILER, WORKLOAD, run_idx, session)
    matmul_source = dpcpp.source(
        MATMUL_SOURCE_TEMPLATE
        .replace('__COST_FILE_PATH__', str(paths['cost_file']))
        .replace('__DEVICE_FILE_PATH__', str(paths['device_file']))
    )

    cf_matmul = dpcpp.CostFunction( matmul_source ).target( args.target )                              \
                                                   .flags( [f'-DWARMUP_RUNS={args.warmup_runs}',
                                                            f'-DMEASUREMENT_RUNS={args.measurement_runs}'] ) \
                                                   .cost_file( str(paths['cost_file']) )

    config, min_cost, tuning_data = Tuner().tuning_parameters( TILE_M, TILE_N, TILE_K, REG_TILE_M, REG_TILE_N, UNROLL_K )  \
                                           .search_technique( AUCBandit() )                                                \
                                           .log_file( str(paths['log_file']) )                                            \
                                           .tune( cf_matmul, Evaluations(args.max_fevals) )

    result = bench.base_result_fields(WORKLOAD, COMPILER, args, run_idx, session, Path(__file__).name,
                                      bench.read_device_name(paths['device_file']), paths['log_file'],
                                      config, min_cost, tuning_data)
    result.update(PROBLEM_SIZE)
    bench.write_result_json(paths['result_file'], **result)

    print(f'run {run_idx}: min_cost={min_cost}, config={config}')
    print(f'  result -> {paths["result_file"]}')
