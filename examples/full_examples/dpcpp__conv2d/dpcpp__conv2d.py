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

WORKLOAD = 'conv2d'
COMPILER = 'dpcpp'

parser = argparse.ArgumentParser(description='2D convolution, ported from SyTuner')
bench.add_common_args(parser, COMPILER, default_output_dir=str(Path(__file__).resolve().parent / 'dir_dumps'))
args = parser.parse_args()

# 2D convolution with shared-memory input staging, ported from SyTuner's
# benchmarks/conv2d/sycl-no-spec-const (same kernel, same 4 tunable parameters
# and restrictions as Conv2dTunerBase). Originally from KernelTuner's OpenCL
# conv2d example. Same 4096x4096/17x17 problem size as the original -- this
# kernel is cheap enough per-run that a full pyATF sweep (full AoT recompile
# every evaluation) still finishes in a reasonable demo time.
CONV2D_SOURCE_TEMPLATE = '''
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>

inline constexpr int kImageHeight  = 4096;
inline constexpr int kImageWidth   = 4096;
inline constexpr int kFilterHeight = 17;
inline constexpr int kFilterWidth  = 17;
inline constexpr int kBorderHeight = (kFilterHeight / 2) * 2;
inline constexpr int kBorderWidth  = (kFilterWidth  / 2) * 2;
inline constexpr int kInputHeight  = kImageHeight + kBorderHeight;
inline constexpr int kInputWidth   = kImageWidth  + kBorderWidth;

#ifndef BLOCK_SIZE_X
#define BLOCK_SIZE_X 16
#endif
#ifndef BLOCK_SIZE_Y
#define BLOCK_SIZE_Y 16
#endif
#ifndef TILE_SIZE_X
#define TILE_SIZE_X 1
#endif
#ifndef TILE_SIZE_Y
#define TILE_SIZE_Y 1
#endif
#ifndef WARMUP_RUNS
#define WARMUP_RUNS 2
#endif
#ifndef MEASUREMENT_RUNS
#define MEASUREMENT_RUNS 5
#endif

inline constexpr int kBlockSizeX = BLOCK_SIZE_X;
inline constexpr int kBlockSizeY = BLOCK_SIZE_Y;
inline constexpr int kTileSizeX  = TILE_SIZE_X;
inline constexpr int kTileSizeY  = TILE_SIZE_Y;
inline constexpr int kShWidth    = kBlockSizeX * kTileSizeX + kBorderWidth;
inline constexpr int kShHeight   = kBlockSizeY * kTileSizeY + kBorderHeight;

class Conv2DKernel {
public:
    Conv2DKernel(float* output, const float* input, const float* filter,
                sycl::local_accessor<float, 1> shInput)
        : output_(output), input_(input), filter_(filter), shInput_(shInput) {}

    void operator()(sycl::nd_item<2> item) const {
        const int ty = (int)item.get_local_id(0);
        const int tx = (int)item.get_local_id(1);
        const int by = (int)item.get_group(0) * kBlockSizeY * kTileSizeY;
        const int bx = (int)item.get_group(1) * kBlockSizeX * kTileSizeX;

        float* sh = shInput_.template get_multi_ptr<sycl::access::decorated::no>().get();

        #pragma unroll
        for (int i = ty; i < kShHeight; i += kBlockSizeY) {
            #pragma unroll
            for (int j = tx; j < kShWidth; j += kBlockSizeX) {
                int y = by + i, x = bx + j;
                float val = 0.0f;
                if (y < kInputHeight && x < kInputWidth) val = input_[(size_t)y * kInputWidth + x];
                sh[i * kShWidth + j] = val;
            }
        }
        sycl::group_barrier(item.get_group());

        float sum[kTileSizeY][kTileSizeX];
        #pragma unroll
        for (int yi = 0; yi < kTileSizeY; ++yi)
            #pragma unroll
            for (int xi = 0; xi < kTileSizeX; ++xi) sum[yi][xi] = 0.0f;

        #pragma unroll
        for (int fi = 0; fi < kFilterHeight; ++fi) {
            #pragma unroll
            for (int fj = 0; fj < kFilterWidth; ++fj) {
                float w = filter_[fi * kFilterWidth + fj];
                #pragma unroll
                for (int yi = 0; yi < kTileSizeY; ++yi) {
                    #pragma unroll
                    for (int xi = 0; xi < kTileSizeX; ++xi) {
                        int shRow = ty + yi * kBlockSizeY + fi;
                        int shCol = tx + xi * kBlockSizeX + fj;
                        sum[yi][xi] += sh[shRow * kShWidth + shCol] * w;
                    }
                }
            }
        }

        #pragma unroll
        for (int yi = 0; yi < kTileSizeY; ++yi) {
            #pragma unroll
            for (int xi = 0; xi < kTileSizeX; ++xi) {
                int y = by + ty + yi * kBlockSizeY;
                int x = bx + tx + xi * kBlockSizeX;
                if (y < kImageHeight && x < kImageWidth) output_[(size_t)y * kImageWidth + x] = sum[yi][xi];
            }
        }
    }

private:
    float* output_;
    const float* input_;
    const float* filter_;
    sycl::local_accessor<float, 1> shInput_;
};

int main() {
    sycl::queue q{ sycl::gpu_selector_v, sycl::property::queue::enable_profiling{} };

    std::vector<float> hostInput((size_t)kInputHeight * kInputWidth);
    std::vector<float> hostFilter((size_t)kFilterHeight * kFilterWidth);
    std::vector<float> hostOutput((size_t)kImageHeight * kImageWidth, 0.0f);
    for (size_t i = 0; i < hostInput.size(); ++i) hostInput[i] = (float)((i % 13) * 0.1f - 0.5f);
    for (size_t i = 0; i < hostFilter.size(); ++i) hostFilter[i] = (float)(((i * 7) % 17) * 0.05f - 0.4f);

    float* devInput  = sycl::malloc_device<float>(hostInput.size(), q);
    float* devFilter = sycl::malloc_device<float>(hostFilter.size(), q);
    float* devOutput = sycl::malloc_device<float>(hostOutput.size(), q);
    q.memcpy(devInput, hostInput.data(), hostInput.size() * sizeof(float));
    q.memcpy(devFilter, hostFilter.data(), hostFilter.size() * sizeof(float));
    q.wait();

    size_t globalY = ((kImageHeight + kBlockSizeY * kTileSizeY - 1) / (kBlockSizeY * kTileSizeY)) * kBlockSizeY;
    size_t globalX = ((kImageWidth  + kBlockSizeX * kTileSizeX - 1) / (kBlockSizeX * kTileSizeX)) * kBlockSizeX;
    sycl::nd_range<2> ndRange(sycl::range<2>(globalY, globalX), sycl::range<2>(kBlockSizeY, kBlockSizeX));
    constexpr size_t shMemSize = (size_t)kShWidth * kShHeight;

    auto launch = [&]() {
        return q.submit([&](sycl::handler &h) {
            sycl::local_accessor<float, 1> shInput(shMemSize, h);
            h.parallel_for(ndRange, Conv2DKernel(devOutput, devInput, devFilter, shInput));
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

    q.memcpy(hostOutput.data(), devOutput, hostOutput.size() * sizeof(float)).wait();
    sycl::free(devInput, q);
    sycl::free(devFilter, q);
    sycl::free(devOutput, q);

    // Sampled correctness check: a full kImageHeight*kImageWidth*filter CPU
    // reference would dominate per-evaluation time, since this whole program
    // reruns from scratch every pyATF evaluation. Check a fixed set of
    // pseudo-random output positions against a direct convolution sum.
    unsigned seed = 12345u;
    for (int s = 0; s < 256; ++s) {
        seed = seed * 1103515245u + 12345u;
        int y = (int)(seed % kImageHeight);
        seed = seed * 1103515245u + 12345u;
        int x = (int)(seed % kImageWidth);
        double sum = 0.0;
        for (int fi = 0; fi < kFilterHeight; ++fi)
            for (int fj = 0; fj < kFilterWidth; ++fj)
                sum += (double)hostInput[(size_t)(y + fi) * kInputWidth + (x + fj)] * (double)hostFilter[fi * kFilterWidth + fj];
        double got = hostOutput[(size_t)y * kImageWidth + x];
        if (std::fabs(sum - got) > 1e-2 * std::max(1.0, std::fabs(sum))) {
            fprintf(stderr, "result check failed at (%d,%d): got %f, expected %f\\n", y, x, got, sum);
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

PROBLEM_SIZE = {'problem_image_height': 4096, 'problem_image_width': 4096,
                'problem_filter_height': 17, 'problem_filter_width': 17}

# device limits for this machine (RTX 2000 Ada), see dpcpp__matmul.py
MAX_WORKGROUP_SIZE = 1024
MAX_SHARED_MEMORY  = 49152  # bytes
IMAGE_HEIGHT = IMAGE_WIDTH = 4096

# Step 1: Generate the Search Space (same 4 tunables + restrictions as
# SyTuner's Conv2dTunerBase.tune_params()/restrictions())
BLOCK_SIZE_X = TP('BLOCK_SIZE_X', Set(*[16 * i for i in range(1, 9)]))   # 16,32,...,128
BLOCK_SIZE_Y = TP('BLOCK_SIZE_Y', Set(*[2 ** i for i in range(6)]))     # 1,2,4,8,16,32
TILE_SIZE_X  = TP('TILE_SIZE_X',  Set(1, 2, 4),
                  lambda BLOCK_SIZE_X, TILE_SIZE_X: IMAGE_WIDTH % (BLOCK_SIZE_X * TILE_SIZE_X) == 0)
TILE_SIZE_Y  = TP('TILE_SIZE_Y',  Set(1, 2, 4),
                  lambda BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y:
                      IMAGE_HEIGHT % (BLOCK_SIZE_Y * TILE_SIZE_Y) == 0
                      and BLOCK_SIZE_X * BLOCK_SIZE_Y <= MAX_WORKGROUP_SIZE
                      and (BLOCK_SIZE_Y * TILE_SIZE_Y + 16) * (BLOCK_SIZE_X * TILE_SIZE_X + 16) * 4 <= MAX_SHARED_MEMORY)

# Step 2 & 3: repeat the whole autotuning procedure from scratch --runs times
# (matching SyTuner's KernelTuner runner default, AUTOTUNE_RUNS=5) -- each run
# gets its own fresh search technique (AUCBandit carries state across a run),
# its own isolated log/result/cost/device files, and its own freshly compiled
# source (the cost/device file paths are baked into the source as literals).
session = bench.new_session_id()
for run_idx in range(args.runs):
    paths = bench.run_paths(args.output_dir, COMPILER, WORKLOAD, run_idx, session)
    conv2d_source = dpcpp.source(
        CONV2D_SOURCE_TEMPLATE
        .replace('__COST_FILE_PATH__', str(paths['cost_file']))
        .replace('__DEVICE_FILE_PATH__', str(paths['device_file']))
    )

    cf_conv2d = dpcpp.CostFunction( conv2d_source ).target( args.target )                              \
                                                   .flags( [f'-DWARMUP_RUNS={args.warmup_runs}',
                                                            f'-DMEASUREMENT_RUNS={args.measurement_runs}'] ) \
                                                   .cost_file( str(paths['cost_file']) )

    config, min_cost, tuning_data = Tuner().tuning_parameters( BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y )  \
                                           .search_technique( AUCBandit() )                                            \
                                           .log_file( str(paths['log_file']) )                                         \
                                           .tune( cf_conv2d, Evaluations(args.max_fevals) )

    result = bench.base_result_fields(WORKLOAD, COMPILER, args, run_idx, session, Path(__file__).name,
                                      bench.read_device_name(paths['device_file']), paths['log_file'],
                                      config, min_cost, tuning_data)
    result.update(PROBLEM_SIZE)
    bench.write_result_json(paths['result_file'], **result)

    print(f'run {run_idx}: min_cost={min_cost}, config={config}')
    print(f'  result -> {paths["result_file"]}')
