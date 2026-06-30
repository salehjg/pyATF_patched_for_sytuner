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

WORKLOAD = 'pnpoly'
COMPILER = 'dpcpp'

parser = argparse.ArgumentParser(description='Point-in-polygon (crossing-number), ported from SyTuner')
bench.add_common_args(parser, COMPILER, default_output_dir=str(Path(__file__).resolve().parent / 'dir_dumps'))
args = parser.parse_args()

# Point-in-polygon (crossing-number test), ported from SyTuner's
# benchmarks/pnpoly/sycl-no-spec-const (same kernel, same 5 tunable parameters
# as PnpolyTunerBase). Originally from KernelTuner's CUDA pnpoly example.
# Same NUM_POINTS (20,000,000) and VERTICES (600) as SyTuner's own defaults.
PNPOLY_SOURCE_TEMPLATE = '''
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <fstream>

inline constexpr int VERTICES   = 600;
inline constexpr int NUM_POINTS = 20000000;

#ifndef BLOCK_SIZE_X
#define BLOCK_SIZE_X 256
#endif
#ifndef TILE_SIZE
#define TILE_SIZE 1
#endif
#ifndef BETWEEN_METHOD
#define BETWEEN_METHOD 0
#endif
#ifndef USE_PRECOMPUTED_SLOPES
#define USE_PRECOMPUTED_SLOPES 0
#endif
#ifndef USE_METHOD
#define USE_METHOD 0
#endif
#ifndef WARMUP_RUNS
#define WARMUP_RUNS 2
#endif
#ifndef MEASUREMENT_RUNS
#define MEASUREMENT_RUNS 5
#endif

inline constexpr int kBlockSizeX = BLOCK_SIZE_X;
inline constexpr int kTileSize   = TILE_SIZE;

#if BETWEEN_METHOD == 0
#define BETWEEN(a, b, c) (((b) > (c)) ? ((b) > (a) && (a) >= (c)) : ((b) <= (a) && (a) < (c)))
#elif BETWEEN_METHOD == 1
#define BETWEEN(a, b, c) ((((b) <= (a)) ^ ((c) <= (a))) != 0)
#elif BETWEEN_METHOD == 2
#define BETWEEN(a, b, c) (((b) - (a)) * ((c) - (a)) < 0.0f)
#elif BETWEEN_METHOD == 3
#define BETWEEN(a, b, c) (sycl::min((b),(c)) <= (a) && (a) < sycl::max((b),(c)))
#endif

class PnPolyKernel {
public:
    PnPolyKernel(int* bitmap, const sycl::float2* points, const sycl::float2* vertices,
                const float* slopes, int n)
        : bitmap_(bitmap), points_(points), vertices_(vertices), slopes_(slopes), n_(n) {}

    void operator()(sycl::nd_item<1> item) const {
        const int globalRange = (int)item.get_global_range(0);
        int i = (int)item.get_global_id(0);

        #pragma unroll TILE_SIZE
        for (int t = 0; t < kTileSize; ++t, i += globalRange) {
            if (i >= n_) break;
            const sycl::float2 p = points_[i];
            int c = 0;
            int k = VERTICES - 1;
#if USE_METHOD == 0
            for (int vi = 0; vi < VERTICES; ++vi) {
                const sycl::float2 vk = vertices_[k];
                const sycl::float2 vj = vertices_[vi];
                if (BETWEEN(p.y(), vj.y(), vk.y())) {
#if USE_PRECOMPUTED_SLOPES == 1
                    float slope = slopes_[vi];
#else
                    float slope = (vk.x() - vj.x()) / (vk.y() - vj.y());
#endif
                    if (p.x() < slope * (p.y() - vj.y()) + vj.x()) c = !c;
                }
                k = vi;
            }
#else
            for (int vi = 0; vi < VERTICES; ++vi) {
                const sycl::float2 vk = vertices_[k];
                const sycl::float2 vj = vertices_[vi];
                if (BETWEEN(p.y(), vj.y(), vk.y())) {
#if USE_PRECOMPUTED_SLOPES == 1
                    float slope = slopes_[vi];
#else
                    float slope = (vk.x() - vj.x()) / (vk.y() - vj.y());
#endif
                    float xIntersect = slope * (p.y() - vj.y()) + vj.x();
                    if (p.x() < xIntersect) {
                        if (vj.y() < vk.y()) c++;
                        else                 c--;
                    }
                }
                k = vi;
            }
            c = (c != 0) ? 1 : 0;
#endif
            bitmap_[i] = c;
        }
    }

private:
    int* bitmap_;
    const sycl::float2* points_;
    const sycl::float2* vertices_;
    const float* slopes_;
    int n_;
};

int main() {
    sycl::queue q{ sycl::gpu_selector_v, sycl::property::queue::enable_profiling{} };

    std::vector<sycl::float2> hostPoints(NUM_POINTS);
    unsigned seed = 777u;
    for (int i = 0; i < NUM_POINTS; ++i) {
        seed = seed * 1103515245u + 12345u;
        float x = ((float)(seed % 40000) / 10000.0f) - 2.0f;  // [-2, 2)
        seed = seed * 1103515245u + 12345u;
        float y = ((float)(seed % 40000) / 10000.0f) - 2.0f;
        hostPoints[i] = sycl::float2(x, y);
    }

    // Polygon: unit-circle approximation, vertices in descending angle order
    std::vector<sycl::float2> hostVertices(VERTICES);
    std::vector<float> vx(VERTICES), vy(VERTICES);
    for (int i = 0; i < VERTICES; ++i) {
        float angle = 2.0f * 3.14159265358979f * (float)(VERTICES - 1 - i) / (float)VERTICES;
        vx[i] = std::cos(angle);
        vy[i] = std::sin(angle);
        hostVertices[i] = sycl::float2(vx[i], vy[i]);
    }
    std::vector<float> hostSlopes(VERTICES);
    hostSlopes[0] = (vx[VERTICES - 1] - vx[0]) / (vy[VERTICES - 1] - vy[0] + 1e-30f);
    for (int i = 1; i < VERTICES; ++i)
        hostSlopes[i] = (vx[i - 1] - vx[i]) / (vy[i - 1] - vy[i] + 1e-30f);

    std::vector<int> hostBitmap(NUM_POINTS, 0);

    sycl::float2* devPoints   = sycl::malloc_device<sycl::float2>(NUM_POINTS, q);
    sycl::float2* devVertices = sycl::malloc_device<sycl::float2>(VERTICES, q);
    float* devSlopes          = sycl::malloc_device<float>(VERTICES, q);
    int* devBitmap             = sycl::malloc_device<int>(NUM_POINTS, q);
    q.memcpy(devPoints, hostPoints.data(), NUM_POINTS * sizeof(sycl::float2));
    q.memcpy(devVertices, hostVertices.data(), VERTICES * sizeof(sycl::float2));
    q.memcpy(devSlopes, hostSlopes.data(), VERTICES * sizeof(float));
    q.wait();

    size_t numPoints = (size_t)((NUM_POINTS + kTileSize - 1) / kTileSize);
    size_t globalX = ((numPoints + kBlockSizeX - 1) / kBlockSizeX) * kBlockSizeX;
    sycl::nd_range<1> ndRange{ sycl::range<1>(globalX), sycl::range<1>(kBlockSizeX) };

    auto launch = [&]() {
        return q.submit([&](sycl::handler &h) {
            h.parallel_for(ndRange, PnPolyKernel(devBitmap, devPoints, devVertices, devSlopes, NUM_POINTS));
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

    q.memcpy(hostBitmap.data(), devBitmap, NUM_POINTS * sizeof(int)).wait();
    sycl::free(devPoints, q);
    sycl::free(devVertices, q);
    sycl::free(devSlopes, q);
    sycl::free(devBitmap, q);

    // Sampled correctness check against one fixed, canonical crossing-number
    // formula -- BETWEEN_METHOD/USE_METHOD/USE_PRECOMPUTED_SLOPES are
    // alternate ways of computing the same test for this simple convex
    // (circle-approximation) polygon, so they should all agree with it.
    // A full NUM_POINTS*VERTICES CPU reference would dominate per-evaluation
    // time, since this whole program reruns from scratch every evaluation.
    unsigned sseed = 999u;
    for (int s = 0; s < 1000; ++s) {
        sseed = sseed * 1103515245u + 12345u;
        int idx = (int)(sseed % NUM_POINTS);
        float px = hostPoints[idx].x(), py = hostPoints[idx].y();
        int c = 0;
        int k = VERTICES - 1;
        for (int vi = 0; vi < VERTICES; ++vi) {
            float vky = vy[k], vjy = vy[vi];
            bool between = (vjy > vky) ? (vjy > py && py >= vky) : (vjy <= py && py < vky);
            if (between) {
                float slope = (vx[k] - vx[vi]) / (vky - vjy);
                if (px < slope * (py - vjy) + vx[vi]) c = !c;
            }
            k = vi;
        }
        if (c != hostBitmap[idx]) {
            fprintf(stderr, "result check failed at point %d: got %d, expected %d\\n", idx, hostBitmap[idx], c);
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

PROBLEM_SIZE = {'problem_num_points': 20000000, 'problem_vertices': 600}

# device limits for this machine (RTX 2000 Ada), see dpcpp__matmul.py
MAX_WORKGROUP_SIZE = 1024
NUM_POINTS = 20_000_000

# Step 1: Generate the Search Space (same 5 tunables + restrictions as
# SyTuner's PnpolyTunerBase.tune_params()/restrictions())
BLOCK_SIZE_X = TP('BLOCK_SIZE_X', Set(*[32 * i for i in range(1, 32)]))            # 32,64,...,992
TILE_SIZE    = TP('TILE_SIZE',    Set(1, *[2 * i for i in range(1, 11)]),          # 1,2,4,...,20
                  lambda BLOCK_SIZE_X, TILE_SIZE:
                      BLOCK_SIZE_X <= MAX_WORKGROUP_SIZE
                      and ((NUM_POINTS + TILE_SIZE - 1) // TILE_SIZE + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X * BLOCK_SIZE_X <= NUM_POINTS)
BETWEEN_METHOD         = TP('BETWEEN_METHOD',         Set(0, 1, 2, 3))
USE_PRECOMPUTED_SLOPES = TP('USE_PRECOMPUTED_SLOPES', Set(0, 1))
USE_METHOD             = TP('USE_METHOD',             Set(0, 1))

# Step 2 & 3: repeat the whole autotuning procedure from scratch --runs times
# (matching SyTuner's KernelTuner runner default, AUTOTUNE_RUNS=5) -- each run
# gets its own fresh search technique (AUCBandit carries state across a run),
# its own isolated log/result/cost/device files, and its own freshly compiled
# source (the cost/device file paths are baked into the source as literals).
session = bench.new_session_id()
for run_idx in range(args.runs):
    paths = bench.run_paths(args.output_dir, COMPILER, WORKLOAD, run_idx, session)
    pnpoly_source = dpcpp.source(
        PNPOLY_SOURCE_TEMPLATE
        .replace('__COST_FILE_PATH__', str(paths['cost_file']))
        .replace('__DEVICE_FILE_PATH__', str(paths['device_file']))
    )

    cf_pnpoly = dpcpp.CostFunction( pnpoly_source ).target( args.target )                              \
                                                   .flags( [f'-DWARMUP_RUNS={args.warmup_runs}',
                                                            f'-DMEASUREMENT_RUNS={args.measurement_runs}'] ) \
                                                   .cost_file( str(paths['cost_file']) )

    config, min_cost, tuning_data = Tuner().tuning_parameters( BLOCK_SIZE_X, TILE_SIZE, BETWEEN_METHOD,
                                                                USE_PRECOMPUTED_SLOPES, USE_METHOD )  \
                                           .search_technique( AUCBandit() )                            \
                                           .log_file( str(paths['log_file']) )                         \
                                           .tune( cf_pnpoly, Evaluations(args.max_fevals) )

    result = bench.base_result_fields(WORKLOAD, COMPILER, args, run_idx, session, Path(__file__).name,
                                      bench.read_device_name(paths['device_file']), paths['log_file'],
                                      config, min_cost, tuning_data)
    result.update(PROBLEM_SIZE)
    bench.write_result_json(paths['result_file'], **result)

    print(f'run {run_idx}: min_cost={min_cost}, config={config}')
    print(f'  result -> {paths["result_file"]}')
