from pyatf import TP, Interval, Tuner
from pyatf.cost_functions import acpp
from pyatf.search_techniques import AUCBandit
from pyatf.abort_conditions import Evaluations

# same program as dpcpp__saxpy.py's -- ACPP's generic/SSCP flow JIT-compiles the
# device code at first launch instead of baking in one architecture at compile
# time, so this one source runs unmodified on NVIDIA/AMD/Intel, no target() call
saxpy_source = acpp.source('''
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#ifndef WPT
#define WPT 1
#endif
#ifndef LS
#define LS 1
#endif

static constexpr int N = 1000;  // must match N below in acpp__saxpy.py

int main() {
    // explicit GPU selector -- ACPP's default selector can silently pick the
    // OpenMP host device instead when more than one backend is visible
    sycl::queue q{ sycl::gpu_selector_v, sycl::property::queue::enable_profiling{} };

    const float a = (float)rand() / RAND_MAX;
    std::vector<float> x(N), y(N), y_gold(N);
    for (int i = 0; i < N; ++i) {
        x[i] = (float)rand() / RAND_MAX;
        y[i] = (float)rand() / RAND_MAX;
        y_gold[i] = y[i] + a * x[i];
    }

    float *dx = sycl::malloc_device<float>(N, q);
    float *dy = sycl::malloc_device<float>(N, q);
    q.memcpy(dx, x.data(), N * sizeof(float));
    q.memcpy(dy, y.data(), N * sizeof(float));
    q.wait();

    const size_t global_size = N / WPT;
    const size_t local_size = LS;

    auto event = q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> it) {
            size_t gid = it.get_global_id(0);
            for (int w = 0; w < WPT; ++w) {
                size_t index = w * global_size + gid;
                dy[index] += a * dx[index];
            }
        });
    });
    event.wait();

    auto t0 = event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto t1 = event.get_profiling_info<sycl::info::event_profiling::command_end>();

    q.memcpy(y.data(), dy, N * sizeof(float)).wait();
    sycl::free(dx, q);
    sycl::free(dy, q);

    for (int i = 0; i < N; ++i) {
        if (std::fabs(y[i] - y_gold[i]) > 1e-3f) {
            fprintf(stderr, "result check failed at %d: got %f, expected %f\\n", i, y[i], y_gold[i]);
            return 1;
        }
    }

    std::ofstream cost_out("/tmp/pyatf_acpp_saxpy_cost.txt");
    cost_out << (t1 - t0);
    return 0;
}
''')

# input size
N = 1000

# Step 1: Generate the Search Space
WPT = TP('WPT', Interval( 1, N ), lambda WPT: N % WPT == 0           )
LS  = TP('LS',  Interval( 1, N ), lambda WPT, LS: (N / WPT) % LS == 0)

# Step 2: Implement a Cost Function
cf_saxpy = acpp.CostFunction( saxpy_source ).cost_file( '/tmp/pyatf_acpp_saxpy_cost.txt' )

# Step 3: Explore the Search Space
tuning_result = Tuner().tuning_parameters( WPT, LS )       \
                       .search_technique( AUCBandit() )    \
                       .tune( cf_saxpy, Evaluations(50) )
