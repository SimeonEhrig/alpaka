#include <alpaka/alpaka.hpp>

#include <vikunja/vikunja.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

template<typename TAccTag>
auto example(TAccTag const&) -> int
{
    // preample
    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Vec = alpaka::Vec<Dim, Idx>;

    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    vikunja::Executor<Acc, alpaka::NonBlocking> exe;
    auto queue = exe.get_default_queue();

    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    // allocate memory

    using DataType = int;
    constexpr Idx problem_size = 10;
    std::vector<DataType> host_data(problem_size);
    std::iota(std::begin(host_data), std::end(host_data), 0);

    for(auto const d : host_data)
    {
        std::cout << std::setw(2) << d << " ";
    }
    std::cout << "\n";

    auto const extent(static_cast<Idx>(problem_size));
    auto bufDevIn = alpaka::allocAsyncBuf<DataType, Idx>(exe, extent);
    auto bufDevOut = alpaka::allocAsyncBuf<DataType, Idx>(exe, extent);

    alpaka::memcpy(queue, bufDevIn, alpaka::createView(devHost, host_data));

    // vikunja::transform<Acc>(queue, bufDevIn, bufDevOut, [] ALPAKA_FN_HOST_ACC(DataType i) { return i * 3; });
    vikunja::transform(exe, bufDevIn, bufDevOut, [] ALPAKA_FN_HOST_ACC(DataType i) { return i * 3; });

    alpaka::memcpy(queue, alpaka::createView(devHost, host_data), bufDevOut);
    alpaka::wait(queue);

    for(auto const d : host_data)
    {
        std::cout << std::setw(2) << d << " ";
    }
    std::cout << "\n";
    return 0;
}

auto main() -> int
{
    int return_value = 0;
    if constexpr(alpaka::AccIsEnabled<alpaka::TagCpuOmp2Blocks>::value)
    {
        return_value += example(alpaka::TagCpuOmp2Blocks{});
    }
    if constexpr(alpaka::AccIsEnabled<alpaka::TagGpuCudaRt>::value)
    {
        return_value += example(alpaka::TagGpuCudaRt{});
    }
    return return_value;
}
