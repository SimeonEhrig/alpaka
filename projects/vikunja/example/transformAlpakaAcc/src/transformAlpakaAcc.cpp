#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <vikunja/vikunja.hpp>

#include <iomanip>
#include <iostream>
#include <vector>

template<typename TAccTag>
auto example(TAccTag const&) -> int
{
    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Vec = alpaka::Vec<Dim, Idx>;

    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;
    std::cout << "Using alpaka accelerator: " << alpaka::getAccName<Acc>() << std::endl;

    using QueueProperty = alpaka::NonBlocking;
    using Queue = alpaka::Queue<Acc, QueueProperty>;

    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    auto const platformAcc = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    Queue queue(devAcc);

    using DataType = int;
    constexpr Idx problem_size = 10;
    std::vector<DataType> host_data(problem_size);
    std::iota(std::begin(host_data), std::end(host_data), 0);

    std::cout << "input -> output\n";
    for(auto const d : host_data)
    {
        std::cout << std::setw(2) << d << " ";
    }
    std::cout << "\n";

    auto const extent(static_cast<Idx>(problem_size));
    auto bufDevIn = alpaka::allocAsyncBuf<DataType, Idx>(queue, extent);
    auto bufDevOut = alpaka::allocAsyncBuf<DataType, Idx>(queue, extent);

    alpaka::memcpy(queue, bufDevIn, alpaka::createView(devHost, host_data));

    vikunja::transform<Acc>(queue, bufDevIn, bufDevOut, [] ALPAKA_FN_HOST_ACC(DataType i) { return i * 3; });

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
    // Execute the example once for each enabled accelerator.
    // If you would like to execute it for a single accelerator only you can use the following code.
    //  \code{.cpp}
    //  auto tag = TagCpuSerial;
    //  return example(tag);
    //  \endcode
    //
    // valid tags:
    //   TagCpuSerial, TagGpuHipRt, TagGpuCudaRt, TagCpuOmp2Blocks, TagCpuTbbBlocks,
    //   TagCpuOmp2Threads, TagCpuSycl, TagCpuTbbBlocks, TagCpuThreads,
    //   TagFpgaSyclIntel, TagGenericSycl, TagGpuSyclIntel
    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag); });
}
