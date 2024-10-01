#pragma once
#include <cassert>
#include <cstddef>
#include <vector>

namespace vikunja
{
    template<typename T>
    concept isExecutor = requires(T t)
    {
        typename T::Acc;
        typename T::Queue;
        t.queues;
        t.get_platform();
        t.get_device();
        t.get_default_queue();
    };

    template<typename TAcc, typename TQueueProperty>
    class Executor
    {
    public:
        using Acc = TAcc;
        using Queue = alpaka::Queue<TAcc, TQueueProperty>;
        std::vector<Queue> queues;

    private:
        alpaka::Platform<Acc> platform;
        alpaka::Dev<Acc> device;

    public:
        Executor(std::size_t const device_id = 0)
            : platform(alpaka::Platform<TAcc>{})
            , device(alpaka::getDevByIdx(platform, device_id))
        {
            queues.emplace_back(device);
        }

        alpaka::Platform<Acc>& get_platform() const noexcept
        {
            return platform;
        }

        alpaka::Dev<Acc>& get_device() const noexcept
        {
            return device;
        }

        Queue& get_default_queue() noexcept
        {
            assert(queues.size() > 0);
            return queues[0];
        }
    };
} // namespace vikunja

namespace alpaka
{
    template<typename TElem, typename TIdx, typename TExtent, typename TVikunjaExecutor>
    requires vikunja::isExecutor<TVikunjaExecutor> ALPAKA_FN_HOST auto allocAsyncBuf(
        TVikunjaExecutor executor,
        TExtent const& extent = TExtent())
    {
        return alpaka::allocAsyncBuf<TElem, TIdx>(executor.get_default_queue(), extent);
    }

    // template<typename TVikunjaExecutor, typename TViewSrc, typename TViewDest>
    // requires vikunja::isExecutor<TVikunjaExecutor> ALPAKA_FN_HOST auto memcpy(
    //     TVikunjaExecutor& executor,
    //     TViewDest& viewDst,
    //     TViewDest const& viewSrc) -> void
    // {
    //     alpaka::memcpy(executor.get_default_queue(), viewDst, viewSrc);
    // }
} // namespace alpaka
