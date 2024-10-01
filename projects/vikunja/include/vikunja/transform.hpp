#pragma once
#include <cassert>
#include <iostream>

namespace vikunja
{
    namespace detail
    {
        namespace transform
        {
            struct TranformKernel
            {
                template<typename TAcc, typename TMdSpanInput, typename TMdSpanOutput, typename TFunctor>
                ALPAKA_FN_ACC auto operator()(
                    TAcc const& acc,
                    TMdSpanInput input,
                    TMdSpanOutput output,
                    TFunctor&& func) const -> void
                {
                    auto const global_idx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
                    auto max_size = input.extent(0);
                    if(global_idx < max_size)
                    {
                        output(global_idx) = func(input(global_idx));
                    }
                }
            };
        } // namespace transform
    } // namespace detail

    // TODO(SimeonEhrig): buf_in cannot be a `TBuf const &` because of `alpaka::experimental::getMdSpan`
    template<typename TAcc, typename TQueue, typename TBuf, typename TFunc>
    void transform(TQueue& queue, TBuf& buf_in, TBuf& buf_out, TFunc&& func)
    {
        // TODO(SimeonEhrig): check for multi dimensional Dim
        auto size_in = alpaka::getExtents(buf_in)[0];
        auto size_out = alpaka::getExtents(buf_out)[0];
        assert(size_in <= size_out);

        using Idx = alpaka::Idx<TAcc>;
        using Vec1D = alpaka::Vec<alpaka::DimInt<1>, Idx>;

        auto mdBufDevIn = alpaka::experimental::getMdSpan(buf_in);
        auto mdBufDevOut = alpaka::experimental::getMdSpan(buf_out);

        Vec1D const elementsPerThread(Vec1D::all(static_cast<Idx>(1)));
        Vec1D const elementsPerGrid(Vec1D::all(static_cast<Idx>(size_in)));

        vikunja::detail::transform::TranformKernel kernel{};

        alpaka::KernelCfg<TAcc> const hostKernelCfg = {elementsPerGrid, elementsPerThread};

        auto const workDiv = alpaka::getValidWorkDiv(
            hostKernelCfg,
            alpaka::getDev(queue),
            kernel,
            mdBufDevIn,
            mdBufDevOut,
            std::forward<TFunc>(func));

        alpaka::exec<TAcc>(queue, workDiv, kernel, mdBufDevIn, mdBufDevOut, std::forward<TFunc>(func));
    }

    template<typename TExecutor, typename TBuf, typename TFunc>
    requires isExecutor<TExecutor> void transform(TExecutor& executor, TBuf& buf_in, TBuf& buf_out, TFunc&& func)
    {
        transform<typename TExecutor::Acc>(executor.get_default_queue(), buf_in, buf_out, std::forward<TFunc>(func));
    }
} // namespace vikunja
