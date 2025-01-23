#pragma once

#include "Vec.hpp"

#include <alpaka/alpaka.hpp>

class IterateAllElementsKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TOutput, typename TInput, typename TIdx>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TOutput output, TInput input, TIdx const& numElements) const -> void
    {
        static_assert(alpaka::Dim<TAcc>::value == 1, "The ExecuteExpressionKernel expects 1-dimensional indices!");

        // The uniformElements range for loop takes care automatically of the blocks, threads and elements in the
        // kernel launch grid.
        for(auto i : alpaka::uniformElements(acc, numElements))
        {
            output[i] = input[i];
        }
    }
};

template<typename TAcc, alpaka::concepts::Queue TQueue, LinMath::concepts::Vec TVec, typename TFuncKernel>
void run_kernel_over_vec(TQueue& queue, TVec& vec, TFuncKernel&& funcKernel)
{
    IterateAllElementsKernel kernel;

    alpaka::KernelCfg<TAcc> const kernelCfg = {vec.size(), 1u};

    auto const workDiv
        = alpaka::getValidWorkDiv(kernelCfg, alpaka::getDev(queue), kernel, vec.data(), funcKernel, vec.size());

    auto const taskKernel = alpaka::createTaskKernel<TAcc>(workDiv, kernel, vec.data(), funcKernel, vec.size());

    alpaka::enqueue(queue, taskKernel);
}

template<typename TData>
class ConstValueKernel
{
    TData value;

public:
    ConstValueKernel(TData value) : value(value)
    {
    }

    ALPAKA_FN_INLINE ALPAKA_FN_ACC TData operator[](std::size_t const) const
    {
        return value;
    }
};

template<typename TAcc, alpaka::concepts::Queue TQueue, LinMath::concepts::Vec TVec, typename TData>
void vec_init_const(TQueue& queue, TVec& vec, TData const value)
{
    run_kernel_over_vec<TAcc>(queue, vec, ConstValueKernel{value});
}

template<typename TData>
class IotaKernel
{
    TData start;
    TData step;

public:
    IotaKernel(TData start, TData step = TData(1)) : start(start), step(step)
    {
    }

    ALPAKA_FN_INLINE ALPAKA_FN_ACC TData operator[](std::size_t const i) const
    {
        return start + i * step;
    }
};

template<typename TAcc, alpaka::concepts::Queue TQueue, LinMath::concepts::Vec TVec, typename TData>
void vec_iota(TQueue& queue, TVec& vec, TData const start, TData const step = TData(1))
{
    run_kernel_over_vec<TAcc>(queue, vec, IotaKernel{start, step});
}

template<typename TAcc, typename TQueue, typename Expr>
LinMath::Vec<typename Expr::Data, TQueue, typename Expr::Idx> eval(TQueue& queue, Expr expr)
{
    LinMath::Vec<typename Expr::Data, TQueue, typename Expr::Idx> vec{queue, expr.size()};
    run_kernel_over_vec<TAcc>(queue, vec, expr);
    return vec;
}
