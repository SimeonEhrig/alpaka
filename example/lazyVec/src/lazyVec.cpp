#include "Execute.hpp"
#include "Vec.hpp"

#include <alpaka/alpaka.hpp>

#include <concepts>
#include <cstddef>
#include <iostream>
#include <vector>

template<typename TTag>
void example(TTag)
{
    using Idx = std::size_t;
    using Dim = alpaka::DimInt<1>;
    using Acc = alpaka::TagToAcc<TTag, Dim, Idx>;
    using DevAcc = alpaka::Dev<Acc>;
    using Queue = alpaka::Queue<Acc, alpaka::NonBlocking>;

    std::cout << "Acc: " << alpaka::getAccName<Acc>() << "\n";

    auto const platform = alpaka::Platform<Acc>{};
    auto const devAcc = alpaka::getDevByIdx(platform, 0);
    Queue queue(devAcc);

    LinMath::Vec vs1(queue, 1000, int{});
    LinMath::Vec vs2(queue, 1000, int{});
    LinMath::Vec vs3(queue, 1000, int{});

    vec_init_const<Acc>(queue, vs1, 3);
    vec_iota<Acc>(queue, vs2, 1, 3);
    vec_init_const<Acc>(queue, vs3, 1);

    static_assert(LinMath::concepts::ExprConstructable<decltype(vs1)>);
    static_assert(LinMath::concepts::ExprConstructable<LinMath::VectorAdd<int, std::size_t, int*, int*>>);
    static_assert(LinMath::concepts::ExprConstructable<
                  LinMath::Expr<int, std::size_t, LinMath::VectorAdd<int, std::size_t, int*, int*>>>);

    LinMath::Expr e1 = vs1 + vs2;
    LinMath::Expr e2 = e1 + vs3;

    LinMath::Vec r = eval<Acc>(queue, e2);

    std::vector<int> result = r.getHostVector(queue);
    for(auto i = 0; i < 10; ++i)
    {
        std::cout << result[i] << " ";
    }
    std::cout << "\n";

    // actual not required, because getHostVector() does an alpaka::wait()
    alpaka::wait(queue);
}

int main(int argc, char** argv)
{
    if constexpr(alpaka::AccIsEnabled<alpaka::TagGpuCudaRt>::value)
    {
        example(alpaka::TagGpuCudaRt{});
    }
    else if(alpaka::AccIsEnabled<alpaka::TagCpuOmp2Blocks>::value)
    {
        example(alpaka::TagCpuOmp2Blocks{});
    }
    else
    {
        example(alpaka::TagCpuSerial{});
    }
    return 0;
}
