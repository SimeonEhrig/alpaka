/* Copyright 2024 Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/acc/Traits.hpp>
#include <alpaka/core/Interface.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <string>

TEMPLATE_LIST_TEST_CASE("isSingleThreadAcc", "[acc]", alpaka::test::TestAccs)
{
    using Acc = TestType;

    // Check that both traits are defined, and that only one is true.
    REQUIRE(alpaka::isSingleThreadAcc<Acc> != alpaka::isMultiThreadAcc<Acc>);

    auto const platform = alpaka::Platform<Acc>{};
    auto const dev = alpaka::getDevByIdx(platform, 0);
    auto const devProps = alpaka::getAccDevProps<Acc>(dev);

    // Compare the runtime properties with the compile time trait.
    INFO("Accelerator: " << alpaka::core::demangled<Acc>);
    if constexpr(alpaka::isSingleThreadAcc<Acc>)
    {
        // Require a single thread per block.
        REQUIRE(devProps.m_blockThreadCountMax == 1);
    }
    else
    {
        // Assume multiple threads per block, but allow a single thread per block.
        // For example, the AccCpuOmp2Threads accelerator may report a single thread on a single core system.
        REQUIRE(devProps.m_blockThreadCountMax >= 1);
    }
}

struct TestTag : public alpaka::interface::Implements<alpaka::InterfaceTag, TestTag>
{
    static auto get_name() -> std::string
    {
        return "TestTag";
    }
};

template<typename TDim, typename TIdx>
struct TestAcc : public alpaka::interface::Implements<alpaka::ConceptAcc, TestAcc<TDim, TIdx>>

{
};

template<typename TDim, typename TIdx>
struct alpaka::trait::IsMultiThreadAcc<TestAcc<TDim, TIdx>> : std::true_type
{
};

template<typename TDim, typename TIdx>
struct alpaka::trait::AccToTag<TestAcc<TDim, TIdx>>
{
    using type = TestTag;
};

template<alpaka::concepts::Acc>
void foo()

{
}

template<alpaka::concepts::Acc TAcc, alpaka::concepts::Tag TRet = alpaka::AccToTag<TAcc>>
auto bar() -> TRet
{
    return alpaka::AccToTag<TAcc>{};
}

template<typename T>
void tagOrAcc()
{
    if constexpr(alpaka::concepts::Tag<T>)
    {
        std::cout << "I'm a Tag\n";
    }
    if constexpr(alpaka::concepts::Acc<T>)
    {
        std::cout << "I'm a Acc\n";
    }
}

TEMPLATE_LIST_TEST_CASE("testConceptAcc", "[acc][concept]", alpaka::test::TestAccs)
{
    using Acc = TestType;
    INFO("Accelerator: " << alpaka::core::demangled<Acc>);

    // STATIC_REQUIRE(alpaka::concepts::Acc<Acc>);
    foo<Acc>();
    using Acc2 = TestAcc<alpaka::DimInt<1>, int>;
    foo<Acc2>();
    INFO("TestAcc::getAccName: " << alpaka::getAccName<Acc2>());
    INFO("alpaka::isSingleThreadAcc: " << std::boolalpha << alpaka::isSingleThreadAcc<Acc2>);
    INFO("alpaka::isMultiThreadAcc: " << std::boolalpha << alpaka::isMultiThreadAcc<Acc2>);
    REQUIRE(true);
    alpaka::concepts::Tag auto t = bar<Acc2>();

    tagOrAcc<decltype(t)>();
    tagOrAcc<Acc2>();
}
