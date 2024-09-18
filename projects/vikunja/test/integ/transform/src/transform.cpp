#include <alpaka/alpaka.hpp>
#include <alpaka/test/acc/TestAccs.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vikunja/vikunja.hpp>

#include <algorithm>
#include <sstream>
#include <type_traits>

using Dim1D = alpaka::DimInt<1>;
using Idx = std::size_t;

using TestAccs = alpaka::test::EnabledAccs<Dim1D, Idx>;

template<typename TData>
auto get_data_type_name()
{
    if constexpr(std::is_same_v<TData, int>)
    {
        return "int";
    }
    else if(std::is_same_v<TData, float>)
    {
        return "float";
    }
    else if(std::is_same_v<TData, double>)
    {
        return "double";
    }
    else
    {
        return typeid(TData).name();
    }
}

template<typename TAcc, typename TData>
void transform_integration_test(Idx const problem_size)
{
    std::stringstream ss;
    ss << "Using alpaka accelerator: " << alpaka::getAccName<TAcc>() << "\n";
    ss << "Datatype: " << get_data_type_name<TData>() << "\n";
    ss << "size: " << problem_size << "\n";
    INFO(ss.str());

    using DeviceAcc = TAcc;
    using Vec = alpaka::Vec<Dim1D, Idx>;

    using QueueProperty = alpaka::NonBlocking;
    using Queue = alpaka::Queue<DeviceAcc, QueueProperty>;

    auto const platformHost = alpaka::PlatformCpu{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0);

    auto const platformAcc = alpaka::Platform<DeviceAcc>{};
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

    Queue queue(devAcc);

    std::vector<TData> host_data(problem_size);
    std::iota(std::begin(host_data), std::end(host_data), 0);
    std::vector<TData> std_host_data = host_data;

    auto const extent(static_cast<Idx>(problem_size));
    auto bufDevIn = alpaka::allocAsyncBuf<TData, Idx>(queue, extent);
    auto bufDevOut = alpaka::allocAsyncBuf<TData, Idx>(queue, extent);

    alpaka::memcpy(queue, bufDevIn, alpaka::createView(devHost, host_data));

    auto functor = [] ALPAKA_FN_HOST_ACC(TData i) { return i * static_cast<TData>(2); };
    vikunja::transform<DeviceAcc>(queue, bufDevIn, bufDevOut, functor);

    alpaka::memcpy(queue, alpaka::createView(devHost, host_data), bufDevOut);

    std::transform(std::begin(std_host_data), std::end(std_host_data), std::begin(std_host_data), functor);
    alpaka::wait(queue);

    CHECK(host_data.size() == std_host_data.size());

    using stype = typename std::vector<TData>::size_type;

    for(stype i = 0; i < host_data.size(); ++i)
    {
        if constexpr(std::is_same_v<TData, int>)
        {
            CHECK(host_data[i] == std_host_data[i]);
        }
        else
        {
            REQUIRE_THAT(host_data[i], Catch::Matchers::WithinRel(std_host_data[i]));
        }
    }
}

TEMPLATE_LIST_TEST_CASE("transformTypeInt", "[integ][transform][int]", TestAccs)
{
    // TODO(SimeonEhrig): check size 0
    auto size = GENERATE(1, 10, 99, 978, 101025);
    transform_integration_test<TestType, int>(static_cast<Idx>(size));
}

TEMPLATE_LIST_TEST_CASE("transformTypeFloat", "[integ][transform][float]", TestAccs)
{
    // TODO(SimeonEhrig): check size 0
    auto size = GENERATE(1, 10, 99, 978, 101025);
    transform_integration_test<TestType, float>(static_cast<Idx>(size));
}

TEMPLATE_LIST_TEST_CASE("transformTypeDouble", "[integ][transform][double]", TestAccs)
{
    // TODO(SimeonEhrig): check size 0
    auto size = GENERATE(1, 10, 99, 978, 101025);
    transform_integration_test<TestType, double>(static_cast<Idx>(size));
}
