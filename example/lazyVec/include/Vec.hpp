#pragma once

#include "Expr.hpp"
#include "Op.hpp"

#include <alpaka/alpaka.hpp>

namespace alpaka::concepts
{
    template<typename T>
    concept Queue = alpaka::isQueue<T>;

    template<typename T>
    concept Device = alpaka::isDevice<T>;
} // namespace alpaka::concepts

namespace LinMath
{
    namespace concepts
    {
        template<typename T>
        concept Vec = requires(T t) {
            typename T::Idx;
            typename T::Extent;
            typename T::Data;
            typename T::Buf;
            {
                t.getBuf()
            } -> std::same_as<typename T::Buf&>;
            {
                t.extends()
            } -> std::same_as<typename T::Extent>;
            {
                t.data()
            } -> std::same_as<typename T::Data*>;
            {
                t.size()
            } -> std::same_as<typename T::Idx>;
        };
    } // namespace concepts

    template<typename TData, typename THasDev, typename TIdx = std::size_t>
    class Vec
    {
    private:
        using Dev = alpaka::Dev<std::decay_t<THasDev>>;

    public:
        using Idx = TIdx;
        using Extent = alpaka::Vec<alpaka::DimInt<1>, Idx>;
        using Data = TData;
        using Buf = alpaka::Buf<Dev, TData, alpaka::DimInt<1>, Idx>;

    private:
        Buf m_storage;

        template<alpaka::concepts::Device T>
        [[nodiscard]] Buf alloc(T& dev, std::size_t const size)
        {
            Extent extent(size);
            return alpaka::allocBuf<Data, Idx, Extent, Dev>(dev, extent);
        }

        template<alpaka::concepts::Queue T>
        [[nodiscard]] Buf alloc(T& queue, std::size_t const size)
        {
            alpaka::Vec<alpaka::DimInt<1>, Idx> extent(size);
            return alpaka::allocAsyncBuf<Data, Idx>(queue, extent);
        }

    public:
        Vec(THasDev& mem_allocator, TIdx const size) : m_storage(alloc(mem_allocator, size))
        {
            static_assert(alpaka::isDevice<THasDev> || alpaka::isQueue<THasDev>);
        }

        Vec(THasDev& mem_allocator, TIdx const size, TData type) : m_storage(alloc(mem_allocator, size))
        {
            static_assert(alpaka::isDevice<THasDev> || alpaka::isQueue<THasDev>);
        }

        [[nodiscard]] Buf& getBuf()
        {
            return m_storage;
        }

        [[nodiscard]] Data* data()
        {
            return alpaka::getPtrNative(m_storage);
        }

        [[nodiscard]] constexpr Data* data() const
        {
            return alpaka::getPtrNative(m_storage);
        }

        [[nodiscard]] constexpr Extent extends() const
        {
            return alpaka::getExtents(m_storage);
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr TData operator[](TIdx const i) const
        {
            return m_storage[i];
        }

        [[nodiscard]] constexpr TIdx size() const
        {
            return extends()[0];
        }

        template<alpaka::concepts::Queue TQueue>
        std::vector<Data> getHostVector(TQueue& queue)
        {
            std::vector<Data> hostVector(size());
            auto const devHost = alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0);
            auto viewHostVector = createView(devHost, hostVector);

            alpaka::memcpy(queue, viewHostVector, m_storage);
            alpaka::wait(queue);

            return hostVector;
        }

        void _()
        {
            static_assert(LinMath::concepts::ExprConstructable<Vec>);
        }

        operator VecAccess<TData, TIdx>()
        {
            return {data(), size()};
        }

        VecAccess<TData, TIdx> op()
        {
            return {data(), size()};
        }
    };
} // namespace LinMath
