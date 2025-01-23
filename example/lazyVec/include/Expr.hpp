#pragma once

#include <alpaka/alpaka.hpp>

#include <type_traits>

namespace LinMath
{
    namespace concepts
    {
        template<typename T>
        concept ExprConstructable = requires(T t, T const ct) {
            typename T::Data;
            typename T::Idx;
            {
                ct.size()
            } -> std::same_as<typename T::Idx>;
            {
                ct[typename T::Idx{0}]
            } -> std::same_as<typename T::Data>;
        };
    } // namespace concepts

    template<typename TData, typename TIdx, typename TOp>
    class Expr
    {
    private:
        TOp m_storage;

    public:
        using Data = TData;
        using Idx = TIdx;
        using Op = TOp;

        Expr(TOp const storage) : m_storage(storage)
        {
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC TData operator[](TIdx const i)
        {
            return m_storage[i];
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr TData operator[](TIdx const i) const
        {
            return m_storage[i];
        }

        ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC TIdx size()
        {
            return m_storage.size();
        }

        ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr TIdx size() const
        {
            return m_storage.size();
        }

        auto op()
        {
            return *this;
        }
    };
} // namespace LinMath
