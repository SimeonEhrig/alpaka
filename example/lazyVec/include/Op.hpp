#pragma once

#include "Expr.hpp"

#include <alpaka/alpaka.hpp>

namespace LinMath
{
    template<typename TData, typename TIdx>
    class Op
    {
    protected:
        TIdx m_size;

    public:
        using Data = TData;
        using Idx = TIdx;

        Op(TIdx const size) : m_size(size)
        {
        }

        ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr TIdx size() const
        {
            return m_size;
        }
    };

    template<typename TData, typename TIdx>
    class UnaryOp : public Op<TData, TIdx>
    {
    public:
        UnaryOp(TIdx const size) : Op<TData, TIdx>(size)
        {
        }
    };

    template<typename TData, typename TIdx, typename L, typename R>
    class BinaryOp : public Op<TData, TIdx>
    {
    protected:
        L m_lhv;
        R m_rhv;

    public:
        BinaryOp(L const a, R const b, TIdx const size) : m_lhv(a), m_rhv(b), Op<TData, TIdx>(size)
        {
        }
    };

    template<typename TData, typename TIdx>
    class VecAccess : public UnaryOp<TData, TIdx>
    {
        TData* m_view;

    public:
        VecAccess(TData* view, TIdx const size) : m_view(view), UnaryOp<TData, TIdx>(size)
        {
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr TData& operator[](TIdx const i) const
        {
            return m_view[i];
        }
    };

    template<typename TData, typename TIdx, typename L, typename R>
    class VectorAdd : public BinaryOp<TData, TIdx, L, R>
    {
    public:
        template<concepts::ExprConstructable TLExpr, concepts::ExprConstructable TRExpr>
        VectorAdd(TLExpr l, TRExpr r)
            : BinaryOp<
                decltype(std::declval<typename TLExpr::Data>() + std::declval<typename TRExpr::Data>()),
                decltype(l.size()),
                decltype(l.op()),
                decltype(r.op())>(l.op(), r.op(), l.size())
        {
            static_assert(std::is_same_v<typename TLExpr::Idx, typename TRExpr::Idx>);
        }

        ALPAKA_FN_INLINE ALPAKA_FN_ACC constexpr TData operator[](TIdx const i) const
        {
            return this->m_lhv[i] + this->m_rhv[i];
        }

        auto op()
        {
            return *this;
        }
    }; // namespace

    template<concepts::ExprConstructable TLExpr, concepts::ExprConstructable TRExpr>
    VectorAdd(TLExpr l, TRExpr r) -> VectorAdd<
        decltype(std::declval<typename TLExpr::Data>() + std::declval<typename TRExpr::Data>()),
        decltype(l.size()),
        decltype(l.op()),
        decltype(r.op())>;

    template<concepts::ExprConstructable L, concepts::ExprConstructable R>
    auto operator+(L& a, R& b)
    {
        using result_type = decltype(std::declval<typename L::Data>() + std::declval<typename R::Data>());
        using OperatorType = decltype(VectorAdd{a, b});
        return Expr<result_type, typename L::Idx, OperatorType>{VectorAdd{a, b}};
    }
} // namespace LinMath
