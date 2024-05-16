/* Copyright 2023 Jan Stephan, Luca Ferragina, Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Sycl.hpp"
#include "alpaka/dev/DevGenericSycl.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/dim/DimIntegralConst.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/mem/buf/BufCpu.hpp"
#include "alpaka/mem/buf/Traits.hpp"
#include "alpaka/mem/view/ViewAccessOps.hpp"
#include "alpaka/vec/Vec.hpp"

#include <memory>
#include <type_traits>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka
{
    //! The SYCL memory buffer.
    template<typename TElem, typename TDim, typename TIdx, typename TPlatform, typename TMemVisibility>
    class BufGenericSycl : public internal::ViewAccessOps<BufGenericSycl<TElem, TDim, TIdx, TPlatform, TMemVisibility>>
    {
    public:
        static_assert(
            !std::is_const_v<TElem>,
            "The elem type of the buffer can not be const because the C++ Standard forbids containers of const "
            "elements!");
        static_assert(!std::is_const_v<TIdx>, "The idx type of the buffer can not be const!");

        //! Constructor
        template<typename TExtent, typename Deleter>
        BufGenericSycl(DevGenericSycl<TPlatform> const& dev, TElem* const pMem, Deleter deleter, TExtent const& extent)
            : m_dev{dev}
            , m_extentElements{getExtentVecEnd<TDim>(extent)}
            , m_spMem(pMem, std::move(deleter))
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            static_assert(
                TDim::value == Dim<TExtent>::value,
                "The dimensionality of TExtent and the dimensionality of the TDim template parameter have to be "
                "identical!");

            static_assert(
                std::is_same_v<TIdx, Idx<TExtent>>,
                "The idx type of TExtent and the TIdx template parameter have to be identical!");
        }

        DevGenericSycl<TPlatform> m_dev;
        Vec<TDim, TIdx> m_extentElements;
        std::shared_ptr<TElem> m_spMem;
    };
} // namespace alpaka

namespace alpaka::trait
{
    //! The BufGenericSycl device type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPlatform, typename TMemVisibility>
    struct DevType<BufGenericSycl<TElem, TDim, TIdx, TPlatform, TMemVisibility>>
    {
        using type = DevGenericSycl<TPlatform>;
    };

    //! The BufGenericSycl device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPlatform, typename TMemVisibility>
    struct GetDev<BufGenericSycl<TElem, TDim, TIdx, TPlatform, TMemVisibility>>
    {
        static auto getDev(BufGenericSycl<TElem, TDim, TIdx, TPlatform, TMemVisibility> const& buf)
        {
            return buf.m_dev;
        }
    };

    //! The BufGenericSycl dimension getter trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPlatform, typename TMemVisibility>
    struct DimType<BufGenericSycl<TElem, TDim, TIdx, TPlatform, TMemVisibility>>
    {
        using type = TDim;
    };

    //! The BufGenericSycl memory element type get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPlatform, typename TMemVisibility>
    struct ElemType<BufGenericSycl<TElem, TDim, TIdx, TPlatform, TMemVisibility>>
    {
        using type = TElem;
    };

    //! The BufGenericSycl extent get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPlatform, typename TMemVisibility>
    struct GetExtents<BufGenericSycl<TElem, TDim, TIdx, TPlatform, TMemVisibility>>
    {
        auto operator()(BufGenericSycl<TElem, TDim, TIdx, TPlatform, TMemVisibility> const& buf) const
        {
            return buf.m_extentElements;
        }
    };

    //! The BufGenericSycl native pointer get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPlatform, typename TMemVisibility>
    struct GetPtrNative<BufGenericSycl<TElem, TDim, TIdx, TPlatform, TMemVisibility>>
    {
        static auto getPtrNative(BufGenericSycl<TElem, TDim, TIdx, TPlatform, TMemVisibility> const& buf)
            -> TElem const*
        {
            return buf.m_spMem.get();
        }

        static auto getPtrNative(BufGenericSycl<TElem, TDim, TIdx, TPlatform, TMemVisibility>& buf) -> TElem*
        {
            return buf.m_spMem.get();
        }
    };

    //! The BufGenericSycl pointer on device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPlatform, typename TMemVisibility>
    struct GetPtrDev<BufGenericSycl<TElem, TDim, TIdx, TPlatform, TMemVisibility>, DevGenericSycl<TPlatform>>
    {
        static auto getPtrDev(
            BufGenericSycl<TElem, TDim, TIdx, TPlatform, TMemVisibility> const& buf,
            DevGenericSycl<TPlatform> const& dev) -> TElem const*
        {
            if(dev == getDev(buf))
            {
                return buf.m_spMem.get();
            }
            else
            {
                throw std::runtime_error("The buffer is not accessible from the given device!");
            }
        }

        static auto getPtrDev(
            BufGenericSycl<TElem, TDim, TIdx, TPlatform, TMemVisibility>& buf,
            DevGenericSycl<TPlatform> const& dev) -> TElem*
        {
            if(dev == getDev(buf))
            {
                return buf.m_spMem.get();
            }
            else
            {
                throw std::runtime_error("The buffer is not accessible from the given device!");
            }
        }
    };

    //! The SYCL memory allocation trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPlatform>
    struct BufAlloc<TElem, TDim, TIdx, DevGenericSycl<TPlatform>>
    {
        template<typename TExtent>
        static auto allocBuf(DevGenericSycl<TPlatform> const& dev, TExtent const& extent)
            -> BufGenericSycl<TElem, TDim, TIdx, TPlatform, alpaka::MemVisibilityTypeList<TPlatform>>
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

#    if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
            if constexpr(TDim::value == 0)
                std::cout << __func__ << " ewb: " << sizeof(TElem) << '\n';
            else if constexpr(TDim::value == 1)
            {
                auto const width = getWidth(extent);

                auto const widthBytes = width * static_cast<TIdx>(sizeof(TElem));
                std::cout << __func__ << " ew: " << width << " ewb: " << widthBytes << '\n';
            }
            else if constexpr(TDim::value == 2)
            {
                auto const width = getWidth(extent);
                auto const height = getHeight(extent);

                auto const widthBytes = width * static_cast<TIdx>(sizeof(TElem));
                std::cout << __func__ << " ew: " << width << " eh: " << height << " ewb: " << widthBytes
                          << " pitch: " << widthBytes << '\n';
            }
            else if constexpr(TDim::value == 3)
            {
                auto const width = getWidth(extent);
                auto const height = getHeight(extent);
                auto const depth = getDepth(extent);

                auto const widthBytes = width * static_cast<TIdx>(sizeof(TElem));
                std::cout << __func__ << " ew: " << width << " eh: " << height << " ed: " << depth
                          << " ewb: " << widthBytes << " pitch: " << widthBytes << '\n';
            }
#    endif

            auto const& [nativeDev, nativeContext] = dev.getNativeHandle();
            TElem* memPtr = sycl::malloc_device<TElem>(
                static_cast<std::size_t>(getExtentProduct(extent)),
                nativeDev,
                nativeContext);
            auto deleter = [ctx = nativeContext](TElem* ptr) { sycl::free(ptr, ctx); };

            return BufGenericSycl<TElem, TDim, TIdx, TPlatform, alpaka::MemVisibilityTypeList<TPlatform>>(
                dev,
                memPtr,
                std::move(deleter),
                extent);
        }
    };

    //! The BufGenericSycl stream-ordered memory allocation capability trait specialization.
    template<typename TDim, typename TPlatform>
    struct HasAsyncBufSupport<TDim, DevGenericSycl<TPlatform>> : std::false_type
    {
    };

    //! The BufGenericSycl offset get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPlatform, typename TMemVisibility>
    struct GetOffsets<BufGenericSycl<TElem, TDim, TIdx, TPlatform, TMemVisibility>>
    {
        auto operator()(BufGenericSycl<TElem, TDim, TIdx, TPlatform, TMemVisibility> const&) const -> Vec<TDim, TIdx>
        {
            return Vec<TDim, TIdx>::zeros();
        }
    };

    //! The pinned/mapped memory allocation trait specialization for the SYCL devices.
    template<typename TPlatform, typename TElem, typename TDim, typename TIdx>
    struct BufAllocMapped
    {
        template<typename TExtent>
        static auto allocMappedBuf(DevCpu const& host, TPlatform const& platform, TExtent const& extent) -> BufCpu<
            TElem,
            TDim,
            TIdx,
            alpaka::meta::Unique<std::tuple<alpaka::MemVisibility<DevCpu>, alpaka::MemVisibility<TPlatform>>>>
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            // Allocate SYCL page-locked memory on the host, mapped into the TPlatform address space and
            // accessible to all devices in the TPlatform.
            auto ctx = platform.syclContext();
            TElem* memPtr = sycl::malloc_host<TElem>(static_cast<std::size_t>(getExtentProduct(extent)), ctx);
            auto deleter = [ctx](TElem* ptr) { sycl::free(ptr, ctx); };

            return BufCpu<
                TElem,
                TDim,
                TIdx,
                alpaka::meta::Unique<std::tuple<alpaka::MemVisibility<DevCpu>, alpaka::MemVisibility<TPlatform>>>>(
                host,
                memPtr,
                std::move(deleter),
                extent);
        }
    };

    //! The BufGenericSycl idx type trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPlatform, typename TMemVisibility>
    struct IdxType<BufGenericSycl<TElem, TDim, TIdx, TPlatform, TMemVisibility>>
    {
        using type = TIdx;
    };

    //! The BufCpu pointer on SYCL device get trait specialization.
    template<typename TElem, typename TDim, typename TIdx, typename TPlatform, typename TMemVisibility>
    struct GetPtrDev<BufCpu<TElem, TDim, TIdx, TMemVisibility>, DevGenericSycl<TPlatform>>
    {
        static auto getPtrDev(BufCpu<TElem, TDim, TIdx, TMemVisibility> const& buf, DevGenericSycl<TPlatform> const&)
            -> TElem const*
        {
            return getPtrNative(buf);
        }

        static auto getPtrDev(BufCpu<TElem, TDim, TIdx, TMemVisibility>& buf, DevGenericSycl<TPlatform> const&)
            -> TElem*
        {
            return getPtrNative(buf);
        }
    };

    template<typename TElem, typename TDim, typename TIdx, typename TPlatform, typename TMemVisibility>
    struct MemVisibility<BufGenericSycl<TElem, TDim, TIdx, TPlatform, TMemVisibility>>
    {
        using type = TMemVisibility;
    };
} // namespace alpaka::trait

#    include "alpaka/mem/buf/sycl/Copy.hpp"
#    include "alpaka/mem/buf/sycl/Set.hpp"

#endif
