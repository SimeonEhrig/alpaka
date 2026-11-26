Using Native Backend Handles
============================

*alpaka* devices and queues are created by *alpaka* through :ref:`device selection <device-selection>` and the ``makeQueue`` method of a ``Device`` object.
Applications that already manage a vendor context and its streams, such as PyTorch, or that exchange work with a third-party library, need to reuse the existing native objects instead.
Linking already existing devices or queues also simplify porting existing CUDA/HIP applications to alpaka.
Creating a second context or stream for the same device breaks the ordering the application established and adds a context switch that is easy to avoid.
The link API adopts an existing native device, stream or queue, so that *alpaka* objects can operate on them.

.. hint::
    There is **no** native handle for the host API, and the ``link*`` methods are therefore not available for it.

Linking a device
----------------

The device selector is created for a concrete API and device kind, as in the :ref:`device selection <device-selection>` section.
Pass the native device to ``linkDevice`` instead of calling ``makeDevice``.
For CUDA and HIP the native device is the device ordinal, for SYCL it is a pair of ``sycl::device`` and ``sycl::context``.

.. literalinclude:: ../../snippets/example/210_nativeHandles.cpp
   :language: cpp
   :start-after: BEGIN-TUTORIAL-generic-device-handle
   :end-before: END-TUTORIAL-generic-device-handle
   :dedent:

Linking a queue
---------------

*alpaka* takes a native stream or queue and wrap it into an *alpaka* object that manages synchronization.
The native stream or queue must be asynchronous.
The queue kinds ``queueKind::blocking`` and ``queueKind::nonBlocking`` are supported.
Therefore, if the queue type is set to ``queueKind::blocking``, *alpaka* performs synchronization after
each *alpaka* API call, even if the underlying native stream is asynchronous.

.. literalinclude:: ../../snippets/example/210_nativeHandles.cpp
   :language: cpp
   :start-after: BEGIN-TUTORIAL-generic-queue-handle
   :end-before: END-TUTORIAL-generic-queue-handle
   :dedent:

The caller owns the native stream or queue.
*alpaka* never destroys or resets an adopted device, stream, or queue.

.. warning::
    Linking the CUDA/HIP legacy default stream is not allowed.

.. hint::
    The ``getNativeHandle`` accessors on the *alpaka* device and queue still return the native objects, so you can pass them to vendor libraries at any point.


Lifetime and synchronization
----------------------------

The *alpaka* device and the *queue* wrap around the native device and the native queue.
The native handle must remain in existence longer than any *alpaka* handle that references it.
In addition, the *alpaka* device and the *alpaka* queue apply their properties to the native device
and the native queue.
For example, when an ``alpaka::Device`` or ``alpaka::Queue`` object is destroyed, it calls the
synchronization function for the associated native device and stream.

.. literalinclude:: ../../snippets/example/210_nativeHandles.cpp
   :language: cpp
   :start-after: BEGIN-TUTORIAL-queue-raii-linked
   :end-before: END-TUTORIAL-queue-raii-linked
   :dedent:


This behavior applies as long as an ``alpaka::Device`` or ``alpaka::Queue`` is associated with a native type.
To avoid synchronization when the ``alpaka::Device`` or ``alpaka::Queue`` is detroyed, you must first
unlink the native type.

.. literalinclude:: ../../snippets/example/210_nativeHandles.cpp
   :language: cpp
   :start-after: BEGIN-TUTORIAL-queue-raii-unlink
   :end-before: END-TUTORIAL-queue-raii-unlink
   :dedent:

Even if you have removed the link to ``alpaka::Device`` or ``alpaka::Queue``, you can still call the
native functions for device synchronization and stream synchronization for the native types.
All tasks including those queued via ``alpaka::Device`` or ``alpaka::Queue`` will be synchronized.

**Important:** When an ``alpaka::Device`` or ``alpaka::Queue`` is destroyed, the underlying native type
is not destroyed, regardless of whether it was still linked or had already been unlinked.

.. hint::

    An unlinked device or queue still refers to the native type.
    Therefore, you can still enqueue new tasks.
    Only control functions such as ``alpaka::wait()`` no longer work.
    For example, if you have registered a queue in an asynchronous host function,
    you can enqueue new tasks at any time.

.. warning::

    ``unlinkDevice()`` and ``unlinkQueue()`` are asynchronous functions.
    Without manual synchronization before unlinking, the device or queue will be in an undefined state.
    After calling the unlink function, the only way to achieve a defined state is to use native functions
    for the respective native type.


Backend notes
-------------

- CUDA and HIP identify a device by an integer, SYCL by a ``(device, context)`` pair.
- A SYCL queue linked as an *alpaka* queue must be in-order.
  ``linkQueue`` throws an exception for an out-of-order queue.
- A SYCL device linked through ``linkDevice`` cannot be mixed with an *alpaka*-managed device on the same platform.
  Linking throws an exception if a managed device for that platform already exists.
- For CUDA and HIP, ``linkDevice`` throws an exception if the device number is already an *alpaka*-managed device, because a managed device cannot be unlinked.

Full examples
-------------

The examples below are the compiled tests.
They create the native handles, link the device and queue, run work, and destroy the native handles after the alpaka wrappers.

CUDA
~~~~

.. literalinclude:: ../../snippets/example/210_nativeHandles.cpp
   :language: cpp
   :start-after: BEGIN-TUTORIAL-nativeCuda
   :end-before: END-TUTORIAL-nativeCuda
   :dedent:

HIP
~~~

.. literalinclude:: ../../snippets/example/210_nativeHandles.cpp
   :language: cpp
   :start-after: BEGIN-TUTORIAL-nativeHip
   :end-before: END-TUTORIAL-nativeHip
   :dedent:

SYCL/oneAPI
~~~~~~~~~~~

.. literalinclude:: ../../snippets/example/210_nativeHandles.cpp
   :language: cpp
   :start-after: BEGIN-TUTORIAL-nativeOneApi
   :end-before: END-TUTORIAL-nativeOneApi
   :dedent:

Complete source file
--------------------

.. raw:: html

   <details class="full-source">
   <summary>210_nativeHandles.cpp</summary>

.. filteredliteralinclude:: ../../snippets/example/210_nativeHandles.cpp
   :language: cpp
   :linenos:

.. raw:: html

   </details>
   <br/>
