
/* SPDX-License-Identifier: Apache-2.0 */
/*
 * NNStreamer Android API
 * Copyright (C) 2019 Samsung Electronics Co., Ltd.
 */

package org.nnsuite.nnstreamer;

/**
 * Provides interfaces to create a custom-filter in the pipeline.<br>
 * <br>
 * To register a new custom-filter, an application should call
 * {@link #create(String, TensorsInfo, TensorsInfo, Callback)}
 * before constructing the pipeline.
 */
public final class CustomFilter implements AutoCloseable {
    private long mHandle = 0;
    private String mName = null;
    private Callback mCallback = null;

    private native long nativeInitialize(String name, TensorsInfo in, TensorsInfo out);
    private native void nativeDestroy(long handle);

    /**
     * Interface definition for a callback to be invoked while processing the pipeline.
     *
     * @see #create(String, TensorsInfo, TensorsInfo, Callback)
     */
    public interface Callback {
        /**
         * Called synchronously while processing the pipeline.
         *
         * NNStreamer filter invokes the given custom-filter callback while processing the pipeline.
         * Note that, if it is unnecessary to execute the input data, return null to drop the buffer.
         *
         * @param in The input data (a single frame, tensor/tensors)
         *
         * @return The output data (a single frame, tensor/tensors)
         */
        TensorsData invoke(TensorsData in);
    }

    /**
     * Creates new custom-filter with input and output tensors information.
     *
     * NNStreamer processes the tensors with 'custom-easy' framework which can execute without the model file.
     * Note that if given name is duplicated in the pipeline or same name already exists,
     * the registration will be failed and throw an exception.
     *
     * @param name     The name of custom-filter
     * @param in       The input tensors information
     * @param out      The output tensors information
     * @param callback The function to be called while processing the pipeline
     *
     * @return {@link CustomFilter} instance
     *
     * @throws IllegalArgumentException if given param is invalid
     * @throws IllegalStateException if failed to initialize custom-filter
     */
    public static CustomFilter create(String name, TensorsInfo in, TensorsInfo out, Callback callback) {
        return new CustomFilter(name, in, out, callback);
    }

    /**
     * Gets the name of custom-filter.
     *
     * @return The name of custom-filter
     */
    public String getName() {
        return mName;
    }

    /**
     * Internal constructor to create and register a custom-filter.
     *
     * @param name     The name of custom-filter
     * @param in       The input tensors information
     * @param out      The output tensors information
     * @param callback The function to be called while processing the pipeline
     *
     * @throws IllegalArgumentException if given param is invalid
     * @throws IllegalStateException if failed to initialize custom-filter
     */
    private CustomFilter(String name, TensorsInfo in, TensorsInfo out, Callback callback) {
        if (name == null) {
            throw new IllegalArgumentException("Given name is null");
        }

        if (in == null || out == null) {
            throw new IllegalArgumentException("Given info is null");
        }

        if (callback == null) {
            throw new IllegalArgumentException("Given callback is null");
        }

        mHandle = nativeInitialize(name, in, out);
        if (mHandle == 0) {
            throw new IllegalStateException("Failed to initialize custom-filter " + name);
        }

        mName = name;
        mCallback = callback;
    }

    /**
     * Internal method called from native while processing the pipeline.
     */
    private TensorsData invoke(TensorsData in) {
        TensorsData out = null;

        if (mCallback != null) {
            out = mCallback.invoke(in);
        }

        return out;
    }

    @Override
    protected void finalize() throws Throwable {
        try {
            close();
        } finally {
            super.finalize();
        }
    }

    @Override
    public void close() {
        if (mHandle != 0) {
            nativeDestroy(mHandle);
            mHandle = 0;
        }
    }

    /**
     * Private constructor to prevent the instantiation.
     */
    private CustomFilter() {}
}