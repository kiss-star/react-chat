/* SPDX-License-Identifier: Apache-2.0 */
/*
 * NNStreamer Android API
 * Copyright (C) 2019 Samsung Electronics Co., Ltd.
 */

package org.nnsuite.nnstreamer;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Provides interfaces to create and execute stream pipelines with neural networks.<br>
 * <br>
 * {@link Pipeline} allows the following operations with NNStreamer:<br>
 * - Create a stream pipeline with NNStreamer plugins, GStreamer plugins.<br>
 * - Interfaces to push data to the pipeline from the application.<br>
 * - Interfaces to pull data from the pipeline to the application.<br>
 * - Interfaces to start/stop/destroy the pipeline.<br>
 * - Interfaces to control switches and valves in the pipeline.<br>
 */
public final class Pipeline implements AutoCloseable {
    private long mHandle = 0;
    private HashMap<String, ArrayList<NewDataCallback>> mSinkCallbacks = new HashMap<>();
    private StateChangeCallback mStateCallback = null;

    private static native boolean nativeCheckElementAvailability(String element);
    private native long nativeConstruct(String description, boolean addStateCb);
    private native void nativeDestroy(long handle);
    private native boolean nativeStart(long handle);
    private native boolean nativeStop(long handle);
    private native boolean nativeFlush(long handle, boolean start);
    private native int nativeGetState(long handle);
    private native boolean nativeInputData(long handle, String name, TensorsData data);
    private native String[] nativeGetSwitchPads(long handle, String name);
    private native boolean nativeSelectSwitchPad(long handle, String name, String pad);
    private native boolean nativeControlValve(long handle, String name, boolean open);
    private native boolean nativeAddSinkCallback(long handle, String name);
    private native boolean nativeRemoveSinkCallback(long handle, String name);
    private native boolean nativeInitializeSurface(long handle, String name, Object surface);
    private native boolean nativeFinalizeSurface(long handle, String name);

    /**
     * Interface definition for a callback to be invoked when a sink node receives new data.
     *
     * @see #registerSinkCallback(String, NewDataCallback)
     */
    public interface NewDataCallback {
        /**
         * Called when a sink node receives new data.
         *
         * If an application wants to accept data outputs of an NNStreamer stream, use this callback to get data from the stream.
         * Note that this is synchronously called and the buffer may be deallocated after the callback is finished.
         * Thus, if you need the data afterwards, copy the data to another buffer and return fast.
         * Do not spend too much time in the callback. It is recommended to use very small tensors at sinks.
         *
         * @param data The output data (a single frame, tensor/tensors)
         */
        void onNewDataReceived(TensorsData data);
    }

    /**
     * Interface definition for a callback to be invoked when the pipeline state is changed.
     * This callback can be registered only when constructing the pipeline.
     *
     * @see State
     * @see #start()
     * @see #stop()
     * @see Pipeline#Pipeline(String, StateChangeCallback)
     */
    public interface StateChangeCallback {
        /**
         * Called when the pipeline state is changed.
         *
         * If an application wants to get the change of pipeline state, use this callback.
         * This callback can be registered when constructing the pipeline.
         * This is synchronously called, so do not spend too much time in the callback.
         *
         * @param state The changed state
         */
        void onStateChanged(Pipeline.State state);
    }

    /**
     * The enumeration for pipeline state.
     * Refer to <a href="https://gstreamer.freedesktop.org/documentation/plugin-development/basics/states.html">GStreamer states</a> for the details.
     */
    public enum State {
        /**
         * Unknown state.
         */
        UNKNOWN,
        /**
         * Initial state of the pipeline.
         */
        NULL,
        /**
         * The pipeline is ready to go to PAUSED.
         */
        READY,
        /**
         * The pipeline is stopped, ready to accept and process data.
         */
        PAUSED,
        /**
         * The pipeline is started and the data is flowing.
         */
        PLAYING
    }

    /**
     * Creates a new {@link Pipeline} instance with the given pipeline description.
     *
     * @param description The pipeline description. Refer to GStreamer manual or
     *                    <a href="https://github.com/nnstreamer/nnstreamer">NNStreamer</a> documentation for examples and the grammar.
     *
     * @throws IllegalArgumentException if given param is null
     * @throws IllegalStateException if failed to construct the pipeline
     */
    public Pipeline(String description) {
        this(description, null);
    }

    /**
     * Creates a new {@link Pipeline} instance with the given pipeline description.
     *
     * @param description The pipeline description. Refer to GStreamer manual or
     *                    <a href="https://github.com/nnstreamer/nnstreamer">NNStreamer</a> documentation for examples and the grammar.
     * @param callback    The function to be called when the pipeline state is changed.
     *                    You may set null if it is not required.
     *
     * @throws IllegalArgumentException if given param is invalid
     * @throws IllegalStateException if failed to construct the pipeline
     */
    public Pipeline(String description, StateChangeCallback callback) {
        if (description == null || description.isEmpty()) {
            throw new IllegalArgumentException("Given description is invalid");
        }

        mStateCallback = callback;

        mHandle = nativeConstruct(description, (callback != null));
        if (mHandle == 0) {
            throw new IllegalStateException("Failed to construct the pipeline");
        }
    }

    /**
     * Checks the element is registered and available on the pipeline.
     *
     * @param element The name of GStreamer element
     *
     * @return true if the element is available
     *
     * @throws IllegalArgumentException if given param is invalid
     */
    public static boolean isElementAvailable(String element) {
        if (element == null || element.isEmpty()) {
            throw new IllegalArgumentException("Given 