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
     * This callback can be registered only when constructing the p