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
    private native boolean nativeInputData(long handle, String