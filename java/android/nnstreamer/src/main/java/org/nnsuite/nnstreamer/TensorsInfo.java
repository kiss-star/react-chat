/* SPDX-License-Identifier: Apache-2.0 */
/*
 * NNStreamer Android API
 * Copyright (C) 2019 Samsung Electronics Co., Ltd.
 */

package org.nnsuite.nnstreamer;

import java.util.ArrayList;

/**
 * Provides interfaces to handle tensors information.
 *
 * @see NNStreamer#TENSOR_RANK_LIMIT
 * @see NNStreamer#TENSOR_SIZE_LIMIT
 * @see NNStreamer.TensorType
 */
public final class TensorsInfo implements AutoCloseable, Cloneable {
    private ArrayList<TensorInfo> mInfoList = new ArrayList<>();

    /**
     * Allocates a new {@link TensorsData} instance with the tensors information.
     *
     * @return {@link TensorsData} instance
     *
     * @throws IllegalStateException if tensors info is empty
     */
    public TensorsData allocate() {
        if (getTensorsCount() == 0) {
            throw new IllegalStateException("Empty tensor info");
        }

        return TensorsData.allocate(this);
    }

    /**
     * Creates a new {@link TensorsInfo} instance cloned from the current tensors information.
     *
     * @return {@link TensorsInfo} instance
     */
    @Override
    public TensorsInfo clone() {
        TensorsInfo cloned = new TensorsInfo();

        for (TensorInfo info : mInfoList) {
            cloned.addTensorInfo(info.getName(), info.getType(), info.getDimension());
        }

        return cloned;
    }

    /**
     * Gets the number of tensors.
     * The maximum number of tensors is {@link NNStreamer#TENSOR_SIZE_LIMIT}.
     *
     * @return The number of tensors
     */
    public int getTensorsCount() {
        return mInfoList.size();
    }

    /**
     * Adds a new tensor information.
     *
     * @param type      The tensor data type
     * @param dimension The tensor dimension
     *
     * @throws IndexOutOfBoundsException when the maximum number of tensors in the list
     * @throws IllegalArgumentException if given param is null or invalid
     */
    public voi