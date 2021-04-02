
package org.nnsuite.nnstreamer;

import android.support.test.runner.AndroidJUnit4;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.Assert.*;

/**
 * Testcases for TensorsData.
 */
@RunWith(AndroidJUnit4.class)
public class APITestTensorsData {
    private TensorsData mData;

    @Before
    public void setUp() {
        APITestCommon.initNNStreamer();

        TensorsInfo info = new TensorsInfo();

        info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{100});
        info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{200});
        info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{300});

        mData = TensorsData.allocate(info);
    }

    @After
    public void tearDown() {
        mData.close();
    }

    @Test
    public void testAllocateByteBuffer() {
        try {
            ByteBuffer buffer = TensorsData.allocateByteBuffer(300);

            assertTrue(APITestCommon.isValidBuffer(buffer, 300));
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testAllocate() {
        try {
            TensorsInfo info = new TensorsInfo();

            info.addTensorInfo(NNStreamer.TensorType.INT16, new int[]{2});
            info.addTensorInfo(NNStreamer.TensorType.UINT16, new int[]{2,2});
            info.addTensorInfo(NNStreamer.TensorType.UINT32, new int[]{2,2,2});

            TensorsData data = TensorsData.allocate(info);

            /* index 0: 2 int16 */
            assertTrue(APITestCommon.isValidBuffer(data.getTensorData(0), 4));

            /* index 1: 2:2 uint16 */
            assertTrue(APITestCommon.isValidBuffer(data.getTensorData(1), 8));

            /* index 2: 2:2:2 uint32 */
            assertTrue(APITestCommon.isValidBuffer(data.getTensorData(2), 32));
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testAllocateEmptyInfo_n() {
        try {
            TensorsInfo info = new TensorsInfo();

            TensorsData.allocate(info);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testAllocateNullInfo_n() {
        try {
            TensorsData.allocate(null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testGetData() {
        try {
            assertTrue(APITestCommon.isValidBuffer(mData.getTensorData(0), 100));
            assertTrue(APITestCommon.isValidBuffer(mData.getTensorData(1), 200));
            assertTrue(APITestCommon.isValidBuffer(mData.getTensorData(2), 300));
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testSetData() {