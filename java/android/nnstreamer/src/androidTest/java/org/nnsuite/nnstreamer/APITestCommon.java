
package org.nnsuite.nnstreamer;

import android.Manifest;
import android.content.Context;
import android.os.Environment;
import android.support.test.InstrumentationRegistry;
import android.support.test.rule.GrantPermissionRule;
import android.support.test.runner.AndroidJUnit4;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.net.ServerSocket;

import static org.junit.Assert.*;

/**
 * Common definition to test NNStreamer API.
 *
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class APITestCommon {
    private static boolean mInitialized = false;

    /**
     * Initializes NNStreamer API library.
     */
    public static void initNNStreamer() {
        if (!mInitialized) {
            try {
                Context context = InstrumentationRegistry.getTargetContext();
                mInitialized = NNStreamer.initialize(context);
            } catch (Exception e) {
                fail();
            }
        }
    }

    /**
     * Gets the context for the test application.
     */
    public static Context getContext() {
        return InstrumentationRegistry.getTargetContext();
    }

    /**
     * Grants required runtime permissions.
     */
    public static GrantPermissionRule grantPermissions() {
        return GrantPermissionRule.grant(Manifest.permission.READ_EXTERNAL_STORAGE);
    }

    /**
     * Gets the File object of tensorflow-lite model.
     * Note that, to invoke model in the storage, the permission READ_EXTERNAL_STORAGE is required.
     */
    public static File getTFLiteImgModel() {
        String root = Environment.getExternalStorageDirectory().getAbsolutePath();
        File model = new File(root + "/nnstreamer/test/imgclf/mobilenet_v1_1.0_224_quant.tflite");
        File meta = new File(root + "/nnstreamer/test/imgclf/metadata/MANIFEST");

        if (!model.exists() || !meta.exists()) {
            fail();
        }

        return model;
    }

    /**
     * Reads raw image file (orange) and returns TensorsData instance.
     */
    public static TensorsData readRawImageData() {
        String root = Environment.getExternalStorageDirectory().getAbsolutePath();
        File raw = new File(root + "/nnstreamer/test/orange.raw");

        if (!raw.exists()) {
            fail();
        }

        TensorsInfo info = new TensorsInfo();
        info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{3,224,224,1});

        int size = info.getTensorSize(0);
        TensorsData data = TensorsData.allocate(info);

        try {
            byte[] content = Files.readAllBytes(raw.toPath());
            if (content.length != size) {
                fail();
            }

            ByteBuffer buffer = TensorsData.allocateByteBuffer(size);
            buffer.put(content);

            data.setTensorData(0, buffer);
        } catch (Exception e) {
            fail();
        }

        return data;
    }

    /**
     * Gets the label index with max score, for tensorflow-lite image classification.
     */
    public static int getMaxScore(ByteBuffer buffer) {
        int index = -1;
        int maxScore = 0;

        if (isValidBuffer(buffer, 1001)) {
            for (int i = 0; i < 1001; i++) {
                /* convert unsigned byte */
                int score = (buffer.get(i) & 0xFF);

                if (score > maxScore) {
                    maxScore = score;
                    index = i;
                }
            }
        }

        return index;
    }

    /**
     * Reads raw float image file (orange) and returns TensorsData instance.
     */
    public static TensorsData readRawImageDataPytorch() {
        String root = Environment.getExternalStorageDirectory().getAbsolutePath();
        File raw = new File(root + "/nnstreamer/pytorch_data/orange_float.raw");

        if (!raw.exists()) {
            fail();
        }

        TensorsInfo info = new TensorsInfo();
        info.addTensorInfo(NNStreamer.TensorType.FLOAT32, new int[]{224, 224, 3, 1});

        int size = info.getTensorSize(0);
        TensorsData data = TensorsData.allocate(info);

        try {
            byte[] content = Files.readAllBytes(raw.toPath());
            if (content.length != size) {
                fail();
            }

            ByteBuffer buffer = TensorsData.allocateByteBuffer(size);
            buffer.put(content);

            data.setTensorData(0, buffer);
        } catch (Exception e) {
            fail();
        }

        return data;
    }

    /**
     * Reads raw float image file (orange) and returns TensorsData instance.
     */
    public static TensorsData readRawImageDataSNAP() {
        String root = Environment.getExternalStorageDirectory().getAbsolutePath();
        File raw = new File(root + "/nnstreamer/imgclf/orange_float_tflite.raw");

        if (!raw.exists()) {
            fail();
        }

        TensorsInfo info = new TensorsInfo();
        info.addTensorInfo(NNStreamer.TensorType.FLOAT32, new int[]{3, 224, 224, 1});