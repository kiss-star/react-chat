
package org.nnsuite.nnstreamer;

import android.os.Environment;
import android.support.test.rule.GrantPermissionRule;
import android.support.test.runner.AndroidJUnit4;

import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.File;
import java.nio.ByteBuffer;

import static org.junit.Assert.*;

/**
 * Testcases for SingleShot.
 */
@RunWith(AndroidJUnit4.class)
public class APITestSingleShot {
    @Rule
    public GrantPermissionRule mPermissionRule = APITestCommon.grantPermissions();

    @Before
    public void setUp() {
        APITestCommon.initNNStreamer();
    }

    @Test
    public void testOptionsInvalidFW_n () {
        try {
            new SingleShot.Options(null, APITestCommon.getTFLiteImgModel());
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testOptionsUnknownFW_n () {
        try {
            new SingleShot.Options(NNStreamer.NNFWType.UNKNOWN, APITestCommon.getTFLiteImgModel());
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testOptionsInvalidModelFile_n () {
        try {
            File f = null;
            new SingleShot.Options(NNStreamer.NNFWType.TENSORFLOW_LITE, f);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testGetInputInfo() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel());
            TensorsInfo info = single.getInputInfo();

            /* input: uint8 3:224:224:1 */
            assertEquals(1, info.getTensorsCount());
            assertEquals(NNStreamer.TensorType.UINT8, info.getTensorType(0));
            assertArrayEquals(new int[]{3,224,224,1}, info.getTensorDimension(0));

            single.close();
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testGetOutputInfo() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel());
            TensorsInfo info = single.getOutputInfo();

            /* output: uint8 1001:1 */
            assertEquals(1, info.getTensorsCount());
            assertEquals(NNStreamer.TensorType.UINT8, info.getTensorType(0));
            assertArrayEquals(new int[]{1001,1,1,1}, info.getTensorDimension(0));

            single.close();
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testSetNullInputInfo_n() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel());
            single.setInputInfo(null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetInvalidInputInfo_n() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel());

            TensorsInfo newInfo = new TensorsInfo();
            newInfo.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,2,2,2});

            single.setInputInfo(newInfo);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSetInputInfo() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteAddModel());
            TensorsInfo info = single.getInputInfo();

            /* input: float32 with dimension 1 */
            assertEquals(1, info.getTensorsCount());
            assertEquals(NNStreamer.TensorType.FLOAT32, info.getTensorType(0));
            assertArrayEquals(new int[]{1,1,1,1}, info.getTensorDimension(0));

            TensorsInfo newInfo = new TensorsInfo();
            newInfo.addTensorInfo(NNStreamer.TensorType.FLOAT32, new int[]{10});

            single.setInputInfo(newInfo);

            info = single.getInputInfo();
            /* input: float32 with dimension 10 */
            assertEquals(1, info.getTensorsCount());
            assertEquals(NNStreamer.TensorType.FLOAT32, info.getTensorType(0));
            assertArrayEquals(new int[]{10,1,1,1}, info.getTensorDimension(0));

            info = single.getOutputInfo();
            /* output: float32 with dimension 10 */
            assertEquals(1, info.getTensorsCount());
            assertEquals(NNStreamer.TensorType.FLOAT32, info.getTensorType(0));
            assertArrayEquals(new int[]{10,1,1,1}, info.getTensorDimension(0));

            single.close();
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testInvoke() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel());
            TensorsInfo info = single.getInputInfo();

            /* let's ignore timeout (set 10 sec) */
            single.setTimeout(10000);

            /* single-shot invoke */
            for (int i = 0; i < 600; i++) {
                /* dummy input */
                TensorsData out = single.invoke(info.allocate());

                /* output: uint8 1001:1 */
                assertEquals(1, out.getTensorsCount());
                assertEquals(1001, out.getTensorData(0).capacity());

                Thread.sleep(30);
            }

            single.close();
        } catch (Exception e) {
            fail();
        }
    }

    /**
     * Run image classification and validate result.
     */
    private void runImageClassification(NNStreamer.NNFWType fw, String custom) {
        try {
            SingleShot single = new SingleShot(APITestCommon.getTFLiteImgModel(), fw, custom);

            /* single-shot invoke */
            TensorsData in = APITestCommon.readRawImageData();
            TensorsData out = single.invoke(in);
            int labelIndex = APITestCommon.getMaxScore(out.getTensorData(0));

            /* check label index (orange) */
            if (labelIndex != 951) {
                fail();
            }

            single.close();
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testClassificationResultTFLite() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        runImageClassification(NNStreamer.NNFWType.TENSORFLOW_LITE, null);

        /* classification with delegates requires tensorflow-lite 2.3.0 */
        if (android.os.Build.VERSION.SDK_INT >= 29) {
            /* NNAPI supports AHardwareBuffer in Android 10. */
            runImageClassification(NNStreamer.NNFWType.TENSORFLOW_LITE, "Delegate:NNAPI");
            runImageClassification(NNStreamer.NNFWType.TENSORFLOW_LITE, "Delegate:GPU");
            runImageClassification(NNStreamer.NNFWType.TENSORFLOW_LITE, "Delegate:XNNPACK");
        }
    }

    @Test
    public void testClassificationResultNNFW() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.NNFW)) {
            /* cannot run the test */
            return;
        }

        runImageClassification(NNStreamer.NNFWType.NNFW, null);
    }

    /**
     * Run dynamic invoke with add.tflite model.
     */
    private void runInvokeDynamic(NNStreamer.NNFWType fw) {
        try {
            File model = APITestCommon.getTFLiteAddModel();
            SingleShot single = new SingleShot(model, fw);
            TensorsInfo info = single.getInputInfo();

            /* single-shot invoke */
            for (int i = 1; i < 5; i++) {
                /* change input information */
                info.setTensorDimension(0, new int[]{i,1,1,1});
                single.setInputInfo(info);

                TensorsData input = TensorsData.allocate(info);
                ByteBuffer inBuffer = input.getTensorData(0);

                for (int j = 0; j < i; j++) {
                    inBuffer.putFloat(j * 4, j + 1.5f);
                }

                input.setTensorData(0, inBuffer);

                /* invoke */
                TensorsData output = single.invoke(input);

                /* output: float32 i:1:1:1 */
                assertEquals(1, output.getTensorsCount());

                ByteBuffer outBuffer = output.getTensorData(0);
                assertEquals(i * Float.BYTES, outBuffer.capacity());

                for (int j = 0; j < i; j++) {
                    float expected = j + 3.5f;
                    assertEquals(expected, outBuffer.getFloat(j * 4), 0.0f);
                }

                Thread.sleep(30);
            }

            single.close();
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testInvokeDynamicTFLite() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }