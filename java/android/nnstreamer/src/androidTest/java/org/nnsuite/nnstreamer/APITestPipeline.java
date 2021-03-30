
package org.nnsuite.nnstreamer;

import android.os.Environment;
import android.support.test.rule.GrantPermissionRule;
import android.support.test.runner.AndroidJUnit4;
import android.view.Surface;
import android.view.SurfaceView;

import org.junit.Before;
import org.junit.Ignore;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;

import java.io.File;
import java.nio.file.Files;
import java.nio.ByteBuffer;
import java.util.Arrays;

import static org.junit.Assert.*;

/**
 * Testcases for Pipeline.
 */
@RunWith(AndroidJUnit4.class)
public class APITestPipeline {
    private int mReceived = 0;
    private boolean mInvalidState = false;
    private Pipeline.State mPipelineState = Pipeline.State.NULL;

    private Pipeline.NewDataCallback mSinkCb = new Pipeline.NewDataCallback() {
        @Override
        public void onNewDataReceived(TensorsData data) {
            if (data == null ||
                data.getTensorsCount() != 1 ||
                data.getTensorData(0).capacity() != 200) {
                mInvalidState = true;
                return;
            }

            TensorsInfo info = data.getTensorsInfo();

            /* validate received data (uint8 2:10:10:1) */
            if (info == null ||
                info.getTensorsCount() != 1 ||
                info.getTensorName(0) != null ||
                info.getTensorType(0) != NNStreamer.TensorType.UINT8 ||
                !Arrays.equals(info.getTensorDimension(0), new int[]{2,10,10,1})) {
                /* received data is invalid */
                mInvalidState = true;
            }

            mReceived++;
        }
    };

    @Rule
    public GrantPermissionRule mPermissionRule = APITestCommon.grantPermissions();

    @Before
    public void setUp() {
        APITestCommon.initNNStreamer();

        mReceived = 0;
        mInvalidState = false;
        mPipelineState = Pipeline.State.NULL;
    }

    @Test
    public void enumPipelineState() {
        assertEquals(Pipeline.State.UNKNOWN, Pipeline.State.valueOf("UNKNOWN"));
        assertEquals(Pipeline.State.NULL, Pipeline.State.valueOf("NULL"));
        assertEquals(Pipeline.State.READY, Pipeline.State.valueOf("READY"));
        assertEquals(Pipeline.State.PAUSED, Pipeline.State.valueOf("PAUSED"));
        assertEquals(Pipeline.State.PLAYING, Pipeline.State.valueOf("PLAYING"));
    }

    @Test
    public void testAvailableElement() {
        try {
            assertTrue(Pipeline.isElementAvailable("tensor_converter"));
            assertTrue(Pipeline.isElementAvailable("tensor_filter"));
            assertTrue(Pipeline.isElementAvailable("tensor_transform"));
            assertTrue(Pipeline.isElementAvailable("tensor_sink"));
            assertTrue(Pipeline.isElementAvailable("join"));
            assertTrue(Pipeline.isElementAvailable("amcsrc"));
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testAvailableElementNullName_n() {
        try {
            Pipeline.isElementAvailable(null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testAvailableElementEmptyName_n() {
        try {
            Pipeline.isElementAvailable("");
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testAvailableElementInvalidName_n() {
        try {
            assertFalse(Pipeline.isElementAvailable("invalid-element"));
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testConstructInvalidElement_n() {
        String desc = "videotestsrc ! videoconvert ! video/x-raw,format=RGB ! " +
                "invalidelement ! tensor_converter ! tensor_sink";

        try {
            new Pipeline(desc);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testConstructNullDescription_n() {
        try {
            new Pipeline(null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testConstructEmptyDescription_n() {
        try {
            new Pipeline("");
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testConstructNullStateCb() {
        String desc = "videotestsrc ! videoconvert ! video/x-raw,format=RGB ! " +
                "tensor_converter ! tensor_sink";

        try (Pipeline pipe = new Pipeline(desc, null)) {
            Thread.sleep(100);
            assertEquals(Pipeline.State.PAUSED, pipe.getState());
            Thread.sleep(100);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testConstructWithStateCb() {
        String desc = "videotestsrc ! videoconvert ! video/x-raw,format=RGB ! " +
                "tensor_converter ! tensor_sink";

        /* pipeline state callback */
        Pipeline.StateChangeCallback stateCb = new Pipeline.StateChangeCallback() {
            @Override
            public void onStateChanged(Pipeline.State state) {
                mPipelineState = state;
            }
        };

        try (Pipeline pipe = new Pipeline(desc, stateCb)) {
            Thread.sleep(100);
            assertEquals(Pipeline.State.PAUSED, mPipelineState);

            /* start pipeline */
            pipe.start();
            Thread.sleep(300);

            assertEquals(Pipeline.State.PLAYING, mPipelineState);

            /* stop pipeline */
            pipe.stop();
            Thread.sleep(300);

            assertEquals(Pipeline.State.PAUSED, mPipelineState);
            Thread.sleep(100);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testGetState() {
        String desc = "videotestsrc ! videoconvert ! video/x-raw,format=RGB ! " +
                "tensor_converter ! tensor_sink";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();
            Thread.sleep(300);

            assertEquals(Pipeline.State.PLAYING, pipe.getState());

            /* stop pipeline */
            pipe.stop();
            Thread.sleep(300);

            assertEquals(Pipeline.State.PAUSED, pipe.getState());
            Thread.sleep(100);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testRegisterNullDataCb_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            pipe.registerSinkCallback("sinkx", null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testRegisterDataCbInvalidName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            pipe.registerSinkCallback("invalid_sink", mSinkCb);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testRegisterDataCbNullName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            pipe.registerSinkCallback(null, mSinkCb);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testRegisterDataCbEmptyName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            pipe.registerSinkCallback("", mSinkCb);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testUnregisterNullDataCb_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            pipe.unregisterSinkCallback("sinkx", null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testUnregisterDataCbNullName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            pipe.unregisterSinkCallback(null, mSinkCb);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testUnregisterDataCbEmptyName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            pipe.unregisterSinkCallback("", mSinkCb);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testUnregisteredDataCb_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            pipe.unregisterSinkCallback("sinkx", mSinkCb);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testUnregisterInvalidCb_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* register callback */
            Pipeline.NewDataCallback cb1 = new Pipeline.NewDataCallback() {
                @Override
                public void onNewDataReceived(TensorsData data) {
                    mReceived++;
                }
            };

            pipe.registerSinkCallback("sinkx", cb1);

            /* unregistered callback */
            pipe.unregisterSinkCallback("sinkx", mSinkCb);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testRemoveDataCb() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            /* register sink callback */
            pipe.registerSinkCallback("sinkx", mSinkCb);

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                pipe.inputData("srcx", info.allocate());
                Thread.sleep(50);
            }

            /* pause pipeline and unregister sink callback */
            Thread.sleep(100);
            pipe.stop();

            pipe.unregisterSinkCallback("sinkx", mSinkCb);
            Thread.sleep(100);

            /* start pipeline again */
            pipe.start();

            /* push input buffer again */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                pipe.inputData("srcx", info.allocate());
                Thread.sleep(50);
            }

            /* stop pipeline */
            pipe.stop();

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertEquals(10, mReceived);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testDuplicatedDataCb() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            pipe.registerSinkCallback("sinkx", mSinkCb);

            /* try to register same cb */
            pipe.registerSinkCallback("sinkx", mSinkCb);

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                pipe.inputData("srcx", info.allocate());
                Thread.sleep(50);
            }

            /* pause pipeline and unregister sink callback */
            Thread.sleep(100);
            pipe.stop();

            pipe.unregisterSinkCallback("sinkx", mSinkCb);
            Thread.sleep(100);

            /* start pipeline again */
            pipe.start();

            /* push input buffer again */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                pipe.inputData("srcx", info.allocate());
                Thread.sleep(50);
            }

            /* stop pipeline */
            pipe.stop();

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertEquals(10, mReceived);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testMultipleDataCb() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            /* register three callbacks */
            Pipeline.NewDataCallback cb1 = new Pipeline.NewDataCallback() {
                @Override
                public void onNewDataReceived(TensorsData data) {
                    mReceived++;
                }
            };

            Pipeline.NewDataCallback cb2 = new Pipeline.NewDataCallback() {
                @Override
                public void onNewDataReceived(TensorsData data) {
                    mReceived++;
                }
            };

            pipe.registerSinkCallback("sinkx", mSinkCb);
            pipe.registerSinkCallback("sinkx", cb1);
            pipe.registerSinkCallback("sinkx", cb2);

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                pipe.inputData("srcx", info.allocate());
                Thread.sleep(50);
            }

            /* pause pipeline and unregister sink callback */
            Thread.sleep(100);
            pipe.stop();

            pipe.unregisterSinkCallback("sinkx", mSinkCb);
            pipe.unregisterSinkCallback("sinkx", cb1);
            Thread.sleep(100);

            /* start pipeline again */
            pipe.start();

            /* push input buffer again */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                pipe.inputData("srcx", info.allocate());
                Thread.sleep(50);
            }

            /* stop pipeline */
            pipe.stop();

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertEquals(40, mReceived);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testPushToTensorTransform() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)5:1:1:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_transform mode=arithmetic option=typecast:float32,add:0.5 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{5,1,1,1});

            /* register callback */
            Pipeline.NewDataCallback cb1 = new Pipeline.NewDataCallback() {
                @Override
                public void onNewDataReceived(TensorsData data) {
                    if (data != null) {
                        TensorsInfo info = data.getTensorsInfo();
                        ByteBuffer buffer = data.getTensorData(0);

                        /* validate received data (float32 5:1:1:1) */
                        if (info == null ||
                            info.getTensorsCount() != 1 ||
                            info.getTensorType(0) != NNStreamer.TensorType.FLOAT32 ||
                            !Arrays.equals(info.getTensorDimension(0), new int[]{5,1,1,1})) {
                            /* received data is invalid */
                            mInvalidState = true;
                        }

                        for (int i = 0; i < 5; i++) {
                            float expected = i * 2 + mReceived + 0.5f;

                            if (expected != buffer.getFloat(i * 4)) {
                                mInvalidState = true;
                            }
                        }

                        mReceived++;
                    }
                }
            };

            pipe.registerSinkCallback("sinkx", cb1);

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 10; i++) {
                TensorsData in = info.allocate();
                ByteBuffer buffer = in.getTensorData(0);

                for (int j = 0; j < 5; j++) {
                    buffer.put(j, (byte) (j * 2 + i));
                }

                in.setTensorData(0, buffer);

                pipe.inputData("srcx", in);
                Thread.sleep(50);
            }

            /* pause pipeline and unregister sink callback */
            Thread.sleep(200);
            pipe.stop();

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertEquals(10, mReceived);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testRunModel() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        File model = APITestCommon.getTFLiteImgModel();
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)3:224:224:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_filter framework=tensorflow-lite model=" + model.getAbsolutePath() + " ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{3,224,224,1});

            /* register sink callback */
            pipe.registerSinkCallback("sinkx", new Pipeline.NewDataCallback() {
                @Override
                public void onNewDataReceived(TensorsData data) {
                    if (data == null || data.getTensorsCount() != 1) {
                        mInvalidState = true;
                        return;
                    }

                    TensorsInfo info = data.getTensorsInfo();

                    if (info == null || info.getTensorsCount() != 1) {
                        mInvalidState = true;
                    } else {
                        ByteBuffer output = data.getTensorData(0);

                        if (!APITestCommon.isValidBuffer(output, 1001)) {
                            mInvalidState = true;
                        }
                    }

                    mReceived++;
                }
            });

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 15; i++) {
                /* dummy input */
                pipe.inputData("srcx", TensorsData.allocate(info));
                Thread.sleep(100);
            }

            /* sleep 500 to invoke */
            Thread.sleep(500);

            /* stop pipeline */
            pipe.stop();

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertTrue(mReceived > 0);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testClassificationResult() {
        if (!NNStreamer.isAvailable(NNStreamer.NNFWType.TENSORFLOW_LITE)) {
            /* cannot run the test */
            return;
        }

        File model = APITestCommon.getTFLiteImgModel();
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)3:224:224:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_filter framework=tensorflow-lite model=" + model.getAbsolutePath() + " ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* register sink callback */
            pipe.registerSinkCallback("sinkx", new Pipeline.NewDataCallback() {
                @Override
                public void onNewDataReceived(TensorsData data) {
                    if (data == null || data.getTensorsCount() != 1) {
                        mInvalidState = true;
                        return;
                    }

                    ByteBuffer buffer = data.getTensorData(0);
                    int labelIndex = APITestCommon.getMaxScore(buffer);

                    /* check label index (orange) */
                    if (labelIndex != 951) {
                        mInvalidState = true;
                    }

                    mReceived++;
                }
            });

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            TensorsData in = APITestCommon.readRawImageData();
            pipe.inputData("srcx", in);

            /* sleep 1000 to invoke */
            Thread.sleep(1000);

            /* stop pipeline */
            pipe.stop();

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertTrue(mReceived > 0);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testInputBuffer() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            /* register sink callback */
            pipe.registerSinkCallback("sinkx", mSinkCb);

            /* start pipeline */
            pipe.start();

            /* push input buffer repeatedly */
            for (int i = 0; i < 2048; i++) {
                /* dummy input */
                pipe.inputData("srcx", TensorsData.allocate(info));
                Thread.sleep(20);
            }

            /* sleep 300 to pass input buffers to sink */
            Thread.sleep(300);

            /* stop pipeline */
            pipe.stop();

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertTrue(mReceived > 0);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testInputVideo() {
        String desc = "appsrc name=srcx ! " +
                "video/x-raw,format=RGB,width=320,height=240,framerate=(fraction)0/1 ! " +
                "tensor_converter ! tensor_sink name=sinkx";

        /* For media format, set meta with exact buffer size. */
        TensorsInfo info = new TensorsInfo();
        /* input data : RGB 320x240 */
        info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{3 * 320 * 240});

        try (Pipeline pipe = new Pipeline(desc)) {
            /* register sink callback */
            pipe.registerSinkCallback("sinkx", new Pipeline.NewDataCallback() {
                @Override
                public void onNewDataReceived(TensorsData data) {
                    if (data == null || data.getTensorsCount() != 1) {
                        mInvalidState = true;
                        return;
                    }

                    /* check received data */
                    TensorsInfo info = data.getTensorsInfo();
                    NNStreamer.TensorType type = info.getTensorType(0);
                    int[] dimension = info.getTensorDimension(0);

                    if (type != NNStreamer.TensorType.UINT8) {
                        mInvalidState = true;
                    }

                    if (dimension[0] != 3 || dimension[1] != 320 ||
                        dimension[2] != 240 || dimension[3] != 1) {
                        mInvalidState = true;
                    }

                    mReceived++;
                }
            });

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                pipe.inputData("srcx", TensorsData.allocate(info));
                Thread.sleep(30);
            }

            /* sleep 200 to invoke */
            Thread.sleep(200);

            /* stop pipeline */
            pipe.stop();

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertTrue(mReceived > 0);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testInputAudio() {
        String desc = "appsrc name=srcx ! " +
                "audio/x-raw,format=S16LE,rate=16000,channels=1 ! " +
                "tensor_converter frames-per-tensor=500 ! tensor_sink name=sinkx";

        /* For media format, set meta with exact buffer size. */
        TensorsInfo info = new TensorsInfo();
        /* input data : 16k sample rate, mono, signed 16bit little-endian, 500 samples */
        info.addTensorInfo(NNStreamer.TensorType.INT16, new int[]{500});

        try (Pipeline pipe = new Pipeline(desc)) {
            /* register sink callback */
            pipe.registerSinkCallback("sinkx", new Pipeline.NewDataCallback() {
                @Override
                public void onNewDataReceived(TensorsData data) {
                    if (data == null || data.getTensorsCount() != 1) {
                        mInvalidState = true;
                        return;
                    }

                    /* check received data */
                    TensorsInfo info = data.getTensorsInfo();
                    NNStreamer.TensorType type = info.getTensorType(0);
                    int[] dimension = info.getTensorDimension(0);

                    if (type != NNStreamer.TensorType.INT16) {
                        mInvalidState = true;
                    }

                    if (dimension[0] != 1 || dimension[1] != 500 ||
                        dimension[2] != 1 || dimension[3] != 1) {
                        mInvalidState = true;
                    }

                    mReceived++;
                }
            });

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 10; i++) {
                /* dummy input */
                pipe.inputData("srcx", TensorsData.allocate(info));
                Thread.sleep(30);
            }

            /* sleep 200 to invoke */
            Thread.sleep(200);

            /* stop pipeline */
            pipe.stop();

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertTrue(mReceived > 0);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testInputInvalidName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            /* start pipeline */
            pipe.start();

            pipe.inputData("invalid_src", TensorsData.allocate(info));
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInputNullName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            /* start pipeline */
            pipe.start();

            pipe.inputData(null, TensorsData.allocate(info));
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInputEmptyName_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            /* start pipeline */
            pipe.start();

            pipe.inputData("", TensorsData.allocate(info));
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInputNullData_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            pipe.inputData("srcx", null);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testInputInvalidData_n() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "tensor_sink name=sinkx";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{4,10,10,2});

            TensorsData in = TensorsData.allocate(info);

            /* push data with invalid size */
            pipe.inputData("srcx", in);
            fail();
        } catch (Exception e) {
            /* expected */
        }
    }

    @Test
    public void testSelectSwitch() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "output-selector name=outs " +
                "outs.src_0 ! tensor_sink name=sinkx async=false " +
                "outs.src_1 ! tensor_sink async=false";

        try (Pipeline pipe = new Pipeline(desc)) {
            TensorsInfo info = new TensorsInfo();
            info.addTensorInfo(NNStreamer.TensorType.UINT8, new int[]{2,10,10,1});

            /* register sink callback */
            pipe.registerSinkCallback("sinkx", mSinkCb);

            /* start pipeline */
            pipe.start();

            /* push input buffer */
            for (int i = 0; i < 15; i++) {
                /* dummy input */
                pipe.inputData("srcx", TensorsData.allocate(info));
                Thread.sleep(50);

                if (i == 9) {
                    /* select pad */
                    pipe.selectSwitchPad("outs", "src_1");
                }
            }

            /* sleep 300 to pass all input buffers to sink */
            Thread.sleep(300);

            /* stop pipeline */
            pipe.stop();

            /* check received data from sink */
            assertFalse(mInvalidState);
            assertEquals(10, mReceived);
        } catch (Exception e) {
            fail();
        }
    }

    @Test
    public void testGetSwitchPad() {
        String desc = "appsrc name=srcx ! " +
                "other/tensor,dimension=(string)2:10:10:1,type=(string)uint8,framerate=(fraction)0/1 ! " +
                "output-selector name=outs " +
                "outs.src_0 ! tensor_sink name=sinkx async=false " +
                "outs.src_1 ! tensor_sink async=false";

        try (Pipeline pipe = new Pipeline(desc)) {
            /* start pipeline */
            pipe.start();

            /* get pad list of output-selector */