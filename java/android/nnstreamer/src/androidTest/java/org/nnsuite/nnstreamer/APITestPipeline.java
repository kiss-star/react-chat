
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