package com.ishland.c2me.opts.accel.opencl.common.workarounds.nvidia;

import com.ishland.c2me.opts.accel.opencl.common.enumeration.OpenCLDeviceMetadata;
import com.ishland.c2me.opts.accel.opencl.common.util.CLUtil;
import org.lwjgl.opencl.CL12;
import org.lwjgl.system.MemoryStack;

import java.nio.IntBuffer;

public class NvidiaWorkarounds {

    public static boolean isNvidia(OpenCLDeviceMetadata metadata) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            IntBuffer vendorIdBuf = stack.callocInt(1);
            CLUtil.checkCLError(CL12.clGetDeviceInfo(metadata.devicePtr, CL12.CL_DEVICE_VENDOR_ID, vendorIdBuf, null));
            int vendorId = vendorIdBuf.get(0);
            return vendorId == 0x10de;
        }
    }

}
