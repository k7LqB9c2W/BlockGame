package com.ishland.c2me.opts.accel.opencl.common.workarounds.intel;

import com.ishland.c2me.opts.accel.opencl.common.enumeration.OpenCLDeviceMetadata;
import com.ishland.c2me.opts.accel.opencl.common.util.CLUtil;
import io.netty.util.internal.PlatformDependent;
import org.lwjgl.opencl.CL12;
import org.lwjgl.system.MemoryStack;

import java.nio.IntBuffer;

public class IntelWorkarounds {

    public static boolean isUsingIntelGPU(OpenCLDeviceMetadata metadata) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            IntBuffer vendorIdBuf = stack.callocInt(1);
            CLUtil.checkCLError(CL12.clGetDeviceInfo(metadata.devicePtr, CL12.CL_DEVICE_VENDOR_ID, vendorIdBuf, null));
            int vendorId = vendorIdBuf.get(0);
            return vendorId == 0x8086;
        }
    }

    public static boolean isUsingGen9OnWindows(OpenCLDeviceMetadata metadata) {
        if (!PlatformDependent.isWindows()) return false;
        if (!isUsingIntelGPU(metadata)) return false;

        String deviceName = CLUtil.getDeviceInfoStringUTF8(metadata.devicePtr, CL12.CL_DEVICE_NAME);
        return deviceName.equals("Intel(R) HD Graphics 500") ||
                deviceName.equals("Intel(R) HD Graphics 505") ||
                deviceName.equals("Intel(R) HD Graphics 510") ||
                deviceName.equals("Intel(R) HD Graphics 515") ||
                deviceName.equals("Intel(R) HD Graphics 520") ||
                deviceName.equals("Intel(R) HD Graphics 530") ||
                deviceName.equals("Intel(R) HD Graphics P530") ||
                deviceName.equals("Intel(R) Iris(R) Graphics 540") ||
                deviceName.equals("Intel(R) Iris(R) Graphics 550") ||
                deviceName.equals("Intel(R) Iris(TM) Pro Graphics 580") ||
                deviceName.equals("Intel(R) Iris(TM) Pro Graphics P555") ||
                deviceName.equals("Intel(R) Iris(TM) Pro Graphics P580") ||
                deviceName.equals("Intel(R) HD Graphics 610") ||
                deviceName.equals("Intel(R) HD Graphics 615") ||
                deviceName.equals("Intel(R) HD Graphics 620") ||
                deviceName.equals("Intel(R) HD Graphics 630") ||
                deviceName.equals("Intel(R) HD Graphics P630") ||
                deviceName.equals("Intel(R) Iris(R) Plus Graphics") || // some devices just use this generic name
                deviceName.equals("Intel(R) Iris(R) Plus Graphics 640") ||
                deviceName.equals("Intel(R) Iris(R) Plus Graphics 645") ||
                deviceName.equals("Intel(R) Iris(R) Plus Graphics 650") ||
                deviceName.equals("Intel(R) Iris(R) Plus Graphics 655") ||
                deviceName.equals("Intel(R) UHD Graphics") || // some devices just use this generic name
                deviceName.equals("Intel(R) UHD Graphics 600") ||
                deviceName.equals("Intel(R) UHD Graphics 605") ||
                deviceName.equals("Intel(R) UHD Graphics 610") ||
                deviceName.equals("Intel(R) UHD Graphics 615") ||
                deviceName.equals("Intel(R) UHD Graphics 617") ||
                deviceName.equals("Intel(R) UHD Graphics 620") ||
                deviceName.equals("Intel(R) UHD Graphics 630") ||
                deviceName.equals("Intel(R) UHD Graphics P630");
    }

    public static boolean isUsingIntelOnLinux(OpenCLDeviceMetadata metadata) {
        return !PlatformDependent.isWindows() && isUsingIntelGPU(metadata);
    }

}
