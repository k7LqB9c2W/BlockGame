package com.ishland.c2me.opts.accel.opencl.common.enumeration;

import com.ishland.c2me.opts.accel.opencl.common.Config;
import com.ishland.c2me.opts.accel.opencl.common.util.CLUtil;
import com.ishland.c2me.opts.accel.opencl.common.workarounds.Blocklists;
import org.lwjgl.PointerBuffer;
import org.lwjgl.opencl.CL;
import org.lwjgl.opencl.CL12;
import org.lwjgl.opencl.CLCapabilities;
import org.lwjgl.system.Configuration;
import org.lwjgl.system.MemoryStack;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

public class OpenCLDeviceLocator {

    private static final Logger LOGGER = LoggerFactory.getLogger(OpenCLDeviceLocator.class);

    private static void tryLoad(String... libNames) {
        Configuration.OPENCL_EXPLICIT_INIT.set(true);

        Throwable t;
        try {
            CL.getFunctionProvider();
            return;
        } catch (Throwable ignored) {
        }

        try {
            CL.create();
            return;
        } catch (Throwable t1) {
            LOGGER.error("{}", t1.toString());
            t = t1;
        }
        for (String libName : libNames) {
            try {
                CL.create(libName);
                LOGGER.info("Successfully loaded OpenCL from {}", libName);
                return;
            } catch (Throwable t1) {
                LOGGER.error("{}", t1.toString());
                t.addSuppressed(t1);
            }
        }
        throw new RuntimeException("Failed to load OpenCL", t);
    }

    public static boolean isAvailable() {
        try {
            tryLoad(
                    "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1",
                    "/usr/lib64/libOpenCL.so.1",
                    "/usr/lib/libOpenCL.so.1",
                    "/run/opengl-driver/lib/libOpenCL.so",
                    "/vendor/lib64/libOpenCL.so"
            );
            return true;
        } catch (Throwable t) {
            LOGGER.error("Failed to initialize OpenCL", t);
            if (!Config.allowIncompatibilityFallback) {
                throw t;
            }
            return false;
        }
    }

    public static List<OpenCLDeviceMetadata> enumerateAll() {
        final List<OpenCLDeviceMetadata> devices = new ArrayList<>();

        if (!isAvailable()) {
            return devices;
        }

        try (final MemoryStack stack = MemoryStack.stackPush()) {
            IntBuffer errorCode = stack.mallocInt(1);

            IntBuffer countTmp = stack.mallocInt(1);
            CLUtil.checkCLError(CL12.clGetPlatformIDs(null, countTmp));
            PointerBuffer platforms = stack.mallocPointer(countTmp.get(0));
            CLUtil.checkCLError(CL12.clGetPlatformIDs(platforms, (IntBuffer) null));

            for (int i = 0; i < platforms.capacity(); ++i) {
                long platform = platforms.get(i);

                CLCapabilities platformCaps;

                String platformName = CLUtil.getPlatformInfoStringUTF8(platform, CL12.CL_PLATFORM_NAME);
                String platformVersion = CLUtil.getPlatformInfoStringUTF8(platform, CL12.CL_PLATFORM_VERSION);

                try {
                    platformCaps = CL.createPlatformCapabilities(platform);
                } catch (Throwable t) {
                    LOGGER.error("Failed to create OpenCL platform capabilities for platform {}", platformName, t);
                    continue;
                }

                LOGGER.info("Found OpenCL platform {} version {}", platformName, platformVersion);

                List<OpenCLDeviceMetadata> devices1;
                try {
                    devices1 = enumeratePlatformDevices(platform, platformCaps);
                } catch (Throwable t) {
                    LOGGER.warn("Failed to enumerate OpenCL devices for platform {}", platformName, t);
                    continue;
                }
                devices.addAll(devices1);
            }
        }

        return devices;
    }

    private static List<OpenCLDeviceMetadata> enumeratePlatformDevices(long platform, CLCapabilities platformCaps) {
        final List<OpenCLDeviceMetadata> devicesList = new ArrayList<>();
        try (final MemoryStack stack = MemoryStack.stackPush()) {
            IntBuffer errorCode = stack.mallocInt(1);

            IntBuffer countTmp = stack.mallocInt(1);
            int clDeviceType = (Config.allowCPUDevices ? CL12.CL_DEVICE_TYPE_CPU : 0) | (Config.allowGPUDevices ? CL12.CL_DEVICE_TYPE_GPU : 0) | (Config.allowAcceleratorDevices ? CL12.CL_DEVICE_TYPE_ACCELERATOR : 0);
            CLUtil.checkCLError(CL12.clGetDeviceIDs(platform, clDeviceType, null, countTmp));
            PointerBuffer devices = stack.callocPointer(countTmp.get(0));
            CLUtil.checkCLError(CL12.clGetDeviceIDs(platform, clDeviceType, devices, (IntBuffer) null));

            for (int i = 0; i < devices.capacity(); ++i) {
                long device = devices.get(i);

                String deviceVendor = CLUtil.getDeviceInfoStringUTF8(device, CL12.CL_DEVICE_VENDOR);
                String deviceName = CLUtil.getDeviceInfoStringUTF8(device, CL12.CL_DEVICE_NAME);
                String deviceVersion = CLUtil.getDeviceInfoStringUTF8(device, CL12.CL_DEVICE_VERSION);
                String deviceExtensions = CLUtil.getDeviceInfoStringUTF8(device, CL12.CL_DEVICE_EXTENSIONS);

                CLCapabilities deviceCaps;
                try {
                    deviceCaps = CL.createDeviceCapabilities(device, platformCaps);
                } catch (Throwable t) {
                    LOGGER.error("Failed to create OpenCL device capabilities for device {}", deviceName, t);
                    continue;
                }

                if (!deviceCaps.OpenCL12) {
                    LOGGER.warn("OpenCL device ({}) version ({}) does not support OpenCL 1.2", deviceName, deviceVersion);
                    continue;
                }
                if (!deviceCaps.cl_khr_fp64) {
                    LOGGER.warn("OpenCL device ({}) version ({}) does not support cl_khr_fp64", deviceName, deviceVersion);
                    continue;
                }

                UUID deviceUUID;
                UUID driverUUID;

                String platformVendor = CLUtil.getPlatformInfoStringUTF8(platform, CL12.CL_PLATFORM_VENDOR);
                String platformName = CLUtil.getPlatformInfoStringUTF8(platform, CL12.CL_PLATFORM_NAME);
                String platformVersion = CLUtil.getPlatformInfoStringUTF8(platform, CL12.CL_PLATFORM_VERSION);
                String platformExtensions = CLUtil.getPlatformInfoStringUTF8(platform, CL12.CL_PLATFORM_EXTENSIONS);

                if (!deviceCaps.cl_khr_device_uuid) {
                    LOGGER.warn("OpenCL device ({}) version ({}) does not support cl_khr_device_uuid, device matching can be unstable", deviceName, deviceVersion);
                    deviceUUID = UUID.nameUUIDFromBytes((deviceVendor + deviceName + deviceVersion + deviceExtensions).getBytes(StandardCharsets.UTF_8));
                    driverUUID = UUID.nameUUIDFromBytes((platformVendor + platformName + platformVersion + platformExtensions).getBytes(StandardCharsets.UTF_8));
                } else {
                    deviceUUID = CLUtil.getDeviceUUID(device);
                    driverUUID = CLUtil.getDriverUUID(device);
                }

                OpenCLDeviceMetadata clDevice = new OpenCLDeviceMetadata(platform, device, platformCaps, deviceCaps, deviceUUID, driverUUID);

                Set<Blocklists.Reference> blockListReasons = Blocklists.getBlockListReasons(clDevice);
                if (!blockListReasons.isEmpty()) {
                    if (Config.disableBuiltinDeviceBlocklist) {
                        LOGGER.error("OpenCL device ({}) version ({}) is blocklisted, but enabling anyways: ", deviceName, deviceVersion);
                        LOGGER.error("[{}]", blockListReasons.stream().map(Enum::name).collect(Collectors.joining(", ")));
                        LOGGER.error("This may cause crashes or other issues later.");
                    } else {
                        LOGGER.warn("OpenCL device ({}) version ({}) is being blocklisted to prevent crashes or other issues: ", deviceName, deviceVersion);
                        LOGGER.warn("[{}]", blockListReasons.stream().map(Enum::name).collect(Collectors.joining(", ")));
                        continue;
                    }
                }

                LOGGER.info("Found OpenCL device ({}) version ({})", deviceName, deviceVersion);
                LOGGER.info("Device UUID: {}", deviceUUID);
                LOGGER.info("Driver UUID: {}", driverUUID);
                devicesList.add(clDevice);
            }
        }

        return devicesList;
    }

}
