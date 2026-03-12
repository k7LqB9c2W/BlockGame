package com.ishland.c2me.opts.accel.opencl.common;

import com.ishland.c2me.base.common.config.ConfigSystem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashSet;
import java.util.UUID;

public class Config {

    private static final Logger LOGGER = LoggerFactory.getLogger(Config.class);

    public static final int maxConcurrentTasksPerDevice = (int) new ConfigSystem.ConfigAccessor()
            .key("openclAccel.maxConcurrentTasksPerDevice")
            .comment("""
                    Maximum number of concurrent tasks per device
                    Increasing this may increase the performance and will increase the VRAM usage
                    """)
            .getLong(32, 32, ConfigSystem.LongChecks.THREAD_COUNT);

    public static final boolean lowPriorityQueues = new ConfigSystem.ConfigAccessor()
            .key("openclAccel.lowPriorityQueues")
            .comment("""
                    Whether to attempt to lower the priority of the command queues
                    This may reduce the fps drop caused by the OpenCL acceleration
                    
                    Requires cl_khr_priority_hints and cl_khr_throttle_hints
                    """)
            .getBoolean(true, true);

    public static final boolean preferFastCompilation = new ConfigSystem.ConfigAccessor()
            .key("openclAccel.preferFastCompilation")
            .comment("""
                    Whether to prefer fast compilation with noinline hints
                    """)
            .getBoolean(true, true);

    public static final boolean allowCPUDevices = new ConfigSystem.ConfigAccessor()
            .key("openclAccel.allowCPUDevices")
            .comment("""
                    Whether to allow the usage of CPU OpenCL devices
                    """)
            .getBoolean(false, false);

    public static final boolean allowGPUDevices = new ConfigSystem.ConfigAccessor()
            .key("openclAccel.allowGPUDevices")
            .comment("""
                    Whether to allow the usage of GPU OpenCL devices
                    """)
            .getBoolean(true, false);

    public static final boolean allowAcceleratorDevices = new ConfigSystem.ConfigAccessor()
            .key("openclAccel.allowAcceleratorDevices")
            .comment("""
                    Whether to allow the usage of Accelerator OpenCL devices
                    """)
            .getBoolean(true, false);

    public static final boolean allowIncompatibilityFallback = new ConfigSystem.ConfigAccessor()
            .key("openclAccel.allowIncompatibilityFallback")
            .comment("""
                    Whether to allow falling back to non-OpenCL world generation if OpenCL initialization fails
                    """)
            .getBoolean(false, false);

    public static final boolean disableBuiltinDeviceBlocklist = new ConfigSystem.ConfigAccessor()
            .key("openclAccel.disableBuiltinDeviceBlocklist")
            .comment("""
                    Overrides the built-in blocklist and enables acceleration on unsupported configurations
                    """)
            .getBoolean(false, false);

    public static final String deviceUUIDBlacklistRaw = new ConfigSystem.ConfigAccessor()
            .key("openclAccel.deviceUUIDBlacklist")
            .comment("""
                    A comma-separated list of device UUIDs to blacklist
                    You can find the UUIDs in the log when the OpenCL devices are enumerated
                    Example: "951e5ce5-ccec-4a37-9ece-d0a800662d8f,3e825b77-fa62-403e-9155-aed1c014b2a2"
                    """)
            .getString("", "");

    public static final String deviceUUIDWhitelistRaw = new ConfigSystem.ConfigAccessor()
            .key("openclAccel.deviceUUIDWhitelist")
            .comment("""
                    A comma-separated list of device UUIDs to whitelist
                    If non-empty, only the devices in this list will be used
                    You can find the UUIDs in the log when the OpenCL devices are enumerated
                    Example: "951e5ce5-ccec-4a37-9ece-d0a800662d8f,3e825b77-fa62-403e-9155-aed1c014b2a2"
                    """)
            .getString("", "");

    public static final HashSet<UUID> deviceUUIDBlacklist = new HashSet<>();
    public static final HashSet<UUID> deviceUUIDWhitelist = new HashSet<>();

    static {
        for (String s : deviceUUIDBlacklistRaw.split(",")) {
            String trimmed = s.trim();
            if (!trimmed.isEmpty()) {
                try {
                    deviceUUIDBlacklist.add(UUID.fromString(trimmed));
                } catch (IllegalArgumentException e) {
                    LOGGER.error("Invalid UUID in openclAccel.deviceUUIDBlacklist: {}", trimmed);
                }
            }
        }
        for (String s : deviceUUIDWhitelistRaw.split(",")) {
            String trimmed = s.trim();
            if (!trimmed.isEmpty()) {
                try {
                    deviceUUIDWhitelist.add(UUID.fromString(trimmed));
                } catch (IllegalArgumentException e) {
                    LOGGER.error("Invalid UUID in openclAccel.deviceUUIDWhitelist: {}", trimmed);
                }
            }
        }
    }

    public static void init() {
        tryChunkyMaxWorkingCount();
    }

    private static void tryChunkyMaxWorkingCount() {
        String chunkyMaxWorkingCount = System.getProperty("chunky.maxWorkingCount", "");
        int value;
        try {
            value = Integer.parseInt(chunkyMaxWorkingCount);
        } catch (NumberFormatException e) {
            value = 0;
        }
        if (value == 0) {
            value = 192;
            System.setProperty("chunky.maxWorkingCount", Integer.toString(value));
        }
        LOGGER.info("chunky.maxWorkingCount: {}", value);
    }

}
