package com.ishland.c2me.opts.accel.opencl.common.workarounds;

import com.ishland.c2me.opts.accel.opencl.common.enumeration.OpenCLDeviceMetadata;
import com.ishland.c2me.opts.accel.opencl.common.workarounds.intel.IntelWorkarounds;
import com.ishland.c2me.opts.accel.opencl.common.workarounds.nvidia.NvidiaWorkarounds;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

public class Workarounds {

    public static Set<Reference> getWorkarounds(OpenCLDeviceMetadata metadata) {
        EnumSet<Reference> set = EnumSet.noneOf(Reference.class);
        if (IntelWorkarounds.isUsingGen9OnWindows(metadata)) {
            set.add(Reference.BUILTIN_TRAP_BROKEN);
        }
        if (IntelWorkarounds.isUsingIntelOnLinux(metadata)) {
            set.add(Reference.INTEL_LINUX_CLEANUP_HANG);
        }
        if (Boolean.getBoolean("com.ishland.c2me.opts.accel.opencl.markBuiltinTrapBroken")) {
            set.add(Reference.BUILTIN_TRAP_BROKEN);
        }
        if (NvidiaWorkarounds.isNvidia(metadata)) {
            set.add(Reference.NVIDIA_INCOMPLETE_CL30_IMPLEMENTATION);
        }
        return Collections.unmodifiableSet(set);
    }

    public enum Reference {

        /**
         * __builtin_trap(); causes some drivers to crash, such as Intel Gen9 iGPUs on Windows.
         */
        BUILTIN_TRAP_BROKEN,

        /**
         * Nvidia ships incomplete OpenCL 3.0 implementation
         * Stuff broken:
         * - -cl-no-subgroup-ifp being thrown as error
         */
        NVIDIA_INCOMPLETE_CL30_IMPLEMENTATION,

        /**
         * The Intel compute driver hangs on cleanup during exit due to JVM running onexit hook with lock
         * and when the driver is releasing the executor, lwjgl tries to acquire the lock causing a deadlock.
         * This intentionally leaks a user event to avoid running the shutdown sequence at the cost of no hotplugging.
         */
        INTEL_LINUX_CLEANUP_HANG,

        ;
    }

}
