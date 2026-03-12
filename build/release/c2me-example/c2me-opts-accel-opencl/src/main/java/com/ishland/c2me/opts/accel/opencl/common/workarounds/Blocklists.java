package com.ishland.c2me.opts.accel.opencl.common.workarounds;

import com.ishland.c2me.opts.accel.opencl.common.enumeration.OpenCLDeviceMetadata;
import com.ishland.c2me.opts.accel.opencl.common.workarounds.amd.AMDBlocklists;

import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

public class Blocklists {

    public static Set<Reference> getBlockListReasons(OpenCLDeviceMetadata metadata) {
        EnumSet<Reference> set = EnumSet.noneOf(Reference.class);
        if (AMDBlocklists.isUsingBrokenGCNOnOfficialAMDDriver(metadata)) {
            set.add(Reference.BROKEN_DRIVER_AMD_GCN);
        }
        if (AMDBlocklists.isUsingAM5IntegratedGPU(metadata)) {
            set.add(Reference.SLOW_AMD_AM5_IGPU_GFX1036);
        }
        return Collections.unmodifiableSet(set);
    }

    public enum Reference {

        /**
         * AMD drivers on GCN is known to crash on compilation
         */
        BROKEN_DRIVER_AMD_GCN,

        /**
         * AMD AM5 iGPU gfx1036 is too slow to be useful
         */
        SLOW_AMD_AM5_IGPU_GFX1036,

        ;
    }

}
