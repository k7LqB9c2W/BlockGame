package com.ishland.c2me.opts.accel.opencl;

import com.ishland.c2me.base.common.config.ConfigSystem;
import com.ishland.c2me.opts.accel.opencl.common.Config;

public class ModuleEntryPoint {

    private static final boolean enabled = new ConfigSystem.ConfigAccessor()
            .key("openclAccel.enabled")
            .comment("""
                    Enable OpenCL acceleration for world generation.
                    """)
            .getBoolean(true, false);

    static {
        Config.init();
    }

}
