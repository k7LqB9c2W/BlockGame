package com.ishland.c2me.opts.accel.opencl.mixin.workarounds;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.ModifyArg;

@Mixin(value = org.lwjgl.opencl.CL.class, remap = false)
public class MixinCL {

    @ModifyArg(method = "createPlatformCapabilities", at = @At(value = "INVOKE", target = "Lorg/lwjgl/opencl/CL10;nclGetDeviceIDs(JJIJJ)I"), index = 1, require = 2)
    private static long modifyDeviceType(long device_type) {
        if (device_type == 0xffffffffffffffffL) {
            // this is a workaround for java sign-extending the device type
            return 0xffffffffL;
        }
        return device_type;
    }

}
