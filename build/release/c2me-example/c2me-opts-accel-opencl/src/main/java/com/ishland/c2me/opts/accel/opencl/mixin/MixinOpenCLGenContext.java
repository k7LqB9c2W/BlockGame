package com.ishland.c2me.opts.accel.opencl.mixin;

import com.ishland.c2me.opts.accel.opencl.ModuleEntryPoint;
import com.ishland.c2me.opts.accel.opencl.common.gen.CLDataUtil;
import com.ishland.c2me.opts.dfc.common.gen.opencl.GeneratedCLSource;
import com.ishland.c2me.opts.dfc.common.gen.opencl.OpenCLGen;
import com.llamalad7.mixinextras.injector.ModifyReturnValue;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;

import java.io.IOException;
import java.io.InputStream;

@Mixin(OpenCLGen.Context.class)
public class MixinOpenCLGenContext {

    @ModifyReturnValue(method = "build", at = @At("RETURN"), remap = false)
    private static GeneratedCLSource modifySource(GeneratedCLSource original) {
        try (InputStream in = ModuleEntryPoint.class.getClassLoader().getResourceAsStream("clsources/c2me_opencl_ext_math.cl")) {
            if (in == null) throw new NullPointerException("Resource not found");
            String header = new String(in.readAllBytes());
            return new GeneratedCLSource(
                    original.getOrdinal(),
                    header + original.getGeneratedSource(),
                    original.getConstData(),
                    CLDataUtil.transformGlobalDynamicDataOffsets(original.getGlobalDynamicDataOffsets()),
                    original.getFlatCachePrefills(),
                    original.getCache2dPrefills(),
                    original.getInterpolatorPrefills(),
                    original.getDefines(),
                    original.getBiomeMappings(),
                    original.getDumpedPath()
            );
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}
