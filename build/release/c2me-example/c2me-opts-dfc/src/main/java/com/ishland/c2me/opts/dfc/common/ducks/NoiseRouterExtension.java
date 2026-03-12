package com.ishland.c2me.opts.dfc.common.ducks;

import net.minecraft.world.gen.densityfunction.DensityFunction;
import net.minecraft.world.gen.noise.NoiseRouter;

public interface NoiseRouterExtension {

    DensityFunction c2me$getFinalFinalDensity();

    void c2me$setFinalFinalDensity(DensityFunction densityFunction);

    void c2me$setOriginalNoiseRouter(NoiseRouter originalNoiseRouter);

    NoiseRouter c2me$getOriginalNoiseRouter();

}
