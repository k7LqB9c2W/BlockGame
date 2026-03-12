package com.ishland.c2me.base.mixin.access;

import net.minecraft.util.math.noise.SimplexNoiseSampler;
import net.minecraft.world.gen.densityfunction.DensityFunctionTypes;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(DensityFunctionTypes.EndIslands.class)
public interface IDensityFunctionTypesEndIslands {

    @Accessor
    SimplexNoiseSampler getSampler();

}
