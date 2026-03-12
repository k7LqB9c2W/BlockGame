package com.ishland.c2me.base.mixin.access;

import net.minecraft.util.math.random.RandomSplitter;
import net.minecraft.world.gen.chunk.AquiferSampler;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;
import org.spongepowered.asm.mixin.gen.Invoker;

@Mixin(AquiferSampler.Impl.class)
public interface IAquiferSamplerImpl {

    @Accessor
    int getStartX();

    @Accessor
    int getStartY();

    @Accessor
    int getStartZ();

    @Accessor
    int getSizeX();

    @Accessor
    int getSizeZ();

    @Accessor
    long[] getBlockPositions();

    @Accessor
    RandomSplitter getRandomDeriver();

    @Accessor
    AquiferSampler.FluidLevelSampler getFluidLevelSampler();

    @Accessor(value = "field_61452")
    int getSamplingYLowPassCutoff();

    @Invoker
    static int invokeGetLocalX(int i) {
        throw new AbstractMethodError();
    }

    @Invoker
    static int invokeGetLocalY(int i) {
        throw new AbstractMethodError();
    }

    @Invoker
    static int invokeGetLocalZ(int i) {
        throw new AbstractMethodError();
    }

}
