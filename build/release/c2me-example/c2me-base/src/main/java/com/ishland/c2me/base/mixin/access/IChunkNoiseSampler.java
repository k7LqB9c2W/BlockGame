package com.ishland.c2me.base.mixin.access;

import net.minecraft.block.BlockState;
import net.minecraft.world.gen.chunk.ChunkNoiseSampler;
import net.minecraft.world.gen.densityfunction.DensityFunctionTypes;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;
import org.spongepowered.asm.mixin.gen.Invoker;

@Mixin(ChunkNoiseSampler.class)
public interface IChunkNoiseSampler {

    @Accessor
    int getStartBlockX();

    @Accessor
    int getStartBlockY();

    @Accessor
    int getStartBlockZ();

    @Accessor
    int getHorizontalCellBlockCount();

    @Accessor
    int getVerticalCellBlockCount();

    @Accessor
    boolean getIsInInterpolationLoop();

    @Accessor
    boolean getIsSamplingForCaches();

    @Accessor
    int getStartBiomeX();

    @Accessor
    int getStartBiomeZ();

    @Accessor
    int getHorizontalCellCount();

    @Accessor
    int getVerticalCellCount();

    @Accessor
    int getMinimumCellY();

    @Accessor
    int getCellBlockX();

    @Accessor
    int getCellBlockY();

    @Accessor
    int getCellBlockZ();

    @Invoker
    BlockState invokeSampleBlockState();

    @Accessor
    int getHorizontalBiomeEnd();

    @Accessor
    DensityFunctionTypes.Beardifying getBeardifying();

    @Accessor
    int getStartCellX();

    @Accessor
    int getStartCellZ();

}
