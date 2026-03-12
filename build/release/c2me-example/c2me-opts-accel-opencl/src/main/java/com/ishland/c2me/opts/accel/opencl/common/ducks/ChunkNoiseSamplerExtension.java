package com.ishland.c2me.opts.accel.opencl.common.ducks;

import net.minecraft.world.gen.chunk.AquiferSampler;
import net.minecraft.world.gen.chunk.ChunkGeneratorSettings;
import net.minecraft.world.gen.noise.NoiseConfig;

public interface ChunkNoiseSamplerExtension {

    AquiferSampler.FluidLevelSampler c2me$getFluidLevelSampler();

    ChunkGeneratorSettings c2me$getChunkGeneratorSettings();

    NoiseConfig c2me$getNoiseConfig();

}
