package com.ishland.c2me.opts.dfc.mixin.equality;

import it.unimi.dsi.fastutil.doubles.DoubleList;
import net.minecraft.util.math.noise.OctavePerlinNoiseSampler;
import net.minecraft.util.math.noise.PerlinNoiseSampler;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;

import java.util.Arrays;
import java.util.Objects;

@Mixin(OctavePerlinNoiseSampler.class)
public class MixinOctavePerlinNoiseSampler {

    @Shadow @Final private PerlinNoiseSampler[] octaveSamplers;

    @Shadow @Final private DoubleList amplitudes;

    @Shadow @Final private double persistence;

    @Shadow @Final private double lacunarity;

    @Override
    public boolean equals(Object object) {
        if (this == object) return true;
        if (object == null || getClass() != object.getClass()) return false;
        MixinOctavePerlinNoiseSampler that = (MixinOctavePerlinNoiseSampler) object;
        return Double.compare(persistence, that.persistence) == 0 && Double.compare(lacunarity, that.lacunarity) == 0 && Arrays.equals(octaveSamplers, that.octaveSamplers) && Arrays.equals(amplitudes.toDoubleArray(), that.amplitudes.toDoubleArray());
    }

    @Override
    public int hashCode() {
        int result = 1;
        result = 31 * result + Arrays.hashCode(octaveSamplers);
        result = 31 * result + Arrays.hashCode(amplitudes.toDoubleArray());
        result = 31 * result + Double.hashCode(persistence);
        result = 31 * result + Double.hashCode(lacunarity);
        return result;
    }
}
