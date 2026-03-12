package com.ishland.c2me.opts.dfc.mixin.equality;

import net.minecraft.util.math.noise.DoublePerlinNoiseSampler;
import net.minecraft.util.math.noise.OctavePerlinNoiseSampler;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;

import java.util.Objects;

@Mixin(DoublePerlinNoiseSampler.class)
public class MixinDoublePerlinNoiseSampler {

    @Shadow @Final private double amplitude;

    @Shadow @Final private OctavePerlinNoiseSampler firstSampler;

    @Shadow @Final private OctavePerlinNoiseSampler secondSampler;

    @Override
    public boolean equals(Object object) {
        if (this == object) return true;
        if (object == null || getClass() != object.getClass()) return false;
        MixinDoublePerlinNoiseSampler that = (MixinDoublePerlinNoiseSampler) object;
        return Double.compare(amplitude, that.amplitude) == 0 && Objects.equals(firstSampler, that.firstSampler) && Objects.equals(secondSampler, that.secondSampler);
    }

    @Override
    public int hashCode() {
        int result = 1;

        result = 31 * result + Double.hashCode(amplitude);
        result = 31 * result + Objects.hashCode(firstSampler);
        result = 31 * result + Objects.hashCode(secondSampler);

        return result;
    }
}
