package com.ishland.c2me.opts.dfc.mixin.equality;

import net.minecraft.registry.entry.RegistryEntry;
import net.minecraft.util.math.noise.DoublePerlinNoiseSampler;
import net.minecraft.world.gen.densityfunction.DensityFunction;
import org.jetbrains.annotations.Nullable;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;

import java.util.Objects;

@Mixin(DensityFunction.Noise.class)
public class MixinDensityFunctionNoise {

    @Shadow @Final private @Nullable DoublePerlinNoiseSampler noise;

    @Shadow @Final private RegistryEntry<DoublePerlinNoiseSampler.NoiseParameters> noiseData;

    @Override
    public boolean equals(Object object) {
        if (this == object) return true;
        if (object == null || getClass() != object.getClass()) return false;
        MixinDensityFunctionNoise that = (MixinDensityFunctionNoise) object;
        return (this.noise != null || that.noise != null) ? Objects.equals(noise, that.noise) : Objects.equals(noiseData, that.noiseData);
    }

    @Override
    public int hashCode() {
        if (this.noise != null) {
            return this.noise.hashCode();
        } else {
            return Objects.hashCode(this.noiseData);
        }
    }

}
