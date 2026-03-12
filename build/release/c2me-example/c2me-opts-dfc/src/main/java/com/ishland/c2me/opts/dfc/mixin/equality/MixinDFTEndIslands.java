package com.ishland.c2me.opts.dfc.mixin.equality;

import net.minecraft.util.math.noise.SimplexNoiseSampler;
import net.minecraft.world.gen.densityfunction.DensityFunctionTypes;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;

import java.util.Objects;

@Mixin(DensityFunctionTypes.EndIslands.class)
public class MixinDFTEndIslands {

    @Shadow @Final private SimplexNoiseSampler sampler;

    @Override
    public boolean equals(Object object) {
        if (this == object) return true;
        if (object == null || getClass() != object.getClass()) return false;
        MixinDFTEndIslands that = (MixinDFTEndIslands) object;
        return Objects.equals(sampler, that.sampler);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(sampler);
    }
}
