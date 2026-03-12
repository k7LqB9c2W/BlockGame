package com.ishland.c2me.opts.dfc.mixin.equality;

import net.minecraft.util.math.noise.SimplexNoiseSampler;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;

import java.util.Arrays;
import java.util.Objects;

@Mixin(SimplexNoiseSampler.class)
public class MixinSimplexNoiseSampler {

    @Shadow @Final private int[] permutation;

    @Shadow @Final public double originX;

    @Shadow @Final public double originY;

    @Shadow @Final public double originZ;

    @Override
    public boolean equals(Object object) {
        if (this == object) return true;
        if (object == null || getClass() != object.getClass()) return false;
        MixinSimplexNoiseSampler that = (MixinSimplexNoiseSampler) object;
        return Double.compare(originX, that.originX) == 0 && Double.compare(originY, that.originY) == 0 && Double.compare(originZ, that.originZ) == 0 && Arrays.equals(permutation, that.permutation);
    }

    @Override
    public int hashCode() {
        int result = 1;
        result = 31 * result + Arrays.hashCode(permutation);
        result = 31 * result + Double.hashCode(originX);
        result = 31 * result + Double.hashCode(originY);
        result = 31 * result + Double.hashCode(originZ);
        return result;
    }
}
