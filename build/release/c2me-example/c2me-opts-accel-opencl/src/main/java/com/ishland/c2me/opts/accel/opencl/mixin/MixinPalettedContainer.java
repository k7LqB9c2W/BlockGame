package com.ishland.c2me.opts.accel.opencl.mixin;

import com.ishland.c2me.opts.accel.opencl.common.ducks.PalettedContainerExtension;
import net.minecraft.world.chunk.PaletteProvider;
import net.minecraft.world.chunk.PalettedContainer;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;

@Mixin(PalettedContainer.class)
public abstract class MixinPalettedContainer<T> implements PalettedContainerExtension<T> {

    @Shadow
    protected abstract void set(int index, T value);

    @Shadow
    @Final
    private PaletteProvider<T> paletteProvider;

    @Override
    public void c2me$setUnsafe(int x, int y, int z, T value) {
        this.set(this.paletteProvider.computeIndex(x, y, z), value);
    }

}
