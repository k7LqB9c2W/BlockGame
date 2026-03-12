package com.ishland.c2me.opts.accel.opencl.mixin.debug.disable_features_surface_rules;

import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.chunk.ChunkGenerating;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

import java.util.concurrent.CompletableFuture;

@Mixin(ChunkGenerating.class)
public class MixinChunkGenerating {

    @Inject(method = {"buildSurface", "generateFeatures"}, at = @At("HEAD"), cancellable = true)
    private static void noopStuff(CallbackInfoReturnable<CompletableFuture<Chunk>> cir) {
        cir.setReturnValue(CompletableFuture.completedFuture(null));
    }


}
