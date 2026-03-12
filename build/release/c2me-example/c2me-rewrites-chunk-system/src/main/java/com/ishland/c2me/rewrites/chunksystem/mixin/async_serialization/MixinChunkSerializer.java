package com.ishland.c2me.rewrites.chunksystem.mixin.async_serialization;

import com.ishland.c2me.rewrites.chunksystem.common.async_chunkio.AsyncSerializationUtil;
import com.ishland.c2me.rewrites.chunksystem.common.async_chunkio.ChunkIoMainThreadTaskUtils;
import com.llamalad7.mixinextras.injector.wrapoperation.Operation;
import com.llamalad7.mixinextras.injector.wrapoperation.WrapOperation;
import net.minecraft.util.math.ChunkSectionPos;
import net.minecraft.world.chunk.ChunkSection;
import net.minecraft.world.chunk.SerializedChunk;
import net.minecraft.world.poi.PointOfInterestStorage;
import org.slf4j.Logger;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;

@Mixin(SerializedChunk.class)
public class MixinChunkSerializer {

    @Shadow @Final private static Logger LOGGER;

    @WrapOperation(method = "convert", at = @At(value = "INVOKE", target = "Lnet/minecraft/world/poi/PointOfInterestStorage;initForPalette(Lnet/minecraft/util/math/ChunkSectionPos;Lnet/minecraft/world/chunk/ChunkSection;)V"))
    private void onPoiStorageInitForPalette(PointOfInterestStorage instance, ChunkSectionPos sectionPos, ChunkSection chunkSection, Operation<Void> original) {
        ChunkIoMainThreadTaskUtils.executeMain(() -> original.call(instance, sectionPos, chunkSection));
    }

    @WrapOperation(method = "fromChunk", at = @At(value = "INVOKE", target = "Lnet/minecraft/world/chunk/ChunkSection;copy()Lnet/minecraft/world/chunk/ChunkSection;"))
    private static ChunkSection avoidSectionCopyOnUnload(ChunkSection instance, Operation<ChunkSection> original) {
        if (AsyncSerializationUtil.duringUnloadSerialization.isBound()) {
            return instance;
        } else {
            return original.call(instance);
        }
    }

}
