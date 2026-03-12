package com.ishland.c2me.base.mixin.access;

import net.minecraft.world.biome.source.util.MultiNoiseUtil;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(MultiNoiseUtil.SearchTree.TreeLeafNode.class)
public interface IMultiNoiseUtilSearchTreeTreeLeafNode<T> {

    @Accessor
    T getValue();

}
