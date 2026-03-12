package com.ishland.c2me.base.mixin.access;

import net.minecraft.world.biome.source.util.MultiNoiseUtil;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(MultiNoiseUtil.SearchTree.TreeBranchNode.class)
public interface IMultiNoiseUtilSearchTreeTreeBranchNode<T> {

    @Accessor
    MultiNoiseUtil.SearchTree.TreeNode<T>[] getSubTree();

}
