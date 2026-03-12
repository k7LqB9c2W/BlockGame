package com.ishland.c2me.base.mixin.access;

import it.unimi.dsi.fastutil.objects.ObjectListIterator;
import net.minecraft.structure.JigsawJunction;
import net.minecraft.util.math.BlockBox;
import net.minecraft.world.gen.StructureWeightSampler;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

import java.util.List;

@Mixin(StructureWeightSampler.class)
public interface IStructureWeightSampler {

    @Accessor(value = "field_61465")
    List<StructureWeightSampler.Piece> getPiecesList();

    @Accessor(value = "field_61466")
    List<JigsawJunction> getJunctionsList();

    @Accessor(value = "field_61467")
    BlockBox getAffectedBox();

    @Accessor
    static float[] getSTRUCTURE_WEIGHT_TABLE() {
        throw new AbstractMethodError();
    }

}
