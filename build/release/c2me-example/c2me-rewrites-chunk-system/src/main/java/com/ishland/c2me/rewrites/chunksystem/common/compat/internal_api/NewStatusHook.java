package com.ishland.c2me.rewrites.chunksystem.common.compat.internal_api;

import com.ishland.c2me.rewrites.chunksystem.common.NewChunkStatus;
import net.minecraft.world.chunk.ChunkStatus;

import java.util.ArrayList;

public class NewStatusHook {

    public static void beforeVanillaStatusRegister(ArrayList<NewChunkStatus> pending, ChunkStatus nextStatus) {
    }

}
