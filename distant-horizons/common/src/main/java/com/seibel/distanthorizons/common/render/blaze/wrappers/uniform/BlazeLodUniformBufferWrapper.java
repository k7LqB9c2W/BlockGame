package com.seibel.distanthorizons.common.render.blaze.wrappers.uniform;

#if MC_VER <= MC_1_21_10
public class BlazeLodUniformBufferWrapper {}

#else

import com.mojang.blaze3d.buffers.Std140Builder;
import com.mojang.blaze3d.buffers.Std140SizeCalculator;
import com.seibel.distanthorizons.core.dataObjects.render.bufferBuilding.LodBufferContainer;
import com.seibel.distanthorizons.core.util.math.Vec3f;
import com.seibel.distanthorizons.core.wrapperInterfaces.render.objects.ILodContainerUniformBufferWrapper;

import java.nio.ByteBuffer;

public class BlazeLodUniformBufferWrapper extends BlazeUniformBufferWrapper implements ILodContainerUniformBufferWrapper
{
	
	private boolean uploaded = false;
	
	
	
	//=============//
	// constructor //
	//=============//
	//region
	
	public BlazeLodUniformBufferWrapper() { super(BlazeLodUniformBufferWrapper.class.getName()); }
	
	//endregion
	
	
	
	//========//
	// ??? //
	//========//
	//region
	
	@Override
	public void createUniformData(LodBufferContainer bufferContainer)
	{
		Vec3f modelOffset = new Vec3f(
			(float) (bufferContainer.minCornerBlockPos.getX()),
			(float) (bufferContainer.minCornerBlockPos.getY()),
			(float) (bufferContainer.minCornerBlockPos.getZ()));
		
		// upload data //
		
		int uniformBufferSize = new Std140SizeCalculator()
			.putVec3() // uModelOffset
			.get();
		
		ByteBuffer buffer = this.getOrCreateBuffer(uniformBufferSize);
		Std140Builder.intoBuffer(buffer)
			.putVec3(modelOffset.x, modelOffset.y, modelOffset.z) // uModelOffset
			.get();
		
	}
	
	@Override
	public void tryUpload()
	{
		if (this.uploaded)
		{
			return;
		}
		
		this.upload();
		
		this.uploaded = true;
	}
	
	//endregion
	
}
#endif