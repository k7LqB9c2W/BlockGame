package com.seibel.distanthorizons.common.render.blaze;

#if MC_VER <= MC_1_21_10
public class BlazeDhRenderApiDefinition {}

#else

import com.seibel.distanthorizons.common.render.blaze.objects.BlazeGenericObjectVertexContainer;
import com.seibel.distanthorizons.common.render.blaze.postProcessing.BlazeDhFarFadeRenderer;
import com.seibel.distanthorizons.common.render.blaze.postProcessing.BlazeDhFogRenderer;
import com.seibel.distanthorizons.common.render.blaze.postProcessing.BlazeDhSsaoRenderer;
import com.seibel.distanthorizons.common.render.blaze.postProcessing.BlazeVanillaFadeRenderer;
import com.seibel.distanthorizons.common.render.blaze.test.BlazeDhTestTriangleRenderer;
import com.seibel.distanthorizons.common.render.blaze.wrappers.buffer.BlazeVertexBufferWrapper;
import com.seibel.distanthorizons.common.render.blaze.wrappers.uniform.BlazeLodUniformBufferWrapper;
import com.seibel.distanthorizons.core.render.renderer.AbstractDebugWireframeRenderer;
import com.seibel.distanthorizons.core.wrapperInterfaces.render.AbstractDhRenderApiDefinition;
import com.seibel.distanthorizons.core.wrapperInterfaces.render.objects.IDhGenericObjectVertexBufferContainer;
import com.seibel.distanthorizons.core.wrapperInterfaces.render.objects.ILodContainerUniformBufferWrapper;
import com.seibel.distanthorizons.core.wrapperInterfaces.render.objects.IVertexBufferWrapper;
import com.seibel.distanthorizons.core.wrapperInterfaces.render.renderPass.*;

public class BlazeDhRenderApiDefinition extends AbstractDhRenderApiDefinition
{
	//=========//
	// getters //
	//=========//
	//region
	
	public String getApiName() { return "Blaze3D"; }
	
	//endregion
	
	
	
	//============//
	// singletons //
	//============//
	//region
	
	@Override public IDhMetaRenderer getMetaRenderer() { return BlazeDhMetaRenderer.INSTANCE; }
	@Override public IDhTerrainRenderer getTerrainRenderer() { return BlazeDhTerrainRenderer.INSTANCE; }
	@Override public IDhSsaoRenderer getSsaoRenderer() { return BlazeDhSsaoRenderer.INSTANCE; }
	@Override public IDhFogRenderer getFogRenderer() { return BlazeDhFogRenderer.INSTANCE; }
	@Override public IDhFarFadeRenderer getFarFadeRenderer() { return BlazeDhFarFadeRenderer.INSTANCE; }
	@Override public AbstractDebugWireframeRenderer getDebugWireframeRenderer() { return BlazeDebugWireframeRenderer.INSTANCE; }
	
	@Override public IDhVanillaFadeRenderer getVanillaFadeRenderer() { return BlazeVanillaFadeRenderer.INSTANCE; }
	@Override public IDhTestTriangleRenderer getTestTriangleRenderer() { return BlazeDhTestTriangleRenderer.INSTANCE; }
	
	//endregion
	
	
	
	//===========//
	// factories //
	//===========//
	//region
	
	@Override public IDhGenericRenderer createGenericRenderer() { return new BlazeDhGenericObjectRenderer(); }
	
	@Override public IVertexBufferWrapper createVboWrapper(String name) { return new BlazeVertexBufferWrapper(name); }
	@Override public ILodContainerUniformBufferWrapper createLodContainerUniformWrapper() { return new BlazeLodUniformBufferWrapper(); }
	@Override public IDhGenericObjectVertexBufferContainer createGenericVboContainer() { return new BlazeGenericObjectVertexContainer(); }
	
	//endregion
	
	
	
}
#endif