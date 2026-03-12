package com.seibel.distanthorizons.common.render.blaze;

#if MC_VER <= MC_1_21_10
public class BlazeDhMetaRenderer {}

#else

import com.mojang.blaze3d.textures.GpuTexture;
import com.seibel.distanthorizons.common.render.blaze.apply.BlazeDhApplyRenderer;
import com.seibel.distanthorizons.common.render.blaze.wrappers.texture.BlazeTextureWrapper;
import com.seibel.distanthorizons.core.render.RenderParams;
import com.seibel.distanthorizons.core.util.ColorUtil;
import com.seibel.distanthorizons.core.wrapperInterfaces.minecraft.IMinecraftRenderWrapper;
import com.seibel.distanthorizons.core.wrapperInterfaces.render.renderPass.IDhMetaRenderer;
import net.minecraft.client.Minecraft;

public class BlazeDhMetaRenderer implements IDhMetaRenderer
{
	public static final BlazeDhMetaRenderer INSTANCE = new BlazeDhMetaRenderer();
	
	
	private BlazeDhApplyRenderer applyRenderer;
	
	public final BlazeTextureWrapper dhDepthTextureWrapper = BlazeTextureWrapper.createDepth("DhDepthTexture");
	public final BlazeTextureWrapper dhColorTextureWrapper = BlazeTextureWrapper.createColor("DhColorTexture");
	
	
	
	//=============//
	// constructor //
	//=============//
	//region
	
	private BlazeDhMetaRenderer() 
	{
		this.applyRenderer = new BlazeDhApplyRenderer(
			"dh_apply_to_mc",
			null,
			"apply/blaze/vert", "apply/blaze/frag"
		);
	}
	
	//endregion
	
	
	
	//=================//
	// pre/post render //
	//=================//
	//region
	
	@Override
	public void runRenderPassSetup(RenderParams renderParams)
	{
		// textures
		this.dhDepthTextureWrapper.tryCreateOrResize();
		this.dhColorTextureWrapper.tryCreateOrResize();
	}
	
	@Override
	public void runRenderPassCleanup(RenderParams renderParams) {}
	
	@Override
	public void applyToMcTexture(RenderParams renderParams)
	{
		GpuTexture mcColorTexture = Minecraft.getInstance().getMainRenderTarget().getColorTexture();
		this.applyRenderer.render(this.dhColorTextureWrapper.texture, this.dhDepthTextureWrapper.texture, mcColorTexture);
	}
	
	//endregion
	
	
	
	//================//
	// clear textures //
	//================//
	//region
	
	@Override
	public void clearDhDepthAndColorTextures(RenderParams renderParams) 
	{
		// TODO use for clear color
		//IMinecraftRenderWrapper r;
		//r.getSkyColor()
		
		this.dhDepthTextureWrapper.clearDepth(1.0f);
		this.dhColorTextureWrapper.clearColor(ColorUtil.argbToInt(1, 1, 1, 1)); 
	}
	
	//endregion
	
	
}
#endif