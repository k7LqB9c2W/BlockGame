package com.seibel.distanthorizons.common.render.blaze.wrappers.texture;

#if MC_VER <= MC_1_21_10
public class BlazeTextureWrapper {}

#else

import com.mojang.blaze3d.buffers.GpuBuffer;
import com.mojang.blaze3d.systems.CommandEncoder;
import com.mojang.blaze3d.systems.GpuDevice;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.textures.*;
import com.seibel.distanthorizons.core.dependencyInjection.SingletonInjector;
import com.seibel.distanthorizons.core.logging.DhLogger;
import com.seibel.distanthorizons.core.logging.DhLoggerBuilder;
import com.seibel.distanthorizons.core.util.ColorUtil;
import com.seibel.distanthorizons.core.wrapperInterfaces.minecraft.IMinecraftRenderWrapper;

import java.util.OptionalDouble;

public class BlazeTextureWrapper
{
	public static final DhLogger LOGGER = new DhLoggerBuilder().build();
	
	private static final IMinecraftRenderWrapper MC_RENDER = SingletonInjector.INSTANCE.get(IMinecraftRenderWrapper.class);
	
	private static final GpuDevice GPU_DEVICE = RenderSystem.getDevice();
	private static final CommandEncoder COMMAND_ENCODER = GPU_DEVICE.createCommandEncoder();
	
	
	public final String name;
	public final TextureFormat textureFormat;
	
	public GpuTexture texture = null;
	public GpuTextureView textureView = null;
	public GpuSampler textureSampler = null;
	
	private int width = -1;
	private int height = -1;
	
	
	
	//==============//
	// constructors //
	//==============//
	//region
	
	public static BlazeTextureWrapper createDepth(String name) { return new BlazeTextureWrapper(name, TextureFormat.DEPTH32); }
	public static BlazeTextureWrapper createColor(String name) { return new BlazeTextureWrapper(name, TextureFormat.RGBA8); }
	
	private BlazeTextureWrapper(String name, TextureFormat textureFormat)
	{
		this.name = name;
		this.textureFormat = textureFormat;
	}
	
	//endregion
	
	
	
	//=========//
	// getters //
	//=========//
	//region
	
	public boolean isEmpty() { return this.texture == null; }
	
	/** @return -1 if the texture is null */
	public int getWidth() { return this.width; }
	/** @return -1 if the texture is null */
	public int getHeight() { return this.height; }
	
	//endregion
	
	
	
	//=======//
	// setup //
	//=======//
	//region
	
	/** does nothing if the texture is already created and the correct size */
	public void tryCreateOrResize()
	{
		this.tryCreateTexture();
		this.tryCreateSampler();
	}
	private void tryCreateTexture()
	{
		int viewWidth = MC_RENDER.getTargetFramebufferViewportWidth();
		int viewHeight = MC_RENDER.getTargetFramebufferViewportHeight();
		
		if (this.texture == null
			|| this.width != viewWidth
			|| this.height != viewHeight)
		{
			if (this.texture != null)
			{
				this.texture.close();
				this.textureView.close();
			}
			
			this.width = viewWidth;
			this.height = viewHeight;
			
			int usage = GpuBuffer.USAGE_HINT_CLIENT_STORAGE 
				| GpuBuffer.USAGE_COPY_DST 
				| GpuBuffer.USAGE_VERTEX 
				| GpuBuffer.USAGE_UNIFORM;
			this.texture = GPU_DEVICE.createTexture(this.name,
				usage,
				this.textureFormat,
				viewWidth, viewHeight,
				/*depthOrLayers*/ 1,  /*mipLevels*/ 1
			);
			this.textureView = GPU_DEVICE.createTextureView(this.texture);
		}
	}
	private void tryCreateSampler()
	{
		if (this.textureSampler == null)
		{
			this.textureSampler = GPU_DEVICE.createSampler(
				AddressMode.CLAMP_TO_EDGE, AddressMode.CLAMP_TO_EDGE, // U,V
				FilterMode.LINEAR, FilterMode.LINEAR, // minFilter, magFilter
				1, // maxAnisotropy 
				OptionalDouble.empty() // maxLod
			);
		}
	}
	
	//endregion
	
	
	
	//==========//
	// clearing //
	//==========//
	//region
	
	/** 
	 * Will throw an exception if not a color texture.
	 * @see ColorUtil#argbToInt 
	 */
	public void clearColor(int clearArgbColor) 
	{
		if (this.texture != null)
		{
			COMMAND_ENCODER.clearColorTexture(this.texture, clearArgbColor);
		}
	}
	
	/** Will throw an exception if not a depth texture. */
	public void clearDepth(float depth) 
	{
		if (this.texture != null)
		{
			COMMAND_ENCODER.clearDepthTexture(this.texture, depth);
		}
	}
	
	//endregion
	
	
	
}
#endif