/*
 *    This file is part of the Distant Horizons mod
 *    licensed under the GNU LGPL v3 License.
 *
 *    Copyright (C) 2020 James Seibel
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU Lesser General Public License as published by
 *    the Free Software Foundation, version 3.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public License
 *    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package com.seibel.distanthorizons.common.render.blaze.postProcessing;

#if MC_VER <= MC_1_21_10
public class BlazeDhFogRenderer {}

#else
	
import com.mojang.blaze3d.buffers.GpuBuffer;
import com.mojang.blaze3d.buffers.GpuBufferSlice;
import com.mojang.blaze3d.buffers.Std140Builder;
import com.mojang.blaze3d.buffers.Std140SizeCalculator;
import com.mojang.blaze3d.pipeline.BlendFunction;
import com.mojang.blaze3d.pipeline.RenderPipeline;
import com.mojang.blaze3d.platform.DepthTestFunction;
import com.mojang.blaze3d.platform.DestFactor;
import com.mojang.blaze3d.platform.PolygonMode;
import com.mojang.blaze3d.platform.SourceFactor;
import com.mojang.blaze3d.shaders.UniformType;
import com.mojang.blaze3d.systems.CommandEncoder;
import com.mojang.blaze3d.systems.GpuDevice;
import com.mojang.blaze3d.systems.RenderPass;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.vertex.VertexFormat;
import com.seibel.distanthorizons.api.enums.rendering.EDhApiFogColorMode;
import com.seibel.distanthorizons.api.enums.rendering.EDhApiHeightFogDirection;
import com.seibel.distanthorizons.api.enums.rendering.EDhApiHeightFogMixMode;
import com.seibel.distanthorizons.common.render.blaze.BlazeDhMetaRenderer;
import com.seibel.distanthorizons.common.render.blaze.apply.BlazeDhApplyRenderer;
import com.seibel.distanthorizons.common.render.blaze.wrappers.texture.BlazeTextureWrapper;
import com.seibel.distanthorizons.common.render.blaze.util.BlazePostProcessUtil;
import com.seibel.distanthorizons.common.render.blaze.util.BlazeUniformUtil;
import com.seibel.distanthorizons.core.config.Config;
import com.seibel.distanthorizons.core.dependencyInjection.SingletonInjector;
import com.seibel.distanthorizons.core.logging.DhLogger;
import com.seibel.distanthorizons.core.logging.DhLoggerBuilder;
import com.seibel.distanthorizons.core.render.RenderParams;
import com.seibel.distanthorizons.core.util.LodUtil;
import com.seibel.distanthorizons.core.util.math.Mat4f;
import com.seibel.distanthorizons.core.wrapperInterfaces.minecraft.IMinecraftClientWrapper;
import com.seibel.distanthorizons.core.wrapperInterfaces.minecraft.IMinecraftRenderWrapper;
import com.seibel.distanthorizons.core.wrapperInterfaces.render.renderPass.IDhFogRenderer;
import net.minecraft.resources.Identifier;

import java.awt.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.OptionalDouble;
import java.util.OptionalInt;

/**
 * Renders fog onto the LODs.
 */
public class BlazeDhFogRenderer implements IDhFogRenderer
{
	public static final DhLogger LOGGER = new DhLoggerBuilder().build(); 
	
	private static final IMinecraftClientWrapper MC = SingletonInjector.INSTANCE.get(IMinecraftClientWrapper.class);
	private static final IMinecraftRenderWrapper MC_RENDER = SingletonInjector.INSTANCE.get(IMinecraftRenderWrapper.class);
	
	private static final GpuDevice GPU_DEVICE = RenderSystem.getDevice();
	private static final CommandEncoder COMMAND_ENCODER = GPU_DEVICE.createCommandEncoder();
	
	public static final BlazeDhFogRenderer INSTANCE = new BlazeDhFogRenderer();
	
	
	private BlazeDhApplyRenderer applyRenderer;
	
	private RenderPipeline pipeline;
	private boolean init = false;
	
	private GpuBuffer fragUniformBuffer;
	
	private GpuBuffer vboGpuBuffer;
	
	public BlazeTextureWrapper fogColorTextureWrapper = BlazeTextureWrapper.createColor("DhFogColorTexture");
	
	
	
	//=============//
	// constructor //
	//=============//
	//region
	
	private BlazeDhFogRenderer() { }
	
	private void tryInit()
	{
		if (this.init)
		{
			return;
		}
		this.init = true;
		
		
		
		
		this.applyRenderer = new BlazeDhApplyRenderer(
			"fog_apply_to_dh",
			new BlendFunction(SourceFactor.SRC_ALPHA, DestFactor.ONE_MINUS_SRC_ALPHA, SourceFactor.ONE, DestFactor.ONE_MINUS_SRC_ALPHA),
			"apply/blaze/vert", "apply/blaze/frag"
		);
		
		RenderPipeline.Builder pipelineBuilder = RenderPipeline.builder();
		{
			pipelineBuilder.withCull(false);
			pipelineBuilder.withDepthWrite(false);
			pipelineBuilder.withDepthTestFunction(DepthTestFunction.NO_DEPTH_TEST);
			pipelineBuilder.withColorWrite(true);
			pipelineBuilder.withoutBlend();
			pipelineBuilder.withPolygonMode(PolygonMode.FILL);
			pipelineBuilder.withLocation(Identifier.parse("distanthorizons:fog_render"));
			
			pipelineBuilder.withVertexShader(Identifier.fromNamespaceAndPath("distanthorizons", "fog/blaze/vert"));
			pipelineBuilder.withFragmentShader(Identifier.fromNamespaceAndPath("distanthorizons", "fog/blaze/frag"));
			
			pipelineBuilder.withSampler("uDhDepthTexture");
			
			pipelineBuilder.withUniform("fragUniformBlock", UniformType.UNIFORM_BUFFER);
			
			pipelineBuilder.withVertexFormat(BlazePostProcessUtil.createVertexFormat(), VertexFormat.Mode.TRIANGLE_FAN);
		}
		this.pipeline = pipelineBuilder.build();
		
		
		this.vboGpuBuffer = BlazePostProcessUtil.createAndUploadScreenVertexData("McFogRenderer");
	}
	
	//endregion
	
	
	
	//========//
	// render //
	//========//
	//region
	
	@Override
	public void render(RenderParams renderParams)
	{
		this.tryInit();
		
		
		if (BlazeDhMetaRenderer.INSTANCE.dhDepthTextureWrapper.isEmpty()
			|| BlazeDhMetaRenderer.INSTANCE.dhColorTextureWrapper.isEmpty())
		{
			return;	
		}
		
		
		
		this.fogColorTextureWrapper.tryCreateOrResize();
		
		
		{
			int uniformBufferSize = new Std140SizeCalculator()
				
				// fog uniforms
				.putVec4() // uFogColor
				.putFloat() //uFogScale
				.putFloat() //uFogVerticalScale
				// only used for debugging
				.putInt() //uFogDebugMode  // 1 = render everything with fog color // 7 = use debug rendering
				.putInt() //uFogFalloffType
				
				// fog config
				.putFloat() // uFarFogStart
				.putFloat() // uFarFogLength
				.putFloat() // uFarFogMin
				.putFloat() // uFarFogRange 
				.putFloat() // uFarFogDensity
				
				// height fog config
				.putFloat() // uHeightFogStart
				.putFloat() // uHeightFogLength
				.putFloat() // uHeightFogMin
				.putFloat() // uHeightFogRange
				.putFloat() // uHeightFogDensity
				
				// ??
				.putInt() // uHeightFogEnabled
				.putInt() // uHeightFogFalloffType
				.putInt() // uHeightBasedOnCamera
				.putFloat() // uHeightFogBaseHeight
				.putInt() // uHeightFogAppliesUp
				.putInt() // uHeightFogAppliesDown
				.putInt() // uUseSphericalFog
				.putInt() // uHeightFogMixingMode
				.putFloat() // uCameraBlockYPos
				
				.putMat4f() // uInvMvmProj
				
				.get();
			
			
			// create data //
			
			
			int lodDrawDistance = Config.Client.Advanced.Graphics.Quality.lodChunkRenderDistanceRadius.get() * LodUtil.CHUNK_WIDTH;
			
			
			Mat4f inverseMvmProjMatrix = new Mat4f(renderParams.dhMvmProjMatrix);
			inverseMvmProjMatrix.invert();
			
			if (renderParams.dhMvmProjMatrix == null)
			{
				return;
			}
			
			
			Color fogColor = this.getFogColor(renderParams.partialTicks);
			
			// fog config
			float farFogStart = Config.Client.Advanced.Graphics.Fog.farFogStart.get();
			float farFogEnd = Config.Client.Advanced.Graphics.Fog.farFogEnd.get();
			float farFogMin = Config.Client.Advanced.Graphics.Fog.farFogMin.get();
			float farFogMax = Config.Client.Advanced.Graphics.Fog.farFogMax.get();
			float farFogDensity = Config.Client.Advanced.Graphics.Fog.farFogDensity.get();
			
			// override fog if underwater
			if (MC_RENDER.isFogStateSpecial())
			{
				// hide everything behind fog
				farFogStart = 0.0f;
				farFogEnd = 0.0f;
			}
			
			
			// height config
			EDhApiHeightFogMixMode heightFogMixingMode = Config.Client.Advanced.Graphics.Fog.HeightFog.heightFogMixMode.get();
			boolean heightFogEnabled = heightFogMixingMode != EDhApiHeightFogMixMode.SPHERICAL && heightFogMixingMode != EDhApiHeightFogMixMode.CYLINDRICAL;
			boolean useSphericalFog = heightFogMixingMode == EDhApiHeightFogMixMode.SPHERICAL;
			EDhApiHeightFogDirection heightFogCameraDirection = Config.Client.Advanced.Graphics.Fog.HeightFog.heightFogDirection.get();
			
			float heightFogStart = Config.Client.Advanced.Graphics.Fog.HeightFog.heightFogStart.get();
			float heightFogEnd = Config.Client.Advanced.Graphics.Fog.HeightFog.heightFogEnd.get();
			float heightFogMin = Config.Client.Advanced.Graphics.Fog.HeightFog.heightFogMin.get();
			float heightFogMax = Config.Client.Advanced.Graphics.Fog.HeightFog.heightFogMax.get();
			float heightFogDensity = Config.Client.Advanced.Graphics.Fog.HeightFog.heightFogDensity.get();
			
			
			// upload data //
			
			ByteBuffer buffer = ByteBuffer.allocateDirect(uniformBufferSize);
			buffer.order(ByteOrder.nativeOrder());
			buffer = Std140Builder.intoBuffer(buffer)
				
				// fog uniforms
				.putVec4(
					fogColor.getRed() / 255.0f, 
					fogColor.getGreen() / 255.0f, 
					fogColor.getBlue() / 255.0f, 
					fogColor.getAlpha() / 255.0f) // uFogColor
				.putFloat(1.f / lodDrawDistance) //uFogScale
				.putFloat(1.f / MC.getWrappedClientLevel().getMaxHeight()) //uFogVerticalScale
				// only used for debugging
				.putInt(0) //uFogDebugMode  // 1 = render everything with fog color // 7 = use debug rendering
				.putInt(Config.Client.Advanced.Graphics.Fog.farFogFalloff.get().value) //uFogFalloffType
				
				// fog config
				.putFloat(farFogStart) // uFarFogStart
				.putFloat(farFogEnd - farFogStart) // uFarFogLength
				.putFloat(farFogMin) // uFarFogMin
				.putFloat(farFogMax - farFogMin) // uFarFogRange 
				.putFloat(farFogDensity) // uFarFogDensity
				
				// height fog config
				.putFloat(heightFogStart) // uHeightFogStart
				.putFloat(heightFogEnd - heightFogStart) // uHeightFogLength
				.putFloat(heightFogMin) // uHeightFogMin
				.putFloat(heightFogMax - heightFogMin) // uHeightFogRange
				.putFloat(heightFogDensity) // uHeightFogDensity
				
				// ??
				.putInt(heightFogEnabled ? 1 : 0) // uHeightFogEnabled
				.putInt(Config.Client.Advanced.Graphics.Fog.HeightFog.heightFogFalloff.get().value) // uHeightFogFalloffType
				.putInt(heightFogCameraDirection.basedOnCamera ? 1 : 0) // uHeightBasedOnCamera
				.putFloat(Config.Client.Advanced.Graphics.Fog.HeightFog.heightFogBaseHeight.get()) // uHeightFogBaseHeight
				.putInt(heightFogCameraDirection.fogAppliesUp ? 1 : 0) // uHeightFogAppliesUp
				.putInt(heightFogCameraDirection.fogAppliesDown ? 1 : 0) // uHeightFogAppliesDown
				.putInt(useSphericalFog ? 1 : 0) // uUseSphericalFog
				.putInt(heightFogMixingMode.value) // uHeightFogMixingMode
				.putFloat((float)MC_RENDER.getCameraExactPosition().y) // uCameraBlockYPos
				
				.putMat4f(inverseMvmProjMatrix.createJomlMatrix()) // uInvMvmProj
				
				.get()
			;
			
			this.fragUniformBuffer = BlazeUniformUtil.createBuffer("fragUniformBlock", uniformBufferSize, this.fragUniformBuffer);
			GpuBufferSlice bufferSlice = new GpuBufferSlice(this.fragUniformBuffer, 0, uniformBufferSize);
			
			COMMAND_ENCODER.writeToBuffer(bufferSlice, buffer);
		}
		
		
		this.renderFogToTexture();
		this.applyRenderer.render(this.fogColorTextureWrapper.texture, BlazeDhMetaRenderer.INSTANCE.dhDepthTextureWrapper.texture, BlazeDhMetaRenderer.INSTANCE.dhColorTextureWrapper.texture);
		
	}
	
	private Color getFogColor(float partialTicks)
	{
		Color fogColor;
		
		if (Config.Client.Advanced.Graphics.Fog.colorMode.get() == EDhApiFogColorMode.USE_SKY_COLOR)
		{
			fogColor = MC_RENDER.getSkyColor();
		}
		else
		{
			fogColor = MC_RENDER.getFogColor(partialTicks);
		}
		
		return fogColor;
	}
	
	private void renderFogToTexture()
	{
		try (RenderPass renderPass = COMMAND_ENCODER.createRenderPass(
			this::getRenderPassName,
			this.fogColorTextureWrapper.textureView, 
			/*optionalClearColorAsInt*/ OptionalInt.empty(),
			/*depthTexture*/ null, 
			/*optionalDepthValueAsDouble*/ OptionalDouble.empty()))
		{
			renderPass.bindTexture("uDhDepthTexture", BlazeDhMetaRenderer.INSTANCE.dhDepthTextureWrapper.textureView, BlazeDhMetaRenderer.INSTANCE.dhDepthTextureWrapper.textureSampler);
			
			renderPass.setUniform("fragUniformBlock", this.fragUniformBuffer);
			
			renderPass.setVertexBuffer(0, this.vboGpuBuffer); // vertex buffer can only be "0" lol
			renderPass.setPipeline(this.pipeline);
			
			renderPass.draw(/*indexStart*/ 0, /*indexCount*/ 4);
		}
	}
	private String getRenderPassName() { return "distantHorizons:McFogRenderer"; }
	
	
	//endregion
	
	
	
}
#endif