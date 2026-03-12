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

package com.seibel.distanthorizons.common.render.blaze.apply;

#if MC_VER <= MC_1_21_10
public class BlazeDhApplyRenderer {}

#else

import com.mojang.blaze3d.buffers.GpuBuffer;
import com.mojang.blaze3d.pipeline.BlendFunction;
import com.mojang.blaze3d.pipeline.RenderPipeline;
import com.mojang.blaze3d.platform.DepthTestFunction;
import com.mojang.blaze3d.platform.PolygonMode;
import com.mojang.blaze3d.shaders.UniformType;
import com.mojang.blaze3d.systems.CommandEncoder;
import com.mojang.blaze3d.systems.GpuDevice;
import com.mojang.blaze3d.systems.RenderPass;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.textures.*;
import com.mojang.blaze3d.vertex.VertexFormat;
import com.seibel.distanthorizons.common.render.blaze.util.BlazeDhVertexFormatUtil;
import com.seibel.distanthorizons.common.render.blaze.wrappers.texture.BlazeTextureViewWrapper;
import com.seibel.distanthorizons.common.render.blaze.util.BlazePostProcessUtil;
import com.seibel.distanthorizons.core.logging.DhLogger;
import com.seibel.distanthorizons.core.logging.DhLoggerBuilder;
import com.seibel.distanthorizons.coreapi.ModInfo;
import net.minecraft.resources.Identifier;
import org.jetbrains.annotations.Nullable;

import java.util.Arrays;
import java.util.OptionalDouble;
import java.util.OptionalInt;

/**
 * Copies the given color texture
 * where the depth (or another attribute) is valid.
 * Often used to apply post processing effects or
 * the DH texture to MC's color texture. <br><br>
 * 
 * @see BlazeDhCopyRenderer
 */
public class BlazeDhApplyRenderer
{
	public static final DhLogger LOGGER = new DhLoggerBuilder().build(); 
	
	private static final GpuDevice GPU_DEVICE = RenderSystem.getDevice();
	private static final CommandEncoder COMMAND_ENCODER = GPU_DEVICE.createCommandEncoder();
	
	private RenderPipeline pipeline;
	
	protected GpuBuffer vboGpuBuffer;
	
	protected final String name;
	protected final String identifierName;
	public String getIdentifierName() { return this.identifierName; }
	
	@Nullable
	private final BlendFunction blendFunction;
	private final String vertexShaderPath;
	private final String fragmentShaderPath;
	
	private final BlazeTextureViewWrapper sourceColorTextureViewWrapper = new BlazeTextureViewWrapper();
	private final BlazeTextureViewWrapper sourceDepthTextureViewWrapper = new BlazeTextureViewWrapper();
	
	private final BlazeTextureViewWrapper destinationColorTextureViewWrapper = new BlazeTextureViewWrapper();
	
	/** 
	 * Can be set for special application shaders that need 
	 * extra information. <br><br>
	 * 
	 * will be an empty array if unneeded 
	 */
	private final String[] uniformNames;
	/** will be an empty array if unneeded */
	private final GpuBuffer[] uniformBuffers;
	
	
	
	//=============//
	// constructor //
	//=============//
	//region
	public BlazeDhApplyRenderer(
		String name,
		@Nullable BlendFunction blendFunction,
		String vertexShaderPath, String fragmentShaderPath
		)
	{
		this(
			name,
			blendFunction,
			vertexShaderPath, fragmentShaderPath,
			new String[0] // no extra uniforms
		);
	}
	public BlazeDhApplyRenderer(
		String name,
		@Nullable BlendFunction blendFunction,
		String vertexShaderPath, String fragmentShaderPath,
		String[] uniformNames
		)
	{
		this.name = name;
		this.identifierName = "distanthorizons:"+this.name;
		this.blendFunction = blendFunction;
		
		this.vertexShaderPath = vertexShaderPath;
		this.fragmentShaderPath = fragmentShaderPath;
		
		this.uniformNames = uniformNames;
		this.uniformBuffers = new GpuBuffer[this.uniformNames.length];
	}
	
	private void tryInit(
		GpuTexture sourceColorTexture,
		GpuTexture sourceDepthTexture,
		GpuTexture destinationColorTexture)
	{
		this.createPipeline();
		this.vboGpuBuffer = BlazePostProcessUtil.createAndUploadScreenVertexData(this.name);
		
		this.sourceColorTextureViewWrapper.tryWrap(sourceColorTexture);
		this.sourceDepthTextureViewWrapper.tryWrap(sourceDepthTexture);
		
		this.destinationColorTextureViewWrapper.tryWrap(destinationColorTexture);
		
	}
	private void createPipeline()
	{
		if (this.pipeline != null)
		{
			return;
		}
		
		VertexFormat vertexFormat = VertexFormat.builder()
			.add("vPosition", BlazeDhVertexFormatUtil.SCREEN_POS)
			.build();
		
		RenderPipeline.Builder pipelineBuilder = RenderPipeline.builder();
		{
			pipelineBuilder.withCull(false);
			pipelineBuilder.withDepthWrite(false);
			pipelineBuilder.withDepthTestFunction(DepthTestFunction.NO_DEPTH_TEST);
			pipelineBuilder.withColorWrite(true);
			
			if (this.blendFunction != null)
			{
				pipelineBuilder.withBlend(this.blendFunction);
			}
			else
			{
				pipelineBuilder.withoutBlend();
			}
			
			pipelineBuilder.withPolygonMode(PolygonMode.FILL);
			pipelineBuilder.withLocation(Identifier.parse(this.identifierName)); // TODO will complain if capital letters are included
			
			// TODO manually validate paths to confirm they exist and end with ".fsh" or ".vsh", MC silently fails if the files are missing/improperly named
			pipelineBuilder.withVertexShader(Identifier.fromNamespaceAndPath("distanthorizons", this.vertexShaderPath));
			pipelineBuilder.withFragmentShader(Identifier.fromNamespaceAndPath("distanthorizons", this.fragmentShaderPath));
			
			for (int i = 0; i < this.uniformNames.length; i++)
			{
				String uniformName = this.uniformNames[i];
				pipelineBuilder.withUniform(uniformName, UniformType.UNIFORM_BUFFER);
			}
			
			pipelineBuilder.withSampler("uSourceColorTexture");
			pipelineBuilder.withSampler("uSourceDepthTexture");
			
			pipelineBuilder.withVertexFormat(vertexFormat, VertexFormat.Mode.TRIANGLE_FAN);
		}
		this.pipeline = pipelineBuilder.build();
	}
	
	
	//endregion
	
	
	
	//========//
	// render //
	//========//
	//region
	
	public void setUniform(String uniformName, GpuBuffer uniformBuffer)
	{
		// the uniform array should be short enough (less than 10 items)
		// where a sequential search should be plenty fast
		for (int i = 0; i < this.uniformNames.length; i++)
		{
			String nameAtIndex = this.uniformNames[i];
			if (nameAtIndex.equals(uniformName))
			{
				this.uniformBuffers[i] = uniformBuffer;
				break;
			}
		}
	}
	
	public void render(
		GpuTexture sourceColorTexture,
		GpuTexture sourceDepthTexture,
		GpuTexture destinationColorTexture)
	{
		this.tryInit(sourceColorTexture, sourceDepthTexture, destinationColorTexture);
		
		try (RenderPass renderPass = COMMAND_ENCODER.createRenderPass(
			this::getIdentifierName,
			this.destinationColorTextureViewWrapper.textureView,
			/*optionalClearColorAsInt*/ OptionalInt.empty(),
			/*depthTexture*/ null,
			/*optionalDepthValueAsDouble*/ OptionalDouble.empty()))
		{
			renderPass.bindTexture("uSourceColorTexture", this.sourceColorTextureViewWrapper.textureView, this.sourceColorTextureViewWrapper.textureSampler);
			renderPass.bindTexture("uSourceDepthTexture", this.sourceDepthTextureViewWrapper.textureView, this.sourceDepthTextureViewWrapper.textureSampler);
			
			for (int i = 0; i < this.uniformNames.length; i++)
			{
				String uniformName = this.uniformNames[i];
				GpuBuffer uniformBuffer = this.uniformBuffers[i];
				if (uniformBuffer == null)
				{
					throw new IllegalStateException("Missing uniform ["+uniformName+"], please set the uniform before rendering.");	
				}
				
				renderPass.setUniform(uniformName, uniformBuffer);
			}
			
			renderPass.setVertexBuffer(0, this.vboGpuBuffer);
			renderPass.setPipeline(this.pipeline);
			
			renderPass.draw(/*indexStart*/ 0, /*indexCount*/ 4);
		}
		
		
		// clear the uniforms after rendering
		// so we can check if they're missing during next frame's rendering
		if (ModInfo.IS_DEV_BUILD)
		{
			Arrays.fill(this.uniformBuffers, null);
		}
	}
	
	//endregion
	
	
	
}
#endif