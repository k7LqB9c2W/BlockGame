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
public class BlazeDhFarFadeRenderer {}

#else
	
import com.mojang.blaze3d.buffers.GpuBuffer;
import com.mojang.blaze3d.buffers.GpuBufferSlice;
import com.mojang.blaze3d.buffers.Std140Builder;
import com.mojang.blaze3d.buffers.Std140SizeCalculator;
import com.mojang.blaze3d.pipeline.RenderPipeline;
import com.mojang.blaze3d.platform.DepthTestFunction;
import com.mojang.blaze3d.platform.PolygonMode;
import com.mojang.blaze3d.shaders.UniformType;
import com.mojang.blaze3d.systems.CommandEncoder;
import com.mojang.blaze3d.systems.GpuDevice;
import com.mojang.blaze3d.systems.RenderPass;
import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.blaze3d.vertex.VertexFormat;
import com.seibel.distanthorizons.common.render.blaze.BlazeDhMetaRenderer;
import com.seibel.distanthorizons.common.render.blaze.apply.BlazeDhCopyRenderer;
import com.seibel.distanthorizons.common.render.blaze.util.BlazePostProcessUtil;
import com.seibel.distanthorizons.common.render.blaze.util.BlazeUniformUtil;
import com.seibel.distanthorizons.common.render.blaze.wrappers.texture.BlazeTextureViewWrapper;
import com.seibel.distanthorizons.common.render.blaze.wrappers.texture.BlazeTextureWrapper;
import com.seibel.distanthorizons.core.logging.DhLogger;
import com.seibel.distanthorizons.core.logging.DhLoggerBuilder;
import com.seibel.distanthorizons.core.render.RenderParams;
import com.seibel.distanthorizons.core.util.RenderUtil;
import com.seibel.distanthorizons.core.util.math.Mat4f;
import com.seibel.distanthorizons.core.wrapperInterfaces.render.renderPass.IDhFarFadeRenderer;
import net.minecraft.client.Minecraft;
import net.minecraft.resources.Identifier;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.OptionalDouble;
import java.util.OptionalInt;

/**
 * Fades out DH's far clip plane
 */
public class BlazeDhFarFadeRenderer implements IDhFarFadeRenderer
{
	public static final DhLogger LOGGER = new DhLoggerBuilder().build(); 
	
	private static final GpuDevice GPU_DEVICE = RenderSystem.getDevice();
	private static final CommandEncoder COMMAND_ENCODER = GPU_DEVICE.createCommandEncoder();
	
	public static final BlazeDhFarFadeRenderer INSTANCE = new BlazeDhFarFadeRenderer();
	
	private RenderPipeline pipeline;
	private boolean init = false;
	
	private GpuBuffer fragUniformBuffer;
	
	private GpuBuffer vboGpuBuffer;
	
	public final BlazeTextureWrapper dhFadeColorTextureWrapper = BlazeTextureWrapper.createColor("DhFadeColorTexture");
	public final BlazeTextureViewWrapper mcColorTextureViewWrapper = new BlazeTextureViewWrapper();
	
	
	
	//=============//
	// constructor //
	//=============//
	//region
	
	private BlazeDhFarFadeRenderer() { }
	
	private void tryInit()
	{
		if (this.init)
		{
			return;
		}
		this.init = true;
		
		
		
		RenderPipeline.Builder pipelineBuilder = RenderPipeline.builder();
		{
			pipelineBuilder.withCull(false);
			pipelineBuilder.withDepthWrite(false);
			pipelineBuilder.withDepthTestFunction(DepthTestFunction.NO_DEPTH_TEST);
			pipelineBuilder.withColorWrite(true);
			pipelineBuilder.withoutBlend();
			pipelineBuilder.withPolygonMode(PolygonMode.FILL);
			pipelineBuilder.withLocation(Identifier.parse("distanthorizons:far_fade"));
			
			pipelineBuilder.withVertexShader(Identifier.fromNamespaceAndPath("distanthorizons", "fade/blaze/vert"));
			pipelineBuilder.withFragmentShader(Identifier.fromNamespaceAndPath("distanthorizons", "fade/blaze/dh_fade"));
			
			pipelineBuilder.withSampler("uMcColorTexture");
			
			pipelineBuilder.withSampler("uDhDepthTexture");
			pipelineBuilder.withSampler("uDhColorTexture");
			
			pipelineBuilder.withUniform("fragUniformBlock", UniformType.UNIFORM_BUFFER);
			
			pipelineBuilder.withVertexFormat(BlazePostProcessUtil.createVertexFormat(), VertexFormat.Mode.TRIANGLE_FAN);
		}
		this.pipeline = pipelineBuilder.build();
		
		
		this.vboGpuBuffer = BlazePostProcessUtil.createAndUploadScreenVertexData("McFadeRenderer");
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
		
		
		
		// textures
		this.dhFadeColorTextureWrapper.tryCreateOrResize();
		this.mcColorTextureViewWrapper.tryWrap(Minecraft.getInstance().getMainRenderTarget().getColorTexture());
		
		{
			int uniformBufferSize = new Std140SizeCalculator()
				.putFloat() // uStartFadeBlockDistance
				.putFloat() // uEndFadeBlockDistance
				.putMat4f() // uDhInvMvmProj
				.get();
			
			
			// create data //
			
			float dhFarClipDistance = RenderUtil.getFarClipPlaneDistanceInBlocks();
			float fadeStartDistance = dhFarClipDistance * 0.5f;
			float fadeEndDistance = dhFarClipDistance * 0.9f;
			
			
			Mat4f dhProjectionMatrix = RenderUtil.createLodProjectionMatrix(renderParams.mcProjectionMatrix);
			Mat4f dhModelViewMatrix = RenderUtil.createLodModelViewMatrix(renderParams.mcModelViewMatrix);
			
			Mat4f inverseDhMvmProjMatrix = new Mat4f(dhProjectionMatrix);
			inverseDhMvmProjMatrix.multiply(dhModelViewMatrix);
			inverseDhMvmProjMatrix.invert();
			
			
			
			// upload data //
			
			ByteBuffer buffer = ByteBuffer.allocateDirect(uniformBufferSize);
			buffer.order(ByteOrder.nativeOrder());
			buffer = Std140Builder.intoBuffer(buffer)
				.putFloat(fadeStartDistance) // uStartFadeBlockDistance
				.putFloat(fadeEndDistance) // uEndFadeBlockDistance
				.putMat4f(inverseDhMvmProjMatrix.createJomlMatrix()) // uDhInvMvmProj
				.get()
			;
			
			this.fragUniformBuffer = BlazeUniformUtil.createBuffer("fragUniformBlock", uniformBufferSize, this.fragUniformBuffer);
			GpuBufferSlice bufferSlice = new GpuBufferSlice(this.fragUniformBuffer, 0, uniformBufferSize);
			
			COMMAND_ENCODER.writeToBuffer(bufferSlice, buffer);
		}
		
		
		this.renderFadeToTexture();
		BlazeDhCopyRenderer.INSTANCE.render(this.dhFadeColorTextureWrapper, BlazeDhMetaRenderer.INSTANCE.dhColorTextureWrapper);
		
	}
	
	private void renderFadeToTexture()
	{
		try (RenderPass renderPass = COMMAND_ENCODER.createRenderPass(
			this::getRenderPassName,
			this.dhFadeColorTextureWrapper.textureView, 
			/*optionalClearColorAsInt*/ OptionalInt.empty(),
			/*depthTexture*/ null, 
			/*optionalDepthValueAsDouble*/ OptionalDouble.empty()))
		{
			// MC texture
			renderPass.bindTexture("uMcColorTexture", this.mcColorTextureViewWrapper.textureView, this.mcColorTextureViewWrapper.textureSampler);
			
			// DH textures
			renderPass.bindTexture("uDhDepthTexture", BlazeDhMetaRenderer.INSTANCE.dhDepthTextureWrapper.textureView, BlazeDhMetaRenderer.INSTANCE.dhDepthTextureWrapper.textureSampler);
			renderPass.bindTexture("uDhColorTexture", BlazeDhMetaRenderer.INSTANCE.dhColorTextureWrapper.textureView, BlazeDhMetaRenderer.INSTANCE.dhColorTextureWrapper.textureSampler);
			
			renderPass.setUniform("fragUniformBlock", this.fragUniformBuffer);
			
			renderPass.setVertexBuffer(0, this.vboGpuBuffer); // vertex buffer can only be "0" lol
			renderPass.setPipeline(this.pipeline);
			
			renderPass.draw(/*indexStart*/ 0, /*indexCount*/ 4);
		}
	}
	private String getRenderPassName() { return "distantHorizons:McFadeRenderer"; }
	
	
	//endregion
	
	
	
}
#endif