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
public class BlazeVanillaFadeRenderer {}

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
import com.seibel.distanthorizons.core.config.Config;
import com.seibel.distanthorizons.core.dependencyInjection.SingletonInjector;
import com.seibel.distanthorizons.core.logging.DhLogger;
import com.seibel.distanthorizons.core.logging.DhLoggerBuilder;
import com.seibel.distanthorizons.core.render.RenderParams;
import com.seibel.distanthorizons.core.util.RenderUtil;
import com.seibel.distanthorizons.core.util.math.Mat4f;
import com.seibel.distanthorizons.core.wrapperInterfaces.minecraft.IMinecraftRenderWrapper;
import com.seibel.distanthorizons.core.wrapperInterfaces.render.renderPass.IDhVanillaFadeRenderer;
import net.minecraft.client.Minecraft;
import net.minecraft.resources.Identifier;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.OptionalDouble;
import java.util.OptionalInt;

/**
 * Fades the vanilla chunks
 * into DH's LODs.
 */
public class BlazeVanillaFadeRenderer implements IDhVanillaFadeRenderer
{
	public static final DhLogger LOGGER = new DhLoggerBuilder().build(); 
	
	private static final IMinecraftRenderWrapper MC_RENDER = SingletonInjector.INSTANCE.get(IMinecraftRenderWrapper.class);
	
	private static final GpuDevice GPU_DEVICE = RenderSystem.getDevice();
	private static final CommandEncoder COMMAND_ENCODER = GPU_DEVICE.createCommandEncoder();
	
	public static final BlazeVanillaFadeRenderer INSTANCE = new BlazeVanillaFadeRenderer();
	
	private RenderPipeline pipeline;
	private boolean init = false;
	
	private GpuBuffer fragUniformBuffer;
	
	private GpuBuffer vboGpuBuffer;
	
	public final BlazeTextureWrapper fadeColorTextureWrapper = BlazeTextureWrapper.createColor("DhVanillaFadeTexture");
	
	public final BlazeTextureViewWrapper mcDepthTextureWrapper = new BlazeTextureViewWrapper();
	public final BlazeTextureViewWrapper mcColorTextureWrapper = new BlazeTextureViewWrapper();
	
	
	
	//=============//
	// constructor //
	//=============//
	//region
	
	private BlazeVanillaFadeRenderer() { }
	
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
			pipelineBuilder.withLocation(Identifier.parse("distanthorizons:mc_vanilla_fade_render"));
			
			pipelineBuilder.withVertexShader(Identifier.fromNamespaceAndPath("distanthorizons", "fade/blaze/vert"));
			pipelineBuilder.withFragmentShader(Identifier.fromNamespaceAndPath("distanthorizons", "fade/blaze/vanilla_fade"));
			
			pipelineBuilder.withSampler("uMcDepthTexture");
			pipelineBuilder.withSampler("uCombinedMcDhColorTexture");
			
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
		this.fadeColorTextureWrapper.tryCreateOrResize();
		
		this.mcDepthTextureWrapper.tryWrap(Minecraft.getInstance().getMainRenderTarget().getDepthTexture());
		this.mcColorTextureWrapper.tryWrap(Minecraft.getInstance().getMainRenderTarget().getColorTexture());
		
		
		{
			int uniformBufferSize = new Std140SizeCalculator()
				.putInt() // uOnlyRenderLods
				.putFloat() // uStartFadeBlockDistance
				.putFloat() // uEndFadeBlockDistance
				.putFloat() // uMaxLevelHeight
				.putMat4f() // uDhInvMvmProj
				.putMat4f() // uMcInvMvmProj
				.get();
			
			
			// create data //
			
			float dhNearClipDistance = RenderUtil.getNearClipPlaneInBlocks();
			// this added value prevents the near clip plane and discard circle from touching, which looks bad
			dhNearClipDistance += 16f;
			
			// measured in blocks
			// these multipliers in James' tests should provide a fairly smooth transition
			// without having underdraw issues
			float fadeStartDistance = dhNearClipDistance * 1.5f;
			float fadeEndDistance = dhNearClipDistance * 1.9f;
			
			
			Mat4f inverseMcModelViewProjectionMatrix = new Mat4f(renderParams.mcProjectionMatrix);
			inverseMcModelViewProjectionMatrix.multiply(renderParams.mcModelViewMatrix);
			inverseMcModelViewProjectionMatrix.invert();
			Mat4f inverseMcMvmProjMatrix = inverseMcModelViewProjectionMatrix;
			
			
			Mat4f dhProjectionMatrix = RenderUtil.createLodProjectionMatrix(renderParams.mcProjectionMatrix);
			Mat4f dhModelViewMatrix = RenderUtil.createLodModelViewMatrix(renderParams.mcModelViewMatrix);
			
			Mat4f inverseDhModelViewProjectionMatrix = new Mat4f(dhProjectionMatrix);
			inverseDhModelViewProjectionMatrix.multiply(dhModelViewMatrix);
			inverseDhModelViewProjectionMatrix.invert();
			Mat4f inverseDhMvmProjMatrix = inverseDhModelViewProjectionMatrix;
			
			
			
			// upload data //
			
			ByteBuffer buffer = ByteBuffer.allocateDirect(uniformBufferSize);
			buffer.order(ByteOrder.nativeOrder());
			buffer = Std140Builder.intoBuffer(buffer)
				.putInt(Config.Client.Advanced.Debugging.lodOnlyMode.get() ? 1 : 0) // uOnlyRenderLods
				.putFloat(fadeStartDistance) // uStartFadeBlockDistance
				.putFloat(fadeEndDistance) // uEndFadeBlockDistance
				.putFloat(renderParams.clientLevelWrapper.getMaxHeight()) // uMaxLevelHeight
				.putMat4f(inverseDhMvmProjMatrix.createJomlMatrix()) // uDhInvMvmProj
				.putMat4f(inverseMcMvmProjMatrix.createJomlMatrix()) // uMcInvMvmProj
				.get()
			;
			
			this.fragUniformBuffer = BlazeUniformUtil.createBuffer("fragUniformBlock", uniformBufferSize, this.fragUniformBuffer);
			GpuBufferSlice bufferSlice = new GpuBufferSlice(this.fragUniformBuffer, 0, uniformBufferSize);
			
			COMMAND_ENCODER.writeToBuffer(bufferSlice, buffer);
		}
		
		
		this.renderFadeToTexture();
		BlazeDhCopyRenderer.INSTANCE.render(this.fadeColorTextureWrapper, this.mcColorTextureWrapper);
		
	}
	
	private void renderFadeToTexture()
	{
		try (RenderPass renderPass = COMMAND_ENCODER.createRenderPass(
			this::getRenderPassName,
			this.fadeColorTextureWrapper.textureView,
			/*optionalClearColorAsInt*/ OptionalInt.empty(),
			/*depthTexture*/ null, 
			/*optionalDepthValueAsDouble*/ OptionalDouble.empty()))
		{
			renderPass.bindTexture("uMcDepthTexture", this.mcDepthTextureWrapper.textureView, this.mcDepthTextureWrapper.textureSampler);
			renderPass.bindTexture("uCombinedMcDhColorTexture", this.mcColorTextureWrapper.textureView, this.mcColorTextureWrapper.textureSampler);
			
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