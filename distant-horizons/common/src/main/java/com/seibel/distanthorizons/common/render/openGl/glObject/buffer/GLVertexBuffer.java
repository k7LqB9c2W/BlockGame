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

package com.seibel.distanthorizons.common.render.openGl.glObject.buffer;

import java.nio.ByteBuffer;

import com.seibel.distanthorizons.common.render.openGl.glObject.GLProxy;
import com.seibel.distanthorizons.core.dataObjects.render.bufferBuilding.LodQuadBuilder;
import com.seibel.distanthorizons.core.wrapperInterfaces.render.objects.IVertexBufferWrapper;
import org.lwjgl.opengl.GL32;

import com.seibel.distanthorizons.api.enums.config.EDhApiGpuUploadMethod;

/**
 * This is a container for a OpenGL
 * VBO (Vertex Buffer Object).
 *
 * @author James Seibel
 * @version 11-20-2021
 */
public class GLVertexBuffer extends GLBuffer implements IVertexBufferWrapper
{
	/**
	 * When uploading to a buffer that is too small, recreate it this many times
	 * bigger than the upload payload
	 */
	protected int vertexCount = 0;
	public int getVertexCount() { return this.vertexCount; }
	
	
	
	//=============//
	// constructor //
	//=============//
	//region
	
	public GLVertexBuffer() { this(GLProxy.getInstance().getGpuUploadMethod() == EDhApiGpuUploadMethod.BUFFER_STORAGE); }
	public GLVertexBuffer(boolean isBufferStorage) { super(isBufferStorage); }
	
	//endregion
	
	
	
	//======================//
	// uploading/destroying //
	//======================//
	//region
	
	@Override
	public int getBufferBindingTarget() { return GL32.GL_ARRAY_BUFFER; }
	
	@Override
	public void upload(ByteBuffer buffer, int vertexCount)
	{
		EDhApiGpuUploadMethod uploadMethod = GLProxy.getInstance().getGpuUploadMethod();
		int maxBufferSize = LodQuadBuilder.getMaxBufferByteSize();
		this.uploadBuffer(buffer, vertexCount, uploadMethod, maxBufferSize);
	}
	
	/**
	 * bufferSize is the number of shared verticies. <br>
	 * This number will be higher when actually rendered since each box's face needs 2 triangles 
	 * with 2 shared verticies. 
	 */
	public void uploadBuffer(ByteBuffer byteBuffer, int vertexCount, EDhApiGpuUploadMethod uploadMethod, int maxExpansionSize)
	{
		if (vertexCount < 0)
		{
			throw new IllegalArgumentException("vertexCount is negative!");
		}
		
		// If size is zero, just ignore it.
		if (byteBuffer.limit() - byteBuffer.position() != 0)
		{
			boolean useBuffStorage = uploadMethod.useBufferStorage;
			super.uploadBuffer(byteBuffer, uploadMethod, maxExpansionSize, useBuffStorage ? 0 : GL32.GL_STATIC_DRAW);
		}
		this.vertexCount = vertexCount;
	}
	
	
	@Override
	public void close() { this.destroyAsync(); }
	@Override
	public void destroyAsync()
	{
		super.destroyAsync();
		this.vertexCount = 0;
	}
	
	//endregion
	
	
	
}