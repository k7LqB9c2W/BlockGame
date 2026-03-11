cbuffer CloudConstants : register(b0)
{
    float4x4 uViewProj;
    float4 uCameraPosTime;
    float4 uLayerParams;
    float4 uShapeParams;
    float4 uTopColor;
    float4 uBottomColor;
}

struct VSOutput
{
    float4 position : SV_POSITION;
    float3 worldPos : POSITION0;
    float3 localPos : TEXCOORD0;
    float3 normal : NORMAL0;
    float coverage : TEXCOORD1;
};

float hash1(float2 p)
{
    return frac(sin(dot(p, float2(127.1f, 311.7f))) * 43758.5453123f);
}

float hash2(float2 p)
{
    return frac(sin(dot(p, float2(269.5f, 183.3f))) * 24634.6345234f);
}

float hash3(float2 p)
{
    return frac(sin(dot(p, float2(419.2f, 371.9f))) * 32514.1276123f);
}

void makeBoxVertex(uint vertexIndex, out float3 localPos, out float3 normal)
{
    const float3 positions[36] = {
        float3(-0.5f,  0.5f, -0.5f), float3( 0.5f,  0.5f, -0.5f), float3( 0.5f,  0.5f,  0.5f),
        float3(-0.5f,  0.5f, -0.5f), float3( 0.5f,  0.5f,  0.5f), float3(-0.5f,  0.5f,  0.5f),
        float3(-0.5f, -0.5f,  0.5f), float3( 0.5f, -0.5f,  0.5f), float3( 0.5f, -0.5f, -0.5f),
        float3(-0.5f, -0.5f,  0.5f), float3( 0.5f, -0.5f, -0.5f), float3(-0.5f, -0.5f, -0.5f),
        float3(-0.5f, -0.5f, -0.5f), float3( 0.5f, -0.5f, -0.5f), float3( 0.5f,  0.5f, -0.5f),
        float3(-0.5f, -0.5f, -0.5f), float3( 0.5f,  0.5f, -0.5f), float3(-0.5f,  0.5f, -0.5f),
        float3( 0.5f, -0.5f,  0.5f), float3(-0.5f, -0.5f,  0.5f), float3(-0.5f,  0.5f,  0.5f),
        float3( 0.5f, -0.5f,  0.5f), float3(-0.5f,  0.5f,  0.5f), float3( 0.5f,  0.5f,  0.5f),
        float3(-0.5f, -0.5f,  0.5f), float3(-0.5f, -0.5f, -0.5f), float3(-0.5f,  0.5f, -0.5f),
        float3(-0.5f, -0.5f,  0.5f), float3(-0.5f,  0.5f, -0.5f), float3(-0.5f,  0.5f,  0.5f),
        float3( 0.5f, -0.5f, -0.5f), float3( 0.5f, -0.5f,  0.5f), float3( 0.5f,  0.5f,  0.5f),
        float3( 0.5f, -0.5f, -0.5f), float3( 0.5f,  0.5f,  0.5f), float3( 0.5f,  0.5f, -0.5f)
    };

    const float3 normals[36] = {
        float3(0.0f, 1.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), float3(0.0f, 1.0f, 0.0f),
        float3(0.0f, 1.0f, 0.0f), float3(0.0f, 1.0f, 0.0f), float3(0.0f, 1.0f, 0.0f),
        float3(0.0f, -1.0f, 0.0f), float3(0.0f, -1.0f, 0.0f), float3(0.0f, -1.0f, 0.0f),
        float3(0.0f, -1.0f, 0.0f), float3(0.0f, -1.0f, 0.0f), float3(0.0f, -1.0f, 0.0f),
        float3(0.0f, 0.0f, -1.0f), float3(0.0f, 0.0f, -1.0f), float3(0.0f, 0.0f, -1.0f),
        float3(0.0f, 0.0f, -1.0f), float3(0.0f, 0.0f, -1.0f), float3(0.0f, 0.0f, -1.0f),
        float3(0.0f, 0.0f, 1.0f), float3(0.0f, 0.0f, 1.0f), float3(0.0f, 0.0f, 1.0f),
        float3(0.0f, 0.0f, 1.0f), float3(0.0f, 0.0f, 1.0f), float3(0.0f, 0.0f, 1.0f),
        float3(-1.0f, 0.0f, 0.0f), float3(-1.0f, 0.0f, 0.0f), float3(-1.0f, 0.0f, 0.0f),
        float3(-1.0f, 0.0f, 0.0f), float3(-1.0f, 0.0f, 0.0f), float3(-1.0f, 0.0f, 0.0f),
        float3(1.0f, 0.0f, 0.0f), float3(1.0f, 0.0f, 0.0f), float3(1.0f, 0.0f, 0.0f),
        float3(1.0f, 0.0f, 0.0f), float3(1.0f, 0.0f, 0.0f), float3(1.0f, 0.0f, 0.0f)
    };

    localPos = positions[vertexIndex];
    normal = normals[vertexIndex];
}

VSOutput main(uint vertexId : SV_VertexID, uint instanceId : SV_InstanceID)
{
    VSOutput output;
    output.position = 0.0f.xxxx;
    output.worldPos = 0.0f.xxx;
    output.localPos = 0.0f.xxx;
    output.normal = float3(0.0f, 1.0f, 0.0f);
    output.coverage = 0.0f;

    const uint prismIndex = instanceId % 3u;
    const uint cloudIndex = instanceId / 3u;
    const float radiusCells = uLayerParams.w;
    const uint gridSize = uint(radiusCells * 2.0f + 1.0f);
    const uint localX = cloudIndex % gridSize;
    const uint localZ = cloudIndex / gridSize;

    const float spacing = uLayerParams.z;
    const float2 windOffset = float2(uCameraPosTime.w * uShapeParams.y, uCameraPosTime.w * uShapeParams.y * 0.18f);
    const float2 anchor = (uCameraPosTime.xz + windOffset) / spacing;
    const int2 baseCell = int2(floor(anchor));
    const int2 cell = baseCell + int2(int(localX) - int(radiusCells), int(localZ) - int(radiusCells));
    const float2 cellCoord = float2(cell);

    const float coverageSeed = hash1(cellCoord);
    float cloudPresent = step(uShapeParams.x, coverageSeed);
    const float sizeSeedA = hash2(cellCoord + 11.0f);
    const float sizeSeedB = hash3(cellCoord + 19.0f);
    const float offsetSeedA = hash1(cellCoord + 37.0f);
    const float offsetSeedB = hash2(cellCoord + 53.0f);
    const float heightSeed = hash3(cellCoord + 71.0f);

    float2 centerXZ = (cellCoord + 0.5f) * spacing - windOffset;
    const float layerY = uLayerParams.x + floor(heightSeed * 2.0f) * 1.0f;
    const float thickness = uLayerParams.y;

    float2 prismSize = float2(0.0f, 0.0f);
    float2 prismOffset = float2(0.0f, 0.0f);

    if (prismIndex == 0u)
    {
        prismSize = float2(16.0f + floor(sizeSeedA * 4.0f) * 4.0f,
                           12.0f + floor(sizeSeedB * 4.0f) * 4.0f);
    }
    else if (prismIndex == 1u)
    {
        prismSize = float2(8.0f + floor(sizeSeedB * 3.0f) * 4.0f,
                           8.0f + floor(sizeSeedA * 2.0f) * 4.0f);
        prismOffset = float2((offsetSeedA < 0.5f ? -1.0f : 1.0f) * (8.0f + prismSize.x * 0.35f),
                             (offsetSeedB - 0.5f) * 6.0f);
        cloudPresent *= step(0.20f, offsetSeedA);
    }
    else
    {
        prismSize = float2(8.0f + floor(sizeSeedA * 2.0f) * 4.0f,
                           8.0f + floor(sizeSeedB * 3.0f) * 4.0f);
        prismOffset = float2((offsetSeedA - 0.5f) * 6.0f,
                             (offsetSeedB < 0.5f ? -1.0f : 1.0f) * (8.0f + prismSize.y * 0.35f));
        cloudPresent *= step(0.24f, offsetSeedB);
    }

    centerXZ += prismOffset;

    float3 boxVertex;
    float3 boxNormal;
    makeBoxVertex(vertexId % 36u, boxVertex, boxNormal);

    const float3 halfExtents = float3(prismSize.x * 0.5f, thickness * 0.5f, prismSize.y * 0.5f) * cloudPresent;
    const float3 worldPos = float3(centerXZ.x, layerY, centerXZ.y) + boxVertex * halfExtents * 2.0f;

    output.worldPos = worldPos;
    output.localPos = boxVertex;
    output.normal = boxNormal;
    output.coverage = cloudPresent;
    output.position = mul(uViewProj, float4(worldPos, 1.0f));
    return output;
}
