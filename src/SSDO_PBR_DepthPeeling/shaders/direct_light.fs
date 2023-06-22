#version 330 core
out vec3 FragColor;

in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedo;
uniform sampler2D texNoise;
uniform sampler2D secondZ;
uniform vec3 lightcolor;
uniform vec3 samples[64];
float occlusion[64];
uniform mat4 projection;
int kernelSize = 64;
float radius = 0.5f;
const float bias=0.025;
const vec2 noiseScale = vec2(800.0/4.0, 600.0/4.0); 
const float Pi=3.1415926;
float Abs(float x)
{
	return x<0?-x:x;
}
bool Occluded(float x, float l, float r)
{
    if(l==r)
        return ( x <= r-bias );
    else
        return ( x <= r-bias ) && ( x >= l+bias );
}
void main()
{             
    vec3 lighting=vec3(0.0f);
    vec3 fragPos = texture(gPosition, TexCoords).xyz;
    vec3 normal = normalize(texture(gNormal, TexCoords).rgb);
    vec3 randomVec = normalize(texture(texNoise, TexCoords * noiseScale).xyz);
 
    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);
    vec3 Diffuse = texture(gAlbedo, TexCoords).rgb;
    for(int i = 0; i < kernelSize; ++i)
    {
        vec3 samplePos = TBN * samples[i];
        samplePos = fragPos + samplePos * radius; 
        
        vec4 offset = vec4(samplePos, 1.0);
        offset = projection * offset; 
        offset.xyz /= offset.w;
        offset.xyz = offset.xyz * 0.5 + 0.5; 
        
        float sampleDepth = texture(gPosition, offset.xy).z;
        float sampleDepth2 = texture(secondZ, offset.xy).z;
        occlusion[i]=(Occluded(samplePos.z, sampleDepth2, sampleDepth)? 0.0 : 1.0);

        vec3 lightDir = normalize(samplePos - fragPos);
        vec3 diffuse = max(dot(normal,lightDir),0.0)*lightcolor*Diffuse;
        lighting+=occlusion[i]*2*diffuse/kernelSize;
 
      }

    FragColor = lighting;
}