#version 330 core
out vec3 FragColor;

in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedo;
uniform sampler2D texNoise;
uniform vec3 lightcolor;
uniform int samplecount;
uniform vec3 samples[256];
float occlusion[256];
uniform mat4 projection;
uniform float radius;
const float bias=0.025;
const vec2 noiseScale = vec2(800.0/4.0, 600.0/4.0); 
const float Pi=3.1415926;
void main()
{             
    vec3 lighting;    
    vec3 fragPos = texture(gPosition, TexCoords).xyz;
    vec3 normal = normalize(texture(gNormal, TexCoords).rgb);
    vec3 randomVec = normalize(texture(texNoise, TexCoords * noiseScale).xyz);
 
    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);
    vec3 Diffuse = texture(gAlbedo, TexCoords).rgb;
    for(int i = 0; i < samplecount; ++i)
    {
      // get sample position
      vec3 samplePos = TBN * samples[i]; // from tangent to view-space
      samplePos = fragPos + samplePos * radius; 
      
      // project sample position (to sample texture) (to get position on screen/texture)
      vec4 offset = vec4(samplePos, 1.0);
      offset = projection * offset; // from view to clip-space
      offset.xyz /= offset.w; // perspective divide
      offset.xyz = offset.xyz * 0.5 + 0.5; // transform to range 0.0 - 1.0
      
      // get sample depth
      float sampleDepth = texture(gPosition, offset.xy).z; // get depth value of kernel sample
      occlusion[i]=(sampleDepth >= samplePos.z+bias ? 0.0 : 1.0);
      vec3 lightDir = normalize(samplePos - fragPos);
      vec3 diffuse = max(dot(normal,lightDir),0.0)*lightcolor*Diffuse;
      //direct-light calculation
      lighting+=occlusion[i]*2*diffuse/samplecount;

    }

    FragColor = lighting;

}