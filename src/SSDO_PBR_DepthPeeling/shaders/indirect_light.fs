#version 330 core
out vec3 Fragcolor;
in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D gAlbedo;
uniform sampler2D texNoise;
uniform sampler2D directLight;
uniform sampler2D secondZ;

uniform vec3 samples[64];

const int kernelSize = 64;
const float radius = 0.5f;

const vec2 noiseScale = vec2(800.0f/4.0f, 600.0f/4.0f);
const float PI=3.14159265;
const float A_s=PI*radius*radius/kernelSize;
const float bias=0.025;
float roughness=0.5;
float metallic=0.5;
uniform mat4 projection;
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
float Clamp(float x)
{
	if(x<0.0f)	return 0.0f;
	if(x>1.0f)	return 1.0f;
	return x;
}
float len(vec3 vct)
{
	return vct.x*vct.x+vct.y*vct.y+vct.z*vct.z;
}
float Abs(const float x)
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
	vec3 fragPos = texture(gPosition, TexCoords).xyz;
	vec3 normal = texture(gNormal, TexCoords).rgb;
	vec3 randomVec = texture(texNoise, TexCoords*noiseScale).xyz;
	vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);
	
	vec3 Albedo=texture(gAlbedo, TexCoords).rgb;
	vec3 indir_light=vec3(0.0f);
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, Albedo, metallic);
    vec3 V=normalize(-fragPos);


    vec3 dir_light=texture(directLight, TexCoords).rgb;
    for(int i = 0; i < kernelSize; ++i)
    {
        vec3 sample = TBN * samples[i];
        sample = fragPos + sample * radius; 
        
        vec4 offset = vec4(sample, 1.0);
        offset = projection * offset;
        offset.xyz /= offset.w;
        offset.xyz = offset.xyz * 0.5 + 0.5;
        
        vec3 sample2 = texture(gPosition, offset.xy).xyz;
        vec3 sample3 = texture(secondZ, offset.xy).xyz;
		if(Occluded(sample.z, sample3.z, sample2.z))
		{
			vec3 L_pixel = texture(directLight, offset.xy).rgb;
			vec3 normal_sender = texture(gNormal, offset.xy).rgb;
			vec3 trans_dir = sample2 - fragPos;
			vec3 normalized_trans_dir = normalize(trans_dir);
                                                  
            vec3 L=normalized_trans_dir;
            vec3 H=normalize(L+V);
            vec3 N=normalize(normal);
            float NDF = DistributionGGX(N, H, roughness);   
            float G   = GeometrySmith(N, V, L, roughness);      
            vec3 F    = fresnelSchlick(clamp(dot(H, V),0.0,1.0), F0);
            vec3 numerator    = NDF * G * F; 
            float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
            vec3 specular = numerator / denominator;
            vec3 kS =F;
            vec3 kD = vec3(1.0) - kS;
            kD=vec3(1.0);
            kD *= 1.0 - metallic;

			float cos_s = max(dot(-normalized_trans_dir,normal_sender), 0.0f);
			float cos_r = max(dot(normalized_trans_dir,normal), 0.0f);
			float d_2=len(trans_dir);
			indir_light +=( kD*Albedo/PI+specular) * L_pixel * A_s * cos_s * cos_r*min(1.0f/d_2,10.0f) ;
		}
    }
	
	
	Fragcolor=dir_light+indir_light;

}