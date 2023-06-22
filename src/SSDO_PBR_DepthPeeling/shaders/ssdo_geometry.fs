#version 330 core
layout (location = 0) out vec3 gPosition;
layout (location = 1) out vec3 gNormal;
layout (location = 2) out vec3 gAlbedo;

uniform sampler2D texture_diffuse1;
uniform bool iscube;

in vec2 TexCoords;
in vec3 FragPos;
in vec3 Normal;

void main()
{    
    gPosition = FragPos;
    gNormal = normalize(Normal);
    gAlbedo.rgb = texture(texture_diffuse1,TexCoords).rgb;
}