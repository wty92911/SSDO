#version 330 core
out vec3 FragColor;

in vec2 TexCoords;

uniform sampler2D ssaoInput;
uniform sampler2D texture_diffuse1;
void main() 
{
    vec2 texelSize = 1.0 / vec2(textureSize(ssaoInput, 0));
    vec3 result = vec3(0.0f);
    for (int x = -2; x < 2; ++x) 
    {
        for (int y = -2; y < 2; ++y) 
        {
            vec2 offset = vec2(float(x), float(y)) * texelSize;
            result += texture(ssaoInput, TexCoords + offset).rgb;
        }
    }
    FragColor = result / (4.0 * 4.0);
}  