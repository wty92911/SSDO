
#ifndef MYSSAO_SSAO_HPP
#define MYSSAO_SSAO_HPP
#include <config.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <learnopengl/filesystem.h>
#include <learnopengl/shader.h>
#include <learnopengl/camera.h>
#include <learnopengl/model.h>

 
const float PI = 3.1415926;
const float sigma = 1.0f;
float gauss[3][3];

class Config{
public:
    bool enable_ssdo = true;
};

class SSAO{
public:

    class Shaders{
    public:
        Shader *dir, *indir, *geo, *blur, *lighting;

        ~Shaders(){
            delete dir;
            delete indir;
            delete geo;
            delete blur;
            delete lighting;
        }
    }shaders;

    std::vector<Model> models;

    glm::vec3 lightPos = glm::vec3(2.0, 4.0, -2.0);
    glm::vec3 lightColor = glm::vec3(0.5, 0.5, 0.5);

    class Buffers{
        public:
        unsigned int gBuffer;
        unsigned int gPosition, gNormal, gAlbedo;
        unsigned int attachments[3] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
        unsigned int rboDepth;
        unsigned int ssdoFBO;
        unsigned int ssdoColorBuffer;
        unsigned int ssdoFBO2, ssdoBlurFBO;
        unsigned int ssdoColorBuffer2, ssdoColorBufferBlur;
        unsigned int noiseTexture;
    }buf;

    bool window_resized = true; // MUST start at true

    std::vector<glm::vec3> ssdoKernel;
    std::vector<glm::vec3> ssdoNoise;

    int SCR_WIDTH = 0;
    int SCR_HEIGHT = 0;
    
    glm::mat4 projection;
    glm::mat4 view;
    glm::mat4 model;

    SSAO(){
        for (int i = 0;i < 3;i++){
            for (int j = 0;j < 3;j++)
            {
                gauss[i][j] = exp(-float((i - 1) * (i - 1) + (j - 1) * (j - 1)) / (2.0f * sigma * sigma)) / (2.0f * PI * sigma * sigma);
            }
        }
    }

    void set_context(glm::mat4 projection, glm::mat4 view, glm::mat4 model){
        this->projection = projection;
        this->view = view;
        this->model = model;
    }

    void set_shaders(Shader *dir, Shader *indir, Shader *geo, Shader *blur, Shader *lighting){
        shaders.dir = dir;
        shaders.indir = indir;
        shaders.geo = geo;
        shaders.blur = blur;
        shaders.lighting = lighting;
    }

    void set_shader_dir(const std::string dir){
        set_shaders(new Shader((dir + "/ssdo.vs").c_str(), (dir + "/direct_light.fs").c_str()),
                    new Shader((dir + "/ssdo.vs").c_str(), (dir + "/indirect_light.fs").c_str()),
                    new Shader((dir + "/geometry.vs").c_str(), (dir + "/geometry.fs").c_str()),
                    new Shader((dir + "/ssdo.vs").c_str(), (dir + "/blur.fs").c_str()),
                    new Shader((dir + "/ssdo.vs").c_str(), (dir + "/lighting.fs").c_str())
        );
    }

    void add_model(const Model & model){
        models.push_back(model);
        std::cout << model.textures_loaded.size() << std::endl;
    }

    void init(){
        make_kernels();
        init_shaders();
    }

    void init_shaders(){
        shaders.dir->use();
        shaders.dir->setInt("gPosition", 0);
        shaders.dir->setInt("gNormal", 1);
        shaders.dir->setInt("gAlbedo", 2);
        shaders.dir->setInt("texNoise", 3);

        shaders.indir->use();
        shaders.indir->setInt("gPosition", 0);
        shaders.indir->setInt("gNormal", 1);
        shaders.indir->setInt("gAlbedo", 2);
        shaders.indir->setInt("texNoise", 3);
        shaders.indir->setInt("directLight", 4);

        shaders.blur->use();
        shaders.blur->setInt("ssdoInput", 0);

        shaders.lighting->use();
        shaders.lighting->setInt("gPosition", 0);
        shaders.lighting->setInt("gNormal", 1);
        shaders.lighting->setInt("gAlbedo", 2);
        shaders.lighting->setInt("ssdo", 3);
    }

    void make_kernels(){
        std::uniform_real_distribution<GLfloat> randomFloats(0.0, 1.0);
        std::default_random_engine generator;
        for (unsigned int i = 0; i < 64; ++i)
        {
            glm::vec3 sample(randomFloats(generator) * 2.0 - 1.0, randomFloats(generator) * 2.0 - 1.0, randomFloats(generator));
            sample = glm::normalize(sample);
            sample *= randomFloats(generator);
            float scale = float(i) / 64.0f;
            scale = lerp(0.1f, 1.0f, scale * scale);
            sample *= scale;
            ssdoKernel.push_back(sample);
        }

        
        for (unsigned int i = 0; i < 16; i++)
        {
            glm::vec3 noise(randomFloats(generator) * 2.0 - 1.0, randomFloats(generator) * 2.0 - 1.0, 0.0f); // rotate around z-axis (in tangent space)
            ssdoNoise.push_back(noise);
        }
    }

    void window_size(int h, int w){
        if(h != SCR_HEIGHT || w != SCR_WIDTH){
            window_resized = true;
        }
        SCR_HEIGHT = h;
        SCR_WIDTH = w;
    }

    void regen_buffers(){
        glGenFramebuffers(1, &buf.gBuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, buf.gBuffer);
        
        glGenTextures(1, &buf.gPosition);
        glBindTexture(GL_TEXTURE_2D, buf.gPosition);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, buf.gPosition, 0);
        // normal 
        glGenTextures(1, &buf.gNormal);
        glBindTexture(GL_TEXTURE_2D, buf.gNormal);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, buf.gNormal, 0);
        // color
        glGenTextures(1, &buf.gAlbedo);
        glBindTexture(GL_TEXTURE_2D, buf.gAlbedo);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, buf.gAlbedo, 0);
    
        
        glDrawBuffers(3, buf.attachments);

        
        glGenRenderbuffers(1, &buf.rboDepth);
        glBindRenderbuffer(GL_RENDERBUFFER, buf.rboDepth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, SCR_WIDTH, SCR_HEIGHT);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, buf.rboDepth);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "Framebuffer not complete!" << std::endl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        
        glGenFramebuffers(1, &buf.ssdoFBO);
        glBindFramebuffer(GL_FRAMEBUFFER, buf.ssdoFBO);
        
        // SSDO color buffer
        glGenTextures(1, &buf.ssdoColorBuffer);
        glBindTexture(GL_TEXTURE_2D, buf.ssdoColorBuffer);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, buf.ssdoColorBuffer, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "SSDO Framebuffer not complete!" << std::endl;

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        
        glGenFramebuffers(1, &buf.ssdoFBO2);glGenFramebuffers(1, &buf.ssdoBlurFBO);
        glBindFramebuffer(GL_FRAMEBUFFER, buf.ssdoFBO2);
        
        glGenTextures(1, &buf.ssdoColorBuffer2);
        glBindTexture(GL_TEXTURE_2D,buf. ssdoColorBuffer2);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, buf.ssdoColorBuffer2, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "SSDO Framebuffer not complete!" << std::endl;
        // make some blur
        glBindFramebuffer(GL_FRAMEBUFFER, buf.ssdoBlurFBO);
        glGenTextures(1, &buf.ssdoColorBufferBlur);
        glBindTexture(GL_TEXTURE_2D, buf.ssdoColorBufferBlur);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, buf.ssdoColorBufferBlur, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "SSAO Blur Framebuffer not complete!" << std::endl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glGenTextures(1, &buf.noiseTexture);
        glBindTexture(GL_TEXTURE_2D, buf.noiseTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 4, 4, 0, GL_RGB, GL_FLOAT, &ssdoNoise[0]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        //glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void draw(){

        //if(window_resized){
        //    window_resized = false;
        //    regen_buffers();
        //}

        //glBindFramebuffer(GL_FRAMEBUFFER, buf.gBuffer);
        shaders.geo->use();
        shaders.geo->setMat4("projection", projection);
        shaders.geo->setMat4("view", view);
        shaders.geo->setInt("invertedNormals", 0);
        shaders.geo->setMat4("model", model);

        for (auto m: models){
            m.Draw(*shaders.geo);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        //directlight
        glBindFramebuffer(GL_FRAMEBUFFER, buf.ssdoFBO);
        glClear(GL_COLOR_BUFFER_BIT);
        shaders.dir->use();
        for (unsigned int i = 0; i < 64; ++i)
            shaders.dir->setVec3("samples[" + std::to_string(i) + "]", ssdoKernel[i]);
        shaders.dir->setMat4("projection", projection);
        glm::vec3 lightPosView = glm::vec3(view * glm::vec4(lightPos, 1.0));
        shaders.dir->setVec3("lightcolor", lightColor);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, buf.gPosition);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, buf.gNormal);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, buf.gAlbedo);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, buf.noiseTexture);
        renderQuad();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        //indireclight
        glBindFramebuffer(GL_FRAMEBUFFER, buf.ssdoFBO2);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        shaders.indir->use();
        for (unsigned int i = 0; i < 64; ++i)
            shaders.indir->setVec3("samples[" + std::to_string(i) + "]", ssdoKernel[i]);
        shaders.indir->setMat4("projection", projection);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, buf.gPosition);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, buf.gNormal);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, buf.gAlbedo);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, buf.noiseTexture);
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, buf.ssdoColorBuffer);
        renderQuad();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        //blur stage
        glBindFramebuffer(GL_FRAMEBUFFER, buf.ssdoBlurFBO);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        shaders.blur->use();
        for (unsigned int i = 0;i < 9;i++)
        {
            shaders.blur->setFloat("gauss[" + std::to_string(i) + "]", gauss[i / 3][i % 3]);
        }
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, buf.ssdoColorBuffer2);
        renderQuad();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);



        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        shaders.lighting->use();
        shaders.lighting->setVec3("light.Position", lightPosView);
        shaders.lighting->setVec3("light.Color", lightColor);
        const float linear = 0.09f;
        const float quadratic = 0.032f;
        shaders.lighting->setFloat("light.Linear", linear);
        shaders.lighting->setFloat("light.Quadratic", quadratic);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, buf.gPosition);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, buf.gNormal);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, buf.gAlbedo);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, buf.ssdoColorBufferBlur);
        renderQuad();
    }

    ~SSAO(){

    }

    float lerp(float a, float b, float f) {
    return a + f * (b - a);
    }

    unsigned int cubeVAO = 0;
    unsigned int cubeVBO = 0;

    unsigned int quadVAO = 0;
    unsigned int quadVBO;
    void renderQuad() {
        if (quadVAO == 0) {
            float quadVertices[] = {
                -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
                -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
                1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
                1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
            };

            glGenVertexArrays(1, &quadVAO);
            glGenBuffers(1, &quadVBO);
            glBindVertexArray(quadVAO);
            glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
            glEnableVertexAttribArray(1);
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
        }
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glBindVertexArray(0);
    }
};


#endif  // MYSSAO_SSAO_HPP