#include <SFML/Window.hpp>
#include <GL/glew.h>
#include <gl/GLU.h>
#include <SFML/OpenGL.hpp>
#include <SFML/System.hpp>

// GLM
#include <glm/gtc/matrix_transform.hpp>

#include <fstream>
#include <iostream>


// horizontal angle : toward -Z
static float horizontalAngle = 3.14f;
// vertical angle : 0, look at the horizon
static float verticalAngle = 0.0f;

static glm::vec3 direction;
static glm::vec3 right;

// position
static glm::vec3 position = glm::vec3(0, 0, 5);
// Mouse Center
sf::Vector2i mouseCenter(860, 540);

static float deltaSpeed = 0;

// Initial Field of View
static float initialFoV = 90.0f;

static float speed = 0.03f;
static float mouseSpeed = 0.00005f;

// One color for each vertex. They were generated randomly.
static const GLfloat g_color_buffer_data[] = {
	0.583f, 0.771f, 0.014f,
	0.609f, 0.115f, 0.436f,
	0.327f, 0.483f, 0.844f,
	0.822f, 0.569f, 0.201f,
	0.435f, 0.602f, 0.223f,
	0.310f, 0.747f, 0.185f,
	0.597f, 0.770f, 0.761f,
	0.559f, 0.436f, 0.730f,
	0.359f, 0.583f, 0.152f,
	0.483f, 0.596f, 0.789f,
	0.559f, 0.861f, 0.639f,
	0.195f, 0.548f, 0.859f,
	0.014f, 0.184f, 0.576f,
	0.771f, 0.328f, 0.970f,
	0.406f, 0.615f, 0.116f,
	0.676f, 0.977f, 0.133f,
	0.971f, 0.572f, 0.833f,
	0.140f, 0.616f, 0.489f,
	0.997f, 0.513f, 0.064f,
	0.945f, 0.719f, 0.592f,
	0.543f, 0.021f, 0.978f,
	0.279f, 0.317f, 0.505f,
	0.167f, 0.620f, 0.077f,
	0.347f, 0.857f, 0.137f,
	0.055f, 0.953f, 0.042f,
	0.714f, 0.505f, 0.345f,
	0.783f, 0.290f, 0.734f,
	0.722f, 0.645f, 0.174f,
	0.302f, 0.455f, 0.848f,
	0.225f, 0.587f, 0.040f,
	0.517f, 0.713f, 0.338f,
	0.053f, 0.959f, 0.120f,
	0.393f, 0.621f, 0.362f,
	0.673f, 0.211f, 0.457f,
	0.820f, 0.883f, 0.371f,
	0.982f, 0.099f, 0.879f
};

// Our vertices. Tree consecutive floats give a 3D vertex; Three consecutive vertices give a triangle.
// A cube has 6 faces with 2 triangles each, so this makes 6*2=12 triangles, and 12*3 vertices
static const GLfloat g_vertex_buffer_data[] = {
	-1.0f, -1.0f, -1.0f, // triangle 1 : begin
	-1.0f, -1.0f, 1.0f,
	-1.0f, 1.0f, 1.0f, // triangle 1 : end
	1.0f, 1.0f, -1.0f, // triangle 2 : begin
	-1.0f, -1.0f, -1.0f,
	-1.0f, 1.0f, -1.0f, // triangle 2 : end
	1.0f, -1.0f, 1.0f,
	-1.0f, -1.0f, -1.0f,
	1.0f, -1.0f, -1.0f,
	1.0f, 1.0f, -1.0f,
	1.0f, -1.0f, -1.0f,
	-1.0f, -1.0f, -1.0f,
	-1.0f, -1.0f, -1.0f,
	-1.0f, 1.0f, 1.0f,
	-1.0f, 1.0f, -1.0f,
	1.0f, -1.0f, 1.0f,
	-1.0f, -1.0f, 1.0f,
	-1.0f, -1.0f, -1.0f,
	-1.0f, 1.0f, 1.0f,
	-1.0f, -1.0f, 1.0f,
	1.0f, -1.0f, 1.0f,
	1.0f, 1.0f, 1.0f,
	1.0f, -1.0f, -1.0f,
	1.0f, 1.0f, -1.0f,
	1.0f, -1.0f, -1.0f,
	1.0f, 1.0f, 1.0f,
	1.0f, -1.0f, 1.0f,
	1.0f, 1.0f, 1.0f,
	1.0f, 1.0f, -1.0f,
	-1.0f, 1.0f, -1.0f,
	1.0f, 1.0f, 1.0f,
	-1.0f, 1.0f, -1.0f,
	-1.0f, 1.0f, 1.0f,
	1.0f, 1.0f, 1.0f,
	-1.0f, 1.0f, 1.0f,
	1.0f, -1.0f, 1.0f
};

/*
bool loadOBJ(
const char * path,
std::vector < glm::vec3 > & out_vertices,
std::vector < glm::vec2 > & out_uvs,
std::vector < glm::vec3 > & out_normals
)
{
std::vector< unsigned int > vertexIndices, uvIndices, normalIndices;
std::vector< glm::vec3 > temp_vertices;
std::vector< glm::vec2 > temp_uvs;
std::vector< glm::vec3 > temp_normals;

FILE * file = fopen(path, "r");
if (file == NULL){
printf("Impossible to open the file !\n");
return false;
}

while (1){

char lineHeader[128];
// read the first word of the line
int res = fscanf(file, "%s", lineHeader);
if (res == EOF)
break; // EOF = End Of File. Quit the loop.

// else : parse lineHeader
if (strcmp(lineHeader, "v") == 0){
glm::vec3 vertex;
fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z);
temp_vertices.push_back(vertex);
}
else if (strcmp(lineHeader, "vt") == 0){
glm::vec2 uv;
fscanf(file, "%f %f\n", &uv.x, &uv.y);
temp_uvs.push_back(uv);
}
else if (strcmp(lineHeader, "vn") == 0){
glm::vec3 normal;
fscanf(file, "%f %f %f\n", &normal.x, &normal.y, &normal.z);
temp_normals.push_back(normal);
}
else if (strcmp(lineHeader, "f") == 0){
std::string vertex1, vertex2, vertex3;
unsigned int vertexIndex[3], uvIndex[3], normalIndex[3];
int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vertexIndex[0], &uvIndex[0], &normalIndex[0], &vertexIndex[1], &uvIndex[1], &normalIndex[1], &vertexIndex[2], &uvIndex[2], &normalIndex[2]);
if (matches != 9){
printf("File can't be read by our simple parser : ( Try exporting with other options\n");
return false;
}
vertexIndices.push_back(vertexIndex[0]);
vertexIndices.push_back(vertexIndex[1]);
vertexIndices.push_back(vertexIndex[2]);
uvIndices.push_back(uvIndex[0]);
uvIndices.push_back(uvIndex[1]);
uvIndices.push_back(uvIndex[2]);
normalIndices.push_back(normalIndex[0]);
normalIndices.push_back(normalIndex[1]);
normalIndices.push_back(normalIndex[2]);
// For each vertex of each triangle
for (unsigned int i = 0; i < vertexIndices.size(); i++){
unsigned int vertexIndex = vertexIndices[i];
glm::vec3 vertex = temp_vertices[vertexIndex - 1];
out_vertices.push_back(vertex);
}
}
}
}
*/