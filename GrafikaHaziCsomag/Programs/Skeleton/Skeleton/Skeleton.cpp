//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Nagy Erik
// Neptun : ILF5H9
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";


// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) {
		shininess = _shininess; 
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

mat4 transpose(mat4 mat) {
	vec4 i(0, 0, 0, 0), j(0, 0, 0, 0), k(0, 0, 0, 0), l(0, 0, 0, 0);
	for (int row = 0; row < 4;row++) {
		vec4 r = mat[row];
		i[row] = r[0];
		j[row] = r[1];
		k[row] = r[2];
		l[row] = r[3];
	}
	return mat4(i, j, k, l);
}

vec4 operator*(mat4 mat, vec4 vec) {
	vec4 result(0, 0, 0, 0);
	for (int i = 0; i < 4;i++) {
		//ez egy sora a matrixnak 
		result[i] = mat[i][0] * vec[0] + mat[i][1] * vec[1] + mat[i][2] * vec[2] + mat[i][3] * vec[3];
	}
	return result;
}


mat4 one = mat4(1, 0, 0, 0,
	0, 1, 0, 0,
	0, 0, 1, 0,
	0, 0, 0, 1);

struct Quadrics :public Intersectable {
protected:
	 mat4 Q;
	 mat4 TransformMatrix = one;
	 mat4 InverzMatrix = one;
	 vec3 rotationAxis;
	 float rotationAngle = 0;
	 bool forClose;
	 mat4 Object;
	 vec3 translate;
	 float zmin, zmax;
	 vec3 plane1, plane2;
public:

	Quadrics(mat4 _Q, float _zmin, float _zmax, mat4 animation,mat4 inverzAnimation, Material* _material, mat4 _Oject = mat4(), bool _forClosing=false) {
		forClose = _forClosing;
		Object = _Oject;
		Q = _Q;
		TransformMatrix = animation;
		plane1 = vec3(0,0,_zmin);
		plane2 = vec3(0,0,_zmax);
		material = _material;
		translate = translate;
		InverzMatrix =inverzAnimation;
		Q = InverzMatrix * Q * transpose(InverzMatrix);
		vec4 plane14 = InverzMatrix * vec4(plane1.x, plane1.y, plane1.z, 1); // maybe le kell osztani a 4. taggal de nem biztos
		vec4 plane24 = InverzMatrix * vec4(plane2.x, plane2.y, plane2.z, 1); // maybe le kell osztani a 4. taggal de nem biztos
		plane1 = vec3(plane14.x, plane14.y, plane14.z);
		plane2 = vec3(plane24.x, plane24.y, plane24.z);
	}

	float f(vec4 r) {//feltételezve hogy r.w =1
		return dot(r*Q,r);
	}

	//gradf = normál vektro implicit felületeknél
	vec3 gradf(vec4 r) {
		vec4 g = r * Q * 2;
		return vec3(g.x, g.y, g.z);
	}

	vec3 gradf(vec4 r, mat4 Mat) {
		vec4 g = r * Mat * 2;
		return vec3(g.x, g.y, g.z);
	}

	//ha az eredmény pozitiv akkor a jó irányban vagyunk
	Hit intersect(const Ray& ray) {
		Hit hit;
		vec4 start4 = vec4(ray.start.x, ray.start.y, ray.start.z, 1);
		vec3 start = vec3(start4.x, start4.y, start4.z);// -translate;
		vec4 d0 = vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0);
		vec4 s1 = vec4(start.x, start.y, start.z, 1);
		float a = dot(d0 * Q, d0);
		float b = dot(s1 * Q, d0) * 2;
		float c = dot(s1 * Q, s1);
		float b2 = b * b;
		float discr = b2 - 4.0f * a * c;

		if (discr < 0) return hit; // nem taálálta el a fénysugár

		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		vec3 p1 = ray.start + ray.dir * t1;
		//float min = plane1.x * p1.x + plane1.y * p1.y + plane1.z * p1.z;
		//float max = plane2.x * p1.x + plane2.y * p1.y + plane2.z * p1.z;
		if (p1.z <plane1.z || p1.z >plane2.z) t1 = -1.0f; // t1 kivul esik a hengeren, de a lapokon még rajta lehet 

		float t2 = (-b - sqrt_discr) / 2.0f / a;
		vec3 p2 = ray.start + ray.dir * t2;
		//min = plane1.x * p2.x + plane1.y * p2.y + plane1.z * p2.z;
		//max = plane2.x * p2.x + plane2.y * p2.y + plane2.z * p2.z;
		if (p2.z < plane1.z || p2.z > plane2.z) t2 = -1.0f;

		if (t1 <= 0 && t2 <= 0) return hit;
		if (t1 <= 0)hit.t = t2;
		else if (t2 <= 0) hit.t = t1;
		else if (t2 < t1) hit.t = t2;
		else hit.t = t1;

		hit.position = start + ray.dir * hit.t;
		if (forClose) {
			vec3 position = hit.position;// +translate;
			vec4 p4 = vec4(position.x, position.y, position.z, 1);
			float val = dot(p4 * Object, p4);
			if (val > 0)
				return Hit();
		}

		hit.normal = normalize(this->gradf(vec4(hit.position.x, hit.position.y, hit.position.z, 1)));
		hit.position = hit.position;// +translate;
		hit.material = material;

		return hit;
	}
};


class Camera {
	vec3 eye, lookat, right, up;
	float fov;

	mat4 rotateEyeZaxis(float dt) {
		return RotationMatrix(dt, vec3(0, 0,1));
	}

public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		fov = _fov;
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}

	void Animate(float dt) {
		vec3 d = eye - lookat;
		vec4 d4 = vec4(d.x, d.y, d.z, 1);
		vec4 rotate4 = d4 * rotateEyeZaxis(dt);
		eye = vec3(rotate4.x, rotate4.y, rotate4.z) + lookat;
		set(eye, lookat, up, fov);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable*> lamp;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
	vec3 La2;

public:
	void Animate(float val) {
		camera.Animate(val);
	}
	void build() {//kozelre : 0.5,-2, 1.8
					//10,0,7
					//3, 5, 4
		vec3 eye = vec3(4,4,3), vup = vec3(0,0,1), lookat = vec3(0,0, 2);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.3f, 0.3f, 0.3f);
		La2 = vec3(0.5f, 0.5f, 0.5f);
		vec3 lightDirection(10, -10, 4), Le(1, 1, 1);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd(0.13f, 0.1f, 0.15f), ks(1, 1, 1);
		vec3 kd2(0.1f, 0.18f, 0.15f), ks2(4,4, 4);
		Material* lampColor = new Material(kd, ks, 120);
		Material* ground = new Material(kd2, ks2, 200);

		//TODO: megcsinálni úgy, hogy a középpontban összehozzuk az elemet majd eltoljuk a megfelelõ pozíciókba, ezután a forgatás megvalósítása
		//hogyan kellene: az intersectable elemeket lehessen transzformálni és arrébb vinni, minden transzformációnak kell az inverze illetve
		//minden tarnszformáció öröklõdik egy alul lévõ transzformáciióból

		float paraFocalDistance = 0.0f;
		float zmin = 1.0f;
		float zmax = 1.1f;
		float a11 = 5.0f;
		float R = a11/70;
		//float R = 0.1f;
		float correction = 0.05f;
		float fedlap = (-1.0f / zmax/1.1f);
		//A talpzat az mindig változatlan

#pragma formak
		mat4 plane = ScaleMatrix(vec3(0.0f, 0.0f, -1.0f));
		mat4 plane2 = ScaleMatrix(vec3(0.0f, 0.0f, fedlap));

		mat4 cylinder = mat4(a11 * 0.8, 0, 0, 0,
			0, a11 * 0.8, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, -1);

		mat4 paraboloid = mat4(a11 * 0.4, 0, 0, 0,
			0, a11 * 0.4, 0, 0,
			0, 0, -1, 0,
			0, 0, 0, 0);

		mat4 sphere = mat4(-1 / (R * R), 0, 0, 0,
			0, -1 / (R * R), 0, 0,
			0, 0, -1 / (R * R), 0,
			0, 0, 0, 1);

		mat4 beam = mat4(-a11 * 120.0f, 0, 0, 0,
			0, -a11 * 120.0f, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1);

		float beamLength = 0.6f;
		float bellSize = 0.65f;
		mat4 Transform = ScaleMatrix(vec3(1, 1, 1));
		mat4 invT = ScaleMatrix(vec3(1,1,1));
		float joint1Pos = zmax + R - correction;//elso joint poziciója         //joint
		float beam1Min = R - correction;
		float beam1Max = beam1Min + beamLength;
		float joint2Pos = beamLength+R-correction;
		float beam2Min = R - correction;
		float beam2Max = beam2Min + beamLength;
		float joint3Pos = beamLength + R - correction;
		float bellPosMin = R - correction;
		float bellPosMax = bellPosMin + bellSize;

		vec4 res = mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			1, 2, 3, 1) * vec4(1, 2, 3, 1);
		printf("%f %f %f %f", res.x, res.y, res.z, res.w);


		lamp.push_back(new Quadrics(plane,0,10.0f,Transform,invT,ground));//alap
		lamp.push_back(new Quadrics(cylinder,zmin,zmax, Transform,invT, lampColor));//lámpa talpzata
		lamp.push_back(new Quadrics(plane2, zmin, 3.0f, Transform,invT, lampColor,cylinder,true));//takarólap


		mat4 Joint1Translation = TranslateMatrix(vec3(0, 0, joint1Pos));
		mat4 Joint1InverzTranslation = TranslateMatrix(vec3(0, 0, -joint1Pos));
		mat4 Joint1Rotation = RotationMatrix(M_PI / 6.0f, vec3(1, 0, 0));
		mat4 Joint1InverzRotation = RotationMatrix(-1.0f * M_PI / 6.0f, vec3(1, 0, 0));

		Transform = /*Joint1Rotation*/ Joint1Translation * Transform;
		invT = invT * Joint1InverzTranslation /*Joint1InverzRotation*/;

		lamp.push_back(new Quadrics(sphere, 0, 4.0f, Transform, invT, lampColor));

		mat4 Beam1Translation = TranslateMatrix(vec3(0, 0, beam1Min));
		mat4 Beam1InverzTranslation = TranslateMatrix(vec3(0, 0, -beam1Min));
		
		Transform = Beam1Translation* Transform;
		invT = invT * Beam1InverzTranslation;
		lamp.push_back(new Quadrics(beam, joint1Pos+beam1Min, joint1Pos+beam1Max, Transform, invT, lampColor));

		mat4 Joint2Translation = TranslateMatrix(vec3(0, 0, joint2Pos));
		mat4 Joint2InverzTranslation = TranslateMatrix(vec3(0, 0, -(joint2Pos)));
		mat4 Joint2Rotation = RotationMatrix(M_PI / 6.0f, vec3(0, 1, 0));
		mat4 Joint2InverzRotation = RotationMatrix(-1.0f * M_PI / 6.0f, vec3(0, 1, 0));


		Transform = /*Joint2Rotation **/Joint2Translation* Transform;
		invT = invT * Joint2InverzTranslation;/**Joint2InverzRotation;*/

		lamp.push_back(new Quadrics(sphere, 0, 10.0f, Transform, invT, lampColor));

		mat4 Beam2Translation = TranslateMatrix(vec3(0, 0, beam2Min));
		mat4 Beam2InverzTranslation = TranslateMatrix(vec3(0, 0, -beam2Min));

		Transform = Beam2Translation* Transform;
		invT = invT * Beam2InverzTranslation;

		float zbeam2Min = joint1Pos + beam1Max + beam1Min + R - correction;
		float zbeam2Max = zbeam2Min + beamLength + R - correction;
		lamp.push_back(new Quadrics(beam, zbeam2Min, zbeam2Max, Transform, invT, lampColor));

		mat4 Joint3Translation = TranslateMatrix(vec3(0, 0, joint3Pos));
		mat4 Joint3InverzTranslation = TranslateMatrix(vec3(0, 0, -joint3Pos));
		Transform = Joint3Translation* Transform;
		invT = invT * Joint3InverzTranslation;

		lamp.push_back(new Quadrics(sphere, 0, 10.0f, Transform, invT, lampColor));

		mat4 BellTranslation = TranslateMatrix(vec3(0, 0, bellPosMin));
		mat4 BellInverzTranslation = TranslateMatrix(vec3(0, 0, -bellPosMin));
		Transform = BellTranslation* Transform;
		invT = invT * BellInverzTranslation;

		float zBellMin = joint1Pos + beam1Max + beam2Max + bellPosMin;
		float zBellMax = joint1Pos + beam1Max + beam2Max + bellPosMin+bellPosMax;
		lamp.push_back(new Quadrics(paraboloid, zBellMin, zBellMax, Transform, invT, lampColor));

		//vec4 lightPos = vec4(0, 0, paraFocalDistance, 1) * Transform;
		//vec4 lightPos = vec4(1, 1, 1);
		//lights.push_back(new Light(vec3(lightPos.x, lightPos.y, lightPos.z), vec3(1, 1, 1)));
			}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : lamp) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : lamp) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;
		vec3 outRadiance = hit.material->ka * La;
		for (Light* light : lights) {
			Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
			float cosTheta = dot(hit.normal, light->direction);
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;
	}
};


GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void LoadTexture(int windowWidth, int windowHeight,std::vector<vec4> image){
		texture.create(windowWidth,windowHeight,image);
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;
std::vector<vec4> image;
// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	/*long timeEnd = glutGet(GLUT_ELAPSED_TIME);
	printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));*/

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	image = std::vector<vec4>(windowWidth * windowHeight);
	scene.render(image);
	fullScreenTexturedQuad->LoadTexture(windowWidth, windowHeight, image);
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}

	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	//scene.Animate(0.1f);
	//long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	//printf("%l",time);
	glutPostRedisplay();
}
