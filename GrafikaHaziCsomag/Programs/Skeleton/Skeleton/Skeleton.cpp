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
const char* const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0
	layout(location = 1) in vec3 vColor;
	
	out vec3 color;

	void main() {
		color = vColor;
		gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	in vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";


GPUProgram gpuProgram; // vertex and fragment shaders
//unsigned int vao;	   // virtual world on the GPU

vec3 LineColor(1.0f, 0.0f, 0.0f);
vec3 PointColor(1.0f, 1.0f, 0.0f);
vec3 CircleColor(0.0f, 1.0f, 1.0f);
vec3 SelectColor(1.0f, 1.0f, 1.0f);

class Camera2D {
private:
	float wScreenWidht;
	float wScreenHeight;
	vec2 wCenter;
public:
	Camera2D(int widht, int height, vec2 center) {
		wScreenWidht = widht;
		wScreenHeight = height;
		wCenter = center;
	};

	mat4 V() {
		return  TranslateMatrix(-wCenter);
	};

	mat4 Vinv() {
		return  TranslateMatrix(wCenter);
	};

	mat4 P() {
		mat4 scale = ScaleMatrix(vec2(2 / wScreenWidht, 2 / wScreenHeight));
		return scale;
	};

	mat4 Pinv() {
		return  ScaleMatrix(vec2(wScreenWidht / 2, wScreenHeight / 2));

	};
};

Camera2D* camera;

class Intersectable {
public:
	virtual std::vector<vec2> Intersect(Intersectable object) = 0;
};

class ControlPoint {
private:
	bool selected;
	vec2 wPos;
	vec3 nColor;
public:
	ControlPoint(vec2 pos, vec3 color) {
		selected = false;
		nColor = color;
		wPos = pos;
		//vec4 position = vec4(pos.x,pos.y,0,1) * camera->Pinv() * camera->Vinv();
		//wPos = vec2(position.x, position.y);
	}

	vec2 getPosition() {
		return wPos;
	}

	void Selected(bool val) {
		selected = val;
	}
	
	vec3 getColor() {
		if (selected) return SelectColor;
		
		return PointColor;
	}
};

class Line {
private:
	unsigned int vaoLine;
	unsigned int vboLine[2];
	vec3 color;
	ControlPoint* wPointA, * wPointB;
	vec3 equation;
	std::vector<vec2> endPoints = std::vector<vec2>();

	void CalcEndPoints() {
		endPoints.clear();
		float x;
		float y;
		if (equation.x != 0 && equation.y != 0) {
			y = (float)(equation.z - (equation.x * -5.0f)) / equation.y;
			float yRight = (float)(equation.z - (equation.x * 5.0f)) / equation.y;

			endPoints.push_back(vec2(-5.0f, y));
			endPoints.push_back(vec2(5.0f, yRight));
		}
		else if (equation.x == 0 && equation.y != 0) {
			y = (float)(equation.z / equation.y);
			x = 5.0f;
			endPoints.push_back(vec2(x, y));
			endPoints.push_back(vec2(-x, y));
		}
		else if (equation.y == 0 && equation.x != 0) {
			x = (float)(equation.z / equation.x);
			y = 5.0f;
			endPoints.push_back(vec2(x, y));
			endPoints.push_back(vec2(x, -y));
		}
	}

public:
	Line(ControlPoint* A, ControlPoint* B, vec3 col) {
		color = col;
		wPointA = A;
		wPointB = B;
		vec2 a = A->getPosition();
		vec2 b = B->getPosition();
		vec2 ab = b - a;
		vec2 normal = vec2(-ab.y, ab.x);
		float point = normal.x * a.x + normal.y * a.y;
		equation = vec3(normal.x, normal.y, point);
		CalcEndPoints();
		this->Create();
	}

	void Create() {
		glGenVertexArrays(1, &this->vaoLine);
		glBindVertexArray(this->vaoLine);
		glGenBuffers(2, &vboLine[0]);
		glBindBuffer(GL_ARRAY_BUFFER, vboLine[0]);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), NULL);
		glBindBuffer(GL_ARRAY_BUFFER, vboLine[1]);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), NULL);
	}

	void Draw() {
		mat4 MVP = camera->P() * camera->V();
		gpuProgram.setUniform(MVP, "MVP");
		std::vector<vec3> colors;
		glBindVertexArray(this->vaoLine);

		for (int i = 0; i < endPoints.size(); i++)
			colors.push_back(LineColor);

		glBindBuffer(GL_ARRAY_BUFFER, vboLine[0]);
		glBufferData(GL_ARRAY_BUFFER, endPoints.size() * sizeof(vec2), &endPoints[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, vboLine[1]);
		glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(vec3), &colors[0], GL_STATIC_DRAW);
		glDrawArrays(GL_LINES, 0, endPoints.size());


	}

	
};

class Circle{
private:
	unsigned int vaoCircle;
	unsigned int vboCircle[2];
	ControlPoint* wCenterPoint;
	float Radius;
	std::vector<vec2> points;

	void CalcPoints() {
		int numofIteration = 150;
		vec2 center = wCenterPoint->getPosition();

		for (size_t i = 0; i < numofIteration; i++)
		{
			float fi = i * 2 * M_PI / numofIteration;
			float x = Radius * cos(fi) + center.x;
			float y = Radius * sin(fi) + center.y;
			points.push_back(vec2(x, y));
		}
	}
public:
	Circle(ControlPoint* center, float R) {
		wCenterPoint = center;
		Radius = R;
		this->Create();
		this->CalcPoints();
	}

	void Create() {
		glGenVertexArrays(1, &this->vaoCircle);
		glBindVertexArray(this->vaoCircle);
		glGenBuffers(2, &vboCircle[0]);
		glBindBuffer(GL_ARRAY_BUFFER, vboCircle[0]);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), NULL);
		glBindBuffer(GL_ARRAY_BUFFER, vboCircle[1]);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), NULL);
	}

	void Draw() {
		glBindVertexArray(this->vaoCircle);
		std::vector<vec3> colors;

		for (int i = 0; i < points.size(); i++)
			colors.push_back(CircleColor);
		//points.push_back(points[0]);
		//colors.push_back(CircleColor);

		glBindBuffer(GL_ARRAY_BUFFER, vboCircle[0]);
		glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(vec2), &points[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, vboCircle[1]);
		glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(vec3), &colors[0], GL_STATIC_DRAW);
		glDrawArrays(GL_LINE_LOOP, 0, points.size());

	}
};

class Paper {
private: 
	vec2 wSize;
	unsigned int vao;
	unsigned int vboControlPoints[2];
	std::vector<ControlPoint*> controlPoints;
	std::vector<Line*> lines;
	std::vector<Circle*> circles;
public:
	char DrawingState;
	float Distance = -1.0f;
	std::vector<ControlPoint*> selectedPoints;
public:
	Paper(vec2 size) {
		float cX = windowWidth / windowWidth - 1;	
		float cY = 1.0f - windowWidth / windowHeight;
		double scale = windowWidth / size.x;

		vec4 center4 = vec4(cX, cY, 0, 1) * camera->Pinv()*camera->Vinv();
		vec4 center2Right4 = center4;
		center2Right4.x += 1.0f;//vec4(cX+1.0f, cY, 0, 1) * camera->Pinv()*camera->Vinv();
		wSize = size;
		ControlPoint* centerPoint = new ControlPoint(vec2(center4.x,center4.y), PointColor);
		ControlPoint* rightOne = new ControlPoint(vec2(center2Right4.x, center2Right4.y), PointColor);
		controlPoints.push_back(centerPoint);
		controlPoints.push_back(rightOne);
		lines.push_back(new Line(centerPoint, rightOne, LineColor));
		this->Create();
	}

	void Create() {
		glGenVertexArrays(1, &this->vao);
		glBindVertexArray(this->vao);
		glGenBuffers(2, &vboControlPoints[0]);
		glBindBuffer(GL_ARRAY_BUFFER, vboControlPoints[0]);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), NULL);
		glBindBuffer(GL_ARRAY_BUFFER, vboControlPoints[1]);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), NULL);

	}

	void Draw() {
		for each (Circle * circle in circles)
			circle->Draw();
		
		for each (Line * line in lines)
			line->Draw();

		mat4 MVP = camera->P() * camera->V();
		gpuProgram.setUniform(MVP, "MVP");
		glBindVertexArray(this->vao);

		std::vector<vec2> points = std::vector<vec2>();
		std::vector<vec3> colors = std::vector<vec3>();
		for (int i = 0; i < controlPoints.size(); i++) {
			points.push_back(controlPoints[i]->getPosition());
			colors.push_back(controlPoints[i]->getColor());
		}

		glBindBuffer(GL_ARRAY_BUFFER, vboControlPoints[0]);
		glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(vec2), &points[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, vboControlPoints[1]);
		glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(vec3), &colors[0], GL_STATIC_DRAW);
		glDrawArrays(GL_POINTS, 0, points.size());
	}

	void SelectPoint(float cX, float cY) {
		vec4 wPoint4 = vec4(cX, cY, 0, 1) * camera->Pinv() * camera->Vinv();
		vec2 wPoint = vec2(wPoint4.x, wPoint4.y);

		for each (ControlPoint* cPoint in controlPoints)
		{
			vec2 pos = cPoint->getPosition();
			vec2 section = pos - wPoint;
			if (length(section) <= 0.1f) {
				cPoint->Selected(true);
				selectedPoints.push_back(cPoint);
				return;
			}
		}

		glutPostRedisplay();
	}

	void ClearSelection() {
		for each (ControlPoint* point in selectedPoints)
		{
			point->Selected(false);
		}
		selectedPoints.clear();
		glutPostRedisplay();
	}

	void AddCircle(Circle* circle) {
		circles.push_back(circle);
		glutPostRedisplay();
	}

};


Paper* paper;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glPointSize(10.0f);
	glLineWidth(5.0f);

	camera = new Camera2D(10, 10,vec2(0, 0));
	paper = new Paper(vec2(10, 10));

	//glGenVertexArrays(1, &vao);	// get 1 vao id
	//glBindVertexArray(vao);		// make it active

	//unsigned int vbo;		// vertex buffer object
	//glGenBuffers(1, &vbo);	// Generate 1 buffer
	//glBindBuffer(GL_ARRAY_BUFFER, vbo);
	//// Geometry with 24 bytes (6 floats or 3 x 2 coordinates)
	//float vertices[] = { 0.0f, 0.0f, 1.0f, 3.0f};
	//glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
	//	sizeof(vertices),  // # bytes
	//	vertices,	      	// address
	//	GL_STATIC_DRAW);	// we do not change later

	//glEnableVertexAttribArray(0);  // AttribArray 0
	//glVertexAttribPointer(0,       // vbo -> AttribArray 0
	//	2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
	//	0, NULL); 		     // stride, offset: tightly packed

	//// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	paper->Draw();


	// Set color to (0, 1, 0) = green
	//int location = glGetUniformLocation(gpuProgram.getId(), "color");
	//glUniform3f(location, 0.0f, 1.0f, 0.0f); // 3 floats

	//float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix, 
	//						  0, 1, 0, 0,    // row-major!
	//						  0, 0, 1, 0,
	//						  0, 0, 0, 1 };

	//location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	//glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

	//glBindVertexArray(vao);  // Draw call
	//glDrawArrays(GL_LINES, 0 /*startIdx*/, 1 /*# Elements*/);

	glutSwapBuffers(); // exchange buffers for double buffering
}


// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 's') paper->DrawingState = 's';
	if (key == 'c') paper->DrawingState = 'c';
	if (key == 'l') paper->DrawingState = 'l';
	if (key == 'i') paper->DrawingState = 'i';
	if (key == 'd') glutPostRedisplay();   // if d, invalidate display, i.e. redraw

	paper->ClearSelection();
	printf("Selected State: %c \n", paper->DrawingState);
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

//	char * buttonStat;
	switch (state) {
	case GLUT_DOWN:
		switch (paper->DrawingState) {
		case 's':
			if (paper->selectedPoints.size() == 2)
				paper->ClearSelection();

			paper->SelectPoint(cX,cY);
			if (paper->selectedPoints.size() == 2) {
				vec2 section = paper->selectedPoints[0]->getPosition() - paper->selectedPoints[1]->getPosition();
				paper->Distance = length(section);
				printf("Distance Taken: %f", paper->Distance);
			}
			break;
		case 'c':
			if (paper->Distance >= 0) {
				paper->SelectPoint(cX, cY);
				if (paper->selectedPoints.size() == 1) {
					paper->AddCircle(new Circle(paper->selectedPoints[0], paper->Distance));
					paper->ClearSelection();
				}
			}
			break;
		}
	break;// buttonStat = "pressed"; break;
	//case GLUT_UP:   buttonStat = "released"; break;
	}

	//switch (button) {
	//case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	//case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	//case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}
