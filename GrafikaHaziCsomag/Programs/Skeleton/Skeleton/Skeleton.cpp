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
	Camera2D(int widht, int height, vec2 center) : wCenter(center), wScreenWidht(widht), wScreenHeight(height) {}

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


class Object {
protected:
	vec3 color;
	vec3 equation;
	std::vector<vec2> points;
	bool selected = false;
public:
	Object(vec3 col, bool sel) :color(col),selected(sel) {}

	std::vector<vec2> getPoints() {
		return points;
	}

	void setSelected(bool val) {
		selected = val;
	}

	bool getSelected() {
		return selected;
	}

	vec3 getColor() {
		if (selected)
			return SelectColor;

		return color;
	}

	vec3 GetEquation() {
		return equation;
	}

	virtual float CalcDistance(vec2 point) = 0;

	bool Pick(float cX, float cY) {
		if (selected)
			return false;

		vec4 wPoint4 = vec4(cX, cY, 0, 1) * camera->Pinv() * camera->Vinv();
		vec2 point = vec2(wPoint4.x, wPoint4.y);
		float absVal = CalcDistance(point);

		if (absVal <= 0.1f) {
			selected = true;
		}

		return selected;
	}

};

class ControlPoint {
private:
	bool selected;
	vec2 wPos;
public:
	ControlPoint(vec2 pos, vec3 color) : selected(false),wPos(pos) {}

	vec2 getPosition() {
		return wPos;
	}

	void setSelected(bool val) {
		selected = val;
	}
	
	bool getSelected() {
		return selected;
	}
	
	vec3 getColor() {
		if (selected) return SelectColor;
		
		return PointColor;
	}
};

class Line : public Object{
private:
	unsigned int vaoLine;
	unsigned int vboLine[2];
	ControlPoint* wPointA, * wPointB;

	void CalcPoints() {
		points.clear();
		float x = -5.0f;
		float y;
		float numofTesselation = 400.0f;
		float scale = 10.0f / numofTesselation;
		for (int i = 0; i < numofTesselation; i++)
		{
			if (equation.y != 0) {
				y = (equation.z - (equation.x * x)) / equation.y;
			}
			else {
				y = 0;
			}
			points.push_back(vec2(x, y));
			x += scale;
		}
	}

public:

	Line(ControlPoint* A, ControlPoint* B): Object(LineColor,false),wPointA(A),wPointB(B){
		vec2 a = A->getPosition();
		vec2 b = B->getPosition();
		vec2 ab = b - a;
		vec2 normal = vec2(-ab.y, ab.x);
		float point = normal.x * b.x + normal.y * b.y;
		equation = vec3(normal.x, normal.y, point);

		CalcPoints();
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
		std::vector<vec3> colors;
		std::vector<vec2> endPoints;
		endPoints.push_back(points[0]);
		endPoints.push_back(points[points.size()-1]);
		glBindVertexArray(this->vaoLine);

		for (int i = 0; i < endPoints.size(); i++)
			colors.push_back(this->getColor());

		glBindBuffer(GL_ARRAY_BUFFER, vboLine[0]);
		glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(vec2), &endPoints[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, vboLine[1]);
		glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(vec3), &colors[0], GL_STATIC_DRAW);
		glDrawArrays(GL_LINES, 0, endPoints.size());
	}

	float CalcDistance(vec2 point) {
		float nxX = equation.x * point.x;
		float nyY = equation.y * point.y;
		float val = nxX + nyY - equation.z;
		float val2 = val * val;
		float absVal = sqrtf(val2);

		return absVal;
	}
};

class Circle :public Object{
private:
	unsigned int vaoCircle;
	unsigned int vboCircle[2];
	ControlPoint* wCenterPoint;
	float Radius;

	void CalcPoints() {
		int numofIteration = 300;
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

	Circle(ControlPoint* center, float R):Object(CircleColor,false),wCenterPoint(center),Radius(R){
		vec2 wCenter = center->getPosition();
		float R2 = R * R;
		equation = vec3(wCenter.x, wCenter.y, R2);

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
			colors.push_back(this->getColor());

		glBindBuffer(GL_ARRAY_BUFFER, vboCircle[0]);
		glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(vec2), &points[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, vboCircle[1]);
		glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(vec3), &colors[0], GL_STATIC_DRAW);
		glDrawArrays(GL_LINE_LOOP, 0, points.size());

	}

	float CalcDistance(vec2 point) {
		float xu = point.x - equation.x;
		float yv = point.y - equation.y;
		float val2 = powf(((xu * xu) + (yv * yv) - equation.z), 2.0f);
		float absVal = sqrtf(val2);

		return absVal;
	}
};

class Paper {
private: 
	unsigned int vao;
	unsigned int vboControlPoints[2];
	std::vector<ControlPoint*> controlPoints;
	std::vector<Line*> lines;
	std::vector<Circle*> circles;

	bool Picked(ControlPoint* cPoint, vec2 wPoint) {
		vec2 pos = cPoint->getPosition();
		vec2 section = pos - wPoint;
		if (length(section) <= 0.1f) {
			return true;
		}

		return false;
	}
public:
	char DrawingState;
	float Distance = -1.0f;
	std::vector<ControlPoint*> selectedPoints;
	std::vector<Line*> selectedLines;
	std::vector<Circle*> selectedCircles;
	int selectedObjects=0;

	Paper(vec2 size) {
		float cX = windowWidth / windowWidth - 1;	
		float cY = 1.0f - windowWidth / windowHeight;
		double scale = windowWidth / size.x;

		vec4 center4 = vec4(cX, cY, 0, 1) * camera->Pinv()*camera->Vinv();
		vec4 center2Right4 = center4;
		center2Right4.x += 1.0f;

		ControlPoint* centerPoint = new ControlPoint(vec2(center4.x,center4.y), PointColor);
		ControlPoint* rightOne = new ControlPoint(vec2(center2Right4.x, center2Right4.y), PointColor);
		controlPoints.push_back(centerPoint);
		controlPoints.push_back(rightOne);
		lines.push_back(new Line(centerPoint, rightOne));
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
		mat4 MVP = camera->P() * camera->V();
		gpuProgram.setUniform(MVP, "MVP");

		for each (Circle * circle in circles)
			circle->Draw();
		
		for each (Line * line in lines)
			line->Draw();

		glBindVertexArray(this->vao);

		std::vector<vec2> points = std::vector<vec2>();
		std::vector<vec3> colors = std::vector<vec3>();
		for each (ControlPoint * point in controlPoints)
		{
			points.push_back(point->getPosition());
			colors.push_back(point->getColor());
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
			if (Picked(cPoint, wPoint) && !cPoint->getSelected()) {
				cPoint->setSelected(true);
				selectedPoints.push_back(cPoint);
			}
		}
	}

	void SelectObject(float cX, float cY) {
		for each (Line * line in lines) {
			if(!line->getSelected())
				if (line->Pick(cX, cY)) {
					this->selectedObjects++;
					selectedLines.push_back(line);
					return;
				}
		}

		for each (Circle * circle in circles) {
			if (!circle->getSelected())
				if (circle->Pick(cX, cY)) {
					this->selectedObjects++;
					selectedCircles.push_back(circle);
					return;
				}
		}
	}

	void ClearSelection() {
		for each (ControlPoint* point in selectedPoints)
			point->setSelected(false);
		
		for each (Line * object in selectedLines)
			object->setSelected(false);

		for each (Circle * object in selectedCircles)
			object->setSelected(false);

		selectedPoints.clear();
		selectedCircles.clear();
		selectedLines.clear();
		selectedObjects = 0;
	}

	void AddCircle(Circle* circle) {
		circles.push_back(circle);
	}

	void AddLine(Line* line) {
		lines.push_back(line);
	}

	void AddControlPoint(vec2 point) {
		for each (ControlPoint * controlPoint in controlPoints) {
			if (Picked(controlPoint, point)) return;
		}
		controlPoints.push_back(new ControlPoint(point, PointColor));
	}

	void Intersect(Line* line1, Line* line2) {
		std::vector<vec2> points = line2->getPoints();
		for each (vec2 point in points)
		{
			float absVal = line1->CalcDistance(point);
			if (absVal <= 0.02f)
				AddControlPoint(point);
		}
	}

	void Intersect(Line* line, Circle* circle) {
		std::vector<vec2> points = circle->getPoints();
		for each (vec2 point in points)
		{
			float absVal = line->CalcDistance(point);
			if (absVal <= 0.02f) {
				AddControlPoint(point);
			}
		}
	}

	void Intersect(Circle* circle1, Circle* circle2) {
		std::vector<vec2> points = circle1->getPoints();
		for each (vec2 point in points)
		{
			float absVal = circle2->CalcDistance(point);
			if (absVal <= 0.02f) {
				AddControlPoint(point);
			}
		}
	}

};

Paper* paper;
// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glPointSize(10.0f);
	glLineWidth(3.0f);

	camera = new Camera2D(10, 10,vec2(0, 0));
	paper = new Paper(vec2(10, 10));

	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	paper->Draw();

	glutSwapBuffers(); // exchange buffers for double buffering
}


// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 's') paper->DrawingState = 's';
	if (key == 'c') paper->DrawingState = 'c';
	if (key == 'l') paper->DrawingState = 'l';
	if (key == 'i') paper->DrawingState = 'i';

	paper->ClearSelection();
	printf("Selected State: %c \n", paper->DrawingState);
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {}

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
				}
				glutPostRedisplay();
			break;
		case 'c':
				if (paper->Distance >= 0) {
					paper->SelectPoint(cX, cY);
					if (paper->selectedPoints.size() == 1) {
						paper->AddCircle(new Circle(paper->selectedPoints[0], paper->Distance));
						paper->ClearSelection();
						glutPostRedisplay();
					}
				}
			break;
		case 'i':
				paper->SelectObject(cX, cY);
				if (paper->selectedObjects == 2) {

					if (paper->selectedLines.size() == 1) {
						paper->Intersect(paper->selectedLines[0], paper->selectedCircles[0]);
					}
					else if (paper->selectedLines.size() == 0) {
						paper->Intersect(paper->selectedCircles[0], paper->selectedCircles[1]);
					}
					else {
						paper->Intersect(paper->selectedLines[0], paper->selectedLines[1]);
					}
					paper->ClearSelection();
				}
				glutPostRedisplay();
			break;
		case 'l':
				paper->SelectPoint(cX, cY);
				if (paper->selectedPoints.size() == 2) {
					paper->AddLine(new Line(paper->selectedPoints[0], paper->selectedPoints[1]));
					paper->ClearSelection();
				}
				glutPostRedisplay();
			break;
		}
	break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {}
