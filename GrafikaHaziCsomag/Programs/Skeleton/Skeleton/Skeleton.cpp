//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
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
vec3 LineColor(1.0f, 0.0f, 0.0f);//a kirajzolt vonalak sz�ne
vec3 PointColor(1.0f, 1.0f, 0.0f);//a kirajzolt Pontok sz�ne
vec3 CircleColor(0.0f, 1.0f, 1.0f);//Kirajzolt k�r�k sz�ne
vec3 SelectColor(1.0f, 1.0f, 1.0f);//Kiv�lasztott objektum sz�ne

//A 2D-s vil�got megjelen�t� kamera amely a modellben megalkotott k�p
//"K�perny�re" t�rt�n� adapt�l�s�rt felel�s, a vil�gkordin�t�kat az MVP m�trix seg�ts�g�vel �tvissz�k a kamera ablakba.
class Camera2D {
private:
	float wScreenWidht;//a kamera ablak sz�less�ge
	float wScreenHeight;//kamera ablak magass�ga
	vec2 wCenter;//kamera ablak k�zepe
public:
	//default konstruktor
	Camera2D(int widht, int height, vec2 center) : wCenter(center), wScreenWidht(widht), wScreenHeight(height) {}

	//a View transzform�ci��rt felel�s fv eltoljuk a k�pet a kamera ablak k�z�ppontj�ba
	mat4 V() {
		return  TranslateMatrix(-wCenter);
	};

	//a View Transzform�ci� inverze
	mat4 Vinv() {
		return  TranslateMatrix(wCenter);
	};

	//a Projection transzform�ci��rt felel�s fv sk�l�zzuk a k�pet a kamera ablak m�reteivel.
	mat4 P() {
		mat4 scale = ScaleMatrix(vec2(2 / wScreenWidht, 2 / wScreenHeight));
		return scale;
	};

	//A Projekci�s transzform�ci� inverze
	mat4 Pinv() {
		return  ScaleMatrix(vec2(wScreenWidht / 2, wScreenHeight / 2));
	};
};//Az inverz transzform�ci�kra a UI-n kerszet�li interkaci� feldolgoz�s�ra van sz�ks�g, a kapott pixel koordin�t�kat �tvigy�k a vil�gkorddin�ta rendszerbe

Camera2D* camera;

/**
* a Vonal �s a K�r �soszt�lya
* tartalmazza az alap fveket �s logik�t
*/
class Object {
protected:
	vec3 color;//az objektum sz�ne
	vec3 equation;//implicit egyenletet t�rol, az x,y koordin�ta az egyenes eset�ben a norm�lvektor x,y koordin�t�ja, 
	//k�r eset�ben a k�z�ppont x,y koordin�t�ja, z �rt�ke pedig mind2 esetben az egyenlet kontans �rt�ke.
	std::vector<vec2> points;//azon pontok melyek kiel�g�tik az egyenletet.
	bool selected = false;//kiv�laszt�s eset�n true-ra �ll�tjuk ezzel jelezve a sz�nv�lt�st
public:
	Object(vec3 col, bool sel) :color(col),selected(sel) {}//alap sz�nek �s �rt�kek be�ll�t�sa

	//visszaadja a k�r/egyenes pontjait, metsz�spontsz�m�t�shoz
	std::vector<vec2> getPoints() {
		return points;
	}

	//be�ll�tja a selected �rt�k�t
	void setSelected(bool val) {
		selected = val;
	}

	//visszaadja, hogy ki van e v�lasztva az adott objektum
	bool getSelected() {
		return selected;
	}

	//Ha selected az objektum -->SelectedColor egy�bk�nt az objektum sz�ne
	vec3 getColor() {
		if (selected)
			return SelectColor;

		return color;
	}

	//visszaadja az egyenes/k�r implicit egyenlet�t
	vec3 GetEquation() {
		return equation;
	}

	//Meghat�rozza, a pont t�vols�g�t, az egyenes/k�r implicit egyenlet�hez viszony�tva
	virtual float CalcDistance(vec2 point) = 0;

	//Meghat�rozza, hogy kattint�s eset�n az adott objektum volt e a kiv�lasztani k�v�nt objektum
	bool Pick(float cX, float cY) {
		if (selected)//ha m�r kiv�lasztottuk akkor nincs mit tenni
			return false;

		vec4 wPoint4 = vec4(cX, cY, 0, 1) * camera->Pinv() * camera->Vinv();//a vil�gkoordin�t�k meghat�roz�sa
		vec2 point = vec2(wPoint4.x, wPoint4.y);
		float absVal = CalcDistance(point);//t�vols�g sz�m�t�s egyeneshez viszony�tva

		if (absVal <= 0.1f) {//ha a k�zel�be esik/teh�t nagy val�sz�n�s�ggel ez volt a v�lasztand� objektum/ akkor visszat�r�nk igaz �rt�kkel
			selected = true;
		}

		return selected;
	}

};

/**
* A k�perny�n megjelen� kontrollPontok(amik egyben metsz�spontok is)
*/
class ControlPoint {
private:
	bool selected;//objektumhoz hasonl�an selected a kiv�laszt�s jelz�s�re
	vec2 wPos;//helyzete vil�gkoordin�t�kban
public:
	ControlPoint(vec2 pos, vec3 color) : selected(false),wPos(pos) {}//pozici� meghat�roz�sa, alap �rt�kek megad�sa

	vec2 getPosition() {
		return wPos;
	}

	void setSelected(bool val) {
		selected = val;
	}
	
	bool getSelected() {
		return selected;
	}

	//hasonl�k�pp mint az objektumok eset�ben
	vec3 getColor() {
		if (selected) return SelectColor;
		
		return PointColor;
	}
};

/**
* A egyenest reprezent�l� oszt�ly
*/
class Line : public Object{
private:
	unsigned int vaoLine;//a egyeneshez tartoz� vao �s vbo felv�tele
	unsigned int vboLine[2];//2 vbo felv�tele �s felt�lt�se a gpu-n az els� vbo a koordin�t�kat tartalmazza rendre, a m�sodik pedig az objektumhoz tartoz� sz�nt
	ControlPoint* wPointA, * wPointB;//az egyenes meghat�zos�r�a szolg�l� 2 KontrollPont

	//az implicit egyenletet kiel�g�t� pontok koordin�t�inak kisz�m�t�sa 
	void CalcPoints() {
		points.clear();
		float x = -5.0f;//a kamera ablak bal sz�l�t�l a kamera ablak jobb sz�l�ig haladunk
		float y;//folyamatosan n�velj�k x �rt�k�t, valamekkora ar�nnyal, majd kisz�m�tjuk az y �rt�k�t �s �gy t�roljuk el vec2 t�pusk�nt
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
	//default kontruktor, kisz�m�tjuk az egyenes egyenlet�t �s az azt kiel�g�t� pontokat
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

	//vao �s vbo-k gener�l�sa, attribPointerek defini�l�sa a gener�lt vbo-k hoz.
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

	//A egyenes kirajzol�sa
	void Draw() {
		std::vector<vec3> colors;//egyenes sz�n�t tartalmaz� vector
		std::vector<vec2> endPoints;//egyenes kezd� �s v�gpontj�t tartalmaz� vektor
		endPoints.push_back(points[0]);
		endPoints.push_back(points[points.size()-1]);
		glBindVertexArray(this->vaoLine);

		for (int i = 0; i < endPoints.size(); i++) {
			colors.push_back(this->getColor());//sz�nek felv�tele annak f�ggv�ny�ben, hogy ki van-e v�lasztva az adott egyenes
		}

		//bufferek felt�lt�se, majd egynes kirajzol�sa.
		glBindBuffer(GL_ARRAY_BUFFER, vboLine[0]);
		glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(vec2), &endPoints[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, vboLine[1]);
		glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(vec3), &colors[0], GL_STATIC_DRAW);
		glDrawArrays(GL_LINES, 0, endPoints.size());
	}

	//Object-ben defini�lt met�dus megval�s�t�sa
	float CalcDistance(vec2 point) {
		//kisz�m�tjuk a pontra az egyenes implicit egyenlet�t, amely az adott "szakasz" �s pont t�vols�ga lesz.
		// nx*X+ny*Y = (nx*X1+ny*Y1)-konstans-
		float nxX = equation.x * point.x;
		float nyY = equation.y * point.y;
		float val = nxX + nyY - equation.z;
		float val2 = val * val;
		float absVal = sqrtf(val2);

		return absVal;
	}
};

//k�rt reprezent�l� objektum a vil�gunkban
class Circle :public Object{
private:
	unsigned int vaoCircle;//vao �s vbok felv�tele hasonl�k�pp az egyeneshez
	unsigned int vboCircle[2];
	ControlPoint* wCenterPoint;
	float Radius;//a k�r sugara 

	//K�r egyenlet�vel kisz�m�tjuk az azt kiel�g�t� pontokat,aemelyek hasonl�an az egyenes eset�ben
	//a metsz�spont �s a kiv�laszt�s meghat�roz�s�hoz sz�ks�ges
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
		equation = vec3(wCenter.x, wCenter.y, R2);//k�r egyenlet�nek kisz�m�t�sa

		this->Create();//Vao �s vbo-k l�trehoz�sa
		this->CalcPoints();//Pontok meghat�roz�sa
	}

	void Create() {
		glGenVertexArrays(1, &this->vaoCircle); //vao gener�l�sa majd bindol�sa 
		glBindVertexArray(this->vaoCircle);
		glGenBuffers(2, &vboCircle[0]);//bufferek gener�l�sa �s pointerek mehat�roz�sa
		glBindBuffer(GL_ARRAY_BUFFER, vboCircle[0]);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), NULL);
		glBindBuffer(GL_ARRAY_BUFFER, vboCircle[1]);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), NULL);
	}

	//k�r kirajzol�sa 
	void Draw() {
		glBindVertexArray(this->vaoCircle);
		std::vector<vec3> colors;

		for (int i = 0; i < points.size(); i++) {
			colors.push_back(this->getColor());//pontok �s a k�rh�z tartoz� sz�n elt�rol�sa
		}

		//majd a gpu-ra �tvivend� adatok felt�lt�se 
		glBindBuffer(GL_ARRAY_BUFFER, vboCircle[0]);
		glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(vec2), &points[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, vboCircle[1]);
		glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(vec3), &colors[0], GL_STATIC_DRAW);
		glDrawArrays(GL_LINE_LOOP, 0, points.size());

	}

	//egyeneshez hasonl�an az egyenlet seg�ts�g�vel a k�r �s a pont t�vols�g�nak meghat�roz�sa
	float CalcDistance(vec2 point) {
		float xu = point.x - equation.x;
		float yv = point.y - equation.y;
		float val2 = powf(((xu * xu) + (yv * yv) - equation.z), 2.0f);
		float absVal = sqrtf(val2);

		return absVal;
	}
};

//a Virtu�lis vil�gunkat/szerkeszt� fel�let�nket reprezent�l� oszt�ly/objektum
class Paper {
private: 
	unsigned int vao;//a controllPontokhoz tartoz� vao
	unsigned int vboControlPoints[2];//a controllPontokhoz tartoz� vbo-k
	std::vector<ControlPoint*> controlPoints;//controllPontokat tartalmaz� vector
	std::vector<Line*> lines;// az egyeneseet tartalmaz� "lista"
	std::vector<Circle*> circles;//k�r�ket tartalmaz� "lista"

	bool Picked(ControlPoint* cPoint, vec2 wPoint) {
		vec2 pos = cPoint->getPosition();// a felhaszn�l� �ltal "megadott pont" �s a kontrollPont t�vols�g�nak meghat�roz�sa
		vec2 section = pos - wPoint;
		if (length(section) <= 0.1f) {//ha kicsi ==> ez a kiv�lasztott kontrollPont
			return true;
		}

		return false;//egy�bk�nt nem 
	}
public:
	char DrawingState; //az aktu�lis �llapotot t�roljuk el benne, k�r razol�s, egyenes rajzol�s t�vm�r�s stb.
	float Distance = -1.0f; // a felvett k�rz�t�vols�got t�roljuk el
	std::vector<ControlPoint*> selectedPoints; // a felhaszn�l�i interkaci�val kiv�lasztott pontokat t�roljuk a list�ban
	std::vector<Line*> selectedLines; // kiv�lasztott egyeneseket t�roljuk
	std::vector<Circle*> selectedCircles; // kiv�lasztott k�r�ket t�roljuk
	int selectedObjects=0; // kiv�lasztott objektumok sz�ma

	//l�trehozzuk a default setupot, vonal rajzol�sa k�z�pre, �s k�z�pre �s t�le jobbra egy pont l�trehoz�sa
	Paper(vec2 size) {
		//k�z�p mint koordin�ta meghat�roz�sa 
		float cX = windowWidth / windowWidth - 1;	
		float cY = 1.0f - windowWidth / windowHeight;
		double scale = windowWidth / size.x;

		vec4 center4 = vec4(cX, cY, 0, 1) * camera->Pinv()*camera->Vinv();//�t transzform�ljuk vil�gkoordin�t�kba
		vec4 center2Right4 = center4;

		center2Right4.x += 1.0f; //l�trehozzuk a kett�vel eltolt pontot.

		//ezeket elt�roljuk, majd felvessz�k az erre a k�t pontra illeszked� egyenest.
		ControlPoint* centerPoint = new ControlPoint(vec2(center4.x,center4.y), PointColor);
		ControlPoint* rightOne = new ControlPoint(vec2(center2Right4.x, center2Right4.y), PointColor);
		controlPoints.push_back(centerPoint);
		controlPoints.push_back(rightOne);
		lines.push_back(new Line(centerPoint, rightOne));
		//vao �s vbok gener�l�sa + pointerek be�ll�t�sa
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
		//MVP m�trix l�trehoz�sa amit uniformk�nt �tadunk a GPU-nak ami ezut�n minden �tvitt pontot beszoroz vele, ezzel
		//�t transzform�lva a vil�gkoordin�t�kat norm�l koordin�ta rendszerbe
		mat4 MVP = camera->P() * camera->V();
		gpuProgram.setUniform(MVP, "MVP");

		//k�r�k kirajzol�sa 
		for (size_t i = 0; i < circles.size(); i++)
		{
			circles[i]->Draw();
		}

		//egyenesek kirajzol�sa
		for (size_t i = 0; i < lines.size(); i++)
		{
			lines[i]->Draw();
		}
		
		glBindVertexArray(this->vao);


		//pontok kirajzol�sa, �s a hozz�juk tartoz� sz�nek meghat�roz�sa
		std::vector<vec2> points = std::vector<vec2>();
		std::vector<vec3> colors = std::vector<vec3>();

		for (size_t i = 0; i < controlPoints.size(); i++) {
			points.push_back(controlPoints[i]->getPosition());
			colors.push_back(controlPoints[i]->getColor());
		}

		//a megfelel� bufferek felt�lt�se a poz�ci� - sz�n p�rokkal
		glBindBuffer(GL_ARRAY_BUFFER, vboControlPoints[0]);
		glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(vec2), &points[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, vboControlPoints[1]);
		glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(vec3), &colors[0], GL_STATIC_DRAW);
		glDrawArrays(GL_POINTS, 0, points.size());
	}

	/**Miut�n a felhaszn�l� kattintott egyet a fel�leten
	* levizsg�ljuk, hogy melyik kontrollponthoz volt k�zel, �s amennyiben megtal�ljuk
	* akkor azt elt�roljuk a selectedPoints list�ba
	*/
	void SelectPoint(float cX, float cY) {
		vec4 wPoint4 = vec4(cX, cY, 0, 1) * camera->Pinv() * camera->Vinv();
		vec2 wPoint = vec2(wPoint4.x, wPoint4.y);


		for (size_t i = 0; i < controlPoints.size(); i++) {
			{
				//ha k�zel van �s m�g nem volt kiv�lasztva akkor elt�roljuk, �s mivel meg is tal�ltuk a keresettet
				//ez�rt ki is l�p�nk a ciklusb�l.
				if (Picked(controlPoints[i], wPoint) && !controlPoints[i]->getSelected()) {
					controlPoints[i]->setSelected(true);
					selectedPoints.push_back(controlPoints[i]);
					break;
				}
			}
		}
	}

	/**
	* megkeress�k a kiv�lasztani k�v�nt objektumot a pontokn�l haszn�lt logik�val
	* amelyikhez k�zel vagyunk azt elt�roljuk, majd mivel megtal�ltuk ez�rt vissza is t�rhet�nk
	* fontos szempont, hogy ha m�r ki van v�lasztva akkor ne lehessen m�gegyszer kiv�lasztani.
	*/
	void SelectObject(float cX, float cY) {
		for (size_t i = 0; i < lines.size(); i++)
		{
			if (!lines[i]->getSelected())//ha m�g nincs kiv�lasztva
				if (lines[i]->Pick(cX, cY)) {//�s k�zel van
					this->selectedObjects++;//akkor elt�roljuk
					selectedLines.push_back(lines[i]);
					return;
				}

		}

		for (size_t i = 0; i < circles.size(); i++)
		{
			if (!circles[i]->getSelected())
				if (circles[i]->Pick(cX, cY)) {
					this->selectedObjects++;
					selectedCircles.push_back(circles[i]);
					return;
				}

		}
	}

	//ha lefutott az intersect vagy �llapotot v�ltunk, vagy b�rmi egy�b ok miatt
	//t�r�lni kellene az egyes jel�l�seket akkor azt ezzel tessz�k meg
	void ClearSelection() {
		//vissza�ll�tjuk az �sszes pont �s objektum selected �rt�k�t false-ra 
		//ez�ltal a sz�ne nem a kijel�l�si sz�n lesz
		//�s �jra el tudjuk t�rolni majd ezeket a pontokat/objektumokat
		for (size_t i = 0; i < selectedPoints.size(); i++) {
			selectedPoints[i]->setSelected(false);
		}
		
		for (size_t i = 0; i < selectedLines.size(); i++) {
			selectedLines[i]->setSelected(false);
		}

		for (size_t i = 0; i < selectedCircles.size(); i++) {
			selectedCircles[i]->setSelected(false);
		}

		selectedPoints.clear();
		selectedCircles.clear();
		selectedLines.clear();
		selectedObjects = 0;
	}

	//k�r felv�tele a "pap�r lapra"
	void AddCircle(Circle* circle) {
		circles.push_back(circle);
	}

	//egyenes rajzol�sa a pap�rlapra
	void AddLine(Line* line) {
		lines.push_back(line);
	}

	//metsz�spont felv�tele a pontok k�z�
	void AddControlPoint(vec2 point) {
		for (size_t i = 0; i < controlPoints.size(); i++)
		{
			//ha a pont egy m�r megl�v� kontrollpont k�zel�ben van akkor val�sz�n�, hogy a k�t pont ugyanaz
			if (Picked(controlPoints[i], point)) return;//de csak akkor, ha m�g nincs felv�ve a pontok k�z�
		}

		controlPoints.push_back(new ControlPoint(point, PointColor));
	}


	//k�t egyenes metsz�spontj�nak kisz�m�t�sa
	void Intersect(Line* line1, Line* line2) {
		std::vector<vec2> points = line2->getPoints();
		//v�gigmegy�nk az egyik egyenes pontjain �s megvizsg�ljuk,hogy milyen k�zel vannak a m�sik egyeneshez
		//jobban mondva, hogy rajta vannak-e,
		//ha igen, akor elt�roljuk mint kontrollPont/metsz�sPont
		for (size_t i = 0; i < points.size(); i++)
		{
			float absVal = line1->CalcDistance(points[i]);
			if (absVal <= 0.02f)
				AddControlPoint(points[i]);
		}
	}
	//egy k�r �s egy egyenes metsz�spontj�nak kisz�m�t�sa
	void Intersect(Line* line, Circle* circle) {
		std::vector<vec2> points = circle->getPoints();
		//megvizsg�ljuk,hogy a k�r pontjai k�z�l melyek vannak "rajta"/vagy nagyon k�zel az egyeneshez
		//ha tal�lunk ilyent akkor az egy metsz�spont �s elt�roljuk.
		for (size_t i = 0; i < points.size(); i++) {
			float absVal = line->CalcDistance(points[i]);
			if (absVal <= 0.02f) {
				AddControlPoint(points[i]);
			}
		}
	}
	//k�t k�r metsz�spontj�nak kisz�m�t�sa
	void Intersect(Circle* circle1, Circle* circle2) {
		std::vector<vec2> points = circle1->getPoints();
		//egyik k�r pontjai k�z�l melyek vannak rajta a m�sik k�r�n, ha tal�lunk ilyent akkor az lesz
		//a k�r�k metsz�spontja
		for (size_t i = 0; i < points.size(); i++) {
			float absVal = circle2->CalcDistance(points[i]);
			if (absVal <= 0.02f) {
				AddControlPoint(points[i]);
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

	camera = new Camera2D(10, 10,vec2(0, 0));//kamera/vil�g l�trehoz�sa 10x10 cm ben
	paper = new Paper(vec2(10, 10));//Pap�r l�trehoz�sa a rajzol�shoz

	//vertex �s Fragmentshaderek bet�lt�se
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	paper->Draw();//pap�ron l�v� dolgok felrajzol�sa

	glutSwapBuffers(); // exchange buffers for double buffering
}


// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 's') paper->DrawingState = 's'; //k�rz�vel t�vols�g felv�tele �llapot
	if (key == 'c') paper->DrawingState = 'c';//k�r rajzol�sa  �llapot
	if (key == 'l') paper->DrawingState = 'l';//egyenes rajzol�s a�llapot
	if (key == 'i') paper->DrawingState = 'i'; // metsz�spont keres�se �llapot

	paper->ClearSelection();// kor�bbi elt�rolt kijel�l�sek t�rl�se
	printf("Selected State: %c \n", paper->DrawingState);//�llapot kiir�sa
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
		//�llapottol f�gg�en cselekv�st v�gz�nk
		switch (paper->DrawingState) {
		case 's':
				//kor�bbi kijel�l�s t�rl�se
				if (paper->selectedPoints.size() == 2)
					paper->ClearSelection();

				//ha m�r megadtuk a lem�rdend� t�vols�got akkor le is m�rj�k
				paper->SelectPoint(cX,cY);
				if (paper->selectedPoints.size() == 2) {
					vec2 section = paper->selectedPoints[0]->getPosition() - paper->selectedPoints[1]->getPosition();//szakasz sz�m�t�sa
					paper->Distance = length(section);//szakasz hossz�nak meghat�roz�sa
				}
				glutPostRedisplay();//k�p �jra rajzol�s
			break;
		case 'c':
			//k�r felrajzol�sa
				if (paper->Distance >= 0) {//ha a lem�rt t�vols�g val�s akkor felrajzoljuk a k�rt
					paper->SelectPoint(cX, cY);// a kiv�lasztott pontba, ha van olyan
					if (paper->selectedPoints.size() == 1) {
						paper->AddCircle(new Circle(paper->selectedPoints[0], paper->Distance));
						paper->ClearSelection();// t�r�lj�k a kor�bbi kiv�laszt�st
						glutPostRedisplay();//�s �jra rajzoljuk a k�pet
					}
				}
			break;
		case 'i':
			//metsz�spont keres�se
				paper->SelectObject(cX, cY);
				if (paper->selectedObjects == 2) {

					//kijel�lt objektumok f�ggv�ny�ben keress�k a metsz�spontokat
					if (paper->selectedLines.size() == 1) {
						paper->Intersect(paper->selectedLines[0], paper->selectedCircles[0]);
					}
					else if (paper->selectedLines.size() == 0) {
						paper->Intersect(paper->selectedCircles[0], paper->selectedCircles[1]);
					}
					else {
						paper->Intersect(paper->selectedLines[0], paper->selectedLines[1]);
					}
					//ha lefutott a metsz�spont keres�s akkor t�r�lj�k a kor�bbi kijel�l�st
					paper->ClearSelection();
				}
				glutPostRedisplay();//�s �jra rajzoljuk a k�pet
			break;
		case 'l':
			//egyenes felrajzol�sa 
				paper->SelectPoint(cX, cY);//k�t pont meghat�roz�sa az egyenesnek
				if (paper->selectedPoints.size() == 2) {
					paper->AddLine(new Line(paper->selectedPoints[0], paper->selectedPoints[1]));//egyenes felv�tele
					paper->ClearSelection();//kor�bbi kijel�l�s t�rl�se
				}
				glutPostRedisplay();//k�p �jra rajzol�sa
			break;
		}
	break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {}
//v�ge :) 