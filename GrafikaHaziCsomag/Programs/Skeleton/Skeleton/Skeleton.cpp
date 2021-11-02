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
vec3 LineColor(1.0f, 0.0f, 0.0f);//a kirajzolt vonalak színe
vec3 PointColor(1.0f, 1.0f, 0.0f);//a kirajzolt Pontok színe
vec3 CircleColor(0.0f, 1.0f, 1.0f);//Kirajzolt körök színe
vec3 SelectColor(1.0f, 1.0f, 1.0f);//Kiválasztott objektum színe

//A 2D-s világot megjelenítõ kamera amely a modellben megalkotott kép
//"Képernyõre" történõ adaptálásért felelõs, a világkordinátákat az MVP mátrix segítségével átvisszük a kamera ablakba.
class Camera2D {
private:
	float wScreenWidht;//a kamera ablak szélessége
	float wScreenHeight;//kamera ablak magassága
	vec2 wCenter;//kamera ablak közepe
public:
	//default konstruktor
	Camera2D(int widht, int height, vec2 center) : wCenter(center), wScreenWidht(widht), wScreenHeight(height) {}

	//a View transzformációért felelõs fv eltoljuk a képet a kamera ablak középpontjába
	mat4 V() {
		return  TranslateMatrix(-wCenter);
	};

	//a View Transzformáció inverze
	mat4 Vinv() {
		return  TranslateMatrix(wCenter);
	};

	//a Projection transzformációért felelõs fv skálázzuk a képet a kamera ablak méreteivel.
	mat4 P() {
		mat4 scale = ScaleMatrix(vec2(2 / wScreenWidht, 2 / wScreenHeight));
		return scale;
	};

	//A Projekciós transzformáció inverze
	mat4 Pinv() {
		return  ScaleMatrix(vec2(wScreenWidht / 2, wScreenHeight / 2));
	};
};//Az inverz transzformációkra a UI-n kerszetüli interkació feldolgozására van szükség, a kapott pixel koordinátákat átvigyük a világkorddináta rendszerbe

Camera2D* camera;

/**
* a Vonal és a Kör õsosztálya
* tartalmazza az alap fveket és logikát
*/
class Object {
protected:
	vec3 color;//az objektum színe
	vec3 equation;//implicit egyenletet tárol, az x,y koordináta az egyenes esetében a normálvektor x,y koordinátája, 
	//kör esetében a középpont x,y koordinátája, z értéke pedig mind2 esetben az egyenlet kontans értéke.
	std::vector<vec2> points;//azon pontok melyek kielégítik az egyenletet.
	bool selected = false;//kiválasztás esetén true-ra állítjuk ezzel jelezve a színváltást
public:
	Object(vec3 col, bool sel) :color(col),selected(sel) {}//alap színek és értékek beállítása

	//visszaadja a kör/egyenes pontjait, metszéspontszámításhoz
	std::vector<vec2> getPoints() {
		return points;
	}

	//beállítja a selected értékét
	void setSelected(bool val) {
		selected = val;
	}

	//visszaadja, hogy ki van e választva az adott objektum
	bool getSelected() {
		return selected;
	}

	//Ha selected az objektum -->SelectedColor egyébként az objektum színe
	vec3 getColor() {
		if (selected)
			return SelectColor;

		return color;
	}

	//visszaadja az egyenes/kör implicit egyenletét
	vec3 GetEquation() {
		return equation;
	}

	//Meghatározza, a pont távolságát, az egyenes/kör implicit egyenletéhez viszonyítva
	virtual float CalcDistance(vec2 point) = 0;

	//Meghatározza, hogy kattintás esetén az adott objektum volt e a kiválasztani kívánt objektum
	bool Pick(float cX, float cY) {
		if (selected)//ha már kiválasztottuk akkor nincs mit tenni
			return false;

		vec4 wPoint4 = vec4(cX, cY, 0, 1) * camera->Pinv() * camera->Vinv();//a világkoordináták meghatározása
		vec2 point = vec2(wPoint4.x, wPoint4.y);
		float absVal = CalcDistance(point);//távolság számítás egyeneshez viszonyítva

		if (absVal <= 0.1f) {//ha a közelébe esik/tehát nagy valószínûséggel ez volt a választandó objektum/ akkor visszatérünk igaz értékkel
			selected = true;
		}

		return selected;
	}

};

/**
* A képernyõn megjelenõ kontrollPontok(amik egyben metszéspontok is)
*/
class ControlPoint {
private:
	bool selected;//objektumhoz hasonlóan selected a kiválasztás jelzésére
	vec2 wPos;//helyzete világkoordinátákban
public:
	ControlPoint(vec2 pos, vec3 color) : selected(false),wPos(pos) {}//pozició meghatározása, alap értékek megadása

	vec2 getPosition() {
		return wPos;
	}

	void setSelected(bool val) {
		selected = val;
	}
	
	bool getSelected() {
		return selected;
	}

	//hasonlóképp mint az objektumok esetében
	vec3 getColor() {
		if (selected) return SelectColor;
		
		return PointColor;
	}
};

/**
* A egyenest reprezentáló osztály
*/
class Line : public Object{
private:
	unsigned int vaoLine;//a egyeneshez tartozó vao és vbo felvétele
	unsigned int vboLine[2];//2 vbo felvétele és feltöltése a gpu-n az elsõ vbo a koordinátákat tartalmazza rendre, a második pedig az objektumhoz tartozó színt
	ControlPoint* wPointA, * wPointB;//az egyenes meghatézosáráa szolgáló 2 KontrollPont

	//az implicit egyenletet kielégítõ pontok koordinátáinak kiszámítása 
	void CalcPoints() {
		points.clear();
		float x = -5.0f;//a kamera ablak bal szélétõl a kamera ablak jobb széléig haladunk
		float y;//folyamatosan növeljük x értékét, valamekkora aránnyal, majd kiszámítjuk az y értékét és így tároljuk el vec2 típusként
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
	//default kontruktor, kiszámítjuk az egyenes egyenletét és az azt kielégítõ pontokat
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

	//vao és vbo-k generálása, attribPointerek definiálása a generált vbo-k hoz.
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

	//A egyenes kirajzolása
	void Draw() {
		std::vector<vec3> colors;//egyenes színét tartalmazó vector
		std::vector<vec2> endPoints;//egyenes kezdõ és végpontját tartalmazó vektor
		endPoints.push_back(points[0]);
		endPoints.push_back(points[points.size()-1]);
		glBindVertexArray(this->vaoLine);

		for (int i = 0; i < endPoints.size(); i++) {
			colors.push_back(this->getColor());//színek felvétele annak függvényében, hogy ki van-e választva az adott egyenes
		}

		//bufferek feltöltése, majd egynes kirajzolása.
		glBindBuffer(GL_ARRAY_BUFFER, vboLine[0]);
		glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(vec2), &endPoints[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, vboLine[1]);
		glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(vec3), &colors[0], GL_STATIC_DRAW);
		glDrawArrays(GL_LINES, 0, endPoints.size());
	}

	//Object-ben definiált metódus megvalósítása
	float CalcDistance(vec2 point) {
		//kiszámítjuk a pontra az egyenes implicit egyenletét, amely az adott "szakasz" és pont távolsága lesz.
		// nx*X+ny*Y = (nx*X1+ny*Y1)-konstans-
		float nxX = equation.x * point.x;
		float nyY = equation.y * point.y;
		float val = nxX + nyY - equation.z;
		float val2 = val * val;
		float absVal = sqrtf(val2);

		return absVal;
	}
};

//kört reprezentáló objektum a világunkban
class Circle :public Object{
private:
	unsigned int vaoCircle;//vao és vbok felvétele hasonlóképp az egyeneshez
	unsigned int vboCircle[2];
	ControlPoint* wCenterPoint;
	float Radius;//a kör sugara 

	//Kör egyenletével kiszámítjuk az azt kielégítõ pontokat,aemelyek hasonlóan az egyenes esetében
	//a metszéspont és a kiválasztás meghatározásához szükséges
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
		equation = vec3(wCenter.x, wCenter.y, R2);//kör egyenletének kiszámítása

		this->Create();//Vao és vbo-k létrehozása
		this->CalcPoints();//Pontok meghatározása
	}

	void Create() {
		glGenVertexArrays(1, &this->vaoCircle); //vao generálása majd bindolása 
		glBindVertexArray(this->vaoCircle);
		glGenBuffers(2, &vboCircle[0]);//bufferek generálása és pointerek mehatározása
		glBindBuffer(GL_ARRAY_BUFFER, vboCircle[0]);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), NULL);
		glBindBuffer(GL_ARRAY_BUFFER, vboCircle[1]);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(vec3), NULL);
	}

	//kör kirajzolása 
	void Draw() {
		glBindVertexArray(this->vaoCircle);
		std::vector<vec3> colors;

		for (int i = 0; i < points.size(); i++) {
			colors.push_back(this->getColor());//pontok és a körhöz tartozó szín eltárolása
		}

		//majd a gpu-ra átvivendõ adatok feltöltése 
		glBindBuffer(GL_ARRAY_BUFFER, vboCircle[0]);
		glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(vec2), &points[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, vboCircle[1]);
		glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(vec3), &colors[0], GL_STATIC_DRAW);
		glDrawArrays(GL_LINE_LOOP, 0, points.size());

	}

	//egyeneshez hasonlóan az egyenlet segítségével a kör és a pont távolságának meghatározása
	float CalcDistance(vec2 point) {
		float xu = point.x - equation.x;
		float yv = point.y - equation.y;
		float val2 = powf(((xu * xu) + (yv * yv) - equation.z), 2.0f);
		float absVal = sqrtf(val2);

		return absVal;
	}
};

//a Virtuális világunkat/szerkesztõ felületünket reprezentáló osztály/objektum
class Paper {
private: 
	unsigned int vao;//a controllPontokhoz tartozó vao
	unsigned int vboControlPoints[2];//a controllPontokhoz tartozó vbo-k
	std::vector<ControlPoint*> controlPoints;//controllPontokat tartalmazó vector
	std::vector<Line*> lines;// az egyeneseet tartalmazó "lista"
	std::vector<Circle*> circles;//köröket tartalmazó "lista"

	bool Picked(ControlPoint* cPoint, vec2 wPoint) {
		vec2 pos = cPoint->getPosition();// a felhasználó által "megadott pont" és a kontrollPont távolságának meghatározása
		vec2 section = pos - wPoint;
		if (length(section) <= 0.1f) {//ha kicsi ==> ez a kiválasztott kontrollPont
			return true;
		}

		return false;//egyébként nem 
	}
public:
	char DrawingState; //az aktuális állapotot tároljuk el benne, kör razolás, egyenes rajzolás távmérés stb.
	float Distance = -1.0f; // a felvett körzõtávolságot tároljuk el
	std::vector<ControlPoint*> selectedPoints; // a felhasználói interkacióval kiválasztott pontokat tároljuk a listában
	std::vector<Line*> selectedLines; // kiválasztott egyeneseket tároljuk
	std::vector<Circle*> selectedCircles; // kiválasztott köröket tároljuk
	int selectedObjects=0; // kiválasztott objektumok száma

	//létrehozzuk a default setupot, vonal rajzolása középre, és középre és tõle jobbra egy pont létrehozása
	Paper(vec2 size) {
		//közép mint koordináta meghatározása 
		float cX = windowWidth / windowWidth - 1;	
		float cY = 1.0f - windowWidth / windowHeight;
		double scale = windowWidth / size.x;

		vec4 center4 = vec4(cX, cY, 0, 1) * camera->Pinv()*camera->Vinv();//át transzformáljuk világkoordinátákba
		vec4 center2Right4 = center4;

		center2Right4.x += 1.0f; //létrehozzuk a kettõvel eltolt pontot.

		//ezeket eltároljuk, majd felvesszük az erre a két pontra illeszkedõ egyenest.
		ControlPoint* centerPoint = new ControlPoint(vec2(center4.x,center4.y), PointColor);
		ControlPoint* rightOne = new ControlPoint(vec2(center2Right4.x, center2Right4.y), PointColor);
		controlPoints.push_back(centerPoint);
		controlPoints.push_back(rightOne);
		lines.push_back(new Line(centerPoint, rightOne));
		//vao és vbok generálása + pointerek beállítása
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
		//MVP mátrix létrehozása amit uniformként átadunk a GPU-nak ami ezután minden átvitt pontot beszoroz vele, ezzel
		//át transzformálva a világkoordinátákat normál koordináta rendszerbe
		mat4 MVP = camera->P() * camera->V();
		gpuProgram.setUniform(MVP, "MVP");

		//körök kirajzolása 
		for (size_t i = 0; i < circles.size(); i++)
		{
			circles[i]->Draw();
		}

		//egyenesek kirajzolása
		for (size_t i = 0; i < lines.size(); i++)
		{
			lines[i]->Draw();
		}
		
		glBindVertexArray(this->vao);


		//pontok kirajzolása, és a hozzájuk tartozó színek meghatározása
		std::vector<vec2> points = std::vector<vec2>();
		std::vector<vec3> colors = std::vector<vec3>();

		for (size_t i = 0; i < controlPoints.size(); i++) {
			points.push_back(controlPoints[i]->getPosition());
			colors.push_back(controlPoints[i]->getColor());
		}

		//a megfelelõ bufferek feltöltése a pozíció - szín párokkal
		glBindBuffer(GL_ARRAY_BUFFER, vboControlPoints[0]);
		glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(vec2), &points[0], GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, vboControlPoints[1]);
		glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(vec3), &colors[0], GL_STATIC_DRAW);
		glDrawArrays(GL_POINTS, 0, points.size());
	}

	/**Miután a felhasználó kattintott egyet a felületen
	* levizsgáljuk, hogy melyik kontrollponthoz volt közel, és amennyiben megtaláljuk
	* akkor azt eltároljuk a selectedPoints listába
	*/
	void SelectPoint(float cX, float cY) {
		vec4 wPoint4 = vec4(cX, cY, 0, 1) * camera->Pinv() * camera->Vinv();
		vec2 wPoint = vec2(wPoint4.x, wPoint4.y);


		for (size_t i = 0; i < controlPoints.size(); i++) {
			{
				//ha közel van és még nem volt kiválasztva akkor eltároljuk, és mivel meg is találtuk a keresettet
				//ezért ki is lépünk a ciklusból.
				if (Picked(controlPoints[i], wPoint) && !controlPoints[i]->getSelected()) {
					controlPoints[i]->setSelected(true);
					selectedPoints.push_back(controlPoints[i]);
					break;
				}
			}
		}
	}

	/**
	* megkeressük a kiválasztani kívánt objektumot a pontoknál használt logikával
	* amelyikhez közel vagyunk azt eltároljuk, majd mivel megtaláltuk ezért vissza is térhetünk
	* fontos szempont, hogy ha már ki van választva akkor ne lehessen mégegyszer kiválasztani.
	*/
	void SelectObject(float cX, float cY) {
		for (size_t i = 0; i < lines.size(); i++)
		{
			if (!lines[i]->getSelected())//ha még nincs kiválasztva
				if (lines[i]->Pick(cX, cY)) {//és közel van
					this->selectedObjects++;//akkor eltároljuk
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

	//ha lefutott az intersect vagy állapotot váltunk, vagy bármi egyéb ok miatt
	//törölni kellene az egyes jelöléseket akkor azt ezzel tesszük meg
	void ClearSelection() {
		//visszaállítjuk az összes pont és objektum selected értékét false-ra 
		//ezáltal a színe nem a kijelölési szín lesz
		//és újra el tudjuk tárolni majd ezeket a pontokat/objektumokat
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

	//kör felvétele a "papír lapra"
	void AddCircle(Circle* circle) {
		circles.push_back(circle);
	}

	//egyenes rajzolása a papírlapra
	void AddLine(Line* line) {
		lines.push_back(line);
	}

	//metszéspont felvétele a pontok közé
	void AddControlPoint(vec2 point) {
		for (size_t i = 0; i < controlPoints.size(); i++)
		{
			//ha a pont egy már meglévõ kontrollpont közelében van akkor valószínû, hogy a két pont ugyanaz
			if (Picked(controlPoints[i], point)) return;//de csak akkor, ha még nincs felvéve a pontok közé
		}

		controlPoints.push_back(new ControlPoint(point, PointColor));
	}


	//két egyenes metszéspontjának kiszámítása
	void Intersect(Line* line1, Line* line2) {
		std::vector<vec2> points = line2->getPoints();
		//végigmegyünk az egyik egyenes pontjain és megvizsgáljuk,hogy milyen közel vannak a másik egyeneshez
		//jobban mondva, hogy rajta vannak-e,
		//ha igen, akor eltároljuk mint kontrollPont/metszésPont
		for (size_t i = 0; i < points.size(); i++)
		{
			float absVal = line1->CalcDistance(points[i]);
			if (absVal <= 0.02f)
				AddControlPoint(points[i]);
		}
	}
	//egy kör és egy egyenes metszéspontjának kiszámítása
	void Intersect(Line* line, Circle* circle) {
		std::vector<vec2> points = circle->getPoints();
		//megvizsgáljuk,hogy a kör pontjai közül melyek vannak "rajta"/vagy nagyon közel az egyeneshez
		//ha találunk ilyent akkor az egy metszéspont és eltároljuk.
		for (size_t i = 0; i < points.size(); i++) {
			float absVal = line->CalcDistance(points[i]);
			if (absVal <= 0.02f) {
				AddControlPoint(points[i]);
			}
		}
	}
	//két kör metszéspontjának kiszámítása
	void Intersect(Circle* circle1, Circle* circle2) {
		std::vector<vec2> points = circle1->getPoints();
		//egyik kör pontjai közül melyek vannak rajta a másik körön, ha találunk ilyent akkor az lesz
		//a körök metszéspontja
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

	camera = new Camera2D(10, 10,vec2(0, 0));//kamera/világ létrehozása 10x10 cm ben
	paper = new Paper(vec2(10, 10));//Papír létrehozása a rajzoláshoz

	//vertex és Fragmentshaderek betöltése
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	paper->Draw();//papíron lévõ dolgok felrajzolása

	glutSwapBuffers(); // exchange buffers for double buffering
}


// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 's') paper->DrawingState = 's'; //körzõvel távolság felvétele állapot
	if (key == 'c') paper->DrawingState = 'c';//kör rajzolása  állapot
	if (key == 'l') paper->DrawingState = 'l';//egyenes rajzolás aállapot
	if (key == 'i') paper->DrawingState = 'i'; // metszéspont keresése állapot

	paper->ClearSelection();// korábbi eltárolt kijelölések törlése
	printf("Selected State: %c \n", paper->DrawingState);//állapot kiirása
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
		//állapottol függõen cselekvést végzünk
		switch (paper->DrawingState) {
		case 's':
				//korábbi kijelölés törlése
				if (paper->selectedPoints.size() == 2)
					paper->ClearSelection();

				//ha már megadtuk a lemérdendõ távolságot akkor le is mérjük
				paper->SelectPoint(cX,cY);
				if (paper->selectedPoints.size() == 2) {
					vec2 section = paper->selectedPoints[0]->getPosition() - paper->selectedPoints[1]->getPosition();//szakasz számítása
					paper->Distance = length(section);//szakasz hosszának meghatározása
				}
				glutPostRedisplay();//kép újra rajzolás
			break;
		case 'c':
			//kör felrajzolása
				if (paper->Distance >= 0) {//ha a lemért távolság valós akkor felrajzoljuk a kört
					paper->SelectPoint(cX, cY);// a kiválasztott pontba, ha van olyan
					if (paper->selectedPoints.size() == 1) {
						paper->AddCircle(new Circle(paper->selectedPoints[0], paper->Distance));
						paper->ClearSelection();// töröljük a korábbi kiválasztást
						glutPostRedisplay();//és újra rajzoljuk a képet
					}
				}
			break;
		case 'i':
			//metszéspont keresése
				paper->SelectObject(cX, cY);
				if (paper->selectedObjects == 2) {

					//kijelölt objektumok függvényében keressük a metszéspontokat
					if (paper->selectedLines.size() == 1) {
						paper->Intersect(paper->selectedLines[0], paper->selectedCircles[0]);
					}
					else if (paper->selectedLines.size() == 0) {
						paper->Intersect(paper->selectedCircles[0], paper->selectedCircles[1]);
					}
					else {
						paper->Intersect(paper->selectedLines[0], paper->selectedLines[1]);
					}
					//ha lefutott a metszéspont keresés akkor töröljük a korábbi kijelölést
					paper->ClearSelection();
				}
				glutPostRedisplay();//és újra rajzoljuk a képet
			break;
		case 'l':
			//egyenes felrajzolása 
				paper->SelectPoint(cX, cY);//két pont meghatározása az egyenesnek
				if (paper->selectedPoints.size() == 2) {
					paper->AddLine(new Line(paper->selectedPoints[0], paper->selectedPoints[1]));//egyenes felvétele
					paper->ClearSelection();//korábbi kijelölés törlése
				}
				glutPostRedisplay();//kép újra rajzolása
			break;
		}
	break;
	}
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {}
//vége :) 