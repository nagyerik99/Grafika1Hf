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


/**
* Dualis szamokat megvalosito struktura
* az automatikus derivalas megvalositasahoz
* mivel ha nem ezt hasznalnam akkor minden egyes objektumnal kor,henger,parabola parameter szerinti derivalast kellene hasznalnom
* Az oran tanult modszerrel es igazabol osztallyal
*/
template<class T> struct Dnum {
	float f; // a tenyleges erteke az elemnek
	T d;  // az elem derivaltjai parameterek szeirnt
	Dnum(float f0 = 0, T d0 = T(0)) {
		f = f0, d = d0; 
	} // default konstruktor a 0 ertek felvetelehez
	
	Dnum operator+(Dnum r) {//osszeadaskor az ertekek es a derivalt ertekek is osszadodnak
		return Dnum(f + r.f, d + r.d); 
	}
	Dnum operator-(Dnum r) {//kivonaskor hasonlokeppen
		return Dnum(f - r.f, d - r.d); 
	}
	Dnum operator*(Dnum r) {//szorzasnal az ertekek szorzodnak a derivaltak pedig a derivalasi szabaly szerint szorzodnak
		return Dnum(f * r.f, f * r.d + d * r.f);
	}
	Dnum operator/(Dnum r) {//hasonlokeppen az osztasra is
		return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
	}
};

//az egyeb alapszintu derivaltak megvalositasa a Sin es Cos az egyes objektumok X,Y,Z parametereinek kiszamolasanal kell
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f) * g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f) * g.d); }

typedef Dnum<vec2> Dnum2;

const int tessellationLevel = 25;


//virtualis vilag beli kamera
struct Camera {
	vec3 wEye, wLookat, wVup;  //virtualis szem pozicio, lookat pont es felfele irany koordinatai
	float fov, asp, fp, bp; //field of view "mekkora szogben latjuk a kepet", aspektus, elso es hatso vagolap
public:
	Camera() {//kamera default konstruktora
		asp = (float)windowWidth / windowHeight;//belso parameterek kiszamitasa
		fov = 75.0f * (float)M_PI / 180.0f;
		fp = 0.1; bp = 20;
	}

	//view transzformacios matrix az eloadason tanult modon
	//kozeppont elmozgatasa a megfelelo pontba
	mat4 V() {
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}

	//projekcios matrix
	mat4 P() {
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp * bp / (bp - fp), 0);
	}
};

/* Az egyes anyagok közös tipusa ebbol szarmazhatna a rücskös illetve a tukrozo feluletu anyagok
de mivel most csak rucskos feluletu anagokkal dolgozunk igy ettol eltekintettem*/
class Material {
public:
	vec3 kd, ks, ka; // diffuz visszaverodesi tenyezo, "ambiens szin" stb.
	float shininess;//shininess parameter
};


/**
* A fenyforrast jelkepezo osztaly melynek lehet pozicioja fenyerossege stb.
*/
class Light {
public:
	vec3 La, Le;
	vec4 wLightPos; // homogeneous coordinates, can be at ideal point
};


/**
* A renderStateben tarolt attributomok lesznek atadva a vertex es fragment shadernek a phonShader megvalositasahoz
* Azert ezt valasztottam mert ahogy oran megtanultuk ez adja vissza legrealisztikusabban a kivant hatast
* Ezzel szemben a Gouraud shader a fenyeket es a tukrozodest elrontana
*/
class RenderState {
public:
	mat4	           MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light> lights;
	vec3	           wEye;
};

//shader(ek) kozos ose amely a uniformok beallitasat valositja meg
class Shader : public GPUProgram {
public:
	virtual void Bind(RenderState state) = 0;

	//unifrmok beallitasa 
	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	//fenyforrasok atadasa a vertex shadernek
	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};

//phongshader megvalositasa
class PhongShader : public Shader {

	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse matrix
		uniform Light[8] lights;    // fenyforrasok
		uniform int   nLights;		//fenyforrasok szama
		uniform vec3  wEye;         // a virtualis vilag szempozicioja

		layout(location = 0) in vec3  vtxPos;            // az objektum pozicioja a modellezesi vilagban
		layout(location = 1) in vec3  vtxNorm;      	 // normalvektora

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // pozicio transformalasa es tovabbkuldese
			
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w; //a fenyforrasok tranformalasa vilagkoordinatakba
			}
		    wView  = wEye * wPos.w - wPos.xyz; //nezopont transformalasa a tovabbi szamolasokhoz
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		//ahhoz hogy tujuk hasznalni a CPU-n felepitett strukturat azt itt is meg kell valositani
		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material; //az objektum "anyaga"
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal); //normal vektor normalizalasa
			vec3 V = normalize(wView);  //view normalizalasa

			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 ka = material.ka;
			vec3 kd = material.kd;

			vec3 radiance = vec3(0, 0, 0);
			//a radiancia kiszamolasa az adott "pontban" minden fenyforrast figyelembe veve
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				//spekularis visszaverodes szamolasa
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				radiance += ka * lights[i].La + (kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			//a kiszamolt radiancia "beallitasa" szinnek
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { 
		create(vertexSource, fragmentSource, "fragmentColor"); 
	}

	void Bind(RenderState state) {//mikor kirajzoljuk a kepet akkor a renderstate ben szereplo uniformnak szant ertekeket allitjuk be
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};


//az egyes formak geometriainak kozos osztalya
class Geometry {
protected:
	unsigned int vao, vbo;//minden geometriahoz kulon vertexarray es vertexbuffer tartozik
public:
	
	Geometry() {
		//vao es vbok lefoglalasa
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	//virtualis draw fuggveny minden tenyleges geometriai objetkum masik ose fogja megvalositani
	virtual void Draw() = 0;

	//lefoglalt vao vbo felszabaditasa
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};


//a rajzolas es a vertexData generalasara kialakitott/fenntartott osztaly
//minden geometriai obejktum belole oroklodik es fellul definialja az eval fuggvenyt ami az
//U es V parameterek alapjan kiszamolja az X, Y,Z koordinatakat es az egyes pontokban vett derivaltakat dualis szamok
class ParamSurface : public Geometry {
	struct VertexData {
		vec3 position, normal;//az atadott pozicioja es normalvektor
	};

	unsigned int nVtxPerStrip, nStrips;
public:
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;

	//vertex adat generalasakor hivodik meg, u v parameterekkel. Meghivja az egyes megvalositott eval fuggvenyeket
	VertexData GenVertexData(float u, float v) {
		VertexData vtxData;
		Dnum2 X, Y, Z;
		Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
		eval(U, V, X, Y, Z);
		vtxData.position = vec3(X.f, Y.f, Z.f);
		vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
		//normalvektor a parameter szerinti derivaltak keresztszorzataval kaphatunk,vagyis egyfelekeppen
		vtxData.normal = cross(drdU, drdV);
		return vtxData;
	}

	//vertex adat generalasa N*(M+1)*2 vertex Ponttal
	void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		//generalt adatok attoltese a gpu-ra
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);


		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL

		//az elso 3 float vec3  az a poziciot hatarozza meg majd offsettel szamolunk a normalvektorhoz is 
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
	}

	//generalt poziciok normalvektorokkal valo szamolas 
	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};


//a gombot megvalostio objektum
class Sphere : public ParamSurface {
	float R; //sugara
public:
	Sphere(float r = 0.05f) 
	{ 
		R = r;
		create(); 
	}

	//kor egy pontjanak kiszamitasa, es a kiszamitott ertekek visszaadasa az X,Y,Z attributumokban
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * (float)M_PI, V = V * (float)M_PI;
		X = Cos(U) * Sin(V)*R; Y = Sin(U) * Sin(V)*R; Z = Cos(V)*R;
	}
};

//henger objektum
class Cylinder : public ParamSurface {

public:
	Cylinder() {
		create(); 
	}

	//egy egyseg nagysagu henger kiszamolasa es ezt utana a scale matrix-al meretezzuk
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * M_PI, V = V * 2 - 1.0f;
		X = Cos(U); Y = Sin(U); Z = V;
	}
};

//Paraboloid objektum
class Paraboloid : public ParamSurface {
	float R; // a parabola "legfelso" korenek sugara, vagyis mennyire teruljon szet a paraboloid
public:

	Paraboloid(float r = 0.2) { 
		R = r;
		create(); 
	}

	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		U = U * 2.0f * M_PI, V = (V * R - 1.0f);
		X = V * Cos(U);
		Y = V * Sin(U);
		Z = V*V;
	}
};

//Sik objektum a lampa erre van rahelyezve
class Plane :public ParamSurface {
private:
	vec3 a, b; // sikot kifeszito ket vektor
	Dnum2 px, py, pz; // sik egy pontjanak koordinatai
public:
	Plane(vec3 _p, vec3 _a, vec3 _b) {
		a = _a;
		b = _b;
		px = Dnum2(_p.x, vec2(0, 0));
		py = Dnum2(_p.y, vec2(0, 0));
		pz = Dnum2(_p.z, vec2(0, 0));
		create();
	}

	//sik pontjanak meghatarozasa a hozzatartozo derivaltakkal egyutt
	void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
		X = (px + U * a.x + V * b.x)*4;
		Y = (py + U * a.y + V * b.y);
		Z = (pz + U * a.z + V * b.z) * 20;
	}
};



//megvalositott objektumok amelyeknek van geometriaja,anyaga es lehet forgatni eltolni skalazni
class Object {
public:
	Shader* shader;
	Material* material;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;

	Object(Shader* _shader, Material* _material, Geometry* _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		material = _material;
		geometry = _geometry;
	}

	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	//objektum kirajzolasa
	//Render state osszerakasa es a geometria kirajzolasa
	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		shader->Bind(state);
		geometry->Draw();
	}
};
 

//a virtualis vilagban elhelyezett lampa 
//object-ekbol epul fel
class Lamp {
	//lampa egyes darabjai
	Object* base, *joint1, *joint2, *joint3, *beam1, *beam2, *bell;
	//anyaga
	Material* lampMaterial;
	//hasznalt geometriak
	Geometry* sphere, * cylinder, *paraboloid;
	//meretek
	float beamLength, bellSize,jointSize,baseSize;
	//hasznalt shader
	Shader* shader;
	std::vector<Object*> lampParts;
public:
	//lampa helyzete a sikhoz kepest
	vec3 PlanePos,LampPos;
public:
	Lamp(Shader* _shader, vec3 planePosition) {
		PlanePos = planePosition;
		lampParts = std::vector<Object*>();
		lampMaterial = new Material;
		lampMaterial->kd = vec3(0.15f, 0.1f, 0.15f);
		lampMaterial->ks = vec3(1, 1, 1);
		lampMaterial->ka = lampMaterial->kd * M_PI;
		lampMaterial->shininess = 150;
		shader = _shader;
		jointSize = 0.05f;
		baseSize = 1.0f;
		beamLength = 0.25f;
		LampPos = PlanePos + vec3(2.5, 0, 2.5);
	}

	//lmapa felepitese darabok osszerakasa majd az egyes darabokat eltaroljuk a lampPart "listaban" amit visszaadunk a scene-nek ami ezutan rendezi a kirajzolasokat
	std::vector<Object*> BuildLamp() {
		float height = 0.1;
		float magicConst = 0.01f;
		lampParts.clear();

		sphere = new Sphere(jointSize);
		cylinder = new Cylinder();
		paraboloid = new Paraboloid(baseSize);


		base = new Object(shader, lampMaterial, paraboloid);
		base->translation = LampPos+vec3(0, height, 0);
		base->scale = vec3(0.4f, 0.4f, 0.1f);
		base->rotationAxis = vec3(1, 0, 0);
		base->rotationAngle = 5*M_PI/2;
		lampParts.push_back(base);
		
		height += jointSize;

		joint1 = new Object(shader, lampMaterial, sphere);
		joint1->translation = LampPos + vec3(0,height, 0);
		joint1->scale = vec3(1.0f, 1.0f, 1.0f);
		lampParts.push_back(joint1);

		height += beamLength + jointSize - magicConst;
		beam1 = new Object(shader, lampMaterial, cylinder);
		beam1->translation = LampPos + vec3(0, height, 0);
		beam1->rotationAxis = vec3(1, 0, 0);
		beam1->rotationAngle = M_PI / 2;
		beam1->scale = vec3(jointSize-magicConst, jointSize-magicConst, beamLength);
		lampParts.push_back(beam1);

		
		height += beamLength + jointSize - magicConst;
		joint2 = new Object(shader, lampMaterial, sphere);
		joint2->translation = LampPos + vec3(0, height, 0);
		joint2->scale = vec3(1.0f, 1.0f, 1.0f);
		lampParts.push_back(joint2);


		height += beamLength + jointSize - magicConst;
		beam2 = new Object(shader, lampMaterial, cylinder);
		beam2->translation = LampPos + vec3(0, height-0.1f, -0.13f);
		beam2->rotationAxis = vec3(1, 0, 0);
		beam2->rotationAngle = M_PI / 3;
		beam2->scale = vec3(jointSize - magicConst, jointSize - magicConst, beamLength);
		lampParts.push_back(beam2);


		height += beamLength + jointSize - magicConst;
		joint3 = new Object(shader, lampMaterial, sphere);
		joint3->translation = LampPos + vec3(0, height-0.15f, -0.27f);
		joint3->scale = vec3(1.0f, 1.0f, 1.0f);
		lampParts.push_back(joint3);



		height += jointSize - magicConst;
		base = new Object(shader, lampMaterial, paraboloid);
		base->translation = LampPos + vec3(0.12f, height-0.15f, -0.28);
		base->scale = vec3(0.4f, 0.4f, 0.3f);
		base->rotationAxis = vec3(1, 1, 0);
		base->rotationAngle =  18*M_PI/4;
		lampParts.push_back(base);

		return lampParts;
	}
};

//virtualis vilagunk szintere
class Scene {
	std::vector<Object*> objects; // az elhelyezett objektumok amiket ki kell rajzolni
	Camera camera; // 3D camera
	std::vector<Light> lights;
	Lamp* lamp; // kirajzolando lampa
public:
	//szinter felepitese a lampa osszerakas asik kirajzolasa, camera parameterek beallitasa
	//shader letrehozasas
	//fenyek meghatarozasa, helyezet tipus erosseg stb.
	void Build() {

		Shader* phongShader = new PhongShader();
		Material* material1 = new Material;
		material1->kd = vec3(0.1f, 0.1f, 0.1f);
		material1->ks = 0;
		material1->ka = material1->kd * M_PI;
		material1->shininess = 150;

		
		Geometry* plane = new Plane(vec3(0, 0, 0), vec3(1, 0, 0), vec3(0, 0, 1));
		Object* planeObject = new Object(phongShader, material1, plane);
		planeObject->translation = vec3(-1, 2, -1);
		planeObject->scale = vec3(3.0f, 3.0f, 3.0f);

		objects.push_back(planeObject);

		lamp = new Lamp(phongShader,planeObject->translation);

		std::vector<Object*> lampParts = lamp->BuildLamp();
		for (int i = 0; i < lampParts.size(); i++) {
			objects.push_back(lampParts[i]);
		}

		
		camera.wEye = vec3(5, 3, 2);
		camera.wLookat = lamp->LampPos;
		camera.wVup = vec3(0, 1, 0);

		
		lights.resize(2);
		lights[0].wLightPos = vec4(-5, 1, -3, 0);
		lights[0].La = vec3(0.2f, 0.2f, 0.2f);
		lights[0].Le = vec3(2, 2, 2);

		vec3 pos = planeObject->translation + vec3(2.5, 0, 2.5);
		vec4 pos4 = vec4(pos.x, pos.y, pos.z, 1);
		pos4 = pos4 * TranslateMatrix(objects[objects.size() - 1]->translation);

		lights[1].wLightPos = pos4-vec4(-0.15,2.15,1.5,0);
		lights[1].La = vec3(0.2f, 0.2f, 0.2f);
		lights[1].Le = vec3(12, 12, 12);
	}


	//a megvalositott vilag kirajzolasa az aktualis kamera poziciobol nezve
	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object* obj : objects) obj->Draw(state);
	}

	//ido fuggvenyetol fuggo animalas
	void Animate() {
		//lookat pont koruli forgatas
		float dt = 0.001f;
		camera.wEye = vec3((camera.wEye.x - camera.wLookat.x) * cos(dt) + (camera.wEye.z - camera.wLookat.z) * sin(dt) + camera.wLookat.x,
			camera.wEye.y,
			-(camera.wEye.x - camera.wLookat.x) * sin(dt) + (camera.wEye.z - camera.wLookat.z) * cos(dt) + camera.wLookat.z);
	}
};

Scene scene;

// Initialization, create an OpenGL context
//scene felepitese viewport beallitasa, melyesteszteles, a hatsolap eldobas tiltatasa mivel animalunk a vilagban mas nezopontbol is nezzuk a vilagot
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
}

// Window has become invalid: Redraw
//Depth buffert is ki kell takaritani
void onDisplay() {
	glClearColor(0.18f, 0.18f, 0.18f, 0.0f);							// hatterszin beallitasa 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	scene.Render();
	glutSwapBuffers();
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { 
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	static float tend = 0;
	const float dt = 0.1f; 
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;

	// aciklusban animalas a szaggatas elkeruleseert kell mivel koran sem garantalja nekunk a kornyezet, hogy mindig ugyan akkor kerul hivasra az onIdle
	for (float t = tstart; t < tend; t += dt) {
		scene.Animate();
	}
	glutPostRedisplay();
}