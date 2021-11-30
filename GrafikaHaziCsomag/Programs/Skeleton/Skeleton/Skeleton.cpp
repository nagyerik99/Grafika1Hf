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


/* Az egyes anyagok közös tipusa ebbol szarmazhatna a rücskös illetve a tukrozo feluletu anyagok
de mivel most csak rucskos feluletu anagokkal dolgozunk igy ettol eltekintettem*/
struct Material {
	vec3 ka, kd, ks; // diffuz visszaverodesi tenxezo, shininess parameter stb.
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) {
		shininess = _shininess; 
	}
};

//egy talalatot reprezental, ha a ray eltalata a feluletet illetve akkor is ha nem olyankor ambiens fennyel fog visszaterni
struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

//sugart reprezentalo struktura amelynek van kezdeti pontja iranya ez tudja elmetszeni a feluletet
struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

//minden olyan objektumot reprezental amit elmetszhet a ray(fenysugar), ennek vizsgalatara van az egyetlen absztrakt fuggvenye az intersect fv.
class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};


//matrix transzponalasat valositja meg, erre az inverzmatrix transzponaltjaval valo szorzas megvalositasahoz van szukseg
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


//matrix szorzasa vec4 transzponalatjaval muveletet valositja meg
vec4 operator*(mat4 mat, vec4& vec) {
	vec4 result(0, 0, 0, 0);
	for (int i = 0; i < 4;i++) {
		//ez egy sora a matrixnak 
		result[i] = mat[i][0] * vec[0] + mat[i][1] * vec[1] + mat[i][2] * vec[2] + mat[i][3] * vec[3];
	}
	return result;
}


//korrekcios ertek amire a raynel es a sik talalat szamitasnal van szuksegunk
const float epsilon = 0.0001f;

//tetszoleges vec3 normalizalasa
vec3 normalize(vec3* vec) {
	return (*vec) * (1 / (length(*vec) + epsilon));
}

//a vilagunkban a sikot reprezentalo objektum jelen esetunkben ez az alap amelyre a lampat tesszuk illetve sikok vagjak el az egyes lampadrabokat is.
struct Plane : public Intersectable {
	vec3 point, normal;
	bool forClose;
	mat4 Object;

	//default konstruktok ertekek felvetel a sik egy pontja es a sik normalvektora plusz a sik anyaga illetve 2 tetszoleges parameter amelyre az alap takarolapjanak szamitasanal van szukseg
	Plane(const vec3& _point, const vec3& _normal, Material* mat, mat4 invMat, mat4 _Oject = mat4(), bool _forClosing = false){
		Object = _Oject;
		forClose = _forClosing;
		material = mat;
		point = _point;
		normal = normalize(_normal);
	}

	//a fenysugar es sik metszesenek vizsgalata
	Hit intersect(const Ray& ray) {
		Hit hit;
		double NdotV = dot(normal, ray.dir);
		if (fabs(NdotV) < epsilon) return hit;
		double t = dot(normal, point - ray.start) / NdotV;
		if (t < epsilon) return hit;//nem metszettuk el a sikot
		hit.t = t;
		hit.position = ray.start + ray.dir * hit.t;  //ha elmetszettuk akkor a talalat helyenek megallapitasa

		if (forClose) {//ha egy masik objektum takarofeluletet akarjuk megvalositani akkor true es megvizsgaljuk hogy az eltalalt pont benne van a kvadratikus objektumban
			vec3 position = hit.position;
			vec4 p4 = vec4(position.x, position.y, position.z, 1);
			float val = dot(p4 * Object, p4);
			if (val < 0)
				return Hit();//ha nincs benne akkor nem "jelezzuk" a talalatot
		}

		hit.normal = normal;
		if (dot(hit.normal, ray.dir) > 0) hit.normal = hit.normal * (-1);
		hit.material = material;
		return hit;//talalati hely normalvektor es anyag meghatarozasa
	}

public:
	vec3 getNormal() {
		return normal;
	}

	void setNormal(vec3 _normal) {
		this->normal = _normal;
	}

	//az adott pont a sik jo oldalan van e 
	bool isPointRightSide(vec3 p) {
		float d = dot(normal, point);
		float val = normal.x * p.x + normal.y * p.y + normal.z * p.z-d;

		return val < 0 ? false : true;
	}
};

// amegfelelo inverzTranszformacios matrixxal megszorozzuk a sik normalvektorat ezzel a transzformacionak megfelelo muveleteket elvegezve rajta
void inverzTransformPlane(mat4 mat, Plane* plane) {
	vec3 normal = plane->getNormal();
	vec4 normal4 = vec4(normal.x, normal.y, normal.z, 1);
	normal4 = mat * normal4;
	normal = vec3(normal4.x, normal4.y, normal4.z);
	normal = normalize(normal);
	plane->setNormal(normal);
}

//Kvadratikus objektuomkat reprezentalo osztaly ilyen objektumokbol epul fel a lampa
struct Quadrics :public Intersectable {
protected:
	 mat4 Q;//az objektumot leiro matrix
	 mat4 InverzMatrix;// a Transzformacio inverzmatrixa
	 Plane* plane1, *plane2;// az objektumot elmetszo sikok amik alapertelemezetten nullptr ek
public:

	Quadrics(mat4 _Q,mat4 inverzAnimation, Material* _material,Plane* _plane1 = nullptr,Plane* _plane2=nullptr) {
		//ha van elmetszo sik akkor a transzformaciot arra is alkalmazni fogjuk ha nincs akkor ertelemszeruen nemm kell
		if (_plane1 != nullptr && _plane2 != nullptr) {
			inverzTransformPlane(inverzAnimation, _plane1);
			inverzTransformPlane(inverzAnimation, _plane2);
		}
		plane1 = _plane1;
		plane2 = _plane2;
		material = _material;
		InverzMatrix =inverzAnimation;
		//a transzformacios muveletek alkalmazasa a kvadratikus objektumra 
		Q = InverzMatrix * _Q * transpose(InverzMatrix);
	}

	float f(vec4 r) {//feltételezve hogy r.w =1
		return dot(r*Q,r);
	}

	//gradf = normál vektor implicit felületeknél
	vec3 gradf(vec4 r) {
		vec4 g = r * Q * 2;
		return vec3(g.x, g.y, g.z);
	}

	vec3 gradf(vec4 r, mat4 Mat) {
		vec4 g = r * Mat * 2;
		return vec3(g.x, g.y, g.z);
	}

	//kvadratikus objektumok elmetszese ray-el vizsgalat
	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 start = vec3(ray.start.x, ray.start.y, ray.start.z);
		vec4 d0 = vec4(ray.dir.x, ray.dir.y, ray.dir.z, 0);
		vec4 s1 = vec4(start.x, start.y, start.z, 1);
		float a = dot(d0 * Q, d0);
		float b = dot(s1 * Q, d0) * 2;
		float c = dot(s1 * Q, s1);
		float b2 = b * b;
		float discr = b2 - 4.0f * a * c;
		//egyenloseg megoldasa
		if (discr < 0) return hit; // nem taálálta el a fénysugár

		//ha eltalata a talalati helyek meghatarozasa
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 mindig
		vec3 p1 = ray.start + ray.dir * t1;

		//megvizsgaljuk hogy a kapott pont az a sik jo oldalan van e es ha nem akkor t=-1 vagyis mintha nem talalta volna el a sugar
		if (plane1 != nullptr && plane2 != nullptr) {
			bool min1 = plane1->isPointRightSide(p1);
			bool max1 = plane2->isPointRightSide(p1);
			if (!min1 || !max1) t1 = -1.0f; // t1 kivul esik a hengeren, de a lapokon még rajta lehet 
		}

		float t2 = (-b - sqrt_discr) / 2.0f / a;
		vec3 p2 = ray.start + ray.dir * t2;
		//itt is hasonlokepp
		if (plane1 != nullptr && plane2 != nullptr) {
			bool min2 = plane1->isPointRightSide(p2);
			bool max2 = plane2->isPointRightSide(p2);
			if (!min2 || !max2) t2 = -1.0f;
		}

		//megnezzuk hohgy van e olyan realis pont ahol eltalalta a sugar az objektumot es a ket hatarolo sik kozott vagyunk
		// ezek kozul pedig a kisebb ertekut vagyis az idoben kozelebbi kell nekunk mivel ez van elol
		if (t1 <= 0 && t2 <= 0) return hit;
		if (t1 <= 0)hit.t = t2;
		else if (t2 <= 0) hit.t = t1;
		else if (t2 < t1) hit.t = t2;
		else hit.t = t1;

		hit.position = start + ray.dir * hit.t;

		hit.normal = normalize(this->gradf(vec4(hit.position.x, hit.position.y, hit.position.z, 1)));
		hit.position = hit.position;
		hit.material = material;

		return hit;
	}
};


//a vilagban elhelyezett kamera 
class Camera {
	//szempozicio felfele irany lokkat pont vagyis hogy hova nezunk
	vec3 eye, lookat, right, up;
	float fov;

	//animacio vagyis a z tengely koruli forgatas ami = a lampa korul forgatas
	mat4 rotateEyeZaxis(float dt) {
		return RotationMatrix(dt, vec3(0, 0,1));
	}

public:
	//kamera kepenek es helyzetenek beallitasa
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		fov = _fov;
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	//fenysugar meghatarozasa
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}

	//kamera animalas vagyis bizonxos idokozonket 0.1 ertekkel forgatjuk az up vagyis a z tengely korul
	void Animate(float dt) {
		vec3 d = eye - lookat;
		vec4 d4 = vec4(d.x, d.y, d.z, 1);
		vec4 rotate4 = d4 * rotateEyeZaxis(dt);
		eye = vec3(rotate4.x, rotate4.y, rotate4.z) + lookat;
		set(eye, lookat, up, fov);
	}
};

// avilagunkban a fenyt reprezentalo objektum 
//ambiens es pontfenyforrasok vannak 
struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

// a szinter ahol elhelyezzuk az egyes objektumokat az elvartaknak megfeleloen
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

	//a ter felepitese objektumok kezdeti elhelyezese megfelelo pozicioba es vagasuk stb.
	void build() {
		vec3 eye = vec3(5,3,5), vup = vec3(0,0,1), lookat = vec3(0,0, 1);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);//kamera beallitaas

		La = vec3(0.3f, 0.3f, 0.3f);//ambiens fenyforras meghatarozasa 
		La2 = vec3(0.5f, 0.5f, 0.5f);
		vec3 lightDirection(5, -5, 7), Le(1, 1, 1);//pontfenyforrras elhelyezese
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd(0.13f, 0.1f, 0.15f), ks(1, 1, 1);
		vec3 kd2(0.1f, 0.18f, 0.15f), ks2(4,4, 4);
		Material* lampColor = new Material(kd, ks, 120);//lampa anyaganak meghatarozas
		Material* ground = new Material(kd2, ks2, 200);// a "padlo" anyaganak meghatarozasa

		float a11 = 5.0f;
		float R = a11/70;
		float correction = 0.05f;


		//formakat leiro mat4 matrixok
#pragma formak
		mat4 cylinder = mat4(-a11 * 0.8, 0, 0, 0,
			0, -a11 * 0.8, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1); //talpzat

		mat4 paraboloid = mat4(a11 * 0.4, 0, 0, 0,
			0, a11 * 0.4, 0, 0,
			0, 0, -1, 0,
			0, 0, 0, 0);//lampa buraja

		mat4 sphere = mat4(-1 / (R * R), 0, 0, 0,
			0, -1 / (R * R), 0, 0,
			0, 0, -1 / (R * R), 0,
			0, 0, 0, 1);//csuklogomb

		mat4 beam = mat4(-a11 * 120.0f, 0, 0, 0,
			0, -a11 * 120.0f, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1);//a clamparudakat leiro henger kvadratikus leirasa

		//kulonbozo meretek es konstansok meghatarozasa a lampa preferalt elhelyezesehez
		float beamLength = 0.9f;
		float bellSize = 1.2f;
		float joint1Pos = 1.1+(R);
		float beam1Max = beamLength+R;
		float joint2Pos = beamLength-R;
		float beam2Min = 7*R;
		float beam2Max = beam2Min + beamLength;
		float joint3Pos = beamLength+R;
		float bellPosMax = R + bellSize;
		float plane2BeamLower = joint1Pos + R + beam2Max + 2 * R;
		float Plane2BeamUpper = plane2BeamLower + beamLength+beamLength;
		float planeBellPosUpper = Plane2BeamUpper + bellSize;

		//Transzformacios es inverzT matrix felvetele
		mat4 Transform = ScaleMatrix(vec3(1, 1, 1));
		mat4 invT = ScaleMatrix(vec3(1, 1, 1));


		//az egyes objektumokat hatarolo sikok meghatarozasa 
		Plane* planeTest = new Plane(vec3(0, 0, 1), vec3(0.0f, 0.0f, -1.0f), ground, invT);
		Plane* planeLower = new Plane(vec3(0, 0, 1.0), vec3(0.0f, 0.0f, 1.0f), ground, invT);

		Plane* planeUpper = new Plane(vec3(0, 0, 1.1), vec3(0.0f, 0.0f, -1.0f), lampColor, invT,cylinder,true);
		Plane* plane1Beam = new Plane(vec3(0, 0, joint1Pos+R), vec3(0.0f, 0.0f, 1.0f), lampColor, invT);

		Plane* plane1BeamUpper = new Plane(vec3(0, 0, joint1Pos+beam1Max), vec3(0.0f, 0.0f, -1.0f), lampColor, invT);
		Plane* plane2Beam = new Plane(vec3(0, 0, plane2BeamLower), vec3(0.0f, 0.0f, 1.0f), lampColor, invT);

		Plane* plane2BeamUpper = new Plane(vec3(0, 0, Plane2BeamUpper), vec3(0.0f, 0.0f, -1.0f), lampColor, invT);
		Plane* planeBell = new Plane(vec3(0, 0, Plane2BeamUpper), vec3(0.0f, 0.0f, 1.0f), lampColor, invT);

		Plane* planeBellUpper = new Plane(vec3(0, 0, planeBellPosUpper), vec3(0.0f, 0.0f, -1.0f), lampColor, invT);

		//talaj felepitese
		lamp.push_back(planeTest);

		//talpzat meghatarozasa
		lamp.push_back(new Quadrics(cylinder,invT,lampColor,planeLower,planeUpper));//lámpa talpzata
		lamp.push_back(planeUpper);

		mat4 Joint1Translation = TranslateMatrix(vec3(0, 0, joint1Pos));
		mat4 Joint1InverzTranslation = TranslateMatrix(vec3(0, 0, -joint1Pos));				//elso csuklogomb elhelyezese es beallitasa
		mat4 Joint1Rotation = RotationMatrix(M_PI / 6.0f, vec3(1, 0, 0));
		mat4 Joint1InverzRotation = RotationMatrix(-1.0f * M_PI / 6.0f, vec3(1, 1, 0));

		Transform = Joint1Rotation* Joint1Translation * Transform;
		invT = invT * Joint1InverzTranslation *Joint1InverzRotation;

		lamp.push_back(new Quadrics(sphere, invT, lampColor));


		mat4 Beam1Translation = TranslateMatrix(vec3(0, 0, R));
		mat4 Beam1InverzTranslation = TranslateMatrix(vec3(0, 0, -R));

		Transform = Beam1Translation* Transform;											//elso lamparud elhelyezese es eltolasa a megfelelo helyre
		invT = invT * Beam1InverzTranslation;
		lamp.push_back(new Quadrics(beam, invT, lampColor, plane1Beam, plane1BeamUpper));


		mat4 Joint2Translation = TranslateMatrix(vec3(0, 0, joint2Pos));
		mat4 Joint2InverzTranslation = TranslateMatrix(vec3(0, 0, -(joint2Pos)));				//2.csuklogom
		mat4 Joint2Rotation = RotationMatrix(M_PI / 5.0f, vec3(1, 0, 0));
		mat4 Joint2InverzRotation = RotationMatrix(-1.0f * M_PI / 5.0f, vec3(1, 0,0));


		Transform = Joint2Rotation *Joint2Translation* Transform;
		invT = invT * Joint2InverzTranslation *Joint2InverzRotation;

		lamp.push_back(new Quadrics(sphere, invT, lampColor));


		lamp.push_back(new Quadrics(beam, invT, lampColor, plane2Beam, plane2BeamUpper)); // 2.lamparud

		mat4 Joint3Translation = TranslateMatrix(vec3(0, 0, joint3Pos));
		mat4 Joint3InverzTranslation = TranslateMatrix(vec3(0, 0, -joint3Pos));

		Transform = Joint3Translation* Transform;											//3.csuklogomb
		invT = invT * Joint3InverzTranslation;
		
		lamp.push_back(new Quadrics(sphere, invT, lampColor));

		mat4 BellTranslation = TranslateMatrix(vec3(0, 0, R));
		mat4 BellInverzTranslation = TranslateMatrix(vec3(0, 0, -R));
		Transform = BellTranslation* Transform;												//lampabura elhelyezese
		invT = invT * BellInverzTranslation;

		lamp.push_back(new Quadrics(paraboloid, invT, lampColor, planeBell, planeBellUpper));
	}

	//miutan elepitettuk a szinteret utana alkalmazzuk a "kép" minden pontjara a sugarkovetest
	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));//adott sugarral a vilag felterkepezese
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {//az elso elmetszett obejktum meghatarozasa
		Hit bestHit;
		for (Intersectable* object : lamp) {
			Hit hit = object->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}


	//az arnyek meghatarozas, itt csak azt kell tudnunk, hogy van e "takaro" felulelt vagy sem
	bool shadowIntersect(Ray ray) {
		for (Intersectable* object : lamp) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	//sugarkovetes megvalositasa
	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = firstIntersect(ray); //elso elmetszett objektum
		if (hit.t < 0) return La;//ha nincs akkor ambiens feny
		vec3 outRadiance = hit.material->ka * La; 
		for (Light* light : lights) {//egyebkent minden fenyforrasbol erkezo rayel megvizsgalni az arnyekoltsagot
			Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
			float cosTheta = dot(hit.normal, light->direction);
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
				outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction);
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
			}
		}
		return outRadiance;//visszaterunk a megfelelo radianciaval
	}
};


GPUProgram gpuProgram; // vertex es fragment shader
Scene scene;//Vilagunk szintere


//a kepet kirajzolo osztaly 
class FullScreenTexturedQuad {
	unsigned int vao;	//vao
	Texture texture; // texturazott kepet valositunk meg
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		//vao es vbo generalasa aktivalasa majd vbo feltoltese az "egesz kepernyovel" vagyis normalizalt koordinatarendszerben a (-1-1)  (1,1) sarokpontokkal rendelkezo negyzet kepe
		glGenVertexArrays(1, &vao);	
		glBindVertexArray(vao);	

		unsigned int vbo;	
		glGenBuffers(1, &vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo); 
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL); 
	}

	//Modositott textura betoltese ujra
	void LoadTexture(int windowWidth, int windowHeight,std::vector<vec4> image){
		texture.create(windowWidth,windowHeight,image);
	}

	//textura felrajzolasa 
	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;
std::vector<vec4> image;


//szinter felepites es gpuprogram initelese
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}


//kepernyo kepenek ujrarajzolasa az ujonnan megalkotott texturaval
void onDisplay() {
	image = std::vector<vec4>(windowWidth * windowHeight);
	scene.render(image);//kep ujra renderelese
	fullScreenTexturedQuad->LoadTexture(windowWidth, windowHeight, image);//betoltese
	fullScreenTexturedQuad->Draw();//rajzolasa
	glutSwapBuffers();//megjelenitese
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	scene.Animate(0.1f);
	glutPostRedisplay();
}
