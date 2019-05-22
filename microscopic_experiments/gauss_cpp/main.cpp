#include <bits/stdc++.h>

//これらは特異値分解のためだけに導入したが、vvdで実装してきた諸々を全て Eigen に移植しても良いかもしれない。
#include <Eigen/Core>
#include <Eigen/SVD>
using Eigen::MatrixXd;
using Eigen::JacobiSVD;

#define pb push_back 
#define mp make_pair 
#define mt make_tuple
#define debug printf("--%d--\n",__LINE__)
using namespace std;
typedef vector<double> vd;
typedef vector<vector<double> > vvd;
typedef vector<int> vi;
typedef vector<vi> vvi;
typedef tuple<vd, vd> tupleDatum;
typedef vector<tupleDatum> tupleData;




template<typename T> T maximum(const vector<T> &x) {T ret = x[0];for(int i=1;i<x.size();i++) ret = max(ret, x[i]); return ret;}
template<typename T> T minimum(const vector<T> &x) {T ret = x[0];for(int i=1;i<x.size();i++) ret = min(ret, x[i]); return ret;}
template<typename T> string to_string(const vector<T> &x){stringstream ss; for(int i=0;i<x.size();i++) ss << (i==0 ? "[" : ", ") << x[i]; ss << "]"; return ss.str();}
string to_string(const vvd &x){
	stringstream ss;
	for(int i=0;i<x.size();i++) {ss << (i==0 ? "[" : " ") << to_string(x[i]) << (i+1==x.size() ? "]" : ",") << endl;}
	return ss.str();
}
bool nearlyEqual(double x, double y){return fabs(x-y)<1e-8;}
vvd zeros(int n, int m){vvd ret(n); for(int i=0;i<n;i++) ret[i] = vd(m); return ret;}
vvd ones(int n, int m){vvd ret = zeros(n, m); for(int i=0;i<n;i++)for(int j=0;j<m;j++) ret[i][j] = 1.0; return ret;}
double drand(){ return (0.0+rand())/RAND_MAX;} // [0,1]
double Uniform(){return (1.0 + rand())/((double)RAND_MAX+2.0);} // (0, 1)
double randn(){return sqrt( -2.0*log(Uniform()) ) * sin( 2.0*M_PI*Uniform() );}
vd randn(int n){vd ret(n); for(int i=0;i<n;i++) ret[i] = randn(); return ret;}
vvd randn(int n, int m){vvd ret = zeros(n, m); for(int i=0;i<n;i++)for(int j=0;j<m;j++) ret[i][j] = randn(); return ret;}
double sum(const vd &x){double ret = 0.0; for(int i=0;i<x.size();i++) ret += x[i]; return ret;}
double norm(const vd &x){double ret = 0.0; for(int i=0;i<x.size();i++) ret += x[i]*x[i]; return sqrt(ret);}
double snorm(const vd &x){double ret = 0.0; for(int i=0;i<x.size();i++) ret += x[i]*x[i]; return ret;}
double dist(const vd &x, const vd &y){double ret = 0.0; for(int i=0;i<x.size();i++) ret += (x[i]-y[i])*(x[i]-y[i]); return sqrt(ret);}
double sdist(const vd &x, const vd &y){double ret = 0.0; for(int i=0;i<x.size();i++) ret += (x[i]-y[i])*(x[i]-y[i]); return ret;}
vd add(const vd &x, const vd &y){assert(x.size()==y.size());vd ret(x.size());for(int i=0;i<ret.size();i++) ret[i] = x[i] + y[i]; return ret;}
vd sub(const vd &x, const vd &y){assert(x.size()==y.size());vd ret(x.size());for(int i=0;i<ret.size();i++) ret[i] = x[i] - y[i]; return ret;}
vd mul(const vd &x, const vd &y){assert(x.size()==y.size());vd ret(x.size());for(int i=0;i<ret.size();i++) ret[i] = x[i] * y[i]; return ret;}
vd mul(const vd &x, double r){vd ret(x.size());for(int i=0;i<ret.size();i++) ret[i] = x[i] * r; return ret;}
vvd add(const vvd &x, const vvd &y){assert(x.size()==y.size() && x[0].size()==y[0].size());vvd ret=zeros(x.size(), x[0].size());for(int i=0;i<ret.size();i++)for(int j=0;j<ret[0].size();j++) ret[i][j] = x[i][j] + y[i][j]; return ret;}
vvd sub(const vvd &x, const vvd &y){assert(x.size()==y.size() && x[0].size()==y[0].size());vvd ret=zeros(x.size(), x[0].size());for(int i=0;i<ret.size();i++)for(int j=0;j<ret[0].size();j++) ret[i][j] = x[i][j] - y[i][j]; return ret;}
vvd mul(const vvd &x, const vvd &y){assert(x.size()==y.size() && x[0].size()==y[0].size());vvd ret=zeros(x.size(), x[0].size());for(int i=0;i<ret.size();i++)for(int j=0;j<ret[0].size();j++) ret[i][j] = x[i][j] * y[i][j]; return ret;}
vvd mul(const vvd &x, double r){vvd ret=zeros(x.size(), x[0].size());for(int i=0;i<ret.size();i++)for(int j=0;j<ret[0].size();j++) ret[i][j] = x[i][j] * r; return ret;}
vvd transpose(const vvd &x){
	vvd ret = zeros(x[0].size(), x.size());
	for(int i=0;i<ret.size();i++)for(int j=0;j<ret[0].size();j++) ret[i][j] = x[j][i];
	return ret;
}
vd sigmoid(const vd &x){
	vd ret(x.size());
	for(int i=0;i<ret.size();i++) ret[i] = 1.0 / (1.0 + exp(-x[i]));
	return ret;
}
vd der_sigmoid(const vd &x){
	vd ret(x.size());
	for(int i=0;i<ret.size();i++) ret[i] = x[i] * (1.0 - x[i]);
	return ret;
}
double crossEntropy(const vd &activatedY, const vd &t){
	double ret = 0.0;
	for(int i=0;i<t.size();i++){
		if (nearlyEqual(t[i],0.0)){
			ret -= log(1.0 - activatedY[i]);
		}else if (nearlyEqual(t[i],1.0)){
			ret -= log(activatedY[i]);
		}else{
			assert(0); // t の各要素は 0 or 1 としている
			//t * log(y) + (1-t) * log(1-y) と一般化して書くこともできるが、ここではしていない。
		}
	}
	return ret / t.size(); //chainerは割っているのでそれに合わせた
}
vd der_sigmoidCrossEntropy(const vd &sigm_y, const vd &t){
	return mul(sub(sigm_y, t), 1.0/t.size()); //chainerは割っているのでそれに合わせた（上も同じ）
}
int countAccuracy(const vd &y, const vd &t){
	//y の正負と t の 0/1 が一致してる個数を返す
	assert(y.size()==t.size());
	int cnt = 0;
	for(int i=0;i<t.size();i++) if (y[i]<0 && nearlyEqual(t[i],0) || y[i]>=0 && nearlyEqual(t[i],1)) cnt++;
	return cnt;
}
vd softmax(const vd &x){
	vd ret(x.size());
	double maxi = maximum(x);
	double denom = 0.0;
	for(int i=0;i<x.size();i++) denom += exp(x[i] - maxi);
	for(int i=0;i<x.size();i++) ret[i] = exp(x[i] - maxi) / denom;
	return ret;
}
vd dot(const vvd &W, const vd &x){
	assert(W[0].size()==x.size());
	vd ret(W.size());
	for(int i=0;i<W.size();i++)for(int j=0;j<W[0].size();j++) ret[i] += W[i][j] * x[j];
	return ret;
}
vvd dot(const vvd &W1, const vvd &W2){
	assert(W1[0].size()==W2.size());
	vvd ret = zeros(W1.size(), W2[0].size());
	for(int i=0;i<ret.size();i++)for(int j=0;j<ret[0].size();j++){
		for(int k=0;k<W2.size();k++){
			ret[i][j] += W1[i][k] * W2[k][j];
		}
	}
	return ret;
}
vvd covdot(const vd &x, const vd &y){
	vvd ret = zeros(x.size(), y.size());
	for(int i=0;i<ret.size();i++)for(int j=0;j<ret[0].size();j++) ret[i][j] = x[i] * y[j];
	return ret;
}

double maxOverlap(const vvd &x){ // x の行ベクトル同士がなすオーバーラップの最大
	vvd cov = dot(x, transpose(x));
	double ret = 0.0;
	for(int i=0;i<cov.size();i++)for(int j=0;j<i;j++){
		ret = max(ret, fabs(cov[i][j]*cov[i][j]/cov[i][i]/cov[j][j]));
	}
	return sqrt(ret);
}
double minNorm(const vvd &x){ // x の行ベクトルのノルムの最小値
	double ret = DBL_MAX;
	for(int i=0;i<x.size();i++) ret = min(ret, snorm(x[i]));
	return sqrt(ret);
}


vd calculateSingularValues(const vvd &x){ //Eigen利用
	//Matrix<float, x.size(), x[0].size()> mat;
	MatrixXd mat(x.size(), x[0].size());
	for(int i=0;i<x.size();i++)for(int j=0;j<x[0].size();j++) mat(i, j) = x[i][j];
	JacobiSVD< MatrixXd > svd(mat);
	vd ret;
	for(int i=0;i<svd.singularValues().size();i++){
		ret.pb(svd.singularValues()[i]);
	}
	sort(ret.begin(), ret.end(), greater<double>());
	return ret;
}


string to_string(const tupleData &data){
	stringstream ss;
	for(int i=0;i<data.size();i++){
		vd x = get<0>(data[i]), y = get<1>(data[i]);
		ss << "(" << to_string(x) << ", " << to_string(y) << ")" << endl;
	}
	ss << "(size: " << data.size() << ")" << endl;
	return ss.str();
}

int speciesToLabel(string s){
	if (s=="setosa") return 0;
	else if (s=="versicolor") return 1;
	return 2;
}
tupleData loadIrisData(const vvi &labelList){
	//例えば、labelList = {{1}, {2}}の時は、setosa は無視され、versicolor は[0]に、virginicaは[1]になります。
	//labelList = {{0, 1}, {2}}の時は、setosa と versicolor は[0]に、virginicaは[1]になります。
	tupleData ret;
	FILE *in = fopen("iris.txt", "r");
	char s[32];
	for(int i=0;i<5;i++) fscanf(in,"%s",s);
	for(int i=0;i<150;i++){
		vd x(4), y(1);
		for(int j=0;j<4;j++) fscanf(in,"%lf",&x[j]);
		fscanf(in,"%s",s);
		string str(s);
		int label = speciesToLabel(str);
		bool flg = false;
		for(int j=0;j<labelList.size();j++){
			if (find(labelList[j].begin(), labelList[j].end(), label) != labelList[j].end()){
				y[0] = j; flg=true; break;
			}
		}
		if (!flg) continue;
		ret.pb(mt(x, y));
	}
	fclose(in);
	return ret;
}

tupleData loadMnistData(const vvi &labelList){
	//例えば、labelList = {{5}, {8}}の時は、5,8以外のラベルをもつデータは無視され、5 は[0]に、8 は[1]になります。
	tupleData ret;
	FILE *in = fopen("mnist.txt", "r");
	for(int i=0;i<60000;i++){
		vd x(784), y(1);
		int label;
		for(int j=0;j<784;j++) fscanf(in,"%lf",&x[j]);
		fscanf(in,"%ld",&label);
		bool flg = false;
		for(int j=0;j<labelList.size();j++){
			if (find(labelList[j].begin(), labelList[j].end(), label) != labelList[j].end()){
				y[0] = j; flg=true; break;
			}
		}
		if (!flg) continue;
		ret.pb(mt(x, y));
	}
	fclose(in);
	return ret;
}

tupleData loadGaussData(){ //ラベルは空です サイズ 60000 のガウしあんデータをランダムに生成して返す(loadMnistDataに揃えてる)
	tupleData ret;
	for(int i=0;i<60000;i++){
		vd x = randn(784), y;
		ret.pb(mt(x, y));
	}
	return ret;
}


tuple<tupleData, tupleData> splitData(const tupleData &data, double ratio){
	tupleData data1, data2;
	vvi idx(3);
	for(int i=0;i<data.size();i++){
		assert(get<1>(data[i]).size()==1);
		idx[(int)(get<1>(data[i])[0])].pb(i); //y[0]をインデックスと思うので注意！
	}
	for(int i=0;i<3;i++){
		random_shuffle(idx[i].begin(), idx[i].end());
		for(int j=0;j<idx[i].size();j++){
			if (j < idx[i].size() * ratio){
				data1.pb(data[idx[i][j]]);
			}else{
				data2.pb(data[idx[i][j]]);
			}
		}
	}
	return mt(data1, data2);
}

tupleData onehottify(const tupleData &data, int numLabel){
	//yをonehottifyする。 例：[1] -> [0, 1, 0] (numLabel==3の場合)
	tupleData ret;
	for(int i=0;i<data.size();i++){
		vd x = get<0>(data[i]), y = get<1>(data[i]);
		assert(y.size()==1);
		int y0 = (int)(y[0] + 0.5);
		assert(0<=y0 && y0<numLabel);
		vd newY(numLabel);
		newY[y0] = 1.0;
		ret.pb(mt(x, newY));
	}
	return ret;
}




class MLP{
public:
	int N, K, O;
	vvd W1, W2;
	vd b1, b2;
	vd x, z, h, y;
	vd lossfrac, accfrac;
	double learningRate = 0.001;
	int updateCounter;
	MLP(int N, int K, int O){
		this->N = N;
		this->K = K;
		this->O = O;
		W1 = zeros(K, N); for(int i=0;i<K;i++)for(int j=0;j<N;j++) W1[i][j] = randn() / sqrt(N);
		W2 = zeros(O, K); for(int i=0;i<O;i++)for(int j=0;j<K;j++) W2[i][j] = randn() / sqrt(K);
		b1 = vd(K);
		b2 = vd(O);
		lossfrac = vd(2);
		accfrac = vd(2);
	}
	vd forward(const vd &x){
		z = add(b1, dot(W1, x));
		h = sigmoid(z);
		return add(b2, dot(W2, h));
	}
	void update(const tupleDatum &datum, string lossType){
		vd x = get<0>(datum), t = get<1>(datum);
		vd y = forward(x);
		//yとtのシグモイドクロスエントロピーの、各変数に関する勾配を計算する
		vd gy, gh, gb1, gb2;
		vvd gW1, gW2;

		if (lossType=="sigmoidCrossEntropy"){
			vd sigm_y = sigmoid(y);
			double loss = crossEntropy(sigm_y, t);
			lossfrac[0] += loss; lossfrac[1] += 1.0;
			int accCnt = countAccuracy(y, t);
			accfrac[0] += accCnt; accfrac[1] += t.size();
			//誤差逆伝播
			//cout << sigm_y.size() << " " << t.size() << endl;
			gy = der_sigmoidCrossEntropy(sigm_y, t);
		}else if (lossType=="squared"){
			double loss = 0.5 * sdist(y, t); //0.5 * ||y - t||^2
			lossfrac[0] += loss; lossfrac[1] += 1.0;
			gy = sub(y, t);
		}else{
			cerr << "Invalid lossType." << endl; exit(0);
		}

		gb2 = gy;
		gW2 = covdot(gy, h);
		gh = dot(transpose(W2), gy);
		gb1 = mul(gh, der_sigmoid(h)); //アダマールせき
		gW1 = covdot(gb1, x);

		/*if (updateCounter<100){ 
			cout << to_string(x) << "  " << to_string(h) << "  " << to_string(y) << "  " << to_string(t) << endl;
			cout << to_string(W1) << endl;
			cout << to_string(W2) << endl;

			cout << to_string(gW1) << endl;
			cout << to_string(gW2) << endl;
		}else if (updateCounter==100) exit(0);*/

		//アップデート(SGD)
		W1 = add(W1, mul(gW1, -learningRate));
		W2 = add(W2, mul(gW2, -learningRate));
		b1 = add(b1, mul(gb1, -learningRate));
		b2 = add(b2, mul(gb2, -learningRate));
		updateCounter++;
	}
	double aveloss(bool clearFlag){
		if (lossfrac[1]==0) return -1.0;
		double ret = lossfrac[0] / lossfrac[1];
		if (clearFlag) lossfrac = vd(2);
		return ret;
	}
	double aveacc(bool clearFlag){
		if (accfrac[1]==0) return -1.0;
		double ret = accfrac[0] / accfrac[1];
		if (clearFlag) accfrac = vd(2);
		return ret;
	}
	double calculateZMaxOverlap(const tupleData &data){
		vvd zs = zeros(K, data.size());
		for(int j=0;j<data.size();j++){
			x = get<0>(data[j]);
			z = add(b1, dot(W1, x));
			for(int i=0;i<K;i++){
				zs[i][j] = z[i];
			}
		}
		return maxOverlap(zs);
	}
	tuple<double, double, double, double, vd> calculateHStats(const tupleData &data){
		vvd hs = zeros(K, data.size());
		for(int j=0;j<data.size();j++){
			x = get<0>(data[j]);
			h = sigmoid(add(b1, dot(W1, x)));
			for(int i=0;i<K;i++){
				hs[i][j] = h[i];
			}
		}
		vvd hsWithOnes = zeros(K+1, data.size());
		for(int i=0;i<K;i++)for(int j=0;j<data.size();j++) hsWithOnes[i][j] = hs[i][j];
		for(int j=0;j<data.size();j++) hsWithOnes[K][j] = 1.0;

		vd hsWithOnesSings = calculateSingularValues(hsWithOnes);
		return mt(maxOverlap(hs), maxOverlap(hsWithOnes), minNorm(hs), minNorm(sub(hs, ones(K, data.size()))), hsWithOnesSings);
	}
};

tupleData passTeacher(const tupleData &data, MLP &teacher){
	tupleData ret;
	for(int i=0;i<data.size();i++){
		vd x = get<0>(data[i]);
		ret.pb(mt(x, teacher.forward(x)));
	}
	return ret;
}



void routine(string filename){
	srand(time(NULL));
	int N=784, K=2, O=1;

	/*vvi labelList(2);
	labelList[0] = vi{0, 1, 2, 3, 4};
	labelList[1] = vi{5, 6, 7, 8, 9};*/

	tupleData gauss = loadGaussData();
	//TODO: random_shuffle は srand に依存するかは実は実装によるらしく、注意！ 
	//shuffleに乱数生成器を渡すようにすべきか。
	//tupleData irisTrain = get<0>(splitData(iris, 0.6));
	tupleData gaussTrain = gauss;
	
	//mnistTrain = onehottify(mnistTrain, labelList.size());
	MLP mlp_t(N, K, O);
	gaussTrain = passTeacher(gaussTrain, mlp_t);
		tuple<double, double, double, double, vd> HStats = mlp_t.calculateHStats(gaussTrain);
		vd HOSV = get<4>(HStats);
		cerr << "teacher HOSV: " << HOSV[0] << " " << HOSV[1] << " " << HOSV[2] << endl;


	cerr << "gaussTrain.size() == " << gaussTrain.size() << endl;
	/*cerr << to_string(get<0>(gaussTrain[0])) << endl;
	cerr << to_string(get<1>(gaussTrain[0])) << endl;*/

	MLP mlp(N, K, O);
	FILE *out = fopen(filename.c_str(), "w");
	fprintf(out, "epoch loss acc W1MaxOvl W1LogNormRat ZMaxOvl HMaxOvl HOnesMaxOvl HMinNorm HOSV0 HOSV1 HOSV2 W2MinNorm\n");
	fprintf(out, "%d %g %g %g %g %g %g %g %g %g %g %g %g\n", -1, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, HOSV[0], HOSV[1], HOSV[2], -1.0);

	for(int epoch=0;epoch<1000;epoch++){
		random_shuffle(gaussTrain.begin(), gaussTrain.end());
		for(int i=0;i<gaussTrain.size();i++){ //オンライン学習
			mlp.update(gaussTrain[i], "squared");
		}
		if (epoch % 1==0){
			double loss, acc, W1MaxOvl, W1LogNormRat, ZMaxOvl, W2MinNorm;
			tuple<double, double, double, double, vd> HStats;
			
			loss = mlp.aveloss(true);
			acc = mlp.aveacc(true);
			W1MaxOvl = maxOverlap(mlp.W1);
			W1LogNormRat = fabs(log(norm(mlp.W1[0]))-log(norm(mlp.W1[1]))); assert(mlp.W1.size()==2);
			ZMaxOvl = mlp.calculateZMaxOverlap(gaussTrain);
			HStats = mlp.calculateHStats(gaussTrain);
			double HMaxOvl = get<0>(HStats), HOnesMaxOvl = get<1>(HStats), HMinNorm = get<2>(HStats), HMinDistFromOnes = get<3>(HStats);
			vd HOSV = get<4>(HStats);
			W2MinNorm = minNorm(transpose(mlp.W2));

			assert(K==2); //特異値を3個目まで表示する関係で (K + 1(バイアスベクトル))
			fprintf(out, "%d %g %g %g %g %g %g %g %g %g %g %g %g\n", epoch, loss, acc, W1MaxOvl, W1LogNormRat, ZMaxOvl, HMaxOvl, HOnesMaxOvl, HMinNorm, HOSV[0], HOSV[1], HOSV[2], W2MinNorm);
			if (epoch % 10==0){
				cout << "epoch: " << epoch << " loss: " << loss << " acc: " << acc
				<< " W1MaxOvl: " << W1MaxOvl
				<< " W1LogNormRat: " << W1LogNormRat				
				<< " ZMaxOvl: " << ZMaxOvl
				<< " HMaxOvl: " << HMaxOvl
				<< " HOnesMaxOvl: " << HOnesMaxOvl
				<< " HMinNorm: " << HMinNorm
				<< " HOSV[0]: " << HOSV[0]
				<< " HOSV[1]: " << HOSV[1]
				<< " HOSV[2]: " << HOSV[2]
				<< " W2MinNorm: " << W2MinNorm <<  endl;
			}
			//if (epoch>=1000000 && loss < 0.03) break; //終了条件は適宜変更すること
		}
	}
	fclose(out);
	return;
}

int main(void){
	for(int i=0;i<20;i++){
		stringstream ss; ss << "out" << i << ".txt";
		routine(ss.str());
	}
	return 0;
}