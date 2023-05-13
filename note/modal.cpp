//#pragma GCC optimize(3,"Ofast","inline")
//#pragma GCC optimize(2)



//int h[N], e[M], ne[M], idx;
// void add(int a,int b){
// 	e[idx] = b;ne[idx] = h[a];h[a] = idx++;
// }

//int h[N], e[M], w[M], ne[M], idx;
//void add(int a,int b,int w){
//	e[idx] = b;w[idx] = w;ne[idx] = h[a];h[a] = idx++;
//}

//lambda
//function<int(int)> f = [&f](int n)

/*最大公约数
int gcd(int a,int b){
    return a%b==0?b:gcd(b,a%b);
}
*/

/*
int p[N];//开
int find(int u){//尽量都抽象为找同集的位置，用find(u)代替u
	if(p[u] != u){
		p[u] = find(p[u]);
	}
	return p[u];
}
*/

/*
    find = [&](int x){
        if(p[x] == x) return x;
        int zx = find(p[x]);
        w[x] += w[p[x]];
        return p[x] = zx;
    };
    auto unite = [&](int a, int b, int c){//将a加到b上，a->b = c
        int pa = find(a);
        w[pa] = - w[a] + c + w[b];
        p[pa] = find(b);
    };
*/

/*
LL qmi(LL a,int b)//快速幂在于思想
{
    LL res=1;
    while(b)//对b进行二进制化,从低位到高位
    {
        if(b&1) res = res *a % mod;
        b >>= 1;
        a = a * a % mod;
    }
    return res;
}
//p0 + .. + pk-1
int sum(int p, int k)
{
    if (k == 1) return 1;
    if (k % 2 == 0) return (1 + qmi(p, k / 2)) * sum(p, k / 2) % mod;
    return (sum(p, k - 1) + qmi(p, k - 1)) % mod;
}

*/

/*
long long C(int n,int m){//n在100以内，m要选较小的一边
    long long res = 1;
    for(int i = 0; i < m; i++) res *= n - i, res /= i + 1;
    return res;
}
*/

/*
int lowbit(int x)  // 返回末尾的1
{return x & -x;}
*/

/*
const int N= 1000010;

int primes[N], cnt;
bool st[N];

void get_primes(int n)
{
    for (int i = 2; i <= n; i ++ )
    {
        if (st[i]) continue;
        primes[cnt ++ ] = i;
        for (int j = i + i; j <= n; j += i)
            st[j] = true;
    }
}
*/
/*
#include <iostream>
#include <cstring>
#include <algorithm>
using namespace std;
typedef long long LL;

const LL N = 1e5+10;
LL a[N],b[N];
LL n,m;

LL lowbit(LL x)  // 返回末尾的1
{
    return x & -x;
}

void add(LL x,LL w){
    for(LL i = x;i <= n;i += lowbit(i)){ //同时维护两个树状数组
        a[i] += w;
        b[i] += x*w;
    }
}

LL query(LL u){
    LL res = 0;
    for(LL i = u;i > 0;i -= lowbit(i)){ //优化的区间查询方案
        res += (u+1) * a[i] - b[i];     //计算公式
    }
    return res;
}


    LL a,b;cin >> a >> b;
    //cout << query(b) - query(a-1) << endl;
    //LL a,b,c;cin >> a >> b >> c;
    //add(a, c);
    //add(b+1, -c);


*/


/*
//qmi(a, p - 2, p)
//费马小定理只能在m是质数是用
//b 存在乘法逆元的充要条件是 b 与模数 m 互质
//若m不一定是质数，要用欧几里得
LL qmi(int a, int b, int p)
{
    LL res = 1;
    while(b){
        if(b & 1) res = res * a % p;
        a = (LL)a * a % p;
        b >>= 1;
    }
    return res;
}
*/
/*

const int N = 1e5 + 10, mod = 1e9 + 7;
int fact[N], infact[N];
//快速幂求逆元
int qmi(int a, int b, int m){
    int ans = 1;
    while (b){
        if(b & 1)
            ans = (LL)a * ans % m;
        a = (LL)a * a % m;
        b >>= 1;
    }
    return ans;
}
void get_fact_infact(){
    fact[0] = 1; //跟总体计算有关，不影响计算的
    infact[0] = 1;
    for (int i = 1; i < N; i ++ ){
        fact[i] = (LL)fact[i - 1] * i % mod;
        infact[i] = (LL)infact[i - 1] * qmi(i, mod - 2, mod) % mod; // i的模mod的乘法逆元
    }
}

LL C(int n,int m){
    return (LL)fact[m] * infact[n] % mod * infact[n - m] % mod;
}
*/

//最大子段板子
/*
//线段树分治
struct node{
	int l,r;
	int sum,ms;//maxsum
    int ml,mr;//maxl,maxr
}tree[N*4];


void PushUp(int i)
{
	tree[i].sum=tree[i<<1].sum+tree[i<<1|1].sum;
	tree[i].ml=max(tree[i<<1].sum+tree[i<<1|1].ml,tree[i<<1].ml);
	tree[i].mr=max(tree[i<<1|1].sum+tree[i<<1].mr,tree[i<<1|1].mr);
	tree[i].ms=max(max(tree[i<<1].ms,tree[i<<1|1].ms),tree[i<<1].mr+tree[i<<1|1].ml);
}
void build(int i,int l,int r)
{
	tree[i].l=l,tree[i].r=r;
	if(l==r)
	{
		tree[i].sum=tree[i].ml=tree[i].mr=tree[i].ms=a[l];
		return ;
	}
	int mid=(l+r)>>1;
	build(i<<1,l,mid);
	build(i<<1|1,mid+1,r);
	PushUp(i);
}
void update(int i,int pos,int val)
{
	if(tree[i].l==tree[i].r)
	{
		tree[i].ms=tree[i].ml=tree[i].mr=tree[i].sum=val;
		return ;
	}
	int mid=(tree[i].l+tree[i].r)>>1;
	if(pos<=mid)
		update(i<<1,pos,val);
	else update(i<<1|1,pos,val);
	PushUp(i);
}
node query(int i,int l,int r)
{
	if(l<=tree[i].l&&tree[i].r<=r)
		return tree[i];
	int mid=(tree[i].l+tree[i].r)>>1;
	if(r<=mid) return query(i<<1,l,r);
	else if(l>mid) return query(i<<1|1,l,r);
	else
	{
		node x=query(i<<1,l,r),y=query(i<<1|1,l,r),res;
		//合并答案 
		res.sum=x.sum+y.sum;
		res.ml=max(x.sum+y.ml,x.ml);
		res.mr=max(y.sum+x.mr,y.mr);
		res.ms=max(max(x.ms,y.ms),x.mr+y.ml);
		return res;
	}
}
*/
/*
//b[j] = max{b[j - 1] + a[j],a[j]}, 1 <= j <= n
int max_subsegment(int a[],int n){
	int result = 0,b = 0;
	int begin = 0,end = 0;//记录最大子段的起始，终止下标 
	for(int i = 0; i < n;i++){
		if(b > 0){
			b += a[i];
		}else{
			b = a[i];
			begin = i;
		}
		if(b > result){
			result = b;	
			end = i;
		}
	}
	
	return result;
}
*/


/*
int a[N];
int lowbit(int x)  // 返回末尾的1
{return x & -x;}

void add(int x,int w){
    for(int i = x;i < N;i += lowbit(i)){
        a[i] += w;
    }
}

int query(int u){
    int res = 0;
    for(int i = u;i > 0;i -= lowbit(i)){
        res += a[i];
    }
    return res;
}

*/
/*
快速排序板子
int a[N];

void quick_sort(int l, int r)
{
    if (l >= r) return;

    int i = l - 1, j = r + 1, x = a[l + r >> 1];
    while (i < j)
    {
        do i ++ ; while (a[i] < x);
        do j -- ; while (a[j] > x);
        if (i < j) swap(a[i], a[j]);
    }

    quick_sort(l, j);
    quick_sort(j + 1, r);
}
*/

/*
归并排序
int a[N], tmp[N];

void merge_sort(int q[], int l, int r)
{
    if (l >= r) return;

    int mid = l + r >> 1;

    merge_sort(q, l, mid), merge_sort(q, mid + 1, r);

    int k = 0, i = l, j = mid + 1;
    while (i <= mid && j <= r)
        if (q[i] <= q[j]) tmp[k ++ ] = q[i ++ ];
        else tmp[k ++ ] = q[j ++ ];
    while (i <= mid) tmp[k ++ ] = q[i ++ ];
    while (j <= r) tmp[k ++ ] = q[j ++ ];

    for (i = l, j = 0; i <= r; i ++, j ++ ) q[i] = tmp[j];
}

*/
/*
//kmp，找字串
#include <iostream>
using namespace std;
const int N = 1e6 + 10;
char s[N],p[N];
int pre[N];                             
int main(){
    int n,m;
    cin >> n >> (p + 1) >> m >> (s + 1);
                                        //把前缀和后缀相等的下标存入ne
                                        //理解ne数组
                                        //变短到可以再伸长
    for(int i = 2,j = 0;i <= n;i ++)
    {
        while(j && p[j+1] != p[i])j = pre[j];
        if(p[j+1] == p[i])j++;
        pre[i] = j;
    }
    //kmp
    for(int i = 1,j = 0;i <= m;i ++)
    {
        while(j && p[j+1] != s[i])j = pre[j];
        if(p[j+1] == s[i])j++;
        if(j == n)
        {
            cout << i - n << " ";
        }
    }
}

*/

/*
trie字符串树

const int N = 100010;

int son[N][26], cnt[N], idx;
char str[N];

void insert(char *str)//用数组模拟这个26叉树
                    //功能为往集合中插入一个字符串
{
    int p = 0;
    for (int i = 0; str[i]; i ++ )
    {
        int u = str[i] - 'a';
        if (!son[p][u]) son[p][u] = ++ idx;
        p = son[p][u];
    }
    cnt[p] ++ ;//p没有重复，见idx
}

int query(char *str)//在集合中查找一个字符串出现的次数
{
    int p = 0;
    for (int i = 0; str[i]; i ++ )
    {
        int u = str[i] - 'a';
        if (!son[p][u]) return 0;
        p = son[p][u];
    }
    return cnt[p];
}

*/
/*
//字典树的按位运用
//找出数组中任意两个异或结果最大的
const int N = 100010, M = 3100010;//M接近N的30倍，n最大是N，N个数字每个存30位idx

int n;
int a[N], son[M][2], idx;

void insert(int x)                //按照二进制安排树
{
    int p = 0;
    for (int i = 30; i >= 0; i -- )
    {
        int &s = son[p][x >> i & 1];//元素从首位开始各位
        if (!s) s = ++ idx;        //s“引用”son[p][x >> i & 1] ,s改变了，节点也跟着改变
        p = s;
    }
}

int search(int x)
{
    int p = 0, res = 0;
    for (int i = 30; i >= 0; i -- )//从最高位开始，找son[p][!s]是否存在（这样让最高位的异或结果是1最大）
    {                              //如此往复直到最后一位
        int s = x >> i & 1;
        if (son[p][!s])
        {
            res += 1 << i;         //成功找到，res加上对应的数
            p = son[p][!s];        //下一层
        }
        else p = son[p][s];        //走原来的下一层
    }
    return res;
}
*/

/*
//字符串hash，普通的自然溢出（容易被刻意卡）
typedef unsigned long long ULL;
const int N = 1e5+5,P = 131;//131 13331
ULL h[N],p[N];
// h[i]前i个字符的hash值
// 字符串变成一个p进制数字，体现了字符+顺序，需要确保不同的字符串对应不同的数字
// P = 131 或  13331 Q=2^64，在99%的情况下不会出现冲突
ULL query(int l,int r){
    return h[r] - h[l-1]*p[r-l+1];
}
int main()
{
    int n,m;scanf("%d %d",&n,&m);
    char *str = new char[n]; scanf("%s",str + 1);
    p[0] = 1;
    for (int i = 1; i <= n; i ++ ){
        h[i] = h[i - 1] * P + str[i];
        p[i] = p[i - 1] * P;
    }
    while (m -- ){
        int l1,r1,l2,r2;
        cin >> l1 >> r1 >> l2 >> r2;
        if(query(l1,r1) == query(l2,r2)) cout << "Yes" << endl;
        else cout << "No" << endl;
    }
}
*/

/*
//字符串hash，普通的自然溢出（容易被刻意卡）
typedef unsigned long long LL;
const int N = 1e5+5,P1 = 131,P2 = 13331;//131 13331
LL h1[N],p1[N];
LL h2[N],p2[N];
// LL mod1 = 1e9+7,mod2 = 1e9 + 9;


LL qu(LL l,LL r,LL h[],LL p[]){
    return h[r] - h[l-1]*p[r-l+1];
}

pair <LL, LL> query(int l,int r){
    return make_pair(qu(l,r,h1,p1),qu(l,r,h2,p2));
}

int main()
{
    int n,m;scanf("%d %d",&n,&m);
    char *str = new char[n+1]; scanf("%s",str + 1);
    p1[0] = p2[0] = 1;
    for (int i = 1; i <= n; i ++ ){
        h1[i] = h1[i - 1] * P1 + str[i];
        p1[i] = p1[i - 1] * P1;
        h2[i] = h2[i - 1] * P2 + str[i];
        p2[i] = p2[i - 1] * P2;
    }
    while (m -- ){
        int l1,r1,l2,r2;
        cin >> l1 >> r1 >> l2 >> r2;
        //cout << query(l1,r1).first << " " << query(l1,r1).second << endl;
        //cout << query(l2,r2).first << " " << query(l2,r2).second << endl;
        if(query(l1,r1) == query(l2,r2)) cout << "Yes" << endl;
        else cout << "No" << endl;
    }
}
*/

/*
拓扑顺序
const int N = 100010;

int n, m;
int h[N], e[N], ne[N], idx;
int d[N];
int q[N];

void add(int a, int b)//邻接表
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

bool topsort()
{
    int hh = 0, tt = -1;//模拟队列

    for (int i = 1; i <= n; i ++ )//入队  入度为0的下标
        if (!d[i])
            q[ ++ tt] = i;

    while (hh <= tt)
    {
        int t = q[hh ++ ];//出队

        for (int i = h[t]; i != -1; i = ne[i])//按链上走
        {
            int j = e[i];//判断下一节点在入度减一后是否要入队
            if (-- d[j] == 0)
                q[ ++ tt] = j;
        }
    }

    return tt == n - 1;//有向无环图全部入队，tt == n - 1
}

int main()
{
    scanf("%d%d", &n, &m);

    memset(h, -1, sizeof h);

    for (int i = 0; i < m; i ++ )
    {
        int a, b;
        scanf("%d%d", &a, &b);
        add(a, b);

        d[b] ++ ;
    }

    if (!topsort()) puts("-1");
    else
    {
        for (int i = 0; i < n; i ++ ) printf("%d ", q[i]);
        puts("");
    }

    return 0;
}
*/
/*
dijkstra邻接表版
const int N = 510;

int n, m;
int g[N][N];
int dist[N];
bool st[N];

int dijkstra()  // 求1号点到n号点的最短路距离，如果从1号点无法走到n号点则返回-1
{
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;
    for (int i = 0; i < n - 1; i ++ )//n次
    {
        int t = -1;
        for (int j = 1; j <= n; j ++ )
            if (!st[j] && (t == -1 || dist[t] > dist[j]))
                t = j;//当前的最短边

        for (int j = 1; j <= n; j ++ )
            dist[j] = min(dist[j], dist[t] + g[t][j]);//更新最短距离

        st[t] = true;
    }

    if (dist[n] == 0x3f3f3f3f) return -1;
    return dist[n];
}


int main()
{
    scanf("%d%d", &n, &m);

    memset(g, 0x3f, sizeof g);
    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);

        g[a][b] = min(g[a][b], c);//邻接矩阵a到b
    }

    printf("%d\n", dijkstra());
    //cout << 1061109567  - 0x3f3f3f3f;

    return 0;
}
*/

/*
dijkstra堆优化版

const int N = 150010;
typedef pair<int, int> PII;
int n, m;
int e[N],w[N],ne[N],h[N],idx = 0;
int dist[N];
bool st[N];
void add(int a, int b, int c)  // 添加一条边a->b，边权为c
{
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

int dijkstra()  // 求1号点到n号点的最短路距离，如果从1号点无法走到n号点则返回-1
{
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;
    priority_queue<PII,vector<PII>,greater<PII> > heap;//小根堆pair<权，点>,按权排序
    heap.push({0,1});
    while(!heap.empty()){
        auto t = heap.top();
        heap.pop();
        
        int ww = t.first,ee = t.second;    //pair<权，点>
        if(st[ee]) continue;
        st[ee] = true;
        
        for(int i = h[ee];i != -1;i = ne[i]){
            int j = e[i];
            if (dist[j] > dist[ee] + w[i])
            {
                dist[j] = dist[ee] + w[i];
                heap.push({dist[j], j});
            }
        }
    }
    if (dist[n] == 0x3f3f3f3f) return -1;
    return dist[n];
}


int main()
{
    scanf("%d%d", &n, &m);

    memset(h, -1, sizeof h);
    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);

        add(a, b, c);
    }

    printf("%d\n", dijkstra());
    //cout << 1061109567  - 0x3f3f3f3f;

    return 0;
}
*/

/*
bellman-ford有负权边的最短路

const int N = 10010;
struct node{
    int a,b,c;
}nodes[N];
int n,m,k;
int dist[510],cpy[520];

void bellman_ford(){
    
    memset(dist,0x3f,sizeof dist);
    dist[1] = 0;
    for(int i = 0;i < k;i++){                             //只走不超过k条边
        memcpy(cpy,dist,sizeof dist);                     //防止串联
        for(int j = 0;j < m;j++){                         //按上一步的结果来更新下一步的最短步数
            struct node xx = nodes[j];                    
            dist[xx.b] = min(dist[xx.b],cpy[xx.a] + xx.c);
        }
    }
}

int main()
{
    cin >> n >> m >> k;
    for (int i = 0; i < m; i ++ )
    cin >> nodes[i].a >> nodes[i].b >> nodes[i].c;
    
    bellman_ford();
    
    if(dist[n] > 0x3f3f3f3f/2)cout << "impossible";          //500 * 10000全是负的也不会超过这个1e9/2
    else cout << dist[n];
}
*/

/*
spfa求最短路，边权可以为负数

const int N = 150010;
typedef pair<int, int> PII;
int n, m;
int e[N],w[N],ne[N],h[N],idx = 0;
int dist[N];
bool st[N];
void add(int a, int b, int c)  // 添加一条边a->b，边权为c
{
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

int spfa()  // 求1号点到n号点的最短路距离，如果从1号点无法走到n号点则返回-1
{
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;
    queue<int> dd;
    dd.push(1);
    while(!dd.empty()){
        int k = dd.front();
        dd.pop();
        st[k] = false;
        for(int i = h[k];i != -1;i = ne[i]){
            int point = e[i];
            if(dist[point] > dist[k] + w[i]){
                dist[point] = dist[k] + w[i];
                if(!st[point]){
                    dd.push(point);
                    st[point] = true;
                }
            }
        }
    }
    return dist[n];
}


int main()
{
    scanf("%d%d", &n, &m);

    memset(h, -1, sizeof h);
    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);

        add(a, b, c);
    }

    int ans = spfa();
    if(ans == 0x3f3f3f3f) cout << "impossible";
    else cout << ans;
    //cout << 1061109567  - 0x3f3f3f3f;

    return 0;
}
*/

/*
spfa判断是否存在负权环
即图中存在一个环，一直绕着他走，加权只会越来越小

const int N = 150010;
typedef pair<int, int> PII;
int n, m;
int e[N],w[N],ne[N],h[N],idx = 0;
int dist[N];
bool st[N];
void add(int a, int b, int c)  // 添加一条边a->b，边权为c
{
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

int spfa()  // 求1号点到n号点的最短路距离，如果从1号点无法走到n号点则返回-1
{
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;
    queue<int> dd;
    dd.push(1);
    while(!dd.empty()){
        int k = dd.front();
        dd.pop();
        st[k] = false;
        for(int i = h[k];i != -1;i = ne[i]){
            int point = e[i];
            if(dist[point] > dist[k] + w[i]){
                dist[point] = dist[k] + w[i];
                if(!st[point]){
                    dd.push(point);
                    st[point] = true;
                }
            }
        }
    }
    return dist[n];
}


int main()
{
    scanf("%d%d", &n, &m);

    memset(h, -1, sizeof h);
    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);

        add(a, b, c);
    }

    int ans = spfa();
    if(ans == 0x3f3f3f3f) cout << "impossible";
    else cout << ans;
    //cout << 1061109567  - 0x3f3f3f3f;

    return 0;
}
*/

/*
Floyd求多源最短路，边权可以为负数
图中不存在负权回路

const int N = 210;
int d[N][N];
int n,m,k,INF = 0x3f3f3f3f;

void floyd(){
    for(int k = 1;k <= n;k++){
        for (int i = 1; i <= n; i ++ ){
            for (int j = 1; j <= n; j ++ ){
                d[i][j] = min(d[i][j],d[i][k] + d[k][j]);//dp
            }
        }
    }
}
int main()
{
    cin >> n >> m >> k;
    for (int i = 1; i <= n; i ++ ){
        for(int j = 1; j <= n;j ++){
            if(i == j)d[i][j] = 0;
            else d[i][j] = INF;
        }
    }
    int x,y,z;
    for(int i = 1;i <= m;i++){
        
        cin >> x >> y >> z;
        d[x][y] = min(d[x][y],z);                          //保留最小的
    }
    floyd();
    for (int i = 1; i <= k; i ++ ){
        cin >> x >> y;
        if(d[x][y] >= INF-1e6)cout << "impossible" << endl;
        else cout << d[x][y] << endl;
    }
    
}

*/

/*
n  个顶点和 n−1 条边构成的无向连通子图被称为 G 的一棵生成树
其中边的权值之和最小的生成树被称为无向图 G 的最小生成树
Prim算法求最小生成树
int n,m;
int g[N][N];
int dist[N];
bool st[N];
//Dijkstra算法是更新到起始点的距离，Prim是更新到集合dist的距离
int prim()
{
    int res = 0;
    memset(dist,0x3f,sizeof(dist));
    dist[1] = 0;
    for(int i= 0;i < n;i ++)
    {
        int t = -1;
        for(int j = 1;j <= n;j++)
        {
            if(!st[j] && (t == -1 || dist[t] > dist[j]))t = j;
        }
        
        if(i && dist[t] == 0x3f3f3f3f)return 0x3f3f3f3f;
        
        if(i)res += dist[t];
        
        for(int j = 1;j <= n;j ++)
        {
            dist[j] = min(dist[j],g[t][j]);
        }
        st[t] = true;
    }
    return res;
}

int main()
{
    cin >> n >> m;
    memset(g,0x3f,sizeof(g));
    for(int i = 0;i < m;i++)
    {
        int u,v,w;
        cin >> u >> v >> w;
        g[u][v] = g[v][u] = min(g[u][v],w);
    }
    
    int ans = prim();
    ans == 0x3f3f3f3f?cout << "impossible" : cout << ans;
}
*/

/*
kruskal算法求最小生成树
pair<int,pair<int,int> > edge[200010];//{w,{u,v}}
int p[100010];

int find(int x)  // 并查集
{
    if (p[x] != x) p[x] = find(p[x]);
    return p[x];
}


int main()
{
    int n,m;cin >> n >> m;
    for (int i = 0; i < m; i ++ )
    cin >> edge[i].second.first >> edge[i].second.second >> edge[i].first;
    
    sort(edge,edge + m);
    
    for(int i = 1;i <= n;i++) p[i] = i;
    
    int ans = 0,cnt = 0;                
                                        //kruskal算法
                                        //因为线段权重已经排好了序，所以一定是边权较小的独立的点先联通
                                        //到达了最小生成树的目的
                                        //在此之后的重复访问点都在同一个并查集内，被无视
    for(int i = 0;i < m;i++)
    {
        int a = find(edge[i].second.first),b = find(edge[i].second.second);
        if(a != b)
        {
            p[a] = b;
            ans += edge[i].first;
            cnt++;
        }
    }
    
    if(cnt < n - 1)cout << "impossible";//所有点“联通”后，联通数是总点数n的n-1，以此来判断是否有连通图
    else cout << ans;
}
*/

/*
染色法判断二分图
/*
二分图：给定无向无权图，分成两部分，相邻的点不在同一个部分当中
该图不可能含有点总数为奇数的环
*/

/*
//st中的1  2表示两种颜色，初始时0没有颜色

const int N = 100010,M = 200010;
int h[N], e[M], ne[M], idx;
int st[N];

void add(int a, int b)  // 添加一条边a->b
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

bool dfs(int u,int color){
    st[u] = color;
    for(int i = h[u];i != -1;i = ne[i]){
        int j = e[i];
        if(!st[j] && !dfs(j,3 - color))return false;//相邻点之间的颜色肯定是不一样的
        else if(st[j] == color)return false;        //如果搜索到相邻点与当前点的颜色相同，退出
    }
    return true;
}

int main()
{
    int n,m;cin >> n >> m;
    
    memset(h, -1, sizeof h);//这句...
    
    for(int i = 0;i < m;i++){
        int u,v;
        scanf("%d%d", &u, &v);
        add(u,v);
        add(v,u);
    }
    bool flag = true;
    for (int i = 1; i <= n; i ++ ){
        if(!st[i] && !dfs(i,1)){                      //从任何一个节点开始涂颜色1，确保所有结点
            flag = false;                             //对他的邻点已涂色或者刚涂色上都搜索过，最终输出是否是二分图
            break;
        }
    }
    if(flag)puts("Yes");
    else puts("No");
}
*/

/*
const int N = 4e4+10,M = N*2;
int h[N], e[M], ne[M], idx;
int depth[N],f[N][16];
int root;

void add(int a, int b)  // 添加一条边a->b
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

void bfs(int u)
{
    memset(depth,0x3f,sizeof depth);
    queue<int> q;
    q.push(u);
    depth[u] = 1;depth[0] = 0;                          //设置0的深度的原因为，往上到祖先的过程很容易出现超出根节点的情况、
                                                        //比如lca（root,root）,显然会出现错误
    while(q.size())                                     //按照树的结构bfs
    {
        int t = q.front();q.pop();
        
        for(int i = h[t];i != -1;i = ne[i])
        {
            int j = e[i];                               //当前节点的子节点
            if(depth[j] > depth[t] + 1)                 //只往深度更深的搜
            {
                depth[j] = depth[t] + 1;
                f[j][0] = t;                            //子节点往上的2^0的祖先
                q.push(j);
                for(int k = 1;k <= 15;k ++)             //倍增思想（类似递推）：节点的2^k的祖先 = 节点的2^(k-1)的祖先的2^(k-1)的祖先
                {
                    f[j][k] = f[f[j][k-1]][k-1];
                }
            }
        }
    }
}

int lca(int a,int b)
{
    if(depth[a] < depth[b])swap(a,b);                   //方便对齐
    for(int k = 15;k >= 0;k --)                         //让a的深度和b相同
    {
        if(depth[f[a][k]] >= depth[b])
        {
            a = f[a][k];
        }
    }
    
    if(a == b)
    {
        return a;
    }
    
    for(int k = 15;k >= 0;k --)                         //往上倍减的第一个共同祖先就是lca
    {
        if(f[a][k] != f[b][k])
        {
            a = f[a][k];
            b = f[b][k];
        }
    }
    return f[a][0];
}


//bfs(root);这样开始从图在f数组中用bfs建立关系


/*
关于树上差分
从根节点到叶子节点同一条路路径上的点是没有难度的
关键就是不同路径上的子节点怎么考虑

情况1：点差分（按照点来差分，边界是点，差分最后累加目的也是看点）
        power[a]++;
        power[b]++;
        power[lcaa]--;
        power[f[lcaa][0]]--;
情况2：线差分（解释如上）
        power[a]++;
        power[b]++;
        power[lcaa]--;

处理前需要用lca模板（构造出倍增数组）
然后再去考虑怎么差分
最后通过一个dfs从子节点回溯，计算累加和
*/
/*
马拉车算法O（N）找最长回文串
string Mannacher(string s)
{
    //插入"#"
    string t="$#";
    for(int i=0;i<s.size();++i)
    {
        t+=s[i];
        t+="#";
    }
    
    vector<int> p(t.size(),0);
    //mx表示某个回文串延伸在最右端半径的下标，id表示这个回文子串最中间位置下标
    //resLen表示对应在s中的最大子回文串的半径，resCenter表示最大子回文串的中间位置
    int mx=0,id=0,resLen=0,resCenter=0;

     //建立p数组
    for(int i=1;i<t.size();++i)
    {
        p[i]=mx>i?min(p[2*id-i],mx-i):1;

        //遇到三种特殊的情况，需要利用中心扩展法
        while(t[i+p[i]]==t[i-p[i]])++p[i];

        //半径下标i+p[i]超过边界mx，需要更新
        if(mx<i+p[i]){
            mx=i+p[i];
            id=i;
        }

        //更新最大回文子串的信息，半径及中间位置
        if(resLen<p[i]){
            resLen=p[i];
            resCenter=i;
        }
        //全部回文串在原位置对应
        //if(p[i]-1)cout << s.substr((i-p[i])/2,p[i]-1) << endl;
    }

    //最长回文子串长度为半径-1，起始位置为中间位置减去半径再除以2
    return s.substr((resCenter-resLen)/2,resLen-1);
}
*/

/*
//中心拓展算法
void countSubstrings(string s,int kk) {
 
        int n = s.size(), ans = 0;
        for (int k = 0; k < 2 * n - 1; ++k)
        {
            int i = k / 2, j = k / 2 + k % 2;
 
            //满足边界条件，且s[i] == s[j]，则代表一个新的回文字符串的诞生，否则，跳出循环
            while ( i >= 0 && j < n && s[i] == s[j] )
            {
                if(j - i + 1 >= kk)pa[ed++] = {i,j};
                --i;
                ++j;
            }
        }
    }
*/

/*

// 匈牙利算法，二分图的最大匹配
// 最坏O（n^2）
const int N = 100010;
int h[510], e[N], ne[N], idx;
bool st[N];     //每次走的状态
int match[N];   //女生的男友

void add(int a,int b){
    e[idx] = b;ne[idx] = h[a];h[a] = idx++;
}

bool find(int man){
    for(int i = h[man];i != -1;i = ne[i]){
        int j = e[i];
        if(!st[j]){
            st[j] = true;
            if(!match[j] || find(match[j])){//女生没有男友 或者 找这女生匹配男生的是否还有选择
                match[j] = man;
                return true;
            }
            
        }
    }
    return false;
}

int main()
{
    memset(h, -1, sizeof h);
    int n1,n2,m;cin >> n1 >> n2 >> m;
    for(int i = 0;i < m;i++){
        int a,b;
        cin >> a >> b;
        add(a, b);
    }
    
    int ans = 0;
    for(int i = 1;i <= n1;i++){
        memset(st, 0, sizeof st);
                                    // 
                                    // 每次归零：如果不归零，前人看上的妹子，后人再选就会发现不能选
                                    // 所以重置后，前人看上的妹子后人再选，进入递归，假设这一次st标记前一个妹子被第二个人选择
                                    // 前人看不了原配，找其他的选择，如果找到了，返回成功
                                    // 
        if(find(i))ans++;
    }
    cout << ans;
}
*/

//dp
/*
01背包
onst int N = 1010;
int v[N];
int w[N];
int dp[N];
//递推前i个物品，能放入当前容量为j的最大价值（dp数组表示）
int main()
{
    int n,m;cin >> n >> m;
    for (int i = 1; i <= n; i ++ )cin >> v[i] >> w[i];
    
    for (int i = 1; i <= n;i ++){
        for(int j = m;j >= 0;j --){
            dp[j] = dp[j];
            if(j-v[i] >= 0)dp[j] = max(dp[j-v[i]] + w[i],dp[j]);
        }
    }
    cout << dp[m];
} 
*/
/*
完全背包
const int N = 1010;

int n, m;
int v[N], w[N];
int f[N];

int main()
{
    cin >> n >> m;
    for (int i = 1; i <= n; i ++ ) cin >> v[i] >> w[i];

    for (int i = 1; i <= n; i ++ )
        for (int j = v[i]; j <= m; j ++ )
            f[j] = max(f[j], f[j - v[i]] + w[i]);

    cout << f[m] << endl;

    return 0;
}
*/
/*
最长上升子序列O（NlogN）
const int N = 1e6 + 10;
int a[N],dp[N],cnt;
int main()
{
    int n;cin >> n;
    for(int i = 0;i < n;i++)scanf("%d",&a[i]);
    
    dp[cnt++] = a[0];
    for(int i = 1;i < n;i++){
        if(a[i] > dp[cnt - 1])dp[cnt++] = a[i];
        else{
            int l = 0,r = cnt - 1;
            while(l < r){//替换掉第一个大于或者等于这个数字的那个数
                int mid = (l + r)/2;
                if(dp[mid] >= a[i])r = mid;
                else l = mid + 1;
            }
            dp[r] = a[i];
        }
        //cout << cnt << endl;
    }
    cout << cnt;
}
*/
/*
区间dp
const int N = 310;

int n;
int s[N];
int f[N][N];

int main()
{
    scanf("%d", &n);
    for (int i = 1; i <= n; i ++ ) scanf("%d", &s[i]);

    for (int i = 1; i <= n; i ++ ) s[i] += s[i - 1];

    for (int len = 2; len <= n; len ++ )
        for (int i = 1; i + len - 1 <= n; i ++ )
        {
            int l = i, r = i + len - 1;
            f[l][r] = 1e8;
            for (int k = l; k < r; k ++ )
                f[l][r] = min(f[l][r], f[l][k] + f[k + 1][r] + s[r] - s[l - 1]);
        }

    printf("%d\n", f[1][n]);
    return 0;
}

*/

/*
数位dp的两种方式
#include <bits/stdc++.h>
using namespace std;
using ll = long long;
ll A[22], cnt, digit, dp[22][22][2][2];
ll dfs(int pos, int cntd, bool limit, bool lead) // cntd表示目前为止已经找到多少个digit
{
    ll ans = 0;
    if (pos == cnt)
        return cntd;
    if (dp[pos][cntd][limit][lead] != -1)
        return dp[pos][cntd][limit][lead];
    for (int v = 0; v <= (limit ? A[pos] : 9); ++v)
        if (lead && v == 0)
            ans += dfs(pos + 1, cntd, limit && v == A[pos], true);
        else
            ans += dfs(pos + 1, cntd + (v == digit), limit && v == A[pos], false);
    dp[pos][cntd][limit][lead] = ans;
    return ans;
}
ll f(ll x)
{
    cnt = 0;
    memset(dp, -1, sizeof(dp));
    memset(A, 0, sizeof(A));
    while (x)
        A[cnt++] = x % 10, x /= 10;
    reverse(A, A + cnt);
    return dfs(0, 0, true, true);
}
int main()
{
    ios::sync_with_stdio(false);
    ll x, y;
    while(cin >> x >> y && x && y){
        if(x > y)swap(x,y);
        for (int i = 0; i <= 9; ++i)
        {
            digit = i;
            ll l = f(x - 1), r = f(y);
            cout << r - l << " " ;
        }
        cout << endl;
    }
    return 0;
}
*/
/*
#include <iostream>
#include <algorithm>
#include <vector>
#define debug0(x) cout << "debug0: " << x << endl
using namespace std;

const int N = 10;


// 第一种情况：
// 000~abc-1, x，999（1000个）
// 第二种情况
// abc，x，
//     1. num[i] < x, 0
//     2. num[i] == x, 0~efg
//     3. num[i] > x, 0~999
// 需要特殊讨论的情况：
//     x为 0 时，前面的部分不能全部为 0 
//     所以就要减掉一个对应10的次数


int power10(int k)
{
    int res = 1;
    while(k--)
    {
        res *= 10;
    }
    return res;
}

int get(vector<int> a,int r,int l)
{
    int res = 0;
    for(int i = r;i >= l;i --)
    {
        res = a[i] + res * 10;
    }
    return res;
}

int count(int x,int flag)
{
    if(x == 0)return 0;
    
    vector<int> a;
    while(x)
    {
        a.push_back(x%10);
        x/=10;
    }
    
    int n = a.size(),res = 0;
    for (int i = n-1 - !flag; i >= 0; i -- )//0特殊处理首位
    {
        //前缀数字长度存在
        if(i < n-1)
        {
            res += get(a,n-1,i+1)*power10(i);
            //0的情况至少要从前缀数组里减去1，因为要从末尾1开始
            if(flag == 0)res -= power10(i);
            
        }
        //前缀数组数字为最大的情况
        if(a[i] == flag)
        {
            res += get(a,i-1,0) + 1;
        }
        else if(a[i] > flag)
        {
            res += power10(i);
        }
    }
    return res;
}

int main()
{
    int a, b;
    while (cin >> a >> b , a&&b)
    {
        if (a > b) swap(a, b);

        for (int i = 0; i <= 9; i ++ )
            cout << count(b, i) - count(a - 1, i) << ' ';
        cout << endl;
    }

    return 0;
}
*/
/*
1.	输出值与预期对比
2.	造特殊样例（数字范围，少量但全面的数据）

*/
/*

//一般定义精度，根据题意可以适当改大或者改小，在精度要求较高的题目需要使用
//long double 的输入输出
scanf("%Lf" , &a);
printf("%.10Lf" , a);
//常用函数:fabsl(a),cosl(a).....
//即在末尾加上了字母l
//常数定义
const double eps = 1e-8;
const double PI = acos(-1.0);

int sgn(double x)//符号函数，eps使用最多的地方
{
    if (fabs(x) < eps)
        return 0;
    if (x < 0)
        return -1;
    else
        return 1;
}

//点类及其相关操作
struct Point
{
    double x, y;
    Point() {}
    Point(double _x, double _y) : x(_x), y(_y) {}
    Point operator-(const Point &b) const { return Point(x - b.x, y - b.y); }
    Point operator+(const Point &b) const { return Point(x + b.x, y + b.y); }

    double operator^(const Point &b) const { return x * b.y - y * b.x; } //叉积
    double operator*(const Point &b) const { return x * b.x + y * b.y; } //点积

    bool operator<(const Point &b) const { return x < b.x || (x == b.x && y < b.y); }
    bool operator==(const Point &b) const { return sgn(x - b.x) == 0 && sgn(y - b.y) == 0; }

    Point Rotate(double B, Point P) //绕着点P，逆时针旋转角度B(弧度)
    {
        Point tmp;
        tmp.x = (x - P.x) * cos(B) - (y - P.y) * sin(B) + P.x;
        tmp.y = (x - P.x) * sin(B) + (y - P.y) * cos(B) + P.y;
        return tmp;
    }
};

double dist(Point a, Point b) { return sqrt((a - b) * (a - b)); } //两点间距离
double len(Point a){return sqrt(a.x * a.x + a.y * a.y);}//向量的长度

struct Line
{
    Point s, e;
    Line() {}
    Line(Point _s, Point _e) : s(_s), e(_e) {}

    //两直线相交求交点
    //第一个值为0表示直线重合，为1表示平行,为2是相交
    //只有第一个值为2时，交点才有意义

    pair<int, Point> operator&(const Line &b) const
    {
        Point res = s;
        if (sgn((s - e) ^ (b.s - b.e)) == 0)
        {
            if (sgn((s - b.e) ^ (b.s - b.e)) == 0)
                return make_pair(0, res); //重合
            else
                return make_pair(1, res); //平行
        }
        double t = ((s - b.s) ^ (b.s - b.e)) / ((s - e) ^ (b.s - b.e));
        res.x += (e.x - s.x) * t;
        res.y += (e.y - s.y) * t;
        return make_pair(2, res);
    }
};

//判断线段是否相交
bool inter(Line l1, Line l2)
{
    return max(l1.s.x, l1.e.x) >= min(l2.s.x, l2.e.x) &&
            max(l2.s.x, l2.e.x) >= min(l1.s.x, l1.e.x) &&
            max(l1.s.y, l1.e.y) >= min(l2.s.y, l2.e.y) &&
            max(l2.s.y, l2.e.y) >= min(l1.s.y, l1.e.y) &&
            sgn((l2.s - l1.e) ^ (l1.s - l1.e)) * sgn((l2.e - l1.e) ^ (l1.s - l1.e)) <= 0 &&
            sgn((l1.s - l2.e) ^ (l2.s - l2.e)) * sgn((l1.e - l2.e) ^ (l2.s - l2.e)) <= 0;
}

//判断直线和线段是否相交
bool Seg_inter_line(Line l1, Line l2)
{
    return sgn((l2.s - l1.e) ^ (l1.s - l1.e)) * sgn((l2.e - l1.e) ^ (l1.s - l1.e)) <= 0;
}

//求点到直线的距离
//返回(点到直线上最近的点，垂足)
Point PointToLine(Point P, Line L)
{
    Point result;
    double t = ((P - L.s) * (L.e - L.s)) / ((L.e - L.s) * (L.e - L.s));
    result.x = L.s.x + (L.e.x - L.s.x) * t;
    result.y = L.s.y + (L.e.y - L.s.y) * t;
    return result;
}

//求点到线段的距离
//返回点到线段上最近的点
Point NearestPointToLineSeg(Point P, Line L)
{
    Point result;
    double t = ((P - L.s) * (L.e - L.s)) / ((L.e - L.s) * (L.e - L.s));
    if (t >= 0 && t <= 1)
    {
        result.x = L.s.x + (L.e.x - L.s.x) * t;
        result.y = L.s.y + (L.e.y - L.s.y) * t;
    }
    else
    {
        if (dist(P, L.s) < dist(P, L.e))
            result = L.s;
        else
            result = L.e;
    }
    return result;
}

//计算多边形面积,点的编号从0~n-1
double CalcArea(Point p[], int n)
{
    double res = 0;
    for (int i = 0; i < n; i++)
        res += (p[i] ^ p[(i + 1) % n]) / 2;
    return fabs(res);
}

//*判断点在线段上
bool OnSeg(Point P, Line L)
{
    return sgn((L.s - P) ^ (L.e - P)) == 0 &&
            sgn((P.x - L.s.x) * (P.x - L.e.x)) <= 0 &&
            sgn((P.y - L.s.y) * (P.y - L.e.y)) <= 0;
}

//求凸包Andrew算法
//p为点的编号
//n为点的数量
//ch为生成的凸包上的点
//返回凸包大小
int ConvexHull(Point *p, int n, Point *ch) //求凸包
{
    sort(p, p + n);
    n = unique(p, p + n) - p; //去重
    int m = 0;
    for (int i = 0; i < n; ++i)
    {
        while (m > 1 && sgn((ch[m - 1] - ch[m - 2]) ^ (p[i] - ch[m - 1])) <= 0)
            --m;
        ch[m++] = p[i];
    }
    int k = m;
    for (int i = n - 2; i >= 0; i--)
    {
        while (m > k && sgn((ch[m - 1] - ch[m - 2]) ^ (p[i] - ch[m - 1])) <= 0)
            --m;
        ch[m++] = p[i];
    }
    if (n > 1)
        m--;
    return m;
}

//极角排序
//叉积：对于 tmp = a x b
//如果b在a的逆时针(左边):tmp > 0
//顺时针(右边): tmp < 0
//同向: tmp = 0
//相对于原点的极角排序
//如果是相对于某一点x,只需要把x当作原点即可
bool mycmp(Point a, Point b)
{
    if (atan2(a.y, a.x) != atan2(b.y, b.x))
        return atan2(a.y, a.x) < atan2(b.y, b.x);
    else
        return a.x < b.x;
}

//判断点在凸多边形内
//要求
//点形成一个凸包，而且按逆时针排序
//如果是顺时针把里面的<0改为>0
//点的编号:0~n-1
//返回值：
//-1:点在凸多边形外
//0:点在凸多边形边界上
//1:点在凸多边形内
int inConvexPoly(Point a, Point p[], int n)
{
    for (int i = 0; i < n; i++)
    {
        if (sgn((p[i] - a) ^ (p[(i + 1) % n] - a)) < 0)
            return -1;
        else if (OnSeg(a, Line(p[i], p[(i + 1) % n])))
            return 0;
    }
    return 1;
}

//判断点是否在凸包内
bool inConvex(Point A, Point *p, int tot)
{
    int l = 1, r = tot - 2, mid;
    while (l <= r)
    {
        mid = (l + r) >> 1;
        double a1 = (p[mid] - p[0]) ^ (A - p[0]);
        double a2 = (p[mid + 1] - p[0]) ^ (A - p[0]);
        if (a1 >= 0 && a2 <= 0)
        {
            if (((p[mid + 1] - p[mid]) ^ (A - p[mid])) >= 0)
                return true;
            return false;
        }
        else if (a1 < 0)
            r = mid - 1;
        else
            l = mid + 1;
    }
    return false;
}

//判断点在任意多边形内
//射线法，poly[]的顶点数要大于等于3,点的编号0~n-1
//返回值
//-1:点在凸多边形外
//0:点在凸多边形边界上
//1:点在凸多边形内
int inPoly(Point p, Point poly[], int n)
{
    int cnt;
    Line ray, side;
    cnt = 0;
    ray.s = p;
    ray.e.y = p.y;
    ray.e.x = -100000000000.0; //-INF,注意取值防止越界

    for (int i = 0; i < n; i++)
    {
        side.s = poly[i];
        side.e = poly[(i + 1) % n];

        if (OnSeg(p, side))
            return 0;

        //如果平行轴则不考虑
        if (sgn(side.s.y - side.e.y) == 0)
            continue;

        if (OnSeg(side.s, ray))
        {
            if (sgn(side.s.y - side.e.y) > 0)
                cnt++;
        }
        else if (OnSeg(side.e, ray))
        {
            if (sgn(side.e.y - side.s.y) > 0)
                cnt++;
        }
        else if (inter(ray, side))
            cnt++;
    }
    if (cnt % 2 == 1)
        return 1;
    else
        return -1;
}

//判断凸多边形
//允许共线边
//点可以是顺时针给出也可以是逆时针给出
//但是乱序无效
//点的编号0，n-1
bool isconvex(Point poly[], int n)
{
    bool s[3];
    memset(s, false, sizeof(s));
    for (int i = 0; i < n; i++)
    {
        s[sgn((poly[(i + 1) % n] - poly[i]) ^ (poly[(i + 2) % n] - poly[i])) + 1] = true;
        if (s[0] && s[2])
            return false;
    }
    return true;
}

//判断凸包是否相离
//凸包a：n个点,凸包b：m个点
//凸包上的点不能出现在另一个凸包内
//凸包上的线段两两不能相交
bool isConvexHullSeparate(int n, int m, Point a[], Point b[])
{
    for (int i = 0; i < n; i++)
        if (inPoly(a[i], b, m) != -1)
            return false;

    for (int i = 0; i < m; i++)
        if (inPoly(b[i], a, n) != -1)
            return false;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            Line l1 = Line(a[i], a[(i + 1) % n]);
            Line l2 = Line(b[j], b[(j + 1) % m]);
            if (inter(l1, l2))
                return false;
        }
    }
    return true;
}

*/