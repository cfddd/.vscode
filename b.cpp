#include <bits/stdc++.h>
#define debug1(a) cout << #a << '=' << a << endl;
#define debug2(a, b) cout << #a << " = " << a << "  " << #b << " = " << b << endl;
#define debug3(a, b, c) cout << #a << " = " << a << "  " << #b << " = " << b << "  " << #c << " = " << c << endl;
#define debug4(a, b, c, d) cout << #a << " = " << a << "  " << #b << " = " << b << "  " << #c << " = " << c << "  " << #d << " = " << d << endl;
#define debug5(a, b, c, d, e) cout << #a << " = " << a << "  " << #b << " = " << b << "  " << #c << " = " << c << "  " << #d << " = " << d << "  " << #e << " = " << e << endl;
#define endl "\n"
#define fi first
#define se second
#define caseT \
    int T;    \
    cin >> T; \
    while (T--)
#define int long long
// #define int __int128
using namespace std;

typedef long long LL;
typedef unsigned long long ULL;
typedef pair<int, int> PII;
typedef pair<LL, LL> PLL;

const double PI = acos(-1.0);
const int N = 1e7 + 10;
const int mod = 998244353;

int b[N][6];
int n, x, y, m, k;
int a[N];

void solve()
{
    cin >> n >> a[1] >> x >> y >> m >> k;
    for (int i = 2; i <= n;i ++)
        a[i] = (a[i - 1] * x + y) % m;//m!,not mod!
    for (int i = 1,suma = 0; i <= n;i ++)
    {
        suma = (suma + a[i]) % mod;
        b[i][1] = (b[i - 1][1] + suma) % mod;
        // debug3(i, 1, b[i][1]);
    }
    for (int i = 2; i <= k;i ++)
    {
        for (int j = 1; j <= n;j ++)
        {
            b[j][i] = (b[j - 1][i - 1] + b[j - 1][i]) % mod;
            // debug3(j, i, b[j][i]);
        }
    }

    int ans = 0;
    for (int j = 1; j <= n; j++)
    {
        ans ^= b[j][k] * j;
        // debug3(j, k, b[j][k]);
    }
    cout << ans << endl;
}

signed main()
{

    ios::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);

    // caseT
    solve();

    return 0;
}
/*
c b b ad abcd ac n y y
*/
