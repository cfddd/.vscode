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

// 常数定义
const double PI = acos(-1.0);
LL qmi(LL a, int b) // 快速幂在于思想
{
    LL res = 1;
    while (b) // 对b进行二进制化,从低位到高位
    {
        if (b & 1)
            res = res * a;
        b >>= 1;
        a = a * a;
    }
    return res;
}
void solve()
{
    int p, q;
    cin >> p >> q;

    if (p % q)
    {
        cout << p << endl;
        return;
    }

    map<int, int> pP, pQ;
    auto tmp = [&](int start, map<int, int> &primes)
    {
        int cnt = 0;
        for (int i = 2; i * i <= start; i++)
            if (start % i == 0)
            {
                cnt = 0;
                while (start % i == 0)
                {
                    cnt++;
                    start /= i;
                }
                if (cnt)
                    primes[i] = cnt;
            }
        if (start > 1)
            primes[start]++;
    };

    // tmp(p, pP);
    tmp(q, pQ);

    int ans = 0;
    for (PII t : pQ)
    {
        int prime = t.fi;
        int cnt = t.se;
        int cntp = 0;
        int tempp = p;
        while (tempp % prime == 0)
            tempp /= prime, cntp++;

        // debug2(prime,cnt);
        if (cnt > cntp)
            ans = max(ans, p);
        else
        {
            int temp = cntp - (cnt - 1);
            int ttt = temp;
            temp = qmi(prime, temp);
            // debug3(p,prime, temp);
            // debug2(prime * prime, temp);
            ans = max(ans, p / temp);
            // debug1(ans);
        }
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

*/
