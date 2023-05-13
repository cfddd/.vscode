#include <iostream>
#include <cstring>
#include <vector>
#define debug3(a, b, c) cout << #a << " = " << a << " " << #b << " = " << b << " " << #c << " = " << c << endl
using namespace std;
using LL = long long;

template <const int T>
struct ModInt
{
    const static int mod = T;
    int x;
    ModInt(int x = 0) : x(x % mod) {}
    ModInt(long long x) : x(int(x % mod)) {}
    int val() { return x; }
    ModInt operator+(const ModInt &a) const
    {
        int x0 = x + a.x;
        return ModInt(x0 < mod ? x0 : x0 - mod);
    }
    ModInt operator-(const ModInt &a) const
    {
        int x0 = x - a.x;
        return ModInt(x0 < 0 ? x0 + mod : x0);
    }
    ModInt operator*(const ModInt &a) const { return ModInt(1LL * x * a.x % mod); }
    ModInt operator/(const ModInt &a) const { return *this * a.inv(); }
    bool operator==(const ModInt &a) const { return x == a.x; };
    bool operator!=(const ModInt &a) const { return x != a.x; };
    void operator+=(const ModInt &a)
    {
        x += a.x;
        if (x >= mod)
            x -= mod;
    }
    void operator-=(const ModInt &a)
    {
        x -= a.x;
        if (x < 0)
            x += mod;
    }
    void operator*=(const ModInt &a) { x = 1LL * x * a.x % mod; }
    void operator/=(const ModInt &a) { *this = *this / a; }
    friend ModInt operator+(int y, const ModInt &a)
    {
        int x0 = y + a.x;
        return ModInt(x0 < mod ? x0 : x0 - mod);
    }
    friend ModInt operator-(int y, const ModInt &a)
    {
        int x0 = y - a.x;
        return ModInt(x0 < 0 ? x0 + mod : x0);
    }
    friend ModInt operator*(int y, const ModInt &a) { return ModInt(1LL * y * a.x % mod); }
    friend ModInt operator/(int y, const ModInt &a) { return ModInt(y) / a; }
    friend ostream &operator<<(ostream &os, const ModInt &a) { return os << a.x; }
    friend istream &operator>>(istream &is, ModInt &t) { return is >> t.x; }

    ModInt pow(int64_t n) const
    {
        ModInt res(1), mul(x);
        while (n)
        {
            if (n & 1)
                res *= mul;
            mul *= mul;
            n >>= 1;
        }
        return res;
    }

    ModInt inv() const
    {
        int a = x, b = mod, u = 1, v = 0;
        while (b)
        {
            int t = a / b;
            a -= t * b;
            swap(a, b);
            u -= t * v;
            swap(u, v);
        }
        if (u < 0)
            u += mod;
        return u;
    }
};
using mint = ModInt<998244353>;

int main()
{

    cin.tie(0);
    cout.tie(0);
    ios::sync_with_stdio(0);

    int n, x, y, m, k;
    cin >> n;
    n = 10;
    vector<int> a(n);
    cin >> a[0] >> x >> y >> m >> k;
    for (int i = 1; i < n; i++)
        a[i] = (1LL * a[i - 1] * x + y) % m,debug3(i,a[i],0);
    vector<vector<mint>> b(k + 1, vector<mint>(n));
    for (int i = 0; i < n; i++)
        b[0][i] = a[i];
    for (int i = 1; i < n; i++)
        b[0][i] += b[0][i - 1];
    for (int i = 1; i <= k; i++)
    {
        
        b[i][0] = i > 1 ? 0 : a[0];
        for (int j = 1; j < n; j++)
        {
            
            b[i][j] = b[i][j - 1] + b[i - 1][j - 1] + (i > 1 ? 0 : a[j]);
            debug3(j, i, b[i][j]);
        }
    }
    LL ans = 0;
    for (int i = 0; i < n; i++)
        ans ^= 1LL * b[k][i].val() * (i + 1),debug3(k,i,b[k][i].val());
    cout << ans << '\n';
}