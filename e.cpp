#include<bits/stdc++.h>
#define debug1(a) cout << #a << '=' << a << endl;
#define debug2(a, b) cout << #a << " = " << a << "  " << #b << " = " << b << endl;
#define debug3(a, b, c) cout << #a << " = " << a << "  " << #b << " = " << b << "  " << #c << " = " << c << endl;
#define debug4(a, b, c, d) cout << #a << " = " << a << "  " << #b << " = " << b << "  " << #c << " = " << c << "  " << #d << " = " << d << endl;
#define debug5(a, b, c, d, e) cout << #a << " = " << a << "  " << #b << " = " << b << "  " << #c << " = " << c << "  " << #d << " = " << d << "  " << #e << " = " << e << endl;
// #define int long long
using namespace std;

/*--------------------Cut-Line Copy_is_Forbiden---------------------*/

class Solution
{
public:
    typedef pair<int, int> PII;
    int minimumCost(vector<int> &start, vector<int> &target, vector<vector<int>> &specialRoads)
    {
        int cntp = specialRoads.size() * 2 + 2;
        vector<PII> p(cntp);
        p[0] = {start[0], start[1]};
        p[1] = {target[0], target[1]};
        for (int i = 0; i < specialRoads.size();i ++)
        {
            p[i * 2 + 2] = {specialRoads[i][0], specialRoads[i][1]};
            p[i * 2 + 3] = {specialRoads[i][2], specialRoads[i][3]};
        }

        sort(p.begin(), p.end());
        p.erase(unique(p.begin(), p.end()),p.end());
        cntp = p.size();
        debug1(cntp);
        vector<vector<int>> edge(cntp, vector<int>(cntp));
        for (int i = 0; i < cntp;i ++)
        {
            for (int j = 0; j < cntp;j ++)
            {
                edge[i][j] = abs(p[i].first - p[j].first) + abs(p[i].second - p[j].second);
            }
        }
        for (int i = 0; i < specialRoads.size();i ++)
        {
            PII a = {specialRoads[i][0], specialRoads[i][1]};
            PII b = {specialRoads[i][2], specialRoads[i][3]};
            int w = specialRoads[i][4];
            int ia = lower_bound(p.begin(), p.end(), a) - p.begin();
            int ib = lower_bound(p.begin(), p.end(), b) - p.begin();
            edge[ia][ib] = min(edge[ia][ib], w);
        }
        int istart = lower_bound(p.begin(), p.end(), make_pair(start[0], start[1])) - p.begin();
        int itarget = lower_bound(p.begin(), p.end(), make_pair(target[0], target[1])) - p.begin();

        vector<int> dist(cntp, 1e9),st(cntp,0);
        dist[istart] = 0;

        for (int i = 0; i < cntp;i ++)
        {
            int t = -1;
            for (int j = 0; j < cntp;j ++)
            {
                if(!st[j] && (t == -1 || dist[t] > dist[j]))
                    t = j;
            }
            for (int j = 0; j < cntp; j++)
            {
                dist[j] = min(dist[j], dist[t] + edge[t][j]);
            }
            st[t] = 1;
            debug4(dist[t], t, p[t].first, p[t].second);
        }

        return dist[itarget];
    }
};